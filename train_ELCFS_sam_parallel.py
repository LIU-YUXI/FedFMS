import os
# os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, [0,1,2,3,4,5,6,7])) # 一般在程序开头设置
import sys
from tqdm import tqdm
from tensorboardX import SummaryWriter
import shutil
import argparse
import logging
import time
import random
import numpy as np
import collections
from collections import OrderedDict
from glob import glob
import cv2
import torch
from torch.autograd import Variable
import torch.optim as optim
from torchvision import transforms
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from pytorch_metric_learning import losses

from networks.unet2d import Unet2D
from utils.losses import dice_loss
from utils.util import _eval_dice, _eval_haus, _connectivity_region_analysis, parse_fn_haus
from dataloaders.fundus_dataloader import Dataset, ToTensor
from torch.utils.data.distributed import DistributedSampler
from models.sam import SamPredictor, sam_model_registry
from models.sam.utils.transforms import ResizeLongestSide
from sam_utils import *
# net = sam_model_registry['vit_b'](args,checkpoint=args.sam_ckpt).to(device)
import copy
torch.distributed.init_process_group(backend="nccl")
local_rank = int(os.environ['LOCAL_RANK'])
torch.cuda.set_device(local_rank)
device = torch.device("cuda", local_rank)
import torch.distributed as dist
'''
parser = argparse.ArgumentParser()
parser.add_argument('--exp', type=str,  default='xxxx', help='model_name')
parser.add_argument('--max_epoch', type=int,  default=100, help='maximum epoch number to train')
parser.add_argument('--client_num', type=int, default=9, help='batch_size per gpu')
parser.add_argument('--batch_size', type=int, default=10, help='batch_size per gpu')
parser.add_argument('--image_size', type=int, default=1024, help='image_size')
parser.add_argument('--clip_value', type=float,  default=100, help='maximum epoch number to train')
parser.add_argument('--meta_step_size', type=float,  default=1e-3, help='maximum epoch number to train')
parser.add_argument('--base_lr', type=float,  default=0.001, help='maximum epoch number to train')
parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
parser.add_argument('--gpu', type=str,  default='0', help='GPU to use')
parser.add_argument('--display_freq', type=int, default=5, help='batch_size per gpu')
parser.add_argument('--unseen_site', type=int, default=8, help='batch_size per gpu')
parser.add_argument('--pretrain', type=str, default=None, help='pretrain model')
parser.add_argument('--sam_ckpt', type=str, default=None, help='pretrain Sam model')
args = parser.parse_args()
'''
import cfg
args = cfg.parse_args()
snapshot_path = "../output2/" + args.exp + "/"
batch_size = args.batch_size * len(args.gpu.split(','))
meta_step_size = args.meta_step_size
clip_value = args.clip_value
base_lr = args.base_lr
client_num = args.client_num
max_epoch = args.max_epoch
display_freq = args.display_freq
# ？？？
client_name = ['1', '4', '5', '6', '13', '16', '18', '20', '21']
data_path = '/mnt/diskB/lyx/FeTS2022_FedDG_1024'
# 还要生成test数据
client_data_list = []
slice_num =[]
for client_idx in range(client_num):
    print('{}/{}/data_npy/*'.format(data_path,client_name[client_idx]))
    client_data_list.append(glob('{}/{}/data_npy/*'.format(data_path,client_name[client_idx])))
    print (len(client_data_list[client_idx]))
    slice_num.append(len(client_data_list[client_idx]))
slice_num = np.array(slice_num)
#volume_size = [384, 384, 3]
unseen_site_idx = args.unseen_site
source_site_idx = [i for i in range(client_num)]
source_site_idx.remove(unseen_site_idx)
client_weight = slice_num[source_site_idx] / np.sum(slice_num[source_site_idx])
client_weight = np.round(client_weight, decimals=2)
client_weight[-1] = 1 - np.sum(client_weight[:-1])
client_weight = np.insert(client_weight, unseen_site_idx, 0)
print(client_weight)
# client_weight= np.full((client_num,), 1/client_num)
# client_weight[-1] = 1 - np.sum(client_weight[:2])
if args.deterministic:
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
print(torch.__version__)
print(torch.cuda.is_available())
if torch.cuda.is_available():
    # 获取可用的 CUDA 设备数量
    num_devices = torch.cuda.device_count()
    print(f"可用的 CUDA 设备数量: {num_devices}")

    # 遍历输出每个设备的名称
    for i in range(num_devices):
        device_name = torch.cuda.get_device_name(i)
        print(f"CUDA 设备 {i}: {device_name}")
else:
    print("没有可用的 CUDA 设备")
# 一样的模型
def update_global_model(net_clients, client_weight):
    # client_num=4
    # Use the true average until the exponential average is more correct
    for param in zip(*list(net_clients[i].parameters() for i in range(client_num))):
    #for param in zip(net_clients[0].parameters(), net_clients[1].parameters(), 
    #                 net_clients[2].parameters(), net_clients[3].parameters()):
        # print(param,param[0])
        new_para = Variable(torch.Tensor(np.zeros(param[0].shape)), requires_grad=False).cuda(local_rank) 
        for i in range(client_num):
            new_para.data.add_(client_weight[i], param[i].data)

        for i in range(client_num):
            param[i].data.mul_(0).add_(new_para.data)
# 为什么要求平均值？
def extract_contour_embedding(contour_list, embeddings):

    average_embeddings_list = []
    for contour in contour_list:
        # print('contour.shape,embeddings.shape',contour.shape,embeddings.shape)
        contour_embeddings = contour * embeddings
        average_embeddings = torch.sum(contour_embeddings, (-1,-2))/(torch.sum(contour, (-1,-2))+1e-8)
        # print (1,contour.shape)
        # print (2,embeddings.shape)
        # print (3,contour_embeddings.shape)
        # print (4,average_embeddings.shape)
        '''
        1 torch.Size([5, 1, 384, 384])
        2 torch.Size([5, 32, 384, 384])
        3 torch.Size([5, 32, 384, 384])
        4 torch.Size([5, 32])
        '''
        average_embeddings_list.append(average_embeddings)
    return average_embeddings_list
'''
def test(site_index, test_net):

    test_data_list = client_data_list[site_index]

    dice_array = []
    haus_array = []
    # print(test_data_list)
    for fid, filename in enumerate(test_data_list):
        # print(fid)
        data = np.load(filename)
        # why expand_dims?
        image = np.expand_dims(data[..., :3].transpose(2, 0, 1), axis=0)
        mask = np.expand_dims(data[..., 3:].transpose(2, 0, 1), axis=0)
        image = torch.from_numpy(image).float()
        pt = random_click(np.array(mask) / 255, 1, 1)
        logit, pred, _ = test_net.test(image,pt,multimask_output=False)
        pred_y = pred.cpu().detach().numpy()
        pred_y[pred_y>0.75] = 1
        pred_y[pred_y<0.75] = 0

        pred_y_0 = pred_y[:, 0:1, ...]
        # pred_y_1 = pred_y[:, 1:, ...]
        processed_pred_y_0 = _connectivity_region_analysis(pred_y_0)
        #processed_pred_y_1 = _connectivity_region_analysis(pred_y_1)
        # processed_pred_y = np.concatenate([processed_pred_y_0, processed_pred_y_1], axis=1)
        processed_pred_y=processed_pred_y_0
        dice_subject = _eval_dice(mask, processed_pred_y)
        # haus_subject = _eval_haus(mask, processed_pred_y)
        dice_array.append(dice_subject)
        # haus_array.append(haus_subject)
    dice_array = np.array(dice_array)
    # print (dice_array.shape)
    dice_avg = np.mean(dice_array, axis=0).tolist()
    # print (dice_avg)
    # haus_avg = np.mean(haus_array, axis=0).tolist()[0]
    # logging.info("OD dice_avg %.4f OC dice_avg %.4f" % (dice_avg[0], dice_avg[1]))
    logging.info("OD dice_avg %.4f" % (dice_avg[0]))
    return dice_avg, dice_array, 0, [0,0]
'''
def test(site_index, test_net):

    test_data_list = client_data_list[site_index]

    dice_array = []
    haus_array = []
    eiou_array = []
    # print(test_data_list)
    test_net.eval()
    ave_res, mix_res = (0,0,0,0), (0,0,0,0)
    for fid, filename in enumerate(test_data_list):
        # print(fid)s
        data = np.load(filename)/ 255.0
        # print('data',data)
        mask_data = np.load(filename.replace("data", "label"))
        # why expand_dims?
        image = np.expand_dims(data[..., :3].transpose(2, 0, 1), axis=0)
        mask = np.expand_dims(mask_data.transpose(2, 0, 1), axis=0)
        # print('mask_data.shape,mask.shape',mask_data.shape,mask.shape)
        # mask在另外一个地方文件夹。。。
        image = torch.from_numpy(image).float()
        mask = torch.from_numpy(mask)
        # print('dmi',mask)
        mask_data=np.squeeze(mask_data)
        mask_data=cv2.resize(mask_data,(image.shape[-1],image.shape[-1]),interpolation=cv2.INTER_NEAREST)
        pt = np.expand_dims(random_click(np.array(mask_data), 1, 1), axis=0)
        # print('pt',pt)
        point_labels = torch.ones(image.size(0))
        if point_labels[0] != -1:
            # point_coords = samtrans.ResizeLongestSide(longsize).apply_coords(pt, (h, w))
            point_coords = pt
            coords_torch = torch.as_tensor(point_coords, dtype=torch.float, device=local_rank)
            labels_torch = torch.as_tensor(point_labels, dtype=torch.int, device=local_rank)
            coords_torch, labels_torch = coords_torch[None, :, :], labels_torch[None, :]
            pt = (coords_torch, labels_torch)
            # print('pt',pt[0].shape,pt[1].shape)
            # imge= net.image_encoder(imgs)
        se, de = test_net.prompt_encoder(
            points=pt,
            boxes=None,
            masks=None
        )
        image,mask = image.cuda(local_rank), mask.cuda(local_rank)
        image_encoded= test_net.image_encoder(image)
        pred, _, _ = test_net.mask_decoder(
            image_embeddings=image_encoded,
            image_pe=test_net.prompt_encoder.get_dense_pe(), #  1x(embed_dim)x(embedding_h)x(embedding_w)
            sparse_prompt_embeddings=se,
            dense_prompt_embeddings=de, 
            multimask_output=False,
        )
        threshold = (0.1, 0.3, 0.5, 0.7, 0.9)
        
        temp = eval_seg(pred, mask, threshold)
        # print(temp)
        # mix_res = tuple([sum(a) for a in zip(mix_res, temp)])
        
        '''
        # print('pred_y.shape',pred.shape)
        # logit, pred, _ = test_net.test(image,pt,multimask_output=False)
        pred_y = pred.cpu().detach().numpy()
        pred_y[pred_y>0.75] = 1
        pred_y[pred_y<0.75] = 0

        pred_y_0 = pred_y[:, 0:1, ...]
        # pred_y_1 = pred_y[:, 1:, ...]
        processed_pred_y_0 = _connectivity_region_analysis(pred_y_0)
        #processed_pred_y_1 = _connectivity_region_analysis(pred_y_1)
        # processed_pred_y = np.concatenate([processed_pred_y_0, processed_pred_y_1], axis=1)
        processed_pred_y=processed_pred_y_0
        dice_subject = _eval_dice(mask.cpu().numpy(), processed_pred_y)
        # haus_subject = _eval_haus(mask, processed_pred_y)
        dice_array.append(dice_subject)
        # haus_array.append(haus_subject)
        '''
        eiou, edice = temp
        dice_array.append(edice)
        eiou_array.append(eiou)
        
    dice_array = np.array(dice_array)
    eiou_array = np.array(eiou_array)
    # print (dice_array.shape)
    dice_avg = np.mean(dice_array, axis=0).tolist()
    eiou_avg = np.mean(eiou_array, axis=0).tolist()
    # print (dice_avg)
    # haus_avg = np.mean(haus_array, axis=0).tolist()[0]
    # logging.info("OD dice_avg %.4f OC dice_avg %.4f" % (dice_avg[0], dice_avg[1]))
    # logging.info("OD dice_avg %.4f, Eiou %.4f" % (dice_avg, eiou_avg))
    return dice_avg, dice_array, eiou_avg, eiou_array

class FedModel(torch.nn.Module):
    """Define a Federated learning model
    
    Attributes:
    """
    def __init__(self, net_current):
        super(FedModel, self).__init__()
        self.net_current = net_current
        pos_weight = torch.ones([1])*2
        criterion_G = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.lossfunc = criterion_G
    @property
    def device(self):
        return self.net_current.device
    def forward(self, volume_batch_raw, volume_batch_trs_1,disc_contour, disc_bg, cup_contour, cup_bg, pt):
        point_labels = torch.ones(volume_batch_raw.size(0))
        if point_labels[0] != -1:
            # point_coords = samtrans.ResizeLongestSide(longsize).apply_coords(pt, (h, w))
            point_coords = pt
            coords_torch = torch.as_tensor(point_coords, dtype=torch.float, device=self.device)
            labels_torch = torch.as_tensor(point_labels, dtype=torch.int, device=self.device)
            coords_torch, labels_torch = coords_torch[None, :, :], labels_torch[None, :]
            pt = (coords_torch, labels_torch)
            # print('pt',pt[0].shape,pt[1].shape)
        with torch.no_grad():
            # imge= net.image_encoder(imgs)
            se, de = self.net_current.prompt_encoder(
                points=pt,
                boxes=None,
                masks=None
            )
        # print('parameters_name',net_current.image_encoder.named_parameters())
        parameters_to_calculate_grad = []
        parameters_name_to_calculate_grad = []
        for n, value in self.net_current.image_encoder.named_parameters():
            if "Adapter" not in n:
                value.requires_grad = False
        # batch_size, out_chans=256, H', W', 感觉out_chans可以小一点
        volume_batch_raw_encoded= self.net_current.image_encoder(volume_batch_raw)
        outputs_soft_inner, masks_inner_embedding, _ = self.net_current.mask_decoder(
            image_embeddings=volume_batch_raw_encoded,
            image_pe=net.prompt_encoder.get_dense_pe(), #  1x(embed_dim)x(embedding_h)x(embedding_w)
            sparse_prompt_embeddings=se,
            dense_prompt_embeddings=de, 
            multimask_output=False,
        )
        loss_inner = self.lossfunc(outputs_soft_inner, label_batch)
        for n, value in net_current.image_encoder.named_parameters():
            if "Adapter" in n:
                parameters_to_calculate_grad.append(value)
                parameters_name_to_calculate_grad.append((n,value))
        grads = torch.autograd.grad(loss_inner, parameters_to_calculate_grad, retain_graph=True,allow_unused=True)
        # print(grads)
        new_parameters_name_to_calculate_grad,new_grads=[],[]
        for index, value in enumerate(parameters_name_to_calculate_grad):
            n,v=value
            if(grads[index]==None):
                pass# print(n,grads[index])
            else:
                new_parameters_name_to_calculate_grad.append(value)
                new_grads.append(grads[index])
        fast_weights = OrderedDict((name, param - torch.mul(meta_step_size, torch.clamp(grad, 0-clip_value, clip_value))) for
                                            ((name, param), grad) in
                                            zip(new_parameters_name_to_calculate_grad, new_grads))
        outer_net=copy_outer_net(fast_weights,net_current)
        with torch.no_grad():
            # imge= net.image_encoder(imgs)
            se, de = outer_net.prompt_encoder(
                points=pt,
                boxes=None,
                masks=None,
            )
        for n, value in outer_net.image_encoder.named_parameters():
            if "Adapter" not in n:
                value.requires_grad = False
        # 好像有两个weights
        volume_batch_trs_1_encoded= outer_net.image_encoder(volume_batch_trs_1)
        outputs_soft_outer_1,masks_outer_embedding, _ = outer_net.mask_decoder(
            image_embeddings=volume_batch_trs_1_encoded,
            image_pe=net.prompt_encoder.get_dense_pe(), #  1x(embed_dim)x(embedding_h)x(embedding_w)
            sparse_prompt_embeddings=se,
            dense_prompt_embeddings=de, 
            multimask_output=False,
        )
        # outer loop evaluation
        # outputs_soft_outer_1, outputs_mask_outer_1, embedding_outer = net_current(volume_batch_trs_1, fast_weights) #alpha
        loss_outer_1_dice = dice_loss(outputs_soft_outer_1, label_batch)
        #print('contour',disc_contour, 'bg',disc_bg, 'contour',cup_contour, 'bg',cup_bg)
        #print('embedding',embedding_inner)
        inner_disc_ct_em, inner_disc_bg_em, inner_cup_ct_em, inner_cup_bg_em = \
            extract_contour_embedding([disc_contour, disc_bg, cup_contour, cup_bg], masks_inner_embedding)
        outer_disc_ct_em, outer_disc_bg_em, outer_cup_ct_em, outer_cup_bg_em = \
            extract_contour_embedding([disc_contour, disc_bg, cup_contour, cup_bg], masks_outer_embedding)

        disc_ct_em = torch.cat((inner_disc_ct_em, outer_disc_ct_em), 0)
        #print('inner_disc_ct_em',inner_disc_ct_em, ' outer_disc_ct_em',outer_disc_ct_em)
        disc_bg_em = torch.cat((inner_disc_bg_em, outer_disc_bg_em), 0)
        # cup_ct_em = torch.cat((inner_cup_ct_em, outer_cup_ct_em), 0)
        # cup_bg_em = torch.cat((inner_cup_bg_em, outer_cup_bg_em), 0)
        disc_em = torch.cat((disc_ct_em, disc_bg_em), 0)
        # print(disc_em)
        # cup_em = torch.cat((cup_ct_em, cup_bg_em), 0)
        label = np.concatenate([np.ones(disc_ct_em.shape[0]), np.zeros(disc_bg_em.shape[0])])
        label = torch.from_numpy(label)
        # none可能是因为除的embedding是0了。。。
        disc_cont_loss = cont_loss_func(disc_em, label)
        # cup_cont_loss = cont_loss_func(cup_em, label)
        cont_loss = disc_cont_loss # + cup_cont_loss
        loss_outer = loss_outer_1_dice + cont_loss * 0.1
        total_loss = 100 * loss_inner  + loss_outer 
        return loss_inner, loss_outer_1_dice, cont_loss, loss_outer, total_loss

def copy_outer_net(fast_weights,net_current):
    # 深拷贝net_current模型
    net_copy = copy.deepcopy(net_current)
    # 将fast_weights中的权重赋值给net_copy模型
    for name, param in net_copy.named_parameters():
        if name in fast_weights:
            param.data = fast_weights[name]
    return net_copy

if __name__ == "__main__":
    ## make logger file
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    if not os.path.exists(snapshot_path + '/model'):
        os.makedirs(snapshot_path + '/model')
    # if os.path.exists(snapshot_path + '/code'):
    #    shutil.rmtree(snapshot_path + '/code')
    # shutil.copytree('.', snapshot_path + '/code', shutil.ignore_patterns(['.git','__pycache__']))

    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    device_list = [0, 1]
    GPUdevice = torch.device('cuda', device_list[0])
    # define dataset, model, optimizer for each client
    def worker_init_fn(worker_id):
        random.seed(args.seed+worker_id)
    dataloader_clients = []
    net_clients = []
    fedmodel_clients = []
    optimizer_clients = []
    for client_idx in range(client_num):
        freq_site_idx = source_site_idx.copy()
        if client_idx != unseen_site_idx:
            freq_site_idx.remove(client_idx)
        dataset = Dataset(client_idx=client_idx, data_path=data_path,freq_site_idx=freq_site_idx,
                                split='train', transform = transforms.Compose([
                                ToTensor(),
                                ]))
        dataloader = DataLoader(dataset, batch_size=batch_size, sampler=DistributedSampler(dataset, shuffle=True), num_workers=2, drop_last=True)#, num_workers=1, pin_memory=True, worker_init_fn=worker_init_fn
        net = sam_model_registry['vit_b'](args,checkpoint=args.sam_ckpt)# .to(GPUdevice)# Unet2D(num_classes=1)
        # net = nn.DataParallel(net, device_ids=device_list) 
        # net = net.cuda()
        # net = net.cuda(GPUdevice)
        '''
        if args.pretrain is not None:
            weights = torch.load(args.pretrain)
            net.load_state_dict(weights,strict=False)
        '''
        # optimizer = torch.optim.Adam(model.parameters())
        # optimizer = torch.optim.Adam(net.parameters(), lr=args.base_lr, betas=(0.9, 0.999))
        dataloader_clients.append(dataloader)
        net_clients.append(net)
        fed_model = FedModel(net)
        fed_model.to(local_rank)
        for ps in fed_model.parameters():
            dist.broadcast(ps, 0)
        # fed_model = nn.DataParallel(fed_model, device_ids=device_list) 
        # fed_model = fed_model.cuda(device_list[0])
        fed_model = torch.nn.parallel.DistributedDataParallel(fed_model,
                                                    device_ids=[local_rank],broadcast_buffers=False,
                                                    output_device=local_rank, find_unused_parameters=True)
        optimizer = torch.optim.Adam(net.parameters(), lr=args.base_lr, betas=(0.9, 0.999))
        fedmodel_clients.append(fed_model)
        optimizer_clients.append(optimizer)
    '''
    for name, param in  net_clients[0].named_parameters():
        print (name)
    '''

    temperature = 0.05
    cont_loss_func = losses.NTXentLoss(temperature)

    # start federated learning
    writer = SummaryWriter(snapshot_path+'/log', filename_suffix='.{timestamp}')
    # dice, dice_array, haus, haus_array = test(unseen_site_idx, net_clients[unseen_site_idx])
    # print(("   OD dice is: {}, std is {}".format(dice, np.std(dice_array[:]))))
    lr_ = base_lr
    for epoch_num in tqdm(range(max_epoch), ncols=70):
        # update_global_model(net_clients, client_weight)
        for client_idx in source_site_idx:
            dataloader_current = dataloader_clients[client_idx]
            net_current = net_clients[client_idx]
            # net_current.train()
            fedmodel_current = fedmodel_clients[client_idx]
            optimizer_current = optimizer_clients[client_idx]
            time1 = time.time()
            iter_num = 0

            for i_batch, sampled_batch in enumerate(dataloader_current):
                time2 = time.time()

                # obtain training data
                volume_batch, label_batch, disc_contour, disc_bg, cup_contour, cup_bg, pt = sampled_batch['image'], sampled_batch['label'], \
                sampled_batch['disc_contour'], sampled_batch['disc_bg'], sampled_batch['cup_contour'], sampled_batch['cup_bg'], sampled_batch['pt']
                volume_batch_raw_np = volume_batch[:, :3, ...]
                volume_batch_trs_1_np = volume_batch[:, 3:6, ...]
                volume_batch_raw, volume_batch_trs_1, label_batch = \
                    volume_batch_raw_np.cuda(local_rank), volume_batch_trs_1_np.cuda(local_rank), label_batch.cuda(local_rank)
                loss_inner, loss_outer_1_dice, cont_loss, loss_outer, total_loss = fedmodel_current(volume_batch_raw, volume_batch_trs_1,disc_contour, disc_bg, cup_contour, cup_bg, pt)
                
                optimizer_current.zero_grad()
                total_loss.backward()
                optimizer_current.step()

                iter_num = iter_num + 1
                if iter_num % display_freq == 0 and local_rank==0:
                    writer.add_scalar('lr', lr_, iter_num)
                    writer.add_scalar('loss/inner', loss_inner, iter_num)
                    # writer.add_scalar('loss/outer', loss_outer, iter_num)
                    writer.add_scalar('loss/total', total_loss, iter_num)
                    logging.info('Epoch: [%d] client [%d] iteration [%d / %d] : inner loss : %f outer dice loss : %f outer cont loss : %f outer loss : %f total loss : %f' % \
                        (epoch_num, client_idx, iter_num, len(dataloader_current), loss_inner.item(), loss_outer_1_dice.item(), cont_loss.item(), loss_outer.item(), total_loss.item()))
                '''
                if iter_num % 20 == 0:
                    image = np.array(volume_batch_raw_np[0, 0:3, :, :], dtype='uint8')
                    writer.add_image('train/RawImage', image, iter_num)

                    image = np.array(volume_batch_trs_1_np[0, 0:3, :, :], dtype='uint8')
                    writer.add_image('train/TrsImage', image, iter_num)

                    image = outputs_soft_inner[0, 0:1, ...].data.cpu().numpy()
                    writer.add_image('train/RawDiskMask', image, iter_num)
                    # image = outputs_soft_inner[0, 1:, ...].data.cpu().numpy()
                    # writer.add_image('train/RawCupMask', image, iter_num)


                    image = np.array(disc_contour[0, 0:1, :, :].data.cpu().numpy())#, dtype='uint8')
                    writer.add_image('train/disc_contour', image, iter_num)

                    image = np.array(disc_bg[0, 0:1, :, :].data.cpu().numpy())#, dtype='uint8')
                    writer.add_image('train/disc_bg', image, iter_num)

                    # image = np.array(cup_contour[0, 0:1, :, :].data.cpu().numpy())#, dtype='uint8')
                    # writer.add_image('train/cup_contour', image, iter_num)

                    # image = np.array(cup_bg[0, 0:1, :, :].data.cpu().numpy())#, dtype='uint8')
                    # writer.add_image('train/cup_bg', image, iter_num)
                '''
                if iter_num % 50 ==0 and local_rank==0:
                    dice, dice_array, haus, haus_array = test(unseen_site_idx, net_current)
                    print(("   OD dice is: {}, std is {}".format(dice, np.std(dice_array[:]))))
            if local_rank==0:
                dice, dice_array, haus, haus_array = test(unseen_site_idx, net_current)
                print(("clinet OD dice is: {}, std is {}".format(dice, np.std(dice_array[:]))))
        ## model aggregation
        if(local_rank==0):
            update_global_model(net_clients, client_weight)

        ## evaluation test unseen
        with open(os.path.join(snapshot_path, 'evaluation_result.txt'), 'a') as f:
            dice_list = []
            haus_list = []
            if local_rank==0:
                print("epoch {} testing , site {}".format(epoch_num, unseen_site_idx), file=f)
                dice, dice_array, haus, haus_array = test(unseen_site_idx, net_clients[unseen_site_idx])
                print(("   OD dice is: {}, std is {}".format(dice[0], np.std(dice_array[:, 0]))), file=f)
            # print(("   OC dice is: {}, std is {}".format(dice[1], np.std(dice_array[:, 1]))), file=f)
            
        ## save model
        if local_rank==0:
            save_mode_path = os.path.join(snapshot_path + '/model', 'epoch_' + str(epoch_num) + '.pth')
            torch.save(net_clients[0].state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))
    dist.destroy_process_group()
    writer.close()


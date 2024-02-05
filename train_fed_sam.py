import os
os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, [0,1,2,3,4,5,6,7])) # 一般在程序开头设置
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
from dataloaders.federated_dataloader import Dataset, ToTensor, ProstateDataset

from models.sam import SamPredictor, sam_model_registry
from models.sam.utils.transforms import ResizeLongestSide
from sam_utils import *
# net = sam_model_registry['vit_b'](args,checkpoint=args.sam_ckpt).to(device)
import copy
from PIL import Image, ImageDraw
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
snapshot_path = "../output/" + args.exp + "/"
batch_size = args.batch_size * len(args.gpu.split(','))
meta_step_size = args.meta_step_size
clip_value = args.clip_value
base_lr = args.base_lr
# client_num = args.client_num
max_epoch = args.max_epoch
display_freq = args.display_freq
# ？？？
# client_name = ['1', '4', '5', '6', '13', '16', '18', '20', '21']
client_name = ['1', '6', '18', '21']
data_path = '/mnt/diskB/name/FeTS2022_FedDG_1024'
if args.data=='Prostate':
    client_name =  ['BIDMC', 'HK', 'I2CVB', 'ISBI', 'ISBI_1.5', 'UCL']
    data_path = '/mnt/diskB/name/Prostate_processed_1024'
elif args.data=='Fundus':
    # client_name =  ['G1020', 'ORIGA', 'REFUGE','Drishti-GS1']# ,'RIM-ONE'
    client_name =  ['REFUGE', 'ORIGA','G1020','Drishti-GS1']# ,'RIM-ONE',
    # client_name =  ['RIM-ONE','Drishti-GS1']# ,
    # Drishti不是1,'RIM-ONE'也不是
    data_path = '/mnt/diskB/name/Fundus_1024'
elif args.data=='Nuclei':
    # client_name = ['MoNuSAC2018','PanNuke2','PanNuke3','TNBC','MoNuSAC2020']
    #client_name = ['TNBC','MoNuSAC2018','MoNuSAC2020']# ,'PanNuke3','PanNuke2'
    # client_name = ['PanNuke3Breast', 'PanNuke3Testis', 'PanNuke3Kidney', 'PanNuke3Bile-duct', 'PanNuke3Lung', 'PanNuke3Skin', 'PanNuke3Stomach',  'PanNuke3HeadNeck', 'PanNuke3Liver', 'PanNuke3Pancreatic', 'PanNuke3Ovarian', 'PanNuke3Esophagus', 'PanNuke3Bladder', 'PanNuke3Thyroid', 'PanNuke3Uterus', 'PanNuke3Colon', 'PanNuke3Prostate', 'PanNuke3Adrenal_gland','PanNuke3Cervix']
    # client_name = ['PanNuke2Adrenal_gland','PanNuke2Esophagus', 'PanNuke3Testis', 'PanNuke3Kidney', 'PanNuke2Thyroid','PanNuke2Liver','PanNuke3Skin','PanNuke3Uterus','MoNuSAC2020','TNBC','MoNuSAC2018']
    # client_name = ['PanNuke2Adrenal_gland','PanNuke2Esophagus', 'PanNuke3Testis', 'PanNuke3Kidney', 'MoNuSAC2020','TNBC','MoNuSAC2018']
    client_name = ['PanNuke2Adrenal_gland','PanNuke2Esophagus', 'PanNuke3Bile-duct','PanNuke3Uterus', 'MoNuSAC2020','TNBC','MoNuSAC2018']
    data_path = '/mnt/diskB/name/Nuclei_1024'
elif args.data=='CTLung':
    client_name = ['1', '2', '3', '4', '5']
    data_path = '/mnt/diskB/name/CTLung_1024'
# 还要生成test数据
client_num = len(client_name)
client_data_list = []
client_val_data_list = []
slice_num =[]
for client_idx in range(client_num):
    print('{}/{}/data_npy/*'.format(data_path,client_name[client_idx]))
    client_data_list.append(glob('{}/{}/data_npy/*'.format(data_path,client_name[client_idx])))
    client_val_data_list.append(glob('{}/{}/val_data_npy/*'.format(data_path,client_name[client_idx])))
    print (len(client_data_list[client_idx]),len(client_val_data_list[client_idx]))
    slice_num.append(len(client_data_list[client_idx]))
# print(client_val_data_list)
slice_num = np.array(slice_num)
#volume_size = [384, 384, 3]
unseen_site_idx = args.unseen_site
client_data_list[unseen_site_idx].extend(client_val_data_list[unseen_site_idx])
print('unseen site data length:',len(client_data_list[unseen_site_idx]))
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
    # param.requires_grad
    for param in zip(*list(net_clients[i].parameters() for i in range(client_num))):
    # for param in zip(*[p for p in (list(net_clients[i].parameters()) for i in range(client_num)) if p.requires_grad]):
    #for param in zip(net_clients[0].parameters(), net_clients[1].parameters(), 
    #                 net_clients[2].parameters(), net_clients[3].parameters()):
        # print(param,param[0])
        if param[0].requires_grad is False:
            continue
        new_para = Variable(torch.Tensor(np.zeros(param[0].shape)), requires_grad=False).cuda(GPUdevice) 
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
def val(site_index, test_net):
    val_data_list = client_val_data_list[site_index]
    dice_array = []
    eiou_array = []
    dice_array_cup = []
    eiou_array_cup = []
    test_net.eval()
    for fid, filename in enumerate(val_data_list):
        data = np.load(filename)/ 255.0
        mask_data = np.load(filename.replace("data", "label"))
        image = np.expand_dims(data[..., :3].transpose(2, 0, 1), axis=0)
        mask = np.expand_dims(mask_data.transpose(2, 0, 1), axis=0)
        # show_element(mask)
        image = torch.from_numpy(image).float().cuda(GPUdevice)
        mask = torch.from_numpy(mask).cuda(GPUdevice)
        # mask_data=mask_data[:,:,0]# np.squeeze(mask_data)
        # mask_data=cv2.resize(mask_data,(image.shape[-1],image.shape[-1]),interpolation=cv2.INTER_NEAREST)
        # pt = np.expand_dims(random_click(np.array(mask_data), 1, 1), axis=0)
        pt = np.expand_dims(np.array([0,0]), axis=0)
        point_labels = torch.ones(image.size(0))
        if point_labels[0] != -1:
            point_coords = pt
            coords_torch = torch.as_tensor(point_coords, dtype=torch.float, device=GPUdevice)
            labels_torch = torch.as_tensor(point_labels, dtype=torch.int, device=GPUdevice)
            coords_torch, labels_torch = coords_torch[None, :, :], labels_torch[None, :]
            pt = (coords_torch, labels_torch)
        se, de = test_net.prompt_encoder(
            points=pt,
            boxes=None,
            masks=None
        )
        # image,mask = image.cuda(GPUdevice), mask.cuda(GPUdevice)
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
        if(pred.shape[1]==2):
            iou_d, iou_c, disc_dice, cup_dice= temp
            dice_array.append(disc_dice)
            eiou_array.append(iou_d)
            dice_array_cup.append(cup_dice)
            eiou_array_cup.append(iou_c)
        else:
            eiou, edice = temp
            dice_array.append(edice)
            eiou_array.append(eiou)
    if args.num_classes==2:
        dice_array = np.array(dice_array)
        eiou_array = np.array(eiou_array)
        dice_array_cup = np.array(dice_array_cup)
        eiou_array_cup = np.array(eiou_array_cup)
        # print (dice_array.shape)
        dice_avg = np.mean(dice_array, axis=0).tolist()
        eiou_avg = np.mean(eiou_array, axis=0).tolist()
        dice_avg_cup = np.mean(dice_array_cup, axis=0).tolist()
        eiou_avg_cup = np.mean(eiou_array_cup, axis=0).tolist()
        logging.info("validate data from client %d Disc Dice %.4f, Disc IOU %.4f, Cup Dice %.4f, Cup IOU %.4f" % (site_index, dice_avg, eiou_avg, dice_avg_cup, eiou_avg_cup))
        return dice_avg,eiou_avg, dice_avg_cup, eiou_avg_cup
    else:
        dice_array = np.array(dice_array)
        eiou_array = np.array(eiou_array)
        # print (dice_array.shape)
        dice_avg = np.mean(dice_array, axis=0).tolist()
        eiou_avg = np.mean(eiou_array, axis=0).tolist()
        logging.info("validate data from client %d OD dice_avg %.4f, Eiou %.4f" % (site_index, dice_avg, eiou_avg))
        return dice_avg,eiou_avg

def validation(test_net):
    dice_avg, eiou_avg, dice_avg_cup, eiou_avg_cup =[], [], [], []
    for i in source_site_idx:
        if args.num_classes==2:
            dice_avg_one, eiou_avg_one, dice_avg_cup_one, eiou_avg_cup_one=val(i,test_net)
            dice_avg.append(dice_avg_one)
            eiou_avg.append(eiou_avg_one)
            dice_avg_cup.append(dice_avg_cup_one)
            eiou_avg_cup.append(eiou_avg_cup_one)
        else:
            dice_avg_one, eiou_avg_one=val(i,test_net)
            dice_avg.append(dice_avg_one)
            eiou_avg.append(eiou_avg_one)
    dice_avg = np.mean(np.array(dice_avg), axis=0)
    eiou_avg = np.mean(np.array(eiou_avg), axis=0)
    if args.num_classes==2:
        dice_avg_cup = np.mean(np.array(dice_avg_cup), axis=0)
        eiou_avg_cup = np.mean(np.array(eiou_avg_cup), axis=0)
        logging.info("Averagy validation: Disc Dice %.4f, Disc IOU %.4f, Cup Dice %.4f, Cup IOU %.4f" % (dice_avg, eiou_avg, dice_avg_cup, eiou_avg_cup))
        return dice_avg, eiou_avg, dice_avg_cup, eiou_avg_cup
    else:
        logging.info("Averagy validation: OD dice_avg %.4f, Eiou %.4f" % (dice_avg, eiou_avg))
        return dice_avg, eiou_avg
def save_image(mask_data,name='pred'):
    mask_show= mask_data[0,0,:,:].copy()
    mask_show=mask_show*255
    image_save = Image.fromarray(mask_show).convert("RGB")
    image_save.save("../output/output-fundus-disc-{}.jpg".format(name))
    mask_show= mask_data[0,1,:,:].copy()
    mask_show=mask_show*255
    image_save = Image.fromarray(mask_show).convert("RGB")
    image_save.save("../output/output-fundus-cup-{}.jpg".format(name))
def draw_result(image,mask):

    # 假设 mask 是一个 (256, 256) 的 NumPy 数组
    # mask = np.random.randint(0, 2, size=(256, 256))

    # 将 mask 调整为 (1024, 1024)
    resized_mask = np.array(Image.fromarray(mask).resize((1024, 1024), Image.NEAREST))

    # 在 image 上绘制轮廓线
    # image = np.random.randint(0, 255, size=(1024, 1024, 3), dtype=np.uint8)  # 假设 image 是一个 (1024, 1024, 3) 的 NumPy 数组
    contours, _ = cv2.findContours(resized_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    image_with_contours = cv2.drawContours(image.copy(), contours, -1, (0, 255, 0), 2)

    # 保存绘制了轮廓线的图像
    cv2.imwrite('./output/funduscontours_image.jpg', image_with_contours)

def test(site_index, test_net):

    test_data_list = client_data_list[site_index]

    dice_array = []
    eiou_array = []
    dice_array_cup = []
    eiou_array_cup = []
    # print(test_data_list)
    test_net.eval()
    ave_res, mix_res = (0,0,0,0), (0,0,0,0)
    # print('test',site_index,len(test_data_list))
    for fid, filename in enumerate(test_data_list):
        # print(fid)
        data = np.load(filename)/ 255.0
        # print('data',data)
        mask_data = np.load(filename.replace("data", "label"))
        # why expand_dims?
        image = np.expand_dims(data[..., :3].transpose(2, 0, 1), axis=0)
        mask = np.expand_dims(mask_data.transpose(2, 0, 1), axis=0)
        # show_element(mask)
        # print('mask_data.shape,mask.shape',mask_data.shape,mask.shape)
        # mask在另外一个地方文件夹。。。
        image = torch.from_numpy(image).float()
        mask = torch.from_numpy(mask)
        '''
        # save mask
        mask_show= mask_data[:,:,0].copy()
        mask_show[mask_show==1]=255
        image_save = Image.fromarray(mask_show)
        image_save.save("../output/output-{}.jpg".format(client_name[unseen_site_idx]))
        '''
        # print('dmi',mask)
        # print(mask_data.shape)
        # mask_data=mask_data[:,:,0]# np.squeeze(mask_data)
        # mask_data=cv2.resize(mask_data,(image.shape[-1],image.shape[-1]),interpolation=cv2.INTER_NEAREST)
        # pt = np.expand_dims(random_click(np.array(mask_data), 1, 1), axis=0)
        pt = np.expand_dims(np.array([0,0]), axis=0)
        # print('pt',pt)
        point_labels = torch.ones(image.size(0))
        if point_labels[0] != -1:
            # point_coords = samtrans.ResizeLongestSide(longsize).apply_coords(pt, (h, w))
            point_coords = pt
            coords_torch = torch.as_tensor(point_coords, dtype=torch.float, device=GPUdevice)
            labels_torch = torch.as_tensor(point_labels, dtype=torch.int, device=GPUdevice)
            coords_torch, labels_torch = coords_torch[None, :, :], labels_torch[None, :]
            pt = (coords_torch, labels_torch)
            # print('pt',pt[0].shape,pt[1].shape)
            # imge= net.image_encoder(imgs)
        se, de = test_net.prompt_encoder(
            points=pt,
            boxes=None,
            masks=None
        )
        image,mask = image.cuda(GPUdevice), mask.cuda(GPUdevice)
        image_encoded= test_net.image_encoder(image)
        pred, _, _ = test_net.mask_decoder(
            image_embeddings=image_encoded,
            image_pe=test_net.prompt_encoder.get_dense_pe(), #  1x(embed_dim)x(embedding_h)x(embedding_w)
            sparse_prompt_embeddings=se,
            dense_prompt_embeddings=de, 
            multimask_output=False,
        )
        # print('pred.shape,mask.shape',pred.shape,mask.shape)
        threshold = (0.1, 0.3, 0.5, 0.7, 0.9)
        
        temp = eval_seg(pred, mask, threshold)
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
    logging.info("OD dice_avg %.4f, Eiou %.4f" % (dice_avg, eiou_avg))
    return dice_avg, dice_array, eiou_avg, eiou_array
        '''
        if(pred.shape[1]==2):
            iou_d, iou_c, disc_dice, cup_dice= temp
            dice_array.append(disc_dice)
            eiou_array.append(iou_d)
            dice_array_cup.append(cup_dice)
            eiou_array_cup.append(iou_c)
        else:
            eiou, edice = temp
            dice_array.append(edice)
            eiou_array.append(eiou)
    if args.num_classes==2:
        dice_array = np.array(dice_array)
        eiou_array = np.array(eiou_array)
        dice_array_cup = np.array(dice_array_cup)
        eiou_array_cup = np.array(eiou_array_cup)
        # print (dice_array.shape)
        dice_avg = np.mean(dice_array, axis=0).tolist()
        eiou_avg = np.mean(eiou_array, axis=0).tolist()
        dice_avg_cup = np.mean(dice_array_cup, axis=0).tolist()
        eiou_avg_cup = np.mean(eiou_array_cup, axis=0).tolist()
        logging.info("Test Disc Dice %.4f, Disc IOU %.4f, Cup Dice %.4f, Cup IOU %.4f" % (dice_avg, eiou_avg, dice_avg_cup, eiou_avg_cup))
        return dice_avg,eiou_avg, dice_avg_cup, eiou_avg_cup
    else:
        dice_array = np.array(dice_array)
        eiou_array = np.array(eiou_array)
        # print (dice_array.shape)
        dice_avg = np.mean(dice_array, axis=0).tolist()
        eiou_avg = np.mean(eiou_array, axis=0).tolist()
        logging.info("Test OD dice_avg %.4f, Eiou %.4f" % (dice_avg, eiou_avg))
        return dice_avg, eiou_avg



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
    if os.path.exists(snapshot_path + '/code'):
        shutil.rmtree(snapshot_path + '/code')
    shutil.copytree('.', snapshot_path + '/code', shutil.ignore_patterns(['.git','__pycache__']))

    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    GPUdevice = torch.device('cuda', int(args.gpu))
    pos_weight = torch.ones([1]).cuda(device=GPUdevice)*2
    criterion_G = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    lossfunc = criterion_G
    # define dataset, model, optimizer for each client
    def worker_init_fn(worker_id):
        random.seed(args.seed+worker_id)
    dataloader_clients = []
    net_clients = []
    optimizer_clients = []
    device_list = [6, 7]
    for client_idx in range(client_num):
        freq_site_idx = source_site_idx.copy()
        if client_idx != unseen_site_idx:
            freq_site_idx.remove(client_idx)
        if args.data=='Prostate':
            dataset = ProstateDataset(client_idx=client_idx, data_path=data_path,freq_site_idx=freq_site_idx,
                                split='train', transform = transforms.Compose([
                                ToTensor(),
                                ]),client_name=client_name)
        else:
            dataset = Dataset(client_idx=client_idx, data_path=data_path,freq_site_idx=freq_site_idx,
                                split='train', transform = transforms.Compose([
                                ToTensor(),
                                ]),client_name=client_name)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,  num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)
        net = sam_model_registry[args.sam_type](args,checkpoint=args.sam_ckpt).to(GPUdevice)# Unet2D(num_classes=1)
        start_epoch=0
        if args.weights != 0:
            print(f'=> resuming from {args.weights}')
            assert os.path.exists(args.weights)
            checkpoint_file = os.path.join(args.weights)
            assert os.path.exists(checkpoint_file)
            loc = 'cuda:{}'.format(args.gpu_device)
            checkpoint = torch.load(checkpoint_file, map_location=loc)
            start_epoch = args.start_epoch
            # best_tol = checkpoint['best_tol']
            
            net.load_state_dict(checkpoint,strict=False)
            # optimizer.load_state_dict(checkpoint['optimizer'], strict=False)
            print(f'=> loaded checkpoint {checkpoint_file} (epoch {start_epoch})')
        # net = nn.DataParallel(net, device_ids=device_list) 
        # net = net.cuda()
        # net = net.cuda(GPUdevice)
        '''
        if args.pretrain is not None:
            weights = torch.load(args.pretrain)
            net.load_state_dict(weights,strict=False)
        '''
        lr=args.base_lr# if client_idx!=client_num-2 else 0.1*args.base_lr
        optimizer = torch.optim.Adam(net.parameters(), lr=lr, betas=(0.9, 0.999))
        dataloader_clients.append(dataloader)
        net_clients.append(net)
        optimizer_clients.append(optimizer)
    '''
    for name, param in  net_clients[0].named_parameters():
        print (name)
    '''

    temperature = 0.05
    cont_loss_func = losses.NTXentLoss(temperature)

    # start federated learning
    writer = SummaryWriter(snapshot_path+'/log')
    lr_ = base_lr
    # test(unseen_site_idx, net_clients[unseen_site_idx])
    # validation(net_clients[unseen_site_idx])
    update_global_model(net_clients, client_weight)
    for epoch_num in tqdm(range(start_epoch,max_epoch), ncols=70):
        # update_global_model(net_clients, client_weight)
        for client_idx in source_site_idx:
            dataloader_current = dataloader_clients[client_idx]
            net_current = net_clients[client_idx]
            net_current.train()
            optimizer_current = optimizer_clients[client_idx]
            time1 = time.time()
            iter_num = 0
            # validation(net_current)
            for i_batch, sampled_batch in enumerate(dataloader_current):
                
                time2 = time.time()

                # obtain training data
                volume_batch, label_batch, pt = sampled_batch['image'], sampled_batch['label'], sampled_batch['pt']
                volume_batch_raw_np = volume_batch[:, :3, ...]
                point_labels = torch.ones(volume_batch.size(0))
                if point_labels[0] != -1:
                    # point_coords = samtrans.ResizeLongestSide(longsize).apply_coords(pt, (h, w))
                    point_coords = pt
                    coords_torch = torch.as_tensor(point_coords, dtype=torch.float, device=GPUdevice)
                    labels_torch = torch.as_tensor(point_labels, dtype=torch.int, device=GPUdevice)
                    coords_torch, labels_torch = coords_torch[None, :, :], labels_torch[None, :]
                    pt = (coords_torch, labels_torch)
                    # print('pt',pt[0].shape,pt[1].shape)
                with torch.no_grad():
                    # imge= net.image_encoder(imgs)
                    se, de = net_current.prompt_encoder(
                        points=pt,
                        boxes=None,
                        masks=None
                    )
                volume_batch_raw, label_batch = \
                    volume_batch_raw_np.cuda(GPUdevice), label_batch.cuda(GPUdevice)
                # print('parameters_name',net_current.image_encoder.named_parameters())
                parameters_to_calculate_grad = []
                parameters_name_to_calculate_grad = []
                for n, value in net_current.image_encoder.named_parameters():
                    if "Adapter" not in n:
                        value.requires_grad = False
                # batch_size, out_chans=256, H', W', 感觉out_chans可以小一点
                volume_batch_raw_encoded= net_current.image_encoder(volume_batch_raw)
                outputs_soft_inner, masks_inner_embedding, _ = net_current.mask_decoder(
                    image_embeddings=volume_batch_raw_encoded,
                    image_pe=net.prompt_encoder.get_dense_pe(), #  1x(embed_dim)x(embedding_h)x(embedding_w)
                    sparse_prompt_embeddings=se,
                    dense_prompt_embeddings=de, 
                    multimask_output=False,
                )
                # print("outputs_soft_inner.shape,label_batch.shape",outputs_soft_inner.shape,label_batch.shape)
                # loss_inner = F.binary_cross_entropy_with_logits(outputs_soft_inner, label_batch)
                loss_inner = lossfunc(outputs_soft_inner, label_batch) #dice
                # loss_inner = dice_loss(outputs_soft_inner, label_batch)
                total_loss = loss_inner

                optimizer_current.zero_grad()
                total_loss.backward()
                optimizer_current.step()

                iter_num = iter_num + 1
                if iter_num % display_freq == 0:
                    writer.add_scalar('lr', lr_, iter_num)
                    writer.add_scalar('loss/inner', loss_inner, iter_num)
                    # writer.add_scalar('loss/outer', loss_outer, iter_num)
                    writer.add_scalar('loss/total', total_loss, iter_num)
                    logging.info('Epoch: [%d] client [%d] iteration [%d / %d] : inner loss : %f' % \
                        (epoch_num, client_idx, iter_num, len(dataloader_current), loss_inner.item()))
                    '''
                    logging.info('Epoch: [%d] client [%d] iteration [%d / %d] : inner loss : %f outer dice loss : %f outer cont loss : %f outer loss : %f total loss : %f' % \
                        (epoch_num, client_idx, iter_num, len(dataloader_current), loss_inner.item(), loss_outer_1_dice.item(), cont_loss.item(), loss_outer.item(), total_loss.item()))
                    '''
                    
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
                if iter_num % 10000==0:
                    val(client_idx,net_current)
                    # break
            # validation(net_current)
        ## model aggregation
        update_global_model(net_clients, client_weight)
        if epoch_num % 5==0 or epoch_num==max_epoch-1:
            ## evaluation test unseen
            with open(os.path.join(snapshot_path, 'evaluation_result.txt'), 'a') as f:
                dice_list = []
                haus_list = []
                print("epoch {} testing , site {}".format(epoch_num, unseen_site_idx), file=f)
                if args.num_classes==2:
                    dice_avg, eiou_avg, dice_avg_cup, eiou_avg_cup=validation(net_clients[unseen_site_idx])
                    print(("Validation OD dice is: {}, OD IOU is {}, OC dice is: {}, OC IOU is {}".format(dice_avg,eiou_avg, dice_avg_cup, eiou_avg_cup)),file=f)
                    dice_avg, eiou_avg, dice_avg_cup, eiou_avg_cup= test(unseen_site_idx, net_clients[unseen_site_idx])
                    print(("Test OD dice is: {}, OD IOU is {}, OC dice is: {}, OC IOU is {}".format(dice_avg,eiou_avg, dice_avg_cup, eiou_avg_cup)),file=f)
                else:
                    dice_avg, eiou_avg=validation(net_clients[unseen_site_idx])
                    print(("Validation OD dice is: {}, IOU is {}".format(dice_avg,eiou_avg)),file=f)
                    dice_avg, eiou_avg= test(unseen_site_idx, net_clients[unseen_site_idx])
                    print(("Test OD dice is: {}, IOU is {}".format(dice_avg,eiou_avg)),file=f)
                # print(("   OC dice is: {}, std is {}".format(dice[1], np.std(dice_array[:, 1]))), file=f)
                
            ## save model
            save_mode_path = os.path.join(snapshot_path + '/model', 'epoch_' + str(epoch_num) + '.pth')
            torch.save(net_clients[0].state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))

    writer.close()


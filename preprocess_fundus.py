import nibabel as nib
from glob import glob
import numpy  as np 
import cv2
import os
from PIL import Image
# client_name = ['RIM-ONE','G1020', 'ORIGA', 'REFUGE','Drishti-GS1']
client_name = ['Drishti-GS1']
client_num = len(client_name)
data_path = '/mnt/diskB/name/Fundus'
target_path = '/mnt/diskB/name/Fundus_1024'
if not os.path.exists(target_path):
    os.makedirs(target_path)
# 还要生成test数据
client_data_list = []
slice_num =[]
def show_element(print_image):
    # 找到不为零的元素的索引
    non_zero_indices = np.nonzero(print_image)        
    # 打印不为零的元素及其索引
    for i in range(len(non_zero_indices[0])):
        row = non_zero_indices[0][i]
        col = non_zero_indices[1][i]
        # row2 = non_zero_indices[2][i]
        # col2 = non_zero_indices[3][i]
        value = print_image[row, col]
        print(f"元素 {value} 在索引 ({row}, {col})")
        
for client_idx in range(client_num):
    # print('{}/{}/data_npy/*'.format(data_path,client_name[client_idx]))
    client_data_list.append(glob('{}/{}/images/*'.format(data_path,client_name[client_idx])))
    print (len(client_data_list[client_idx]))
    slice_num.append(len(client_data_list[client_idx]))
for client_idx in range(client_num):
    dir_name = '{}/{}/data_npy/'.format(target_path,client_name[client_idx])
    label_dir_name = '{}/{}/label_npy/'.format(target_path,client_name[client_idx])
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    if not os.path.exists(label_dir_name):
        os.makedirs(label_dir_name)
    if client_name[client_idx]=='RIM-ONE':
        for fid, filename in enumerate(client_data_list[client_idx]):
            nii_file_path = filename
            img_data = np.array(Image.open(nii_file_path).convert('RGB'))
            labelname = nii_file_path.replace('images','labels')
            label_od_name = labelname[:-4]+'-1-Disc-T.png'
            label_oc_name = labelname[:-4]+'-1-Cup-T.png'
            od = np.array(Image.open(label_od_name))
            oc = np.array(Image.open(label_oc_name))
            # 打印图像数据的形状
            # print("图像数据形状:", img_data.shape,od.shape,oc.shape)
            image_file_name = os.path.basename(filename)
            rgb_image=cv2.resize(img_data, (1024,1024), interpolation=cv2.INTER_LINEAR)
            image_new_name = image_file_name+'.npy'
            # print(rgb_image.shape)
            save_path = os.path.join(dir_name,image_new_name)
                # print(save_path)
            np.save(save_path,rgb_image)
            # print(od.shape,oc.shape)
            od = cv2.resize(od, (256,256), interpolation=cv2.INTER_NEAREST)        
            oc = cv2.resize(oc, (256,256), interpolation=cv2.INTER_NEAREST)  
            '''
            label=oc.copy()
            # label[label < 100] = 0
            label[label != 0] = 1
            # save mask
            mask_show= label[:,:].copy()
            mask_show[mask_show==1]=255
            image_save = Image.fromarray(mask_show)
            image_save.save("../output/output-RIMONE-post.jpg")
            '''
            label = np.concatenate(( np.expand_dims(od.astype(np.uint8),axis=-1) ,np.expand_dims(oc.astype(np.uint8),axis=-1)),axis=-1)
            label[label != 0] = 1
            # print(od,oc)
            # label = np.expand_dims(label.astype(np.uint8),axis=-1)   
            # print(label.shape)
            save_path = os.path.join(label_dir_name,image_new_name)
            # print(save_path)
            np.save(save_path,label)
    elif client_name[client_idx]=='Drishti-GS1':
        for fid, filename in enumerate(client_data_list[client_idx]):
            nii_file_path = filename
            img_data = np.array(Image.open(nii_file_path).convert('RGB'))
            labelname = nii_file_path.replace('images','labels')
            label_od_name = labelname[:-4]+'_ODsegSoftmap.png'
            label_oc_name = labelname[:-4]+'_cupsegSoftmap.png'
            od = np.array(Image.open(label_od_name))
            oc = np.array(Image.open(label_oc_name))
            # 打印图像数据的形状
            # print("图像数据形状:", img_data.shape,od.shape,oc.shape)
            image_file_name = os.path.basename(filename)
            rgb_image=cv2.resize(img_data, (1024,1024), interpolation=cv2.INTER_LINEAR)
            image_new_name = image_file_name+'.npy'
            # print(rgb_image.shape)
            save_path = os.path.join(dir_name,image_new_name)
                # print(save_path)
            np.save(save_path,rgb_image)
            # print(od.shape,oc.shape)
            od = cv2.resize(od, (256,256), interpolation=cv2.INTER_NEAREST)        
            oc = cv2.resize(oc, (256,256), interpolation=cv2.INTER_NEAREST)  
            '''
            label=od.copy()
            label[label < 100] = 0
            label[label != 0] = 1
            # save mask
            mask_show= label[:,:].copy()
            mask_show[mask_show==1]=255
            image_save = Image.fromarray(mask_show)
            image_save.save("../output/output-DGSod-post-od.jpg")
            label=oc.copy()
            label[label < 100] = 0
            label[label != 0] = 1
            # save mask
            mask_show= label[:,:].copy()
            mask_show[mask_show==1]=255
            image_save = Image.fromarray(mask_show)
            image_save.save("../output/output-DGSoc-post-oc.jpg")
            '''
            od[od < 100] = 0
            od[od != 0] = 1
            oc[oc < 100] = 0
            oc[oc != 0] = 1
            od=od-oc
            label = np.concatenate(( np.expand_dims(od.astype(np.uint8),axis=-1) ,np.expand_dims(oc.astype(np.uint8),axis=-1)),axis=-1)
            label[label != 0] = 1
            # show_element(label)
            # print(od,oc)
            # label = np.expand_dims(label.astype(np.uint8),axis=-1)   
            # print(label.shape)
            save_path = os.path.join(label_dir_name,image_new_name)
            # print(save_path)
            np.save(save_path,label)
    else:
        for fid, filename in enumerate(client_data_list[client_idx]):
            # 读取.nii.gz文件
            nii_file_path = filename
            img_data = np.array(Image.open(filename).convert('RGB'))
            labelname = nii_file_path.replace('images','labels')
            labelname = labelname.replace('jpg','png')
            label_data = np.array(Image.open(labelname, mode='r'))
            # 打印图像数据的形状
            # print("图像数据形状:", img_data.shape,label_data.shape)
            image_file_name = os.path.basename(filename)
            rgb_image=cv2.resize(img_data, (1024,1024), interpolation=cv2.INTER_LINEAR)
            image_new_name = image_file_name+'.npy'
            # print(rgb_image.shape)
            save_path = os.path.join(dir_name,image_new_name)
                # print(save_path)
            np.save(save_path,rgb_image)
            label = label_data
            od = (label==1.).astype(np.uint8)
            oc = (label==2.).astype(np.uint8)
            # print(od.shape,oc.shape)
            od = cv2.resize(od, (256,256), interpolation=cv2.INTER_NEAREST)        
            oc = cv2.resize(oc, (256,256), interpolation=cv2.INTER_NEAREST) 
            '''
            label=od.copy()
            # label[label < 100] = 0
            # save mask
            mask_show= label[:,:].copy()
            mask_show[mask_show==1]=255
            image_save = Image.fromarray(mask_show)
            image_save.save("../output/output-%s-post.jpg"%client_name[client_idx])
            label=oc.copy()
            # label[label < 100] = 0
            # save mask
            mask_show= label[:,:].copy()
            mask_show[mask_show==1]=255
            image_save = Image.fromarray(mask_show)
            image_save.save("../output/output-%s-post.jpg"%client_name[client_idx])
            '''
            label = np.concatenate(( np.expand_dims(od.astype(np.uint8),axis=-1) ,np.expand_dims(oc.astype(np.uint8),axis=-1)),axis=-1)
            # label = np.expand_dims(label.astype(np.uint8),axis=-1)   
            # print(label.shape)
            save_path = os.path.join(label_dir_name,image_new_name)
            # print(save_path)
            np.save(save_path,label)
    print(client_num," finished")


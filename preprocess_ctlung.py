import os
import shutil

import numpy  as np 
import cv2
import os
from glob import glob
from PIL import Image
client_name = ['1', '2', '3', '4', '5']
client_num = len(client_name)
data_path = '/mnt/diskB/name/CTLung'
target_path = '/mnt/diskB/name/CTLung_1024'
if not os.path.exists(target_path):
    os.makedirs(target_path)
# 还要生成test数据
client_data_list = []
slice_num =[]
for client_idx in range(client_num):
    print('{}/{}/data_npy/*'.format(data_path,client_name[client_idx]))
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
    for fid, filename in enumerate(client_data_list[client_idx]):
        img_data = np.array(Image.open(filename))
        labelname = filename.replace('images','masks')
        label_data = np.array(Image.open(labelname))
        # 打印图像数据的形状
        # print("图像数据形状:", img_data.shape,label_data.shape)
        image_file_name = os.path.basename(filename)
        rgb_image=cv2.resize(img_data, (1024,1024), interpolation=cv2.INTER_LINEAR)
        rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_GRAY2RGB)
        image_new_name = image_file_name+'.npy'
        # print(rgb_image.shape,rgb_image)
        save_path = os.path.join(dir_name,image_new_name)
        # print(save_path)
        rgb_image=np.array(rgb_image)
        # print(rgb_image.shape)
        rgb_image=rgb_image[:,:,:3]
        np.save(save_path,rgb_image)
        label = label_data[:,:,2]
        label = cv2.resize(label, (256,256), interpolation=cv2.INTER_NEAREST)   
        # print(label.shape)
        label = label
        label = np.expand_dims(label.astype(np.uint8),axis=-1)   
        # print(label.shape)
        label[label < 150] = 0        
        label[label >= 150] = 1
        save_path = os.path.join(label_dir_name,image_new_name)
        # print(save_path)
        
        np.save(save_path,label)
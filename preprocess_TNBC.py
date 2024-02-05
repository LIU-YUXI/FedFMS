import os
import shutil
'''
# 源文件夹路径
source_folder = '/mnt/diskB/name/TNBC_NucleiSegmentation'

# 目标文件夹路径
target_folder_slide = '/mnt/diskB/name/TNBC/images'
target_folder_gt = '/mnt/diskB/name/TNBC/labels'
if not os.path.exists(target_folder_slide):
    os.makedirs(target_folder_slide)
if not os.path.exists(target_folder_gt):
    os.makedirs(target_folder_gt)
# 遍历源文件夹下的子文件夹
for subfolder in os.listdir(source_folder):
    subfolder_path = os.path.join(source_folder, subfolder)
    
    # 如果是 Slide 开头的文件夹，将其中的文件移动到 Slide 目标文件夹
    if subfolder.startswith('Slide'):
        for filename in os.listdir(subfolder_path):
            source_file_path = os.path.join(subfolder_path, filename)
            target_file_path = os.path.join(target_folder_slide, filename)
            shutil.move(source_file_path, target_file_path)
    
    # 如果是 GT 开头的文件夹，将其中的文件移动到 GT 目标文件夹
    elif subfolder.startswith('GT'):
        for filename in os.listdir(subfolder_path):
            source_file_path = os.path.join(subfolder_path, filename)
            target_file_path = os.path.join(target_folder_gt, filename)
            shutil.move(source_file_path, target_file_path)
'''
import numpy  as np 
import cv2
import os
from glob import glob
from PIL import Image
client_name = 'TNBC'
data_path = '/mnt/diskB/name/TNBC'
target_path = '/mnt/diskB/name/Nuclei_1024'
if not os.path.exists(target_path):
    os.makedirs(target_path)
dir_name = '{}/{}/data_npy/'.format(target_path,client_name)
label_dir_name = '{}/{}/label_npy/'.format(target_path,client_name)
if not os.path.exists(dir_name):
    os.makedirs(dir_name)
if not os.path.exists(label_dir_name):
    os.makedirs(label_dir_name)
client_data_list=glob('{}/images/*'.format(data_path))
print (len(client_data_list))
for fid, filename in enumerate(client_data_list):
    img_data = np.array(Image.open(filename))
    labelname = filename.replace('images','labels')
    label_data = np.array(Image.open(labelname))
    # 打印图像数据的形状
    # print("图像数据形状:", img_data.shape,label_data.shape)
    image_file_name = os.path.basename(filename)
    rgb_image=cv2.resize(img_data, (1024,1024), interpolation=cv2.INTER_LINEAR)
    image_new_name = image_file_name+'.npy'
    # print(rgb_image.shape,rgb_image)
    save_path = os.path.join(dir_name,image_new_name)
    # print(save_path)
    rgb_image=np.array(rgb_image)
    rgb_image=rgb_image[:,:,:3]
    np.save(save_path,rgb_image)
    label = label_data
    label = cv2.resize(label, (256,256), interpolation=cv2.INTER_NEAREST)   
    
    label = np.expand_dims(label.astype(np.uint8),axis=-1)   
    # print(label.shape)
    label[label != 0] = 1
    save_path = os.path.join(label_dir_name,image_new_name)
    # print(save_path)
    
    np.save(save_path,label)
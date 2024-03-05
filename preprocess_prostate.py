import nibabel as nib
from glob import glob
import numpy  as np 
import cv2
import os
client_name = ['BIDMC', 'HK', 'I2CVB', 'ISBI', 'ISBI_1.5', 'UCL']
client_num = len(client_name)
data_path = '/mnt/diskB/name/Prostate_processed/'
target_path = '/mnt/diskB/name/Prostate_processed_1024'
if not os.path.exists(target_path):
    os.makedirs(target_path)
# 还要生成test数据
client_data_list = []
slice_num =[]
def convert_from_nii_to_png(img):
    high = np.quantile(img,0.99)
    low = np.min(img)
    img = np.where(img > high, high, img)
    lungwin = np.array([low * 1., high * 1.])
    newimg = (img - lungwin[0]) / (lungwin[1] - lungwin[0])  
    newimg = (newimg * 255).astype(np.uint8)
    return newimg
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
    for fid, filename in enumerate(client_data_list[client_idx]):
        # 读取.nii.gz文件
        nii_file_path = filename# '/mnt/diskB/name/Prostate_processed/Prostate_processed/BIDMC/images/Case02.nii.gz'
        img = nib.load(nii_file_path)
        # 获取图像数据
        img_data = img.get_fdata()
        lab = nib.load(nii_file_path.replace('images','labels'))
        label_data = lab.get_fdata()
        # 打印图像数据的形状
        # print("图像数据形状:", img_data.shape,label_data.shape)
        image_file_name = os.path.basename(filename)
        img_data = convert_from_nii_to_png(img_data)
        for i in range(img_data.shape[-1]):
            gray_image = np.uint8(np.array(img_data[:,:,i]))
            rgb_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2RGB)
            rgb_image=cv2.resize(rgb_image, (1024,1024), interpolation=cv2.INTER_LINEAR)
            image_new_name = image_file_name+str(i)+'.npy'
            # print(rgb_image.shape)
            save_path = os.path.join(dir_name,image_new_name)
            # print(save_path)
            np.save(save_path,rgb_image)
            label = np.array(label_data[:,:,i])
            label = cv2.resize(label, (256,256), interpolation=cv2.INTER_NEAREST)        
            label = np.expand_dims(label.astype(np.uint8),axis=-1)   
            # print(label.shape)
            save_path = os.path.join(label_dir_name,image_new_name)
            # print(save_path)
            np.save(save_path,label)
    print(client_num," finished")
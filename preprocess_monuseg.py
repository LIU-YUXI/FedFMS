import os
import numpy  as np 
import cv2
import os
from glob import glob
from PIL import Image
client_name = 'MoNuSAC2018'
data_path = '/mnt/diskB/name/Monuseg2018/kmms_training/kmms_training'
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
    labelname = filename.replace('images','labels').replace('.tif',' .png')
    label_data = np.array(Image.open(labelname))
    image_file_name = os.path.basename(filename)
    rgb_image=cv2.resize(img_data, (1024,1024), interpolation=cv2.INTER_LINEAR)
    image_new_name = image_file_name+'.npy'
    # print(rgb_image.shape)
    save_path = os.path.join(dir_name,image_new_name)
    # print(save_path)
    rgb_image=rgb_image[:,:,:3]
    np.save(save_path,rgb_image)
    label = label_data
    label = cv2.resize(label, (256,256), interpolation=cv2.INTER_NEAREST)        
    # save mask
    mask_show= label.copy()
    # mask_show[mask_show==1]=255
    image_save = Image.fromarray(mask_show)
    image_save.save("../output/output-MoNuSAC2018-pre.jpg")
    label = np.expand_dims(label.astype(np.uint8),axis=-1)   
    # print(label.shape)
    save_path = os.path.join(label_dir_name,image_new_name)
    # print(save_path)
    
    label[label < 100] = 0
    label[label != 0] = 1
    # save mask
    mask_show= label[:,:,0].copy()
    mask_show[mask_show==1]=255
    image_save = Image.fromarray(mask_show)
    image_save.save("../output/output-MoNuSAC2018-post.jpg")
    np.save(save_path,label)
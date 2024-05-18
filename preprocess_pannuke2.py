from glob import glob
import numpy  as np 
import cv2
import os
client_name = 'PanNuke2'
data_path = '/mnt/diskB/name/PanNuke2'
target_path = '/mnt/diskB/name/Nuclei_1024'
if not os.path.exists(target_path):
    os.makedirs(target_path)
types = np.load('/mnt/diskB/name/PanNuke2/Images/types.npy')
print(types)
images = np.load('/mnt/diskB/name/PanNuke2/Images/images.npy')
file_number = len(images)
print(len(images))
masks = np.load('/mnt/diskB/name/PanNuke2/Masks/masks.npy')
print(len(masks))
for fid in range(file_number):
    img_data = images[fid]
    label_data = masks[fid]
    img_type = types[fid]
    dir_name = '{}/{}/data_npy/'.format(target_path,client_name+img_type)
    label_dir_name = '{}/{}/label_npy/'.format(target_path,client_name+img_type)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    if not os.path.exists(label_dir_name):
        os.makedirs(label_dir_name)
    image_file_name = types[fid]+str(fid)
    rgb_image=cv2.resize(img_data, (1024,1024), interpolation=cv2.INTER_LINEAR)
    image_new_name = image_file_name+'.npy'
    # print(rgb_image.shape)
    save_path = os.path.join(dir_name,image_new_name)
    # print(save_path)
    np.save(save_path,rgb_image)
    summed_mask = np.sum(label_data[:,:,:-1], axis=-1, keepdims=False)
    label = summed_mask
    label = cv2.resize(label, (256,256), interpolation=cv2.INTER_NEAREST)        
    label = np.expand_dims(label.astype(np.uint8),axis=-1)   
    # print(label.shape)
    label[label != 0] = 1
    save_path = os.path.join(label_dir_name,image_new_name)
    # print(save_path)
    np.save(save_path,label)
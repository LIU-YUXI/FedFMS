from PIL import Image
import numpy as np

# 假设您有一个名为image_data的numpy数组，表示图像
# image_data的形状应为(height, width, channels)，通常为(height, width, 3)或(height, width, 4)
# 如果是灰度图像，则为(height, width, 1)
image_data = np.load("/mnt/diskB/lyx/FeTS2022_FedDG/1/data_npy/FeTS2022_00131.nii.gz99.npy")
# 创建一个PIL图像对象
image_pil = Image.fromarray(image_data[:,:,:3].astype(np.uint8))# [:,:,[2,1,0]]
# 保存图像为jpg文件
image_pil.save('../output/output0.jpg')

image_data = np.load("/mnt/diskB/lyx/FeTS2022_FedDG/4/data_npy/FeTS2022_00561.nii.gz100.npy")
# 创建一个PIL图像对象
image_pil = Image.fromarray(image_data[:,:,:3].astype(np.uint8))# [:,:,[2,1,0]]
# 保存图像为jpg文件
image_pil.save('../output/output1.jpg')


image_data = np.load("/mnt/diskB/lyx/FeTS2022_FedDG/5/data_npy/FeTS2022_00100.nii.gz100.npy")
# 创建一个PIL图像对象
image_pil = Image.fromarray(image_data[:,:,:3].astype(np.uint8))# [:,:,[2,1,0]]
# 保存图像为jpg文件
image_pil.save('../output/output2.jpg')

image_data = np.load("/mnt/diskB/lyx/FeTS2022_FedDG/6/data_npy/FeTS2022_00120.nii.gz100.npy")
# 创建一个PIL图像对象
image_pil = Image.fromarray(image_data[:,:,:3].astype(np.uint8))# [:,:,[2,1,0]]
# 保存图像为jpg文件
image_pil.save('../output/output3.jpg')

image_data = np.load("/mnt/diskB/lyx/FeTS2022_FedDG/13/data_npy/FeTS2022_01489.nii.gz100.npy")
# 创建一个PIL图像对象
image_pil = Image.fromarray(image_data[:,:,:3].astype(np.uint8))# [:,:,[2,1,0]]
# 保存图像为jpg文件
image_pil.save('../output/output4.jpg')

image_data = np.load("/mnt/diskB/lyx/FeTS2022_FedDG/16/data_npy/FeTS2022_00115.nii.gz100.npy")
# 创建一个PIL图像对象
image_pil = Image.fromarray(image_data[:,:,:3].astype(np.uint8))# [:,:,[2,1,0]]
# 保存图像为jpg文件
image_pil.save('../output/output5.jpg')

image_data = np.load("/mnt/diskB/lyx/FeTS2022_FedDG/18/data_npy/FeTS2022_00000.nii.gz100.npy")
# 创建一个PIL图像对象
image_pil = Image.fromarray(image_data[:,:,:3].astype(np.uint8))# [:,:,[2,1,0]]
# 保存图像为jpg文件
image_pil.save('../output/output6.jpg')

image_data = np.load("/mnt/diskB/lyx/FeTS2022_FedDG/20/data_npy/FeTS2022_00101.nii.gz100.npy")
# 创建一个PIL图像对象
image_pil = Image.fromarray(image_data[:,:,:3].astype(np.uint8))# [:,:,[2,1,0]]
# 保存图像为jpg文件
image_pil.save('../output/output7.jpg')

image_data = np.load("/mnt/diskB/lyx/FeTS2022_FedDG/20/data_npy/FeTS2022_00101.nii.gz100.npy")
# 创建一个PIL图像对象
image_pil = Image.fromarray(image_data[:,:,:3].astype(np.uint8))# [:,:,[2,1,0]]
# 保存图像为jpg文件
image_pil.save('../output/output8.jpg')
import os
import random
import shutil
from glob import glob
client_name = ['BIDMC', 'HK', 'I2CVB', 'ISBI', 'ISBI_1.5']
def divide(client_idx):
	data_path = '/mnt/diskB/lyx/Prostate_processed_1024'
	# image_list = glob('{}/{}/data_npy/*'.format(data_path,client_name[client_idx]))
	# label_dir='{}/{}/label_npy'.format(data_path,client_name[client_idx])
	# 源文件夹和目标文件夹的路径
	source_folder = '{}/{}/data_npy'.format(data_path,client_name[client_idx])
	target_folder = '{}/{}/val_data_npy'.format(data_path,client_name[client_idx])
	source_label_folder = '{}/{}/label_npy'.format(data_path,client_name[client_idx])
	target_label_folder = '{}/{}/val_label_npy'.format(data_path,client_name[client_idx])
	# 创建目标文件夹（如果不存在）
	if not os.path.exists(target_folder):
		os.makedirs(target_folder)
	if not os.path.exists(target_label_folder):
		os.makedirs(target_label_folder)
	# 获取源文件夹中的所有图像文件
	image_files = [f for f in os.listdir(source_folder) if f.endswith(('.npy'))]

	# 计算要移动的图像数量（10%）
	num_images_to_move = int(len(image_files) * 0.1)

	# 随机选择要移动的图像文件
	random_images = random.sample(image_files, num_images_to_move)

	# 移动选定的图像文件到目标文件夹
	for image_file in random_images:
		source_path = os.path.join(source_folder, image_file)
		target_path = os.path.join(target_folder, image_file)
		shutil.move(source_path, target_path)
		file_name = os.path.basename(source_path)
		label_path = os.path.join(source_label_folder,file_name)
		shutil.move(label_path, target_label_folder)

	print(f'成功移动了{num_images_to_move}个图像文件到验证集文件夹。')

random.seed(0)
for i in range(len(client_name)):
    divide(i)
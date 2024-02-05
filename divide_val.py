import os
import random
import shutil
from glob import glob
# client_name = ['BIDMC', 'HK', 'I2CVB', 'ISBI', 'ISBI_1.5']
# client_name =  ['RIM-ONE','REFUGE', 'ORIGA','G1020','Drishti-GS1']
# client_name = ['MoNuSAC2018','PanNuke2','PanNuke3','TNBC']
# client_name = ['PanNuke2Thyroid','PanNuke2Liver','PanNuke3Skin','PanNuke3Uterus']
# client_name = ['1', '4', '5', '6', '13', '16', '18', '20']
# client_name = ['PanNuke3Breast', 'PanNuke3Testis', 'PanNuke3Kidney', 'PanNuke3Bile-duct', 'PanNuke3Lung', 'PanNuke3Skin', 'PanNuke3Stomach',  'PanNuke3HeadNeck', 'PanNuke3Liver', 'PanNuke3Pancreatic', 'PanNuke3Ovarian', 'PanNuke3Esophagus', 'PanNuke3Bladder', 'PanNuke3Thyroid', 'PanNuke3Uterus', 'PanNuke3Colon', 'PanNuke3Prostate', 'PanNuke3Adrenal_gland']
# client_name =  ['Drishti-GS1']
# client_name = ['21']
# client_name = ['BIDMC', 'HK', 'I2CVB', 'ISBI', 'ISBI_1.5', 'UCL']
# client_name = ['MoNuSAC2018']
# data_path = '/mnt/diskB/name/Fundus_1024'
# data_path = '/mnt/diskB/name/Nuclei_1024'
# client_name = ['21']
# data_path = '/mnt/diskB/name/FeTS2022_FedDG_1024'
client_name = ['1', '2', '3', '4', '5']
data_path = '/mnt/diskB/name/CTLung_1024'
def move(source_file, destination_file):
	# 复制并覆盖目标文件
	shutil.copy2(source_file, destination_file)
	# 删除源文件
	os.remove(source_file)
def divide(client_idx):
	# data_path = '/mnt/diskB/name/Prostate_processed_1024'
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
		# print(source_path,target_folder)
		move(source_path, target_folder)
		file_name = os.path.basename(source_path)
		label_path = os.path.join(source_label_folder,file_name)
		move(label_path, target_label_folder)

	print(f'成功移动了{num_images_to_move}个图像文件到验证集文件夹。')

random.seed(0)
for i in range(len(client_name)):
    divide(i)
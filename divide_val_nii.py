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
# data_path = '/mnt/diskB/xxx/Fundus_1024'
# data_path = '/mnt/diskB/xxx/Nuclei_1024'
# client_name = ['21']
# data_path = '/mnt/diskB/xxx/FeTS2022_FedDG_1024'
client_name = ['1', '2', '3', '4', '5']
data_path = '/mnt/diskB/xxx/Liver2000_1024'
def move(source_file, destination_file):
	# 复制并覆盖目标文件
	shutil.copy2(source_file, destination_file)
	# 删除源文件
	os.remove(source_file)
def divide(client_idx):
	# data_path = '/mnt/diskB/xxx/Prostate_processed_1024'
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
	file_names = os.listdir(source_folder)
	# 根据文件名中的种类分类文件
	niis = {}
	for file_name in file_names:
		nii = file_name.split('.')[0]  # 获取文件所属种类
		if nii not in niis:
			niis[nii] = []
		niis[nii].append(file_name)
	nii_number=len(list(niis.keys()))
	print(nii_number)
	# 1/10 of the nii files are randomly selected as validation
	selected_niis = random.sample(list(niis.keys()), nii_number//10+1)
	num_images_to_move = 0
	# 将选定种类的文件移动到目标文件夹
	for nii in selected_niis:
		file_list = niis[nii]
		for file_name in file_list:
			source_path = os.path.join(source_folder, file_name)
			target_path = os.path.join(target_folder, file_name)
			shutil.move(source_path, target_path)
			source_path = os.path.join(source_label_folder, file_name)
			target_path = os.path.join(target_label_folder, file_name)
			shutil.move(source_path, target_path)
			num_images_to_move += 1
	print(f'成功移动了{num_images_to_move}个图像文件到验证集文件夹。')
random.seed(0)
for i in range(len(client_name)):
    divide(i)
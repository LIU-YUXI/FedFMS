import os
import random
import shutil
from glob import glob
# client_name =  ['REFUGE', 'ORIGA','G1020','Drishti-GS1']
# client_name = ['1', '6', '18', '21']
# client_name = ['PanNuke2Adrenal_gland','PanNuke2Esophagus', 'PanNuke3Bile-duct','PanNuke3Uterus', 'MoNuSAC2020','TNBC','MoNuSAC2018']
client_name = ['BIDMC', 'HK', 'I2CVB', 'ISBI', 'ISBI_1.5', 'UCL']
# data_path = '/mnt/diskB/name/Fundus_1024'
# data_path = '/mnt/diskB/name/Nuclei_1024'
# data_path = '/mnt/diskB/name/FeTS2022_1024'
def move(source_file, destination_file):
	# 复制并覆盖目标文件
	shutil.copy2(source_file, destination_file)
	# 删除源文件
	os.remove(source_file)
def divide(client_idx):
	data_path = '/mnt/diskB/xxx/Prostate_processed_1024'
	# image_list = glob('{}/{}/data_npy/*'.format(data_path,client_name[client_idx]))
	# label_dir='{}/{}/label_npy'.format(data_path,client_name[client_idx])
	# The path to the source and destination folders
	source_folder = '{}/{}/data_npy'.format(data_path,client_name[client_idx])
	target_folder = '{}/{}/val_data_npy'.format(data_path,client_name[client_idx])
	source_label_folder = '{}/{}/label_npy'.format(data_path,client_name[client_idx])
	target_label_folder = '{}/{}/val_label_npy'.format(data_path,client_name[client_idx])
	if not os.path.exists(target_folder):
		os.makedirs(target_folder)
	if not os.path.exists(target_label_folder):
		os.makedirs(target_label_folder)
	file_names = os.listdir(source_folder)
	# Sort files according to the category in the file name
	niis = {}
	for file_name in file_names:
		nii = file_name.split('.')[0]  # Gets the type of the file
		if nii not in niis:
			niis[nii] = []
		niis[nii].append(file_name)
	nii_number=len(list(niis.keys()))
	print(nii_number)
	# 1/10 of the nii files are randomly selected as validation
	selected_niis = random.sample(list(niis.keys()), nii_number//10+1)
	num_images_to_move = 0
	# Moves the selected type of file to the destination folder
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
	print(f'Successfully moved {num_images_to_move} image files to validation set folder.')
random.seed(0)
for i in range(len(client_name)):
    divide(i)
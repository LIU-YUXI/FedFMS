import os
import numpy as np
import cv2

# 输入和输出文件夹路径
def save(data_name):
	data_input_folder = "/mnt/diskB/lyx/Fundus_1024/{}/data_npy/".format(data_name)
	data_output_folder = "/mnt/diskB/lyx/Fundus_1024_JPG/{}/data_npy/".format(data_name)

	label_input_folder = "/mnt/diskB/lyx/Fundus_1024/{}/label_npy/".format(data_name)
	label_output_folder = "/mnt/diskB/lyx/Fundus_1024_JPG/{}/label_npy/".format(data_name)

	# 如果输出文件夹不存在，创建它们
	os.makedirs(data_output_folder, exist_ok=True)
	os.makedirs(label_output_folder, exist_ok=True)

	# 处理数据文件夹下的npy文件
	data_npy_files = [f for f in os.listdir(data_input_folder) if f.endswith('.npy')]

	for data_npy_file in data_npy_files:
		# 构建数据npy文件的完整路径
		data_npy_path = os.path.join(data_input_folder, data_npy_file)

		# 从npy文件中加载数据
		data = np.load(data_npy_path)
		bgr_data = cv2.cvtColor(data, cv2.COLOR_RGB2BGR)
		# 构建保存jpg文件的完整路径
		jpg_file = os.path.splitext(data_npy_file)[0] + '.jpg'
		jpg_path = os.path.join(data_output_folder, jpg_file)

		# 保存图像为jpg文件
		cv2.imwrite(jpg_path, bgr_data)

	print("数据文件夹转换完成！")

	# 处理标签文件夹下的npy文件
	label_npy_files = [f for f in os.listdir(label_input_folder) if f.endswith('.npy')]

	for label_npy_file in label_npy_files:
		# 构建标签npy文件的完整路径
		label_npy_path = os.path.join(label_input_folder, label_npy_file)

		# 从npy文件中加载数据
		label_data = np.load(label_npy_path)

		# 将值为1的像素转化为255
		label_data[label_data == 1] = 255

		# 构建保存jpg文件的完整路径
		jpg_file = os.path.splitext(label_npy_file)[0] + '-od.jpg'
		jpg_path = os.path.join(label_output_folder, jpg_file)

		# 保存处理后的图像为jpg文件
		cv2.imwrite(jpg_path, label_data[:,:,0].astype(np.uint8))

		# 构建保存jpg文件的完整路径
		jpg_file = os.path.splitext(label_npy_file)[0] + '-oc.jpg'
		jpg_path = os.path.join(label_output_folder, jpg_file)

		# 保存处理后的图像为jpg文件
		cv2.imwrite(jpg_path, label_data[:,:,1].astype(np.uint8))

	print("标签文件夹转换完成！")
client_name=['Drishti-GS1']# 'REFUGE', 'ORIGA','G1020',
for name in client_name:
    save(name)
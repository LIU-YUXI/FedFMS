import numpy as np
''''
type = np.load('/mnt/diskB/lyx/panuke3/Images/types.npy')
print(type)
images = np.load('/mnt/diskB/lyx/PanNuke3/Images/images.npy')
print(images[0],len(images))
masks = np.load('/mnt/diskB/lyx/PanNuke3/Masks/masks.npy')
print(masks[0].shape)
'''
from glob import glob
import os
client_name=glob('/mnt/diskB/lyx/Nuclei_1024/*')
name = []
for c in client_name:
    name.append(os.path.basename(c))
print(name)

print(len(['PanNuke3Breast', 'PanNuke3Testis', 'PanNuke3Kidney', 'PanNuke3Bile-duct', 'PanNuke3Lung', 'PanNuke3Skin', 'PanNuke3Stomach',  'PanNuke3HeadNeck', 'PanNuke3Liver', 'PanNuke3Pancreatic', 'PanNuke3Ovarian', 'PanNuke3Esophagus', 'PanNuke3Bladder', 'PanNuke3Thyroid', 'PanNuke3Uterus', 'PanNuke3Colon', 'PanNuke3Prostate', 'PanNuke3Adrenal_gland','PanNuke3Cervix']))
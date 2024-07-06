import numpy as np
from PIL import Image
import scipy.misc
import matplotlib.pyplot as plt

import numpy as np
import SimpleITK as sitk
import os
import numpy as np
from glob import glob
import time
import shutil
from PIL import Image
import cv2
def convert_from_nii_to_png(img):
    high = np.quantile(img,0.99)
    low = np.min(img)
    img = np.where(img > high, high, img)
    lungwin = np.array([low * 1., high * 1.])
    newimg = (img - lungwin[0]) / (lungwin[1] - lungwin[0])  
    newimg = (newimg * 255).astype(np.uint8)
    return newimg

class Brain(object):
    def __init__(self, site, base_path=None, split='train', transform=None):
        channels = {'1':3, '4':3, '5':3, '6':3, '13':3, '16':3, '18':3, '20':3, '21':3}
        assert site in list(channels.keys())
        self.split = split
        base_path = base_path if base_path is not None else'/mnt/diskB/name/FeTS2022'
        save_base_path = '/mnt/diskB/name/FeTS2022_FedDG_1024'
        sitedir = os.path.join(base_path, site)
        save_sitedir = os.path.join(save_base_path, site)
        imgsdir = os.path.join(sitedir, 'images')
        labelsdir = os.path.join(sitedir, 'labels')
        savesdir = os.path.join(save_sitedir, 'data_npy')
        labelnpydir = os.path.join(save_sitedir, 'label_npy')
        if not os.path.exists(savesdir):
            os.makedirs(savesdir)
        if not os.path.exists(labelnpydir):
            os.makedirs(labelnpydir)
        freqsdir = os.path.join(save_sitedir, 'freq_amp_npy')
        if not os.path.exists(freqsdir):
            os.makedirs(freqsdir)
        ossitedir = os.listdir(imgsdir) #np.load("../data/prostate/{}-dir.npy".format(site)).tolist()
        #print(ossitedir)
        # np.random.seed(2023)  
        lens = len(ossitedir)
        images, labels = [], []
        save_path, label_path, freq_path = [],[],[]
        for j,sample in enumerate(ossitedir):
            #if os.path.getsize(sampledir) < 1024 * 1024 and sampledir.endswith("_segmentation.nii.gz"):
            imgdir = os.path.join(imgsdir, sample)
            labeldir = os.path.join(labelsdir, sample)
            savedir = os.path.join(savesdir,sample)
            savelabeldir = os.path.join(labelnpydir,sample)
            savefreqdir = os.path.join(freqsdir,sample)
            label_v = sitk.ReadImage(labeldir)
            image_v = sitk.ReadImage(imgdir)
            label_v = sitk.GetArrayFromImage(label_v)
            #label_v[label_v != 4] = 0
            #label_v[label_v == 4] = 1
            label_v[label_v == 4] = 1
            label_v[label_v != 1] = 0
            image_v = sitk.GetArrayFromImage(image_v)
            image_v = convert_from_nii_to_png(image_v)
            # 155 240 240
            # print('image_v',image_v.shape)
            for i in range(1, label_v.shape[0] - 1):
                label = np.array(label_v[i, :, :])
                if (np.all(label == 0)) and i%5 != 0:
                    continue
                image = np.array(image_v[i-1:i+2, :, :])
                image = np.transpose(image,(1,2,0))
                
                labels.append(label)
                images.append(image)
                save_path.append(savedir+str(i)+'.npy')
                label_path.append(savelabeldir+str(i)+'.npy')
                freq_path.append(savefreqdir+str(i)+'.npy')
            #if(j>=100):
            #    break
        labels = np.array(labels).astype(int)
        images = np.array(images)
        
        print(site, split, images.shape)

        self.images, self.labels = images, labels
        self.savepathes,self.labelpathes,self.freqpathes = save_path,label_path, freq_path
        self.transform = transform
        self.channels = channels[site]
        self.labels = np.expand_dims(self.labels.astype(np.int64),axis=-1)
        # print('self.images[0], self.labels[0]',self.images[0].shape, self.labels[0].shape)
        print(self.savepathes[0],self.labelpathes[0])
    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        image=cv2.resize(image, (1024,1024), interpolation=cv2.INTER_LINEAR)
        label=cv2.resize(label, (256,256), interpolation=cv2.INTER_NEAREST)
        label = np.expand_dims(label.astype(np.int64),axis=-1)
        # print('image,label',image.shape,label.shape)
        # save_image=np.concatenate((image,label),axis=-1)
        np.save(self.savepathes[idx],image)
        np.save(self.labelpathes[idx],label)
        return # image, label
    def preprocess(self):
        for i in range(len(self.images)):
            self.__getitem__(i)
            # if(i>10):
            #    break
            
sites = ['1', '4', '5', '6', '13', '16', '18', '20', '21']
for site in sites:
    print('begin ',site)
    trainset = Brain(site=site)
    trainset.preprocess()
    print('end',site)
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
# 用这个把所有图片的幅度谱提取出来然后保存成npy
def extract_amp_spectrum(trg_img):

    fft_trg_np = np.fft.fft2( trg_img, axes=(-2, -1) )# 通过指定 axes=(-2, -1) 来对图像的最后两个维度进行傅里叶变换。
    amp_target, pha_trg = np.abs(fft_trg_np), np.angle(fft_trg_np)

    return amp_target# 幅度谱

# 通过在指定的截取窗口内，按照给定的比例交换本地图像和目标图像的幅度谱，从而实现频域上的插值操作。
def amp_spectrum_swap( amp_local, amp_target, L=0.1 , ratio=0):
    
    a_local = np.fft.fftshift( amp_local, axes=(-2, -1) )
    a_trg = np.fft.fftshift( amp_target, axes=(-2, -1) )

    _, h, w = a_local.shape
    b = (  np.floor(np.amin((h,w))*L)  ).astype(int)
    c_h = np.floor(h/2.0).astype(int)
    c_w = np.floor(w/2.0).astype(int)

    h1 = c_h-b
    h2 = c_h+b+1
    w1 = c_w-b
    w2 = c_w+b+1

    a_local[:,h1:h2,w1:w2] = a_local[:,h1:h2,w1:w2] * ratio + a_trg[:,h1:h2,w1:w2] * (1- ratio)
    a_local = np.fft.ifftshift( a_local, axes=(-2, -1) )
    return a_local

def freq_space_interpolation( local_img, amp_target, L=0 , ratio=0):
    
    local_img_np = local_img 

    # get fft of local sample
    fft_local_np = np.fft.fft2( local_img_np, axes=(-2, -1) )

    # extract amplitude and phase of local sample
    amp_local, pha_local = np.abs(fft_local_np), np.angle(fft_local_np)

    # swap the amplitude part of local image with target amplitude spectrum
    amp_local_ = amp_spectrum_swap( amp_local, amp_target, L=L , ratio=ratio)

    # get transformed image via inverse fft
    fft_local_ = amp_local_ * np.exp( 1j * pha_local )# 根据交换后的幅度谱和原始相位谱，重新构建频域表示 fft_local_。
    local_in_trg = np.fft.ifft2( fft_local_, axes=(-2, -1) ) #逆傅里叶
    local_in_trg = np.real(local_in_trg) # 将转换后的图像取实部，得到最终的插值结果。

    return local_in_trg

def draw_image(image):
    plt.imshow(image, cmap='gray')
    plt.subplots_adjust(top=1,bottom=0,left=0,right=1,hspace=0,wspace=0)

    plt.xticks([])
    plt.yticks([])
    
    return 0

def load_image( infilename ) :
    img = Image.open( infilename )
    img.load()
    data = np.asarray( img, dtype="int32" )
    return data
'''
im_local = Image.open("demo_samples/fundus_client4.jpg")
im_trg_list = [Image.open("demo_samples/fundus_client1.png"),
         Image.open("demo_samples/fundus_client2.jpg"),
         Image.open("demo_samples/fundus_client3.jpg")]

im_local = im_local.resize( (384,384), Image.BICUBIC )
im_local = np.asarray(im_local, np.float32)
im_local = im_local.transpose((2, 0, 1))

plt.figure(figsize=(18,3))
# 一一对应？
for client_idx,im_trg in enumerate(im_trg_list):
    im_trg = im_trg.resize( (384,384), Image.BICUBIC )
    im_trg = np.asarray(im_trg, np.float32)
    im_trg = im_trg.transpose((2, 0, 1))

    L = 0.003

    # visualize local data, target data, amplitude spectrum of target data
    plt.figure(figsize=(18,3))
    plt.subplot(1,8,1)
    draw_image((im_local / 255).transpose((1, 2, 0)))    
    plt.xlabel("Local Image", fontsize=12)

    plt.subplot(1,8,2)
    draw_image((im_trg / 255).transpose((1, 2, 0)))
    plt.xlabel("Target Image (Client {})".format(client_idx), fontsize=12)
    
    # amplitude spectrum of target data
    amp_target = extract_amp_spectrum(im_trg)
    amp_target_shift = np.fft.fftshift( amp_target, axes=(-2, -1) )
    
    plt.subplot(1,8,3)
    draw_image(np.clip((np.log(amp_target_shift)/ np.max(np.log(amp_target_shift))).transpose((1, 2, 0)), 0, 1))
    plt.xlabel("Target Amp (Client {})".format(client_idx), fontsize=12)
    
    # continuous frequency space interpolation
    for idx, i in enumerate([0.2, 0.4, 0.6, 0.8, 1.0]):
        plt.subplot(1,8,idx+4)
        local_in_trg = freq_space_interpolation(im_local, amp_target, L=L, ratio=1-i)
        local_in_trg = local_in_trg.transpose((1,2,0))
        draw_image((np.clip(local_in_trg / 255, 0, 1)))
        plt.xlabel("Interpolation Rate: {}".format(i), fontsize=12)
    plt.show()
'''

class Brain(object):
    def __init__(self, site, base_path=None, split='train', transform=None):
        channels = {'1':3, '4':3, '5':3, '6':3, '13':3, '16':3, '18':3, '20':3, '21':3}
        assert site in list(channels.keys())
        self.split = split
        base_path = base_path if base_path is not None else'/mnt/diskB/lyx/FeTS2022'
        save_base_path = '/mnt/diskB/lyx/FeTS2022_FedDG_1024'
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
        # np.random.seed(2023)  # 先定义一个随机数种子
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
        print(self.savepathes[0],self.labelpathes[0],self.freqpathes[0])
    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        image=cv2.resize(image, (1024,1024), interpolation=cv2.INTER_LINEAR)
        label=cv2.resize(label, (256,256), interpolation=cv2.INTER_NEAREST)
        label = np.expand_dims(label.astype(np.int64),axis=-1)
        # print('image,label',image.shape,label.shape)
        # 先把图片+mask通道的存一下
        # save_image=np.concatenate((image,label),axis=-1)
        np.save(self.savepathes[idx],image)
        np.save(self.labelpathes[idx],label)
        # 存频率谱
        # amp=extract_amp_spectrum(image)
        # print('amp',amp.shape)
        # np.save(self.freqpathes[idx],amp)
        '''
        if self.transform is not None:
            if self.split == 'train':
                R1 = RandomRotate90()
                image, label = R1(image, label)
                R2 = RandomFlip()
                image, label = R2(image, label)

            image = np.transpose(image,(2, 0, 1))
            image = torch.Tensor(image)
            
            label = self.transform(label)
        '''
        '''
        im_local=image
        

        im_local = im_local.resize( (384,384), Image.BICUBIC )
        im_local = np.asarray(im_local, np.float32)
        im_local = im_local.transpose((2, 0, 1))
        # 一一对应？
        for client_idx,im_trg in enumerate(im_trg_list):
            im_trg = im_trg.resize( (384,384), Image.BICUBIC )
            im_trg = np.asarray(im_trg, np.float32)
            im_trg = im_trg.transpose((2, 0, 1))

            L = 0.003

            # visualize local data, target data, amplitude spectrum of target data
            plt.figure(figsize=(18,3))
            plt.subplot(1,8,1)
            draw_image((im_local / 255).transpose((1, 2, 0)))    
            plt.xlabel("Local Image", fontsize=12)

            plt.subplot(1,8,2)
            draw_image((im_trg / 255).transpose((1, 2, 0)))
            plt.xlabel("Target Image (Client {})".format(client_idx), fontsize=12)
            
            # amplitude spectrum of target data
            amp_target = extract_amp_spectrum(im_trg)
            amp_target_shift = np.fft.fftshift( amp_target, axes=(-2, -1) )
            
            plt.subplot(1,8,3)
            draw_image(np.clip((np.log(amp_target_shift)/ np.max(np.log(amp_target_shift))).transpose((1, 2, 0)), 0, 1))
            plt.xlabel("Target Amp (Client {})".format(client_idx), fontsize=12)
            
            # continuous frequency space interpolation
            for idx, i in enumerate([0.2, 0.4, 0.6, 0.8, 1.0]):
                plt.subplot(1,8,idx+4)
                local_in_trg = freq_space_interpolation(im_local, amp_target, L=L, ratio=1-i)
                local_in_trg = local_in_trg.transpose((1,2,0))
                draw_image((np.clip(local_in_trg / 255, 0, 1)))
                plt.xlabel("Interpolation Rate: {}".format(i), fontsize=12)
            plt.show()
        '''
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
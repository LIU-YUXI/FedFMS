import os
import torch
import numpy as np
from glob import glob
from torch.utils.data import Dataset
import h5py
import itertools
from torch.utils.data.sampler import Sampler
import random 
from scipy import ndimage
from scipy.ndimage import _ni_support
from scipy.ndimage.morphology import distance_transform_edt, binary_erosion,\
    generate_binary_structure
import cv2
from PIL import Image
from sam_utils import random_click,show_element
current_path = os.getcwd()
class ProstateDataset(Dataset):
    """ LA Dataset """
    def __init__(self,data_path=None, client_idx=None, freq_site_idx=None, split='train', transform=None, client_name=None):
        self.transform = transform
        self.client_name = ['BIDMC', 'HK', 'I2CVB', 'ISBI', 'ISBI_1.5', 'UCL'] if client_name is None else client_name 
        self.freq_list_clients = []
        if split=='train':
            data_path = data_path if data_path is not None else '/mnt/diskB/lyx/Prostate_processed_1024'
            # Reads the file list for image and label
            self.image_list = glob('{}/{}/data_npy/*'.format(data_path,self.client_name[client_idx]))
            self.label_dir='{}/{}/label_npy'.format(data_path,self.client_name[client_idx])
        self.freq_site_index = freq_site_idx

        print("total {} slices".format(len(self.image_list)))

    def __len__(self):
        return len(self.image_list)
    def __getitem__(self, idx):
        raw_file = self.image_list[idx]
        mask_patches = []
        image_patch = np.load(raw_file)
        file_name = os.path.basename(raw_file)
        label_path = os.path.join(self.label_dir,file_name)
        mask_patch = np.load(label_path)
        image_patches = image_patch.copy()
        # prompt is cancelled, so pt is set to 0
        pt=np.array([0,0])
        image_patches = image_patches.transpose(2, 0, 1)
        mask_patches = mask_patch.transpose(2, 0, 1)
        sample = {"image": image_patches.astype(np.float32)/ 255.0, "label": mask_patches.astype(np.float32), "pt":pt}
        return sample
class Dataset(Dataset):
    """ LA Dataset """
    def __init__(self,data_path=None, client_idx=None, freq_site_idx=None, split='train', transform=None, client_name=None):
        self.transform = transform
        self.client_name = ['1', '4', '5', '6', '13', '16', '18', '20', '21'] if client_name is None else client_name 
        self.freq_list_clients = []
        if split=='train':
            # Reads the file list for image and label
            data_path = data_path if data_path is not None else '/mnt/diskB/lyx/FeTS2022_FedDG_1024'
            self.image_list = glob('{}/{}/data_npy/*'.format(data_path,self.client_name[client_idx]))
            self.label_dir='{}/{}/label_npy'.format(data_path,self.client_name[client_idx])
        self.freq_site_index = freq_site_idx
        self.client_idx=client_idx
        print("total {} slices".format(len(self.image_list)))

    def __len__(self):
        return len(self.image_list)
    def __getitem__(self, idx):
        raw_file = self.image_list[idx]
        
        mask_patches = []

        image_patch = np.load(raw_file)
        file_name = os.path.basename(raw_file)
        label_path = os.path.join(self.label_dir,file_name)
        mask_patch = np.load(label_path)
        image_patches = image_patch.copy()
        pt=np.array([0,0])
        image_patches = image_patches.transpose(2, 0, 1)
        mask_patches = mask_patch.transpose(2, 0, 1)
        sample = {"image": image_patches.astype(np.float32)/ 255.0, "label": mask_patches.astype(np.float32), "pt":pt}
        return sample


# The following are data-enhanced functions that will work in the future and not be used for the time being

def _get_coutour_sample(y_true):
    disc_mask = np.expand_dims(y_true[..., 0], axis=2)

    disc_erosion = ndimage.binary_erosion(disc_mask[..., 0], iterations=5).astype(disc_mask.dtype)
    disc_dilation = ndimage.binary_dilation(disc_mask[..., 0], iterations=5).astype(disc_mask.dtype)
    disc_contour = np.expand_dims(disc_mask[..., 0] - disc_erosion, axis = 2)
    disc_bg = np.expand_dims(disc_dilation - disc_mask[..., 0], axis = 2)
    cup_mask = np.expand_dims(y_true[..., 0], axis=2)

    cup_erosion = ndimage.binary_erosion(cup_mask[..., 0], iterations=5).astype(cup_mask.dtype)
    cup_dilation = ndimage.binary_dilation(cup_mask[..., 0], iterations=5).astype(cup_mask.dtype)
    cup_contour = np.expand_dims(cup_mask[..., 0] - cup_erosion, axis = 2)
    cup_bg = np.expand_dims(cup_dilation - cup_mask[..., 0], axis = 2)

    return [disc_contour.transpose(2, 0, 1), disc_bg.transpose(2, 0, 1), cup_contour.transpose(2, 0, 1), cup_bg.transpose(2, 0, 1)]

class CenterCrop(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        # pad the sample if necessary
        if label.shape[0] <= self.output_size[0] or label.shape[1] <= self.output_size[1] or label.shape[2] <= \
                self.output_size[2]:
            pw = max((self.output_size[0] - label.shape[0]) // 2 + 3, 0)
            ph = max((self.output_size[1] - label.shape[1]) // 2 + 3, 0)
            pd = max((self.output_size[2] - label.shape[2]) // 2 + 3, 0)
            image = np.pad(image, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
            label = np.pad(label, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)

        (w, h, d) = image.shape

        w1 = int(round((w - self.output_size[0]) / 2.))
        h1 = int(round((h - self.output_size[1]) / 2.))
        d1 = int(round((d - self.output_size[2]) / 2.))

        label = label[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        image = image[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]

        return {'image': image, 'label': label}


class RandomCrop(object):
    """
    Crop randomly the image in a sample
    Args:
    output_size (int): Desired output size
    """

    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        # pad the sample if necessary
        if label.shape[0] <= self.output_size[0] or label.shape[1] <= self.output_size[1] or label.shape[2] <= \
                self.output_size[2]:
            pw = max((self.output_size[0] - label.shape[0]) // 2 + 3, 0)
            ph = max((self.output_size[1] - label.shape[1]) // 2 + 3, 0)
            pd = max((self.output_size[2] - label.shape[2]) // 2 + 3, 0)
            image = np.pad(image, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
            label = np.pad(label, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)

        (w, h, d) = image.shape
        # if np.random.uniform() > 0.33:
        #     w1 = np.random.randint((w - self.output_size[0])//4, 3*(w - self.output_size[0])//4)
        #     h1 = np.random.randint((h - self.output_size[1])//4, 3*(h - self.output_size[1])//4)
        # else:
        w1 = np.random.randint(0, w - self.output_size[0])
        h1 = np.random.randint(0, h - self.output_size[1])
        d1 = np.random.randint(0, d - self.output_size[2])

        label = label[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        image = image[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        return {'image': image, 'label': label}


class RandomRotFlip(object):
    """
    Crop randomly flip the dataset in a sample
    Args:
    output_size (int): Desired output size
    """

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        k = np.random.randint(0, 4)
        image = np.rot90(image, k)
        label = np.rot90(label, k)
        axis = np.random.randint(0, 2)
        image = np.flip(image, axis=axis).copy()
        label = np.flip(label, axis=axis).copy()

        return {'image': image, 'label': label}


class RandomNoise(object):
    def __init__(self, mu=0, sigma=0.1):
        self.mu = mu
        self.sigma = sigma

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        noise = np.clip(self.sigma * np.random.randn(image.shape[0], image.shape[1], image.shape[2]), -2*self.sigma, 2*self.sigma)
        noise = noise + self.mu
        image = image + noise
        return {'image': image, 'label': label}


class CreateOnehotLabel(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        onehot_label = np.zeros((self.num_classes, label.shape[0], label.shape[1], label.shape[2]), dtype=np.float32)
        for i in range(self.num_classes):
            onehot_label[i, :, :, :] = (label == i).astype(np.float32)
        return {'image': image, 'label': label,'onehot_label':onehot_label}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image = sample['image']
        image = image.reshape(1, image.shape[0], image.shape[1], image.shape[2]).astype(np.float32)
        if 'onehot_label' in sample:
            return {'image': torch.from_numpy(image), 'label': torch.from_numpy(sample['label']).long(),
                    'onehot_label': torch.from_numpy(sample['onehot_label']).long()}
        else:
            return {'image': torch.from_numpy(image), 'label': torch.from_numpy(sample['label']).long()}


class TwoStreamBatchSampler(Sampler):
    """Iterate two sets of indices

    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through
    as many times as needed.
    """
    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch)
            in zip(grouper(primary_iter, self.primary_batch_size),
                    grouper(secondary_iter, self.secondary_batch_size))
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size

def iterate_once(iterable):
    return np.random.permutation(iterable)


def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)
    return itertools.chain.from_iterable(infinite_shuffles())


def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)

def low_freq_mutate_np( amp_src, amp_trg, L=0.1 ):
    a_src = np.fft.fftshift( amp_src, axes=(-2, -1) )
    a_trg = np.fft.fftshift( amp_trg, axes=(-2, -1) )

    _, h, w = a_src.shape
    b = (  np.floor(np.amin((h,w))*L)  ).astype(int)
    c_h = np.floor(h/2.0).astype(int)
    c_w = np.floor(w/2.0).astype(int)
    # print (b)
    h1 = c_h-b
    h2 = c_h+b+1
    w1 = c_w-b
    w2 = c_w+b+1

    ratio = random.randint(1,10)/10
    # print('a_src,a_trg',a_src.shape,a_trg.shape)
    # return a_src
    a_src[:,h1:h2,w1:w2] = a_trg[:,h1:h2,w1:w2]
    # a_src[:,h1:h2,w1:w2] = a_src[:,h1:h2,w1:w2] * ratio + a_trg[:,h1:h2,w1:w2] * (1- ratio)
    # a_src[:,h1:h2,w1:w2] = a_trg[:,h1:h2,w1:w2]
    a_src = np.fft.ifftshift( a_src, axes=(-2, -1) )
    # a_trg[:,h1:h2,w1:w2] = a_src[:,h1:h2,w1:w2]
    # a_trg = np.fft.ifftshift( a_trg, axes=(-2, -1) )
    return a_src
    
def source_to_target_freq( src_img, amp_trg, L=0.1 ):
    # exchange magnitude
    # input: src_img, trg_img
    src_img = src_img.transpose((2, 0, 1))
    src_img_np = src_img #.cpu().numpy()
    fft_src_np = np.fft.fft2( src_img_np, axes=(-2, -1) )

    # extract amplitude and phase of both ffts
    amp_src, pha_src = np.abs(fft_src_np), np.angle(fft_src_np)

    # mutate the amplitude part of source with target
    amp_src_ = low_freq_mutate_np( amp_src, amp_trg, L=L )

    # mutated fft of source
    fft_src_ = amp_src_ * np.exp( 1j * pha_src )

    # get the mutated image
    src_in_trg = np.fft.ifft2( fft_src_, axes=(-2, -1) )
    src_in_trg = np.real(src_in_trg)

    return src_in_trg.transpose(1, 2, 0)


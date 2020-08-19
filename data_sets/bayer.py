import cv2
import random
import os.path
import sys
import pickle
import numpy as np
from glob import glob
from torch.utils.data import Dataset, DataLoader, Subset
sys.path.append('../')
from image_utils import random_dual_augmentation, coll_fn_rand_rot90_float, random_crop, base_collate_fn
from typing import List, Dict
from dataclasses import dataclass
import colour_demosaicing

__author__ = "Assaf Genosar"


def file_name_order(path):
    fname = os.path.basename(path)
    idx = fname.split('.')[0]
    if idx.isdigit():
        return int(fname.split('.')[0])
    else:
        return idx


def list_images(folder_path: str, pattern = '') -> List[str]:
    """return a sorted list of png files"""
    files_list = glob(os.path.join(folder_path, '*'+pattern+'*'))
    files_list.sort(key=file_name_order)
    return files_list




@dataclass
class DatasetParams:
    """stores dataset parameters and validate input"""
    img_dir: str
    ref_dir: str

class Bayer(Dataset):
    """ Texture data format dataset """
    def __init__(self, data_sets: dict, patch_size: int, bi_linear_demosaic: bool, seed= 42, shuffle=True, train_split=0.8):
        """
        Parameters
        ----------
        data_sets:
            img_dir: str - directory of sampled images
            ref_dir: str
        patch_size: int
        shuffle: bool
            shuffle the samples order

        train_split:
            portion of the training samples
        """
        self.data_sets: Dict[str, DatasetParams] = {}
        for set_name, prm in data_sets.items():
            self.data_sets[set_name] = DatasetParams(**prm)
        self.data_map = {}
        self.data: List = []   # list of patches
        self.train_split = train_split
        self.train_idx: np.ndarray = None
        self.test_idx: np.ndarray = None
        self.train_split = train_split
        self.bi_linear_demosaic = bi_linear_demosaic
        self.shuffle = shuffle
        self.seed = seed
        self.patch_size = patch_size

    def prepare_data(self):
        last_idx = 0
        for dataset_name, prm in self.data_sets.items():
            ref_images = list_images(prm.ref_dir)
            for img_path in list_images(prm.img_dir):
                ref_path = img_path.replace(prm.img_dir, prm.ref_dir)
                if ref_path in ref_images:
                    self.data.append((img_path, ref_path))

            print('added {} images from {} dataset'.format(len(self.data) - last_idx, dataset_name))
            last_idx = len(self.data)
        if self.shuffle:
            # random.seed = self.seed
            random.shuffle(self.data)
        train_size = int(len(self.data) * self.train_split)
        self.train_idx = np.arange(train_size)
        self.test_idx = np.arange(train_size, len(self.data))
        with open('/tmp/test_images.txt', 'w') as f:
            for idx in self.test_idx:
                f.write(str(self.data[idx][0]) + os.linesep)
        print('wrote test images to: /tmp/test_images.txt')


    def random_sample_patch(self, im, lbl, do_transpose=True):
        y0, y1, x0, x1 = 0, im.shape[0] - 1, 0, im.shape[1] - 1
        sy = random.randint(y0, y1 - self.patch_size)
        sx = random.randint(x0, x1 - self.patch_size)
        sy = sy - (sy % 2)
        sx = sx - (sx % 2)
        im_p = im[sy:sy+self.patch_size, sx:sx+self.patch_size]
        lbl_p = lbl[sy:sy+self.patch_size, sx:sx+self.patch_size]
        if do_transpose:
            if self.bi_linear_demosaic:
                im_p = np.transpose(im_p, (2, 0, 1))
            lbl_p = np.transpose(lbl_p, (2, 0, 1))
        return im_p.astype(np.float32) / 255, lbl_p.astype(np.float32) / 255

    def __getitem__(self, item):
        img_path, ref_path = self.data[item]

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if self.bi_linear_demosaic:
            img = colour_demosaicing.demosaicing_CFA_Bayer_bilinear(img, pattern='GRBG')
            img = np.clip(img, 0, 255)
        # img = cv2.cvtColor(img, cv2.COLOR_BAYER_RG2RGB)
        ref = cv2.cvtColor(cv2.imread(ref_path), cv2.COLOR_BGR2RGB)
        sample, label = self.random_sample_patch(img, ref)
        return sample, label

    def __len__(self):
        return len(self.data)

    def get_data_loaders(self, batch_size):
        self.prepare_data()
        train_loaders = [DataLoader(dataset=Subset(self, indices=self.train_idx), pin_memory=True, num_workers=4,
                                    batch_size=batch_size, collate_fn=base_collate_fn, shuffle=True)]
        test_loaders = [DataLoader(dataset=Subset(self, indices=self.test_idx), pin_memory=False, shuffle=False,
                                   batch_size=batch_size, collate_fn=base_collate_fn)]
        return train_loaders, test_loaders



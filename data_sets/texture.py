import re
import os.path
import random
import numpy as np
from skimage.io import imread
# from PIL import Image
from collections import defaultdict
from glob import glob
from torch.utils.data import Dataset, DataLoader, Subset
import sys
sys.path.append('../')
from image_utils import random_dual_augmentation, coll_fn_rand_rot90_float, random_crop
# from skimage.io import imsave
from typing import List, Dict
from dataclasses import dataclass
import torch

__author__ = "Assaf Genosar"

# def crop_center(img, target):
#     cropy, cropx = target
#     y, x = img.shape[:2]
#     startx = x//2-(cropx//2)
#     starty = y//2-(cropy//2)
#     return img[starty:starty+cropy, startx:startx+cropx]




@dataclass
class DatasetParams:
    """stores dataset parameters and validate input"""
    root_dir: str
    psi_prefix: str
    gt_prefix: str
    gt_suffix: str
    img_suffix: str
    psi_start: int
    psi_end: int
    psi_step: int
    split_method: str

class TextureDataset(Dataset):
    """ Texture data format dataset """
    def __init__(self, data_sets: dict, patch_size: int, shuffle=True, sigma=2, seed=42, train_split=0.8, egde_crop=10):
        """
        Parameters
        ----------
        data_sets:
            root_path: str
                    path to psi directories
            psi_range: iterable pair
                psi range (maximal is [-4, 10])
            psi_step: int
            psi_prefix: str
            split_method: str
                scene_<name of scene> - will use single scene for validation and the rest for training
                even - will randomly split to training and validation according to training split ratio - in every scene
                random - will randomly split to training and validation (according to training split ratio) -
                         the collection of all samples
        shuffle: bool
            shuffle the samples order
        mu: float
            added noise mean
        sigma: float
            added noise sigma (0 is do not add)
        seed: int
            random's seed
        train_split:
            portion of the training samples
        """
        self.data_sets: Dict[str, DatasetParams] = {}
        for dataset, dataset_prm in data_sets.items():
            self.data_sets[dataset] = DatasetParams(**dataset_prm)
        self.data_map = {}
        self.data: List = []
        self.seed = seed
        self.train_split = train_split
        self.train_idx = defaultdict(list)
        self.test_idx = defaultdict(list)
        self.mu = 0
        self.sigma = sigma / 255
        self.train_split = train_split
        self.shuffle = shuffle
        self.patch_size = patch_size
        self.edge_border = egde_crop

    def map_data(self, dataset_name):
        psi_dirs = {}
        prm = self.data_sets[dataset_name]
        gt_dir = None
        for dir_name in os.listdir(prm.root_dir):
            if prm.psi_prefix in dir_name:
                try:
                    psi_dirs[int(eval(dir_name.replace(prm.psi_prefix, '')))] = os.path.join(prm.root_dir, dir_name)
                except Exception:
                    continue
            elif prm.gt_prefix in dir_name:
                gt_dir = os.path.join(prm.root_dir, dir_name)
        assert gt_dir is not None, "ERROR - GT folder is missing for dataset {}, root folder: {}".format(
            dataset_name, prm.root_dir)
        for gt_path in glob(os.path.join(gt_dir,'*', '*', '*' + prm.gt_suffix)):
            for psi in self.data_map[dataset_name]:
                assert psi in psi_dirs, 'ERROR - psi {} folder is missing in {}'.format(psi, prm.root_dir)
                img_name = os.path.basename(gt_path).replace(prm.gt_suffix, prm.img_suffix)
                im_path = os.path.join(psi_dirs[psi], img_name)
                if os.path.exists(im_path):
                    self.data_map[dataset_name][psi].append(len(self.data))
                    self.data.append((im_path, gt_path))

        for psi in self.data_map[dataset_name]:
            if self.shuffle:
                random.shuffle(self.data_map[dataset_name][psi])
            train_len = int(len(self.data_map[dataset_name][psi]) * self.train_split)
            self.train_idx[dataset_name].extend(self.data_map[dataset_name][psi][:train_len])
            self.test_idx[dataset_name].extend(self.data_map[dataset_name][psi][train_len:])
            if self.shuffle:
                random.shuffle(self.train_idx[dataset_name])
                random.shuffle(self.test_idx[dataset_name])

    def prepare_data(self):
        random.seed(self.seed)
        for dataset, prm in self.data_sets.items():
            self.data_map[dataset] = {}
            for psi in range(prm.psi_start, prm.psi_end + prm.psi_step, prm.psi_step):
                self.data_map[dataset][psi] = []  # list of indexes in data
            # prepare the data according to split method
            if 'even' in prm.split_method:
                self.map_data(dataset)


    def __getitem__(self, item):
        img_path, gt_path = self.data[item]
        img = imread(img_path)
        gt = imread(gt_path)
        img_c, gt_c = random_crop(img, gt, self.patch_size, self.edge_border)
        sample, label = random_dual_augmentation(img_c/255 , gt_c/255 , self.sigma, pad_divisor=32, do_transpose=True)
        # sample_t = torch.as_tensor(sample, dtype=torch.float32)
        # label_t = torch.as_tensor(label, dtype=torch.float32)
        return sample, label

    def __len__(self):
        return len(self.data)

    def get_data_loaders(self, batch_size):
        self.prepare_data()
        train_loaders = []
        test_loaders = []
        for data_set in self.data_map:
            train_loaders.append(
                DataLoader(dataset=Subset(self, indices=self.train_idx[data_set]), pin_memory=True, shuffle=True,
                           batch_size=batch_size, num_workers=8, collate_fn=coll_fn_rand_rot90_float))
            test_loaders.append(DataLoader(dataset=Subset(self, indices=self.test_idx[data_set]), pin_memory=False,
                                           batch_size=batch_size, num_workers=2, collate_fn=coll_fn_rand_rot90_float))
        return train_loaders, test_loaders



import re
import os.path
import random
import torch
import numpy as np
from skimage.io import imread
from skimage.util import random_noise, img_as_float64
# from PIL.Image
from collections import defaultdict
from glob import glob
from torch.utils.data import Dataset, DataLoader, random_split, Subset
import torchvision.transforms as tsfm


__author__ = "Assaf Genosar"

def round_down(num, divisor):
    return num - (num % divisor)

def crop_center(img, target):
    cropy, cropx = target
    y,x = img.shape[:2]
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)
    return img[starty:starty+cropy,startx:startx+cropx]

def collate_fn(data):
    """Creates mini-batch tensors from the list of tuples (image, caption).
    Args:
        data: list of tuple (image, depth).
            - image: torch tensor of shape (3, w, h) torch.float64.
            - depth: torch tensor of shape (w, h) torch.long.
    Returns:
        images: torch tensor of shape (batch_size, 3, w, h).
        depths: torch tensor of shape (batch_size, w, h).
    """
    images, depths = zip(*data)
    # Merge tensors (from tuple of 3D tensor to 4D tensor).
    images = torch.stack(images, 0).float()
    # images = torch.nn.functional.pad(images)
    depths = torch.stack(depths, 0).long()
    return images, depths

def depth_read(filename):
    """ Read depth data from file, return as numpy array.
    this method (with minor modifications by me) originally came with the:
    MPI-Sintel low-level computer vision benchmark.
    For more details about the benchmark, please visit www.mpi-sintel.de
    CHANGELOG:  v1.0 (2015/02/03): First release
    Copyright (c) 2015 Jonas Wulff
    Max Planck Institute for Intelligent Systems, Tuebingen, Germany"""
    tag_float = 202021.25
    with open(filename, 'rb') as f:
        check = np.fromfile(f, dtype=np.float32,  count=1)[0]
        assert check == tag_float, 'depth_read:: Wrong tag in flow file (should be: {}, is: {}). Big-endian machine?'\
            .format(tag_float, check)
        width = np.fromfile(f, dtype=np.int32, count=1)[0]
        height = np.fromfile(f, dtype=np.int32, count=1)[0]
        size = width * height
        assert 0 < size < 100000000, ' depth_read:: Wrong input size (width = {}, height = {}).'.format(width, height)
        depth = np.fromfile(f, dtype=np.float32, count=-1).reshape((height, width))
    return depth


class SintelDataset(Dataset):
    """ Sitel's data format dataset """
    def __init__(self, img_dir: list, depth_dir: str, img_suffix: str, depth_suffix: str,
                 split_method: str, shuffle=True, mu=0, sigma=0.1, seed=42, train_split=None, scene_separator=None):
        """

        Parameters
        ----------
        img_dir: str
            path to images directory
        depth_dir: str
            path to depth maps directory
        img_suffix: str
        depth_suffix: str
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
        scene_separator: str
            regular expression which will indicate the different scenes
        """
        self.data = []  # pairs of (img, depth)
        self.data_map = defaultdict(list)   # list of indexes in data
        self.seed = seed
        self.split_method = split_method
        self.train_split = train_split
        self.train_idx = None
        self.test_idx = None
        self.mu = mu
        self.sigma = sigma
        random.seed(seed)


        # prepare the data according to split method
        if 'scene' in split_method or 'even' in split_method:
            assert scene_separator is not None, "Error - scene separator argument is missing"
            self.map_data(img_dir, depth_dir, img_suffix, depth_suffix, scene_separator)
            assert len(self.data) > 0, "ERROR - dataset is empty - check dataset parameters"

        if 'scene' in split_method:
            test_scene = split_method.replace('scene_', '')
            self.test_idx = np.array(self.data_map[test_scene], dtype=np.int)
            self.train_idx = np.zeros(len(self.data) - self.test_idx.size, dtype=np.int)
            train_ptr = 0
            for scene, ids in self.data_map.items():
                if scene != test_scene:
                    self.train_idx[train_ptr:train_ptr + len(ids)] = ids
        elif 'even' in split_method:
            assert train_split is not None, "Error - train_split argument is missing"
            self.train_idx = np.zeros(int(len(self.data) * train_split), dtype=np.int)
            self.test_idx = np.zeros(len(self.data) - self.train_idx.size, dtype=np.int)
            train_ptr = 0
            test_ptr = 0
            for scene, ids in self.data_map.items():
                if shuffle:
                    random.shuffle(ids)
                train_portion = min(int(len(ids)*train_split + 0.5), self.train_idx.size - train_ptr)
                test_portion = len(ids) - train_portion
                self.train_idx[train_ptr:train_ptr+train_portion] = ids[:train_portion]
                self.test_idx[test_ptr:test_ptr + test_portion] = ids[train_portion:]
                train_ptr += train_portion
                test_ptr += test_portion
        elif 'random' in split_method:
            assert train_split is not None, "Error - train_split argument is missing"
            for img_path in glob(os.path.join(img_dir, '*' + img_suffix)):
                depth_file = os.path.basename(img_path).replace(img_suffix, depth_suffix)
                depth_path = os.path.join(depth_dir, depth_file)
                if os.path.exists(depth_path):
                    self.data.append((img_path, depth_path))

        # here i'm assuming that all the data is homogeneous
        first_im = imread(self.data[0][0])
        width, height = first_im.shape[:2]
        self.target_size = (round_down(width, 32), round_down(height, 32))
        # self.im_transform = tsfm.Compose([tsfm.CenterCrop(target_size), tsfm.ToTensor()])

    def __getitem__(self, item):
        img_path, dpt_path = self.data[item]
        img = imread(img_path)
        img = crop_center(img, self.target_size)
        img = img.transpose(2, 0, 1)
        if self.sigma > 0:
            img = random_noise(img, mode='gaussian', seed=self.seed, var=self.sigma**2)
        else:
            # random noise will convert the image to float
            img = img_as_float64(img)
        img = torch.as_tensor(img, dtype=torch.float64)
        dpt = depth_read(dpt_path)
        dpt = crop_center(dpt, self.target_size)
        return torch.as_tensor(img, dtype=torch.float64), torch.as_tensor(dpt, dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def map_data(self, img_dir, depth_dir, img_suffix, depth_suffix, scene_separator):
        for img_path in glob(os.path.join(img_dir, '*' + img_suffix)):
            img_name = os.path.basename(img_path)
            scene = re.split(scene_separator, img_name)[0] + re.search(scene_separator, img_name).group(0)
            depth_file = os.path.basename(img_path).replace(img_suffix, depth_suffix)
            depth_path = os.path.join(depth_dir, depth_file)
            if os.path.exists(depth_path):
                self.data.append((img_path, depth_path))
                self.data_map[scene].append(len(self.data) - 1)

    def get_data_loaders(self, batch_size):
        if self.split_method == 'random':
            train_size = int(self.train_split * len(self))
            test_size = len(self) - train_size
            torch.manual_seed(self.seed)
            train_dataset, test_dataset = random_split(self, [train_size, test_size])
            train_data_loader = DataLoader(train_dataset, batch_size=batch_size)
            test_data_loader = DataLoader(test_dataset, batch_size=batch_size)
            return train_data_loader, test_data_loader
        else:
            train_loader = DataLoader(dataset=Subset(self, indices=self.train_idx), batch_size=batch_size,
                                      collate_fn=collate_fn)
            test_loader = DataLoader(dataset=Subset(self, indices=self.test_idx), batch_size=batch_size,
                                     collate_fn=collate_fn)
            return train_loader, test_loader




import re
import os.path
import random
import numpy as np
from skimage.io import imread
from collections import defaultdict
from glob import glob
from torch.utils.data import Dataset, DataLoader, Subset
import sys
sys.path.append('../')
from image_utils import random_dual_augmentation, collate_fn_random_rot90
from skimage.io import imsave


__author__ = "Assaf Genosar"

# def crop_center(img, target):
#     cropy, cropx = target
#     y, x = img.shape[:2]
#     startx = x//2-(cropx//2)
#     starty = y//2-(cropy//2)
#     return img[starty:starty+cropy, startx:startx+cropx]


def map_data(d_map: defaultdict, data: list, params: dict):
    data_len = 0
    for img_path in glob(os.path.join(params['img_dir'], '*' + params['img_suffix'])):
        img_name = os.path.basename(img_path)
        scene = re.split(params['scene_separator'],
                         img_name)[0] + re.search(params['scene_separator'], img_name).group(0)
        depth_file = os.path.basename(img_path).replace(params['img_suffix'], params['depth_suffix'])
        depth_path = os.path.join(params['depth_dir'], depth_file)
        if os.path.exists(depth_path):
            data.append((img_path, depth_path))
            d_map[scene].append(len(data) - 1)
            data_len += 1
    params.update({'_len_': data_len})


class SintelDataset(Dataset):
    """ Sitel's data format dataset """
    def __init__(self, device: str, data_sets: dict, analyze_weights=True,
                 shuffle=True, mu=0, sigma=0.1, seed=42, train_split=None):
        """
        Parameters
        ----------
        data_sets:
            img_dir: str
                path to images directory
            depth_dir: str
                path to depth maps directory
            img_suffix: str
            depth_suffix: str
            scene_separator: str
                regular expression which will indicate the different scenes
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
        self.data = []  # list pairs of (img, depth) per data set
        self.data_map = {}
        self.seed = seed
        self.train_split = train_split
        self.train_idx = {}
        self.test_idx = {}
        self.mu = mu
        self.sigma = sigma
        self.device = device
        self.data_sets = data_sets
        self.analyze_weights = analyze_weights
        self.train_split = train_split
        self.shuffle = shuffle
        # if 'cuda' in self.device.type:
        #     torch.multiprocessing.set_start_method('spawn', force=True)
        random.seed(seed)

    def prepare_data(self):
        for data_set, data_params in self.data_sets.items():
            d_map = self.data_map[data_set] = defaultdict(list)  # list of indexes in data
            split_method = data_params['split_method']
            # prepare the data according to split method
            if 'scene' in split_method or 'even' in split_method:
                map_data(d_map, self.data, data_params)
                assert len(self.data) > 0, "ERROR - dataset is empty - check dataset parameters"
                if 'scene' in split_method:
                    test_scene = split_method.replace('scene_', '')
                    test_idx = np.array(d_map[test_scene], dtype=np.int)
                    train_idx = self.train_idx[data_set] = np.zeros(data_params['_len_'] - test_idx.size, dtype=np.int)
                    train_ptr = 0
                    for scene, ids in d_map.items():
                        if scene != test_scene:
                            train_idx[train_ptr:train_ptr + len(ids)] = ids
                elif 'even' in split_method:
                    assert self.train_split is not None, "Error - train_split argument is missing"
                    train_ptr = 0
                    test_ptr = 0
                    train_idx = np.zeros(int(data_params['_len_'] * self.train_split), dtype=np.int)
                    test_idx = np.zeros(data_params['_len_'] - train_idx.size, dtype=np.int)
                    for scene, ids in d_map.items():
                        if self.shuffle:
                            random.shuffle(ids)
                        train_portion = min(int(len(ids)*self.train_split + 0.5), train_idx.size - train_ptr)
                        test_portion = len(ids) - train_portion
                        train_idx[train_ptr:train_ptr+train_portion] = ids[:train_portion]
                        test_idx[test_ptr:test_ptr + test_portion] = ids[train_portion:]
                        train_ptr += train_portion
                        test_ptr += test_portion

                self.test_idx[data_set] = test_idx
                self.train_idx[data_set] = train_idx
            # elif 'random' in split_method:
            #     assert train_split is not None, "Error - train_split argument is missing"
            #     for data_name, p in data_sets.items():
            #         for img_path in glob(os.path.join(p['img_dir'], '*' + p['img_suffix'])):
            #             depth_file = os.path.basename(img_path).replace(p['img_suffix'], p['depth_suffix'])
            #             depth_path = os.path.join(p['depth_dir'], depth_file)
            #             if os.path.exists(depth_path):
            #                 self.data.append((img_path, depth_path))
            print('added data set {} with {} images'.format(data_set, data_params['_len_']))
        if self.analyze_weights:
            h = self.analyze_depth(display=False)
            self.weights = 1 / (h + 1)
            self.weights[0] = 0
            self.weights = self.weights / np.max(self.weights)
            print('weights:', self.weights)

    def depth_read(self, filename):
        """ Read depth data from file, return as numpy array.
        this method (with minor modifications by me) originally came with the:
        MPI-Sintel low-level computer vision benchmark.
        For more details about the benchmark, please visit www.mpi-sintel.de
        CHANGELOG:  v1.0 (2015/02/03): First release
        Copyright (c) 2015 Jonas Wulff
        Max Planck Institute for Intelligent Systems, Tuebingen, Germany"""
        tag_float = 202021.25
        with open(filename, 'rb') as f:
            check = np.fromfile(f, dtype=np.float32, count=1)[0]
            assert check == tag_float, 'depth_read:: Wrong tag in flow file ' \
                                       '(should be: {}, is: {}). Big-endian machine?'.format(tag_float, check)
            width = np.fromfile(f, dtype=np.int32, count=1)[0]
            height = np.fromfile(f, dtype=np.int32, count=1)[0]
            size = width * height
            assert 0 < size < 100000000, ' depth_read:: Wrong input size (width = {}, height = {}).'.format(width,
                                                                                                            height)
            depth = np.fromfile(f, dtype=np.float32, count=-1).reshape((height, width))
        return depth

    def __getitem__(self, item):
        img_path, dpt_path = self.data[item]
        img = imread(img_path).astype(np.float32) / 255
        dpt = self.depth_read(dpt_path)
        sample, label = random_dual_augmentation(image=img, label=dpt, sigma=self.sigma / 255, pad_divisor=32)
        return sample, label

    def __len__(self):
        return len(self.data)

    def analyze_depth(self, display: bool = False):
        print('analyzing labels...')
        h = np.zeros(16, dtype=np.int64)
        bins = range(17)
        for _, dpt_path in self.data:
            dpt = self.depth_read(dpt_path).astype(np.uint8)
            hist, _ = np.histogram(dpt.ravel(), bins=bins)
            h += hist
        if display:
            import matplotlib.pyplot as plt
            plt.bar(range(16), h)
            plt.show()
        return h

    def get_data_loaders(self, batch_size):
        # if self.split_method == 'random':
        #     train_size = int(self.train_split * len(self))
        #     test_size = len(self) - train_size
        #     torch.manual_seed(self.seed)
        #     train_dataset, test_dataset = random_split(self, [train_size, test_size])
        #     train_data_loader = DataLoader(train_dataset, batch_size=batch_size)
        #     test_data_loader = DataLoader(test_dataset, batch_size=batch_size)
        #     return train_data_loader, test_data_loader
        # else:
        self.prepare_data()
        train_loaders = []
        test_loaders = []
        for data_set in self.data_map:
            train_loaders.append(
                DataLoader(dataset=Subset(self, indices=self.train_idx[data_set]), pin_memory=False, shuffle=True,
                           batch_size=batch_size, collate_fn=collate_fn_random_rot90, num_workers=2))
            test_loaders.append(DataLoader(dataset=Subset(self, indices=self.test_idx[data_set]), pin_memory=False,
                                           batch_size=batch_size, collate_fn=collate_fn_random_rot90, num_workers=2))
        return train_loaders, test_loaders



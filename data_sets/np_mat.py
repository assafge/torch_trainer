import numpy as np
from torch.utils.data import Dataset, random_split, DataLoader
from torch import manual_seed
from image_utils import random_dual_augmentation, MyNoiseFlowWrapper


class NumpyMatDataset(Dataset):
    def __init__(self, file_path, seed: int, train_split: float, sigma, data_lbl='data', labels_lbl='labels'):
        self.data = None
        self.labels = None
        self.seed = seed
        self.train_split = train_split
        self.sigma = sigma / 255
        np.random.seed(seed)
        self.file_path = file_path
        self.data_lbl = data_lbl
        self.labels_lbl = labels_lbl
        self.augmentations = [np.flipud, np.fliplr]
        self.noise_flow = None

    def __getitem__(self, item):
        img = self.data[item]
        lbl = self.labels[item]
        return random_dual_augmentation(img, lbl, 0, do_transpose=False, augmentations=self.augmentations)

    def __len__(self):
        return len(self.data)

    def get_data_loaders(self, batch_size):
        mat = np.load(self.file_path)
        self.data = mat[self.data_lbl]
        self.labels = mat[self.labels_lbl]
        # torch transformation accepts HWC only
        # self.data = np.transpose(self.data, (0, 2, 3, 1))
        # self.labels = np.transpose(self.labels, (0, 2, 3, 1))
        train_size = int(self.train_split * len(self))
        test_size = len(self) - train_size
        manual_seed(self.seed)
        self.noise_flow = MyNoiseFlowWrapper()
        train_dataset, test_dataset = random_split(self, [train_size, test_size])
        train_data_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=self.noise_flow.nf_collate_fn_random_rot90)
        test_data_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=self.noise_flow.nf_collate_fn_random_rot90)
        return train_data_loader, test_data_loader

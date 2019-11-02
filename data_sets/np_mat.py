import numpy as np
from torch.utils.data import Dataset, random_split, DataLoader
from torch import manual_seed, as_tensor
import torchvision.transforms as tsfm
from PIL import Image


class NumpyMatDataset(Dataset):
    def __init__(self, file_path, seed: int, train_split: float, sigma, data_lbl='data', labels_lbl='labels'):
        self.data = None
        self.labels = None
        self.seed = seed
        self.train_split = train_split
        self.sigma = sigma
        np.random.seed(seed)
        self.file_path = file_path
        self.data_lbl = data_lbl
        self.labels_lbl = labels_lbl

    def __getitem__(self, item):
        img = self.data[item]
        lbl = self.labels[item]

        # augmentations: #
        # torch transformations cannot be applied to a pair of images, so I've decided to implement in numpy.
        gauss = np.random.normal(0, self.sigma, img.shape).astype(np.float32)
        noisy = img + gauss
        noisy = np.clip(noisy, 0, 1)
        for augment in [np.flipud, np.fliplr]:
            coin = np.random.rand()
            if coin > 0.5:
                noisy = np.ascontiguousarray(augment(noisy))
                lbl = np.ascontiguousarray(augment(lbl))
        noisyt = as_tensor(noisy)
        lblt = as_tensor(lbl)
        return noisyt, lblt

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
        train_dataset, test_dataset = random_split(self, [train_size, test_size])
        train_data_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=2)
        test_data_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=2)
        return train_data_loader, test_data_loader

import numpy as np
from torch.utils.data import Dataset, random_split, DataLoader
from torch import manual_seed


class NumpyMatDataset(Dataset):
    def __init__(self, file_path, seed: int, train_split: float, data_lbl='data', labels_lbl='labels'):
        mat = np.load(file_path)
        self.data = mat[data_lbl]
        self.labels = mat[data_lbl]
        self.seed = seed
        self.train_split = train_split

    def __getitem__(self, item):
        return self.data[item], self.labels[item]

    def __len__(self):
        return self.data.shape[0]

    def get_data_loaders(self, batch_size):
        train_size = int(self.train_split * len(self))
        test_size = len(self) - train_size
        manual_seed(self.seed)
        train_dataset, test_dataset = random_split(self, [train_size, test_size])
        train_data_loader = DataLoader(train_dataset, batch_size=batch_size)
        test_data_loader = DataLoader(test_dataset, batch_size=batch_size)
        return train_data_loader, test_data_loader

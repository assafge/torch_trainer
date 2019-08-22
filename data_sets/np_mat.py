import numpy as np
from torch.utils.data import Dataset, DataLoader, Sampler

class NumpyMatDataset(Dataset):
    def __init__(self, file_path, data_lbl='data', labels_lbl='labels'):
        with open(file_path) as f:
            mat = np.load(f)
        self.data = mat[data_lbl]
        self.labels = mat[data_lbl]

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return self.data.shape[0]

    def get_data_loaders(self):
        train_sampler = Sampler(self.data)
        test_sampler = Sampler(self.data)

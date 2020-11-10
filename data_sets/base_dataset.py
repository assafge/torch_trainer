from torch.utils.data import Dataset
from abc import abstractmethod


class BaseDataset(Dataset):
    def __init__(self, root_dir: str, in_channels: int, out_channels: int, train_split: float):
        self.root_dir = root_dir
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.train_split = train_split

    @abstractmethod
    def get_data_loaders(self):
        pass

    @abstractmethod
    def prepare_data(self):
        pass

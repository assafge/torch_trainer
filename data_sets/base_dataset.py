from torch.utils.data import Dataset
from abc import abstractmethod
from dataclasses import dataclass
from typing import Dict


class BaseDataset(Dataset):
    def __init__(self, root_dir: str, in_channels: int, out_channels: int, train_split: float,
                 data_sets, params_type: dataclass):
        self.root_dir = root_dir
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.train_split = train_split
        self.data_sets: Dict[str, params_type] = {}
        for set_name, prm in data_sets.items():
            self.data_sets[set_name] = params_type(**prm)

    @abstractmethod
    def get_data_loaders(self, batch_size: int):
        pass

    @abstractmethod
    def prepare_data(self):
        pass

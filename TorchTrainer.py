import torch
from utils import read_yaml, get_class
import os
from shutil import rmtree
from time import time
from glob import glob
from torch.utils.tensorboard import SummaryWriter
from collections import namedtuple

class LossSum(namedtuple):
    loss: float = 0
    writer: SummaryWriter
    norm_factor: float


class ConfigurationStruct:
    class Struct:
        def __init__(self, entries: dict):
            self.__dict__.update(entries)
            for key, val in self.__dict__:
                if 'kargs' in key and val is None:
                    self.__dict__[key] = {}


class TorchTrainer:
    """wrapper class for torch models training"""
    def __init__(self, cfg: ConfigurationStruct, root, gpu_index: int, logger):
        self.logger = logger
        self.cfg = cfg
        self.device = torch.device("cuda:" + str(gpu_index) if torch.cuda.is_available() else "cpu")
        self.model: torch.nn.Module = None
        self.optimizer = None
        self.epoch: int = 0
        self.root = root

    @classmethod
    def new_train(cls, out_path, model_cfg, optimizer_cfg, dataset_cfg, gpu_index, logger=None):
        sections = {'model': model_cfg,
                    'optimizer': optimizer_cfg,
                    'data': dataset_cfg}
        config_dict = {}
        for cfg_section, cfg_path in sections.items():
            config_dict[cfg_section] = read_yaml(cfg_path)
        config = ConfigurationStruct(config_dict)

        model_dir = config.model.type + time().strftime("_%d%b%y_%H:%M")
        root = os.path.join(out_path, model_dir)
        if os.path.isdir(root):
            print("root directory is already exist - will delete the previous and create new")
            rmtree(root)
        os.makedirs(root)
        os.makedirs(os.path.join(root,'checkpoints'))
        cls = TorchTrainer(config, gpu_index, root, logger)
        cls.init_nn()

    @classmethod
    def warm_startup(cls, root, gpu_index, logger = None):
        config_dict =  read_yaml(os.path.join(root, 'cfg.yaml'))
        for cfg_section, cfg_dict in config_dict.items():
            config_dict[cfg_section] = ConfigurationStruct(cfg_dict)
        config = ConfigurationStruct(config_dict)
        cls = TorchTrainer(config, gpu_index, root, logger)
        cls.init_nn()
        cls.load_model()

    def init_nn(self):
        model_cls = get_class(self.cfg.model.type, file_path=self.cfg.model.path)
        self.model = model_cls(self.model.kargs)
        optim_cls = get_class(self.cfg.optimizer.type, module_path='torch.optim')
        self.optimizer = model_cls(self.optimizer.kargs)

    def load_checkpoint(self):
        latest = None
        epoch = -1
        for cp_file in glob(os.path.join(self.root, 'checkpoints', 'checkpoint_*.pth')):
            cp_epoch = int(cp_file.split('_')[1].split('.')[0])
            if cp_epoch > epoch:
                epoch = cp_epoch
                latest = cp_file

        checkpoint = torch.load(latest)
        self.epoch = epoch
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # for state in self.optimizer.state.values():
        #     for k, v in state.items():
        #         if isinstance(v, torch.Tensor):
        #             state[k] = v.to(self.device)
        self.model.to(self.device)

    def save_checkpoint(self):
        sub_dir = os.path.join(self.root, 'checkpoints')
        if not os.path.isdir(sub_dir):
            os.makedirs(sub_dir)
        torch.save({'epoch': self.epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict()},
                   os.path.join(sub_dir, 'checkpoint_{}.pth'.format(self.epoch)))

    def get_data_loaders(self):
        dataset_cls = get_class(self.data.type, file_path=self.cfg.data.path)
        dataset = dataset_cls(self.cfg.data.kargs)
        train_loader, test_loader = dataset.get_data_loaders()
        return train_loader, test_loader

    def train(self):
        criterion_cls = get_class(self.model.loss, module_path='torch.nn')
        criterion = criterion_cls(**self.cfg.model.loss_kargs.kargs)
        self.model.zero_grad()
        train_loader, test_loader = self.get_data_loaders()
        train = LossSum(writer=SummaryWriter(os.path.join(self.root, 'train')), norm_facotr=len(train_loader))
        test = LossSum(writer=SummaryWriter(os.path.join(self.root, 'test')), norm_facotr=len(test_loader))

        while self.epoch < self.model.epochs:
            train.loss = test.loss = 0

            self.model.train()
            for i, (x, y) in enumerate(train_loader):
                data, labels = x.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, labels)
                loss.backward()
                self.optimizer.step()
                if i % (len(train_loader) // 4) == 0:
                    print('Step: {} Train loss: {:3.3}'.format(i, loss.item()))
                train.loss += loss.item()

            self.model.eval()
            for i, (x, y) in enumerate(train_loader):
                data, labels = x.to(self.device), y.to(self.device)
                with torch.no_grad():
                    out = self.model(data)
                loss = criterion(out, labels)
                if i % (len(train_loader) // 4) == 0:
                    print('Step: {} / Val. loss: {:3.3}'.format(i, loss.item()))
                test.loss += loss.item()

            train.writer.add_scalar(tag='Loss', scalar_value=train.loss / train.norm_factor, global_step=self.epoch)
            train.writer.add_scalar(tag='Loss', scalar_value=train.loss / test.norm_factor, global_step=self.epoch)
            self.save_checkpoint()


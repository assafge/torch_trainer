import torch
from utils import read_yaml, get_class
import os
from shutil import rmtree
from time import time
from glob import glob


class ConfigurationStruct:
    class Struct:
        def __init__(self, **entries):
            self.__dict__.update(entries)
            for key, val in self.__dict__:
                if 'kargs' in key and val is None:
                    self.__dict__[key] = {}

class TorchTrainer:
    """wrapper class for torch models training"""
    def __init__(self, cfg: ConfigurationStruct, gpu_index: int, logger):
        self.logger = logger_obj
        self.cfg = cfg
        self.device = torch.device("cuda:" + str(gpu_index) if torch.cuda.is_available() else "cpu")
        self.model = None
        self.optimizer = None

    @classmethod
    def new_train(cls, out_path, model_cfg, optimizer_cfg, dataset_cfg, gpu_index, logger=None):
        sections = {'model': model_cfg
                    'optimizer': optimizer_cfg
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
        cls = TorchTrainer(config, gpu_index, logger)
        cls.init_nn()


    @classmethod
    def warm_startup(cls, root, gpu_index, logger = None):
        config_dict =  read_yaml(os.path.join(root, 'cfg.yaml'))
        for cfg_section, cfg_dict in config_dict.items():
            config_dict[cfg_section] = ConfigurationStruct(cfg_dict)
        config = ConfigurationStruct(config_dict)
        cls = TorchTrainer(config, gpu_index, logger)
        cls.init_nn()
        cls.load_model()

    def init_nn(self):
        model_cls = get_class(self.cfg.model.type, file_path=self.cfg.model.path)
        self.model = model_cls(self.model.kargs)
        optim_cls = get_class(self.cfg.optimizer.type, module_path='torch.optim')
        self.optimizer = model_cls(self.optimizer.kargs)

    def load_model(self):
        filename = None
        epoch = -1
        for checkpoint in (os.path.join(root, 'checkpoint_*.pth')):
            cp_epoch = int(checkpoint.split('_'[1].split('.'[0])))
            if cp_epoch > epoch
                epoch = cp_epoch
                filename = filename

        checkpoint = torch.load(checkpoint)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(self.device)

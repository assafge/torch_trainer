import torch
from trainer_utils import read_yaml, get_class, print_progress
import os
from shutil import rmtree
from datetime import datetime
from glob import glob
from torch.utils.tensorboard import SummaryWriter
from performance_measurements import PerformanceMeasurement
from typing import List
import yaml


class ConfigurationStruct:
    def __init__(self, entries: dict):
        self.__dict__.update(entries)
        for key, val in self.__dict__.items():
            if 'kargs' in key and val is None:
                self.__dict__[key] = {}


class UtilityObj:
    def __init__(self, loader: torch.utils.data.DataLoader, writer_path, measurements: dict = {}):
        self.loader = loader
        self.writer = SummaryWriter(writer_path)
        self.loss = 0.0
        self.measurements: List[str, PerformanceMeasurement] = []

        if measurements:
            for meas in measurements.values():
                cls_ = get_class(meas['type'], meas['path'])
                self.measurements.append(cls_())

    def measure(self, outputs, labels):
        for m in self.measurements:
            m.add_measurement(outputs, labels)

    def write_step(self, step):
        for m in self.measurements:
            m.write_step(writer=self.writer, mini_batches=len(self.loader), step=step)

    @property
    def size(self):
        return len(self.loader)



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
        sections = {'model': read_yaml(model_cfg),
                    'optimizer': read_yaml(optimizer_cfg),
                    'data': read_yaml(dataset_cfg)}
        config_dict = {}
        for cfg_section, cfg in sections.items():
            config_dict[cfg_section] = ConfigurationStruct(cfg)
        config = ConfigurationStruct(config_dict)

        model_dir = config.model.type + datetime.now().strftime("_%d%b%y_%H%M")
        root = os.path.join(out_path, model_dir)
        if os.path.isdir(root):
            print("root directory is already exist - will delete the previous and create new")
            rmtree(root)
        os.makedirs(root)
        os.makedirs(os.path.join(root, 'checkpoints'))
        with open(os.path.join(root, 'cfg.yaml'), 'w') as f:
            yaml.dump(data=sections, stream=f)
        cls = TorchTrainer(cfg=config, root=root, gpu_index=gpu_index, logger=logger)
        cls.init_nn()
        cls.model.to(cls.device)
        return cls

    @classmethod
    def warm_startup(cls, root, gpu_index, logger = None):
        config_dict = read_yaml(os.path.join(root, 'cfg.yaml'))
        for cfg_section, cfg_dict in config_dict.items():
            config_dict[cfg_section] = ConfigurationStruct(cfg_dict)
        config = ConfigurationStruct(config_dict)
        cls = TorchTrainer(cfg=config, root=root, gpu_index=gpu_index, logger=logger)
        cls.init_nn()
        cls.load_checkpoint()
        return cls

    def init_nn(self):
        model_cls = get_class(self.cfg.model.type, file_path=self.cfg.model.path)
        self.model = model_cls(**self.cfg.model.kargs)
        optim_cls = get_class(self.cfg.optimizer.type, module_path='torch.optim')
        self.optimizer = optim_cls(self.model.parameters(), **self.cfg.optimizer.kargs)

    def load_checkpoint(self):
        latest = None
        epoch = -1
        for cp_path in glob(os.path.join(self.root, 'checkpoints', 'checkpoint_*.pth')):
            cp_file = os.path.basename(cp_path)
            cp_epoch = int(cp_file.split('_')[1].split('.')[0])
            if cp_epoch > epoch:
                epoch = cp_epoch
                latest = cp_path

        checkpoint = torch.load(latest)
        self.epoch = epoch
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(self.device)
        self.model.to(self.device)

    def save_checkpoint(self):
        fpath = os.path.join(self.root, 'checkpoints', 'checkpoint_{}.pth'.format(self.epoch))
        # print('saving checkpoint {}'.format(fpath))
        torch.save({'epoch': self.epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict()}, fpath)

    def init_training_obj(self):
        """create helper objects in order to make the train function more clear"""
        dataset_cls = get_class(self.cfg.data.type, file_path=self.cfg.data.path)
        dataset = dataset_cls(**self.cfg.data.kargs)
        train_loader, test_loader = dataset.get_data_loaders(self.cfg.model.batch_size)
        train = UtilityObj(loader=train_loader, writer_path=os.path.join(self.root, 'train'))
        test = UtilityObj(loader=test_loader, writer_path=os.path.join(self.root, 'test'),
                          measurements=self.cfg.model.perfomance_measurements)
        return train, test

    def train(self):
        criterion_cls = get_class(self.cfg.model.loss, module_path='torch.nn')
        criterion = criterion_cls(**self.cfg.model.loss_kargs)
        self.model.zero_grad()
        train, test = self.init_training_obj()

        while self.epoch < self.cfg.model.epochs:
            train.loss = test.loss = 0.0

            self.model.train()
            for i, (x, y) in enumerate(train.loader):
                data, labels = x.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, labels)
                loss.backward()
                self.optimizer.step()
                print_progress(iteration=i, total=train.size, prefix='Epoch {} train'.format(self.epoch), length=50,
                               suffix='loss=%0.3f' % loss.item() if loss.item() > 0.1 else 'loss=%0.3e' % loss.item())
                train.loss += loss.item()
            avr_loss = train.loss / train.size
            print_progress(iteration=i + 1, total=train.size, prefix='Epoch {} train'.format(self.epoch), length=50,
                           suffix='loss=%0.3f' % avr_loss if avr_loss > 0.1 else 'loss=%0.3e' % avr_loss)
            train.writer.add_scalar(tag='Loss', scalar_value=avr_loss, global_step=self.epoch)

            self.model.eval()
            for i, (x, y) in enumerate(test.loader):
                data, labels = x.to(self.device), y.to(self.device)
                with torch.no_grad():
                    out = self.model(data)
                loss = criterion(out, labels)
                test.loss += loss.item()
                print_progress(iteration=i, total=test.size, prefix='Epoch {} test '.format(self.epoch), length=50,
                               suffix='loss=%0.3f' % loss.item() if loss.item() > 0.1 else 'loss=%0.3e' % loss.item())
                test.measure(outputs=out, labels=labels)
            avr_loss = test.loss / test.size
            print_progress(iteration=i + 1, total=test.size, prefix='Epoch {} test '.format(self.epoch), length=50,
                           suffix='loss=%0.3f' % avr_loss if avr_loss > 0.1 else 'loss=%0.3e' % avr_loss)
            test.writer.add_scalar(tag='Loss', scalar_value=avr_loss, global_step=self.epoch)
            test.write_step(step=self.epoch)
            self.save_checkpoint()
            self.epoch += 1


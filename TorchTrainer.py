import torch
from trainer_utils import read_yaml, get_class, print_progress, retrieve_name
import os
from shutil import rmtree
from datetime import datetime
from time import time
from torch.utils.tensorboard import SummaryWriter
from performance_measurements import PerformanceMeasurement
from typing import List
import yaml
import numpy as np
import matplotlib.pyplot as plt


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
        self.name = os.path.basename(writer_path)
        self.pname = self.name + (' ' * (7 - len(self.name)))
        self.measurements: List[str, PerformanceMeasurement] = []
        self.stime = time()

        if measurements:
            for meas in measurements.values():
                cls_ = get_class(meas['type'], meas['path'])
                self.measurements.append(cls_())

    def measure(self, outputs, labels, step):
        for m in self.measurements:
            m.add_measurement(outputs, labels, step)

    def write_step(self, step):
        for m in self.measurements:
            m.write_step(writer=self.writer, mini_batches=len(self.loader), step=step)

    def init_loop(self):
        self.loss = 0
        self.stime = time()

    @property
    def size(self):
        return len(self.loader)

    def print_loss(self, val, idx, epoch):
        if idx == self.size:
            avr = self.loss / self.size
            print_progress(iteration=idx, total=self.size, prefix='Epoch {} {}'.format(epoch, self.pname), length=40,
                           suffix='average loss=%.3f, loop time=%.2f' % (avr, time() - self.stime) if avr > 0.1
                           else 'average loss=%.3e, loop time=%.2f' % (avr, time() - self.stime))
        else:
            print_progress(iteration=idx, total=self.size, prefix='Epoch {} {}'.format(epoch, self.pname), length=40,
                           suffix=' running loss=%0.3f' % val if val > 0.1 else ' running loss=%0.3e' % val)

    def add_debug_image(self, labels, outputs, step):
        im = (outputs[0].argmax(axis=0).cpu().numpy() * (255 / 15)).astype(np.uint8)
        ref = (labels[0].cpu().numpy() * (255 / 15)).astype(np.uint8)
        self.writer.add_image('result', im, global_step=step, dataformats='HW')
        self.writer.add_image('ref', ref, global_step=step, dataformats='HW')


class TorchTrainer:
    """wrapper class for torch models training"""
    def __init__(self, cfg: ConfigurationStruct, root, gpu_index: int, logger):
        self.logger = logger
        self.cfg = cfg
        if int(gpu_index) >= 0 and torch.cuda.is_available():
            self.device = torch.device("cuda:" + str(gpu_index))
            print('using device: ', torch.cuda.get_device_name(self.device))
        else:
            self.device = torch.device("cpu")
            print('using cpu')
        self.model: torch.nn.Module = None
        self.optimizer = None
        self.epoch: int = 0
        self.root = root
        self.dataset = None

    @classmethod
    def new_train(cls, out_path, model_cfg, optimizer_cfg, dataset_cfg, gpu_index, exp_name, logger=None):
        sections = {'model': read_yaml(model_cfg),
                    'optimizer': read_yaml(optimizer_cfg),
                    'data': read_yaml(dataset_cfg)}
        config_dict = {}
        for cfg_section, cfg in sections.items():
            config_dict[cfg_section] = ConfigurationStruct(cfg)
        config = ConfigurationStruct(config_dict)

        model_dir = config.model.type + datetime.now().strftime("_%d%b%y_%H%M")
        if len(exp_name):
            model_dir += '_' + exp_name
        root = os.path.join(out_path, model_dir)
        if os.path.isdir(root):
            print("root directory is already exist - will delete the previous and create new")
            rmtree(root)
        os.makedirs(root)
        print('writing results to directory: %s' % root)
        os.makedirs(os.path.join(root, 'checkpoints'))
        with open(os.path.join(root, 'cfg.yaml'), 'w') as f:
            yaml.dump(data=sections, stream=f)
        cls = TorchTrainer(cfg=config, root=root, gpu_index=gpu_index, logger=logger)
        cls.init_nn()
        cls.model.to(cls.device)
        return cls

    @classmethod
    def warm_startup(cls, root, gpu_index, logger=None):
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
        dataset_cls = get_class(self.cfg.data.type, file_path=self.cfg.data.path)
        self.dataset = dataset_cls(**self.cfg.data.kargs)

    def load_checkpoint(self):
        latest = None
        epoch = -1
        cp_path = os.path.join(self.root, 'checkpoints', 'checkpoint.pth')
        checkpoint = torch.load(cp_path)
        self.epoch = checkpoint['epoch']
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(self.device)
        self.model.to(self.device)

    def save_checkpoint(self, better):
        if better:
            fpath = os.path.join(self.root, 'checkpoints', 'checkpoint.pth'.format(self.epoch))
            torch.save({'epoch': self.epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict()}, fpath)
        # if self.epoch % 50 == 0:
        #     fpath = os.path.join(self.root, 'checkpoints', 'checkpoint_{}.pth'.format(self.epoch))
        #     # print('saving checkpoint {}'.format(fpath))
        #     torch.save({'epoch': self.epoch,
        #                 'model_state_dict': self.model.state_dict(),
        #                 'optimizer_state_dict': self.optimizer.state_dict()}, fpath)

    def init_training_obj(self):
        """create helper objects in order to make the train function more clear"""
        train_loader, test_loader = self.dataset.get_data_loaders(self.cfg.model.batch_size)
        train = UtilityObj(loader=train_loader, writer_path=os.path.join(self.root, 'train'))
        test = UtilityObj(loader=test_loader, writer_path=os.path.join(self.root, 'test'),
                          measurements=self.cfg.model.perfomance_measurements)
        return train, test

    def train(self):
        criterion_cls = get_class(self.cfg.model.loss, module_path='torch.nn')
        criterion = criterion_cls(**self.cfg.model.loss_kargs)
        self.model.zero_grad()
        train, test = self.init_training_obj()
        best_loss = test.loss = 2 ** 16

        while self.epoch < self.cfg.model.epochs:
            if test.loss > best_loss:
                best_loss = test.loss
            self.model.train()
            train.init_loop()
            for i, (x, y) in enumerate(train.loader):
                data, labels = x.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, labels)
                loss.backward()
                self.optimizer.step()
                train.loss += loss.item()
                train.print_loss(loss.item(), i, self.epoch)
            train.print_loss(train.loss, i + 1, self.epoch)
            train.writer.add_scalar(tag='Loss', scalar_value=train.loss / train.size, global_step=self.epoch)

            if self.epoch % 5 == 0:
                self.model.eval()
                test.init_loop()
                for i, (x, y) in enumerate(test.loader):
                    data, labels = x.to(self.device), y.to(self.device)
                    with torch.no_grad():
                        out = self.model(data)
                    loss = criterion(out, labels)
                    test.loss += loss.item()
                    test.print_loss(loss.item(), i, self.epoch)
                    test.measure(outputs=out, labels=labels, step=self.epoch)
                test.print_loss(test.loss, i + 1, self.epoch)
                test.writer.add_scalar(tag='Loss', scalar_value=test.loss / test.size, global_step=self.epoch)
                test.write_step(step=self.epoch)

            # self.save_checkpoint(test.loss < best_loss)
            self.save_checkpoint(True)
            self.epoch += 1

    def inference(self, img):
        img_t = torch.Tensor(img).to(device=self.device)
        output = self.model(img_t)
        return output.to('cpu').numpy()


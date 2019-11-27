import torch
from trainer_utils import read_yaml, get_class, print_progress, plot_confusion_matrix
import os
from shutil import rmtree
from datetime import datetime
from time import time
from torch.utils.tensorboard import SummaryWriter
from performance_measurements import PerformanceMeasurement
from typing import List
import yaml
import numpy as np

class ConfigurationStruct:
    def __init__(self, entries: dict):
        self.__dict__.update(entries)
        for key, val in self.__dict__.items():
            if 'kargs' in key and val is None:
                self.__dict__[key] = {}


class UtilityObj:
    def __init__(self, loaders, writer_path, measurements: dict = {}):
        if type(loaders) is list:
            self.loaders = loaders
        elif type(loaders) is torch.utils.data.DataLoader:
            self.loaders = [loaders]
        else:
            print('error - data loader is invalid')
        self.writer = SummaryWriter(writer_path)
        self.loss = 0.0
        self.index = 0
        self.name = os.path.basename(writer_path)
        self.pname = self.name + (' ' * (7 - len(self.name)))
        self.measurements: List[PerformanceMeasurement] = []
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
            m.write_step(writer=self.writer, mini_batches=self.size, step=step)

    def init_loop(self):
        self.loss = 0
        self.index = 0
        self.stime = time()

    @property
    def size(self):
        return sum([len(loader) for loader in self.loaders])

    def print_loss(self, val, epoch):
        if self.index == self.size:
            loop_time = time() - self.stime
            avr = self.loss / self.size
            print_progress(iteration=self.index, total=self.size, prefix='Epoch {} {}'.format(epoch, self.pname), length=40,
                           suffix='average loss=%.3f, loop time=%.2f' % (avr, loop_time) if avr > 0.1
                           else 'average loss=%.3e, loop time=%.2f' % (avr, loop_time))
        else:
            print_progress(iteration=self.index, total=self.size, prefix='Epoch {} {}'.format(epoch, self.pname), length=40,
                           suffix=' running loss=%0.3f' % val if val > 0.1 else ' running loss=%0.3e' % val)
        self.index += 1

    def add_debug_image(self, outputs, labels, step, suffix=''):
        im = (outputs[0].argmax(dim=0).cpu().numpy() * (255 / 15)).astype(np.uint8)
        ref = (labels[0].cpu().numpy() * (255 / 15)).astype(np.uint8)
        self.writer.add_image('result'+suffix, im, global_step=step, dataformats='HW')
        self.writer.add_image('ref'+suffix, ref, global_step=step, dataformats='HW')

    def add_confusion_matrix(self, outputs, labels, step, suffix=''):
        pred = outputs.argmax(dim=1).cpu().numpy().astype(np.uint8).ravel()
        ref = labels.cpu().numpy().astype(np.uint8).ravel()
        im = plot_confusion_matrix(ref, pred, [str(i) for i in range(16)])
        self.writer.add_image('confusion_mat'+suffix, im, global_step=step, dataformats='HWC')


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
        self.running = True

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
        dataset_cls = get_class(self.cfg.data.type, file_path=self.cfg.data.path)
        self.dataset = dataset_cls(**self.cfg.data.kargs)
        optim_cls = get_class(self.cfg.optimizer.type, module_path='torch.optim')
        self.optimizer = optim_cls(self.model.parameters(), **self.cfg.optimizer.kargs)

    def load_checkpoint(self):
        latest = None
        epoch = -1
        cp_path = os.path.join(self.root, 'checkpoints', 'last_checkpoint.pth')
        if not os.path.exists(cp_path):
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
        else:
            fpath = os.path.join(self.root, 'checkpoints', 'last_checkpoint.pth'.format(self.epoch))
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
        train_loaders, test_loaders = self.dataset.get_data_loaders(self.cfg.model.batch_size)
        train = UtilityObj(loaders=train_loaders, writer_path=os.path.join(self.root, 'train'))
        test = UtilityObj(loaders=test_loaders, writer_path=os.path.join(self.root, 'test'),
                          measurements=self.cfg.model.perfomance_measurements)
        return train, test

    def train(self):
        if 'weights' in vars(self.dataset):
            self.cfg.model.loss_kargs.update({
                'weight': torch.as_tensor(data=self.dataset.weights, dtype=torch.float, device=self.device)})
        criterion_cls = get_class(self.cfg.model.loss, module_path='torch.nn')
        criterion = criterion_cls(**self.cfg.model.loss_kargs)
        # w = torch.as_tensor(data=self.dataset.weights, dtype=torch.float, device=self.device)
        # criterion = myCrossEntropy.WeightedPerPixelCrossEntropyLoss

        self.model.zero_grad()
        train, test = self.init_training_obj()
        best_loss = 2 ** 16
        self.running = True

        while self.epoch <= self.cfg.model.epochs and self.running:
            self.model.train()
            train.init_loop()
            for train_loader in train.loaders:
                if not self.running:
                    break
                for x, y in train_loader:
                    if not self.running:
                        break
                    data, labels = x.to(self.device), y.to(self.device)
                    output = self.model(data)
                    self.optimizer.zero_grad()
                    loss = criterion(output, labels)
                    loss.backward()
                    self.optimizer.step()
                    train.loss += loss.item()
                    train.print_loss(loss.item(), self.epoch)
                    if train.index == train.size - 1:
                        test.add_debug_image(output, labels, step=self.epoch, suffix='_train')

            train.print_loss(train.loss, self.epoch)
            train.writer.add_scalar(tag='Loss', scalar_value=train.loss / train.size, global_step=self.epoch)

            if self.epoch % 3 == 0 and self.running:
                self.model.eval()
                test.init_loop()
                for test_loader in test.loaders:
                    for x, y in test_loader:
                        data, labels = x.to(self.device), y.to(self.device)
                        with torch.no_grad():
                            output = self.model(data)
                            loss = criterion(output, labels)
                            test.loss += loss.item()
                            test.measure(outputs=output, labels=labels, step=self.epoch)
                            test.print_loss(loss.item(), self.epoch)
                            if test.index == test.size - 1:
                                test.add_debug_image(output, labels, step=self.epoch, suffix='_test')
                                test.add_confusion_matrix(output, labels, step=self.epoch)
                test.print_loss(test.loss, self.epoch)
                test.writer.add_scalar(tag='Loss', scalar_value=test.loss / test.size, global_step=self.epoch)
                test.write_step(step=self.epoch)
                if test.loss < best_loss:
                    best_loss = test.loss
                    self.save_checkpoint(True)
                else:
                    self.save_checkpoint(False)
                # self.save_checkpoint(test.loss < best_loss)

            self.epoch += 1
        #
        # if not self.running:
        #     self.epoch -= 1
        #     self.save_checkpoint(False)
        # else:


        self.running = False


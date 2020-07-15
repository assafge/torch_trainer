#! /home/assaf/EDOF/venv/bin/python

import torch
from general_utils import read_yaml, get_class, print_progress
import os
from shutil import rmtree
from datetime import datetime
from time import time
from torch.utils.tensorboard import SummaryWriter
from traces import Trace
from typing import List
import yaml

# debug
import numpy as np
import cv2


class ConfigurationStruct:
    def __init__(self, entries: dict):
        self.__dict__.update(entries)
        for key, val in self.__dict__.items():
            if 'kargs' in key and val is None:
                self.__dict__[key] = {}


class UtilityObj:
    def __init__(self, loaders, writer_path, measurements=None):
        if measurements is None:
            measurements = {}
        if type(loaders) is list:
            self.loaders = loaders
        elif type(loaders) is torch.utils.data.DataLoader:
            self.loaders = [loaders]
        else:
            print('error - data loader is invalid')
        self.writer = SummaryWriter(writer_path)
        self.aggregated_loss = 0.0
        self.index = 0
        self.name = os.path.basename(writer_path)
        self.pname = self.name + (' ' * (7 - len(self.name)))
        self.traces: List[Trace] = []
        self.stime = time()

        if measurements:
            for meas in measurements.values():
                cls_ = get_class(meas['type'], meas['path'])
                self.traces.append(cls_(self.writer, self.size))

    def step(self, loss, inputs, pred, labels, epoch):
        for m in self.traces:
            m.add_measurement(inputs, pred, labels)
        print_progress(iteration=self.index, total=self.size, prefix='Epoch {} {}'.format(epoch, self.pname), length=40,
                       suffix=' running loss=%0.3f' % loss if loss > 0.1 else ' running loss=%0.3e' % loss)
        self.aggregated_loss += loss
        self.index += 1

    def epoch_step(self, epoch):
        for m in self.traces:
            m.write_epoch(epoch)
        loop_time = time() - self.stime
        avr = self.aggregated_loss / self.size
        print_progress(iteration=self.size, total=self.size, prefix='Epoch {} {}'.format(epoch, self.pname), length=40,
                       suffix='average loss=%.3f, loop time=%.2f' % (avr, loop_time) if avr > 0.1
                       else 'average loss=%.3e, loop time=%.2f' % (avr, loop_time))
        self.writer.add_scalar('Loss', avr, global_step=epoch)

    def init_loop(self):
        self.aggregated_loss = 0
        self.index = 0
        self.stime = time()

    @property
    def size(self):
        return sum([len(loader) for loader in self.loaders])



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
    def warm_startup(cls, root, gpu_index, strict, logger=None):
        config_dict = read_yaml(os.path.join(root, 'cfg.yaml'))
        for cfg_section, cfg_dict in config_dict.items():
            config_dict[cfg_section] = ConfigurationStruct(cfg_dict)
        config = ConfigurationStruct(config_dict)
        cls = TorchTrainer(cfg=config, root=root, gpu_index=gpu_index, logger=logger)
        cls.init_nn()
        cls.load_checkpoint(strict)
        return cls

    def init_nn(self):
        model_cls = get_class(self.cfg.model.type, file_path=self.cfg.model.path)
        self.model = model_cls(**self.cfg.model.kargs)
        dataset_cls = get_class(self.cfg.data.type, file_path=self.cfg.data.path)
        self.dataset = dataset_cls(**self.cfg.data.kargs)
        optim_cls = get_class(self.cfg.optimizer.type, module_path='torch.optim')
        self.optimizer = optim_cls(self.model.parameters(), **self.cfg.optimizer.kargs)

    def load_checkpoint(self, strict):
        latest = None
        epoch = -1
        cp_path = os.path.join(self.root, 'checkpoints', 'last_checkpoint.pth')
        if not os.path.exists(cp_path):
            cp_path = os.path.join(self.root, 'checkpoints', 'checkpoint.pth')
        checkpoint = torch.load(cp_path, map_location=self.device)
        self.epoch = checkpoint['epoch']
        self.model.load_state_dict(checkpoint['model_state_dict'], strict=strict)
        if not strict:
            print('non strict', self.model.fine_tune_)
            if self.model.fine_tune is not None and self.cfg.model.fine_tune_kargs is not None:
                self.model.fine_tune(**self.cfg.model.fine_tune_kargs)
            optim_cls = get_class(self.cfg.optimizer.type, module_path='torch.optim')
            self.optimizer = optim_cls(self.model.parameters(), **self.cfg.optimizer.kargs)
        else:
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
                          measurements=self.cfg.model.test_traces)
        return train, test

    def train(self):
        if 'weights' in vars(self.dataset):
            if self.dataset.weights is not None:
                self.cfg.model.loss_kargs.update({
                    'weight': torch.as_tensor(data=self.dataset.weights, dtype=torch.float, device=self.device)})
        criterion_cls = get_class(self.cfg.model.loss, module_path=self.cfg.model.loss_module_path if
                                  'loss_module_path' in vars(self.cfg.model) else 'torch.nn')
        criterion = criterion_cls(**self.cfg.model.loss_kargs)

        self.model.zero_grad()
        train, test = self.init_training_obj()
        best_loss = 2 ** 16
        self.running = True

        train.init_loop()
        while self.epoch <= self.cfg.model.epochs and self.running:
            self.model.train()
            for train_loader in train.loaders:
                if not self.running:
                    break
                train.init_loop()
                tt = time()
                for x, y in train_loader:
                    inputs, labels = x.to(self.device), y.to(self.device)
                    outputs = self.model(inputs)
                    self.optimizer.zero_grad()
                    loss = criterion(outputs, labels)
                    loss.backward()
                    self.optimizer.step()
                    train.step(loss.item(), inputs, outputs, labels, self.epoch)
            train.epoch_step(self.epoch)

            if self.epoch % 3 == 0 and self.running:
                self.model.eval()
                test.init_loop()
                for test_loader in test.loaders:
                    for x, y in test_loader:
                        inputs, labels = x.to(self.device), y.to(self.device)
                        with torch.no_grad():
                            outputs = self.model(inputs)
                            loss = criterion(outputs, labels)
                            test.step(loss.item(), inputs, outputs, labels, self.epoch)
                # self.save_to_debug(inputs, outputs, labels)
                test.epoch_step(self.epoch)

            if test.aggregated_loss < best_loss:
                best_loss = test.aggregated_loss
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


    def save_to_debug(self, inputs, outputs, labels):
        out_path = '/tmp/minibatch_debug'
        if not os.path.isdir(out_path):
            os.makedirs(out_path)
        for i in range(outputs.shape[0]):
            inp = np.transpose((inputs[i].cpu().numpy() * 255).astype(np.uint8), (2, 1, 0))
            pred = np.transpose((outputs[i].cpu().numpy() * 255).astype(np.uint8), (2, 1, 0))
            lbl = np.transpose((labels[i].cpu().numpy() * 255).astype(np.uint8), (2, 1, 0))
            cv2.imwrite('{}/{}_input.png'.format(out_path, i), cv2.cvtColor(inp, cv2.COLOR_RGB2BGR))
            cv2.imwrite('{}/{}_predict.png'.format(out_path, i), cv2.cvtColor(pred, cv2.COLOR_RGB2BGR))
            cv2.imwrite('{}/{}_label.png'.format(out_path, i), cv2.cvtColor(lbl, cv2.COLOR_RGB2BGR))



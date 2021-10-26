import torch
from general_utils import read_yaml, get_class
import os
from shutil import rmtree
from datetime import datetime
from time import time
from torch.utils.tensorboard import SummaryWriter
from traces import Trace
from typing import List
import yaml
from tqdm import tqdm, trange
from collections import OrderedDict
# import pandas as pd
# from pathlib import Path
import numpy as np
import cv2


class ConfigurationStruct:
    def __init__(self, entries: dict):
        self.__dict__.update(entries)
        for key, val in self.__dict__.items():
            if 'kargs' in key and val is None:
                self.__dict__[key] = {}


class UtilityObj:
    def __init__(self, loaders, writer_path, measurements=None, losses_names=None):
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
        self.steps = 0
        self.name = os.path.basename(writer_path)
        self.pname = self.name + (' ' * (7 - len(self.name)))
        self.traces: List[Trace] = []
        self.losses = OrderedDict({l: 0.0 for l in losses_names})

        if measurements:
            for meas in measurements.values():
                cls_ = get_class(meas['type'], meas['path'])
                self.traces.append(cls_(self.writer, self.size))

    def step(self, loss, inputs, pred, labels, losses=None):
        for m in self.traces:
            m.add_measurement(inputs, pred, labels)
        self.aggregated_loss += loss
        self.steps += 1
        if self.losses and losses:
            for agg_loss_type, loss_res in zip(self.losses.keys(), losses):
                self.losses[agg_loss_type] += loss_res.item()

    def epoch_step(self, epoch):
        for m in self.traces:
            try:
                m.write_epoch(epoch)
            except Exception:
                pass
        avr = self.aggregated_loss / self.size
        self.writer.add_scalar('Loss', avr, global_step=epoch)
        for loss_name, loss_val in self.losses.items():
            self.writer.add_scalar(f'Losses/{loss_name}', loss_val/self.size, global_step=epoch)
        self.init_loop()

    def init_loop(self):
        self.aggregated_loss = 0.0
        self.steps = 0
        for loss_name in self.losses.keys():
            self.losses[loss_name] = 0.0

    @property
    def size(self):
        return max(self.steps, 1)


class TorchTrainer:
    """wrapper class for torch models training"""

    def __init__(self, cfg: ConfigurationStruct, root, gpu_index: int):
        self.cfg = cfg
        if int(gpu_index) >= 0 and torch.cuda.is_available():
            self.device = torch.device("cuda:" + str(gpu_index))
            print('using device: ', torch.cuda.get_device_name(self.device))
        else:
            self.device = torch.device("cpu")
            print('using cpu')
        self.model: torch.nn.Module = None
        self.optimizer = None
        self.start_epoch: int = 0
        self.root = root
        self.dataset = None
        self.running = False
        self.loss_debug = OrderedDict()

    @classmethod
    def new_train(cls, out_path, model_cfg, optimizer_cfg, dataset_cfg, gpu_index, exp_name):
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
        cls = TorchTrainer(cfg=config, root=root, gpu_index=gpu_index)
        cls.init_nn()
        cls.model.to(cls.device)
        return cls

    @classmethod
    def warm_startup(cls, root, gpu_index, strict, best=False):
        config_dict = read_yaml(os.path.join(root, 'cfg.yaml'))
        for cfg_section, cfg_dict in config_dict.items():
            config_dict[cfg_section] = ConfigurationStruct(cfg_dict)
        config = ConfigurationStruct(config_dict)
        cls = TorchTrainer(cfg=config, root=root, gpu_index=gpu_index)
        cls.init_nn()
        cls.load_checkpoint(strict, best=best)
        return cls

    def init_nn(self):
        model_cls = get_class(self.cfg.model.type, file_path=self.cfg.model.path)
        self.model = model_cls(**self.cfg.model.kargs)
        dataset_cls = get_class(self.cfg.data.type, file_path=self.cfg.data.path)
        self.dataset = dataset_cls(self.root, self.cfg.model.in_channels, self.cfg.model.out_channels,
                                   **self.cfg.data.kargs)
        optim_cls = get_class(self.cfg.optimizer.type, module_path='torch.optim')
        self.optimizer = optim_cls(self.model.parameters(), **self.cfg.optimizer.kargs)

    def load_checkpoint(self, strict, best=False):
        if best:
            cp_path = os.path.join(self.root, 'checkpoints', 'checkpoint.pth')
        else:
            cp_path = os.path.join(self.root, 'checkpoints', 'last_checkpoint.pth')
            if not os.path.exists(cp_path):
                cp_path = os.path.join(self.root, 'checkpoints', 'checkpoint.pth')
        print('loading checkpoint', cp_path)
        checkpoint = torch.load(cp_path, map_location=self.device)
        self.start_epoch = checkpoint['epoch']
        self.model.load_state_dict(checkpoint['model_state_dict'], strict=strict)
        # if not strict:
        #     print('non strict', self.model.fine_tune_)
        #     if self.model.fine_tune is not None and self.cfg.model.fine_tune_kargs is not None:
        #         self.model.fine_tune(**self.cfg.model.fine_tune_kargs)
        #     optim_cls = get_class(self.cfg.optimizer.type, module_path='torch.optim')
        #     self.optimizer = optim_cls(self.model.parameters(), **self.cfg.optimizer.kargs)
        # else:
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(self.device)
        self.model.to(self.device)

    def save_checkpoint(self, better, epoch):
        if better:
            fpath = os.path.join(self.root, 'checkpoints', 'checkpoint.pth'.format(self.start_epoch))
        else:
            fpath = os.path.join(self.root, 'checkpoints', 'last_checkpoint.pth'.format(self.start_epoch))
        torch.save({'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict()}, fpath)
        # if self.epoch % 50 == 0:
        #     fpath = os.path.join(self.root, 'checkpoints', 'checkpoint_{}.pth'.format(self.epoch))
        #     # print('saving checkpoint {}'.format(fpath))
        #     torch.save({'epoch': self.epoch,
        #                 'model_state_dict': self.model.state_dict(),
        #                 'optimizer_state_dict': self.optimizer.state_dict()}, fpath)

    def init_training_obj(self, loss_names):
        """create helper objects in order to make the train function more clear"""
        train_loaders, test_loaders = self.dataset.get_data_loaders(self.cfg.model.batch_size)
        train = UtilityObj(loaders=train_loaders, writer_path=os.path.join(self.root, 'train'), losses_names=loss_names)
        test = UtilityObj(loaders=test_loaders, writer_path=os.path.join(self.root, 'test'),
                          measurements=self.cfg.model.test_traces, losses_names=loss_names)
        return train, test

    def init_loss_func(self):
        criterias = []
        factors = []
        loss_names = []
        use_consistency = False
        for loss_name, loss_cfg in self.cfg.model.loss.items():
            loss_names.append(loss_name)
            module_path = loss_cfg['module_path'] if 'module_path' in loss_cfg else 'torch.nn'
            criterion_cls = get_class(loss_name, module_path)
            class_instance = criterion_cls(**loss_cfg['kargs'])
            if class_instance is torch.nn.Module:
                class_instance.to(self.device)
            criterias.append(class_instance)
            if 'factor' in loss_cfg:
                factors.append(eval(loss_cfg['factor']) if type(loss_cfg['factor']) is str else loss_cfg['factor'])
            else:
                factors.append(1)
            if use_consistency:
                loss_names.append('Consistency')

        def crit(inputs, preds, labels):
            losses = []
            for factor, criteria in zip(factors, criterias):
                losses.append(factor * criteria(preds, labels))
            if use_consistency:
                consistency = torch.nn.functional.mse_loss(inputs, ((preds[:, :3] + preds[:, 3:]) / 2))
                losses.append(consistency)
            return losses
        return crit, loss_names

    def train(self):
        self.model.zero_grad()
        crit, loss_names = self.init_loss_func()
        train, test = self.init_training_obj(loss_names)
        best_loss = 2 ** 16
        self.running = True

        ep_prog = trange(self.start_epoch, self.cfg.model.epochs, desc='epochs', ncols=120)
        for epoch in ep_prog:
            if epoch % 50 == 0:
                test_freq = int(0.5 + (-7 * np.log(epoch+1/self.cfg.model.epochs + 2)))
            self.model.train()
            for train_loader in train.loaders:
                prog = tqdm(train_loader, desc='train', leave=False, ncols=100)
                for x, y in prog:
                    inputs, labels = x.to(self.device), y.to(self.device)
                    outputs = self.model(inputs)
                    self.optimizer.zero_grad()
                    losses = crit(inputs, outputs, labels)
                    loss = sum(losses)
                    loss.backward()
                    self.optimizer.step()
                    train.step(loss, inputs, outputs, labels, losses)
                    prog.set_description(f'train loss {loss.item():.2}')
            train.epoch_step(epoch)

            if epoch % test_freq == 0:
                self.model.eval()
                with torch.no_grad():
                    for test_loader in test.loaders:
                        prog = tqdm(test_loader, desc='validation', leave=False, ncols=100)
                        for x, y in prog:
                            inputs, labels = x.to(self.device), y.to(self.device)
                            outputs = self.model(inputs)
                            losses = crit(inputs, outputs, labels)
                            loss = sum(losses)
                            test.step(loss.item(), inputs, outputs, labels, losses)
                            prog.set_description(f'validation loss {loss.item():.2}')
                if test.aggregated_loss < best_loss:
                    best_loss = test.aggregated_loss
                    self.save_checkpoint(better=True, epoch=epoch)
                else:
                    self.save_checkpoint(better=False, epoch=epoch)
                test.epoch_step(epoch)

    def debug_save(self, inputs, outputs, labels):
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

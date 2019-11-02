import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import abc


class PerformanceMeasurement(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def add_measurement(self, outputs: torch.Tensor, labels: torch.Tensor, step):
        pass

    @abc.abstractmethod
    def write_step(self, writer: SummaryWriter, mini_batches, step):
        pass


class PixelWiseAccuracy(PerformanceMeasurement):
    def __init__(self):
        self.sum = 0.0

    def add_measurement(self, outputs: torch.Tensor, labels: torch.Tensor, step):
        self.sum += (int(torch.sum(outputs.argmax(dim=1) == labels.data).to('cpu')) / labels.data.nelement()) * 100

    def write_step(self, writer: SummaryWriter, mini_batches: int, step: int):
        writer.add_scalar(tag=self.__class__.__name__, scalar_value=self.sum / mini_batches, global_step=step)
        self.sum = 0.0


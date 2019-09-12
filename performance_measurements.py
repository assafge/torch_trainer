import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np


class PerformanceMeasurement:
    def add_measurement(self, outputs: torch.Tensor, labels: torch.Tensor):
        pass

    def write_step(self, writer: SummaryWriter, mini_batches, step):
        pass


class PixelWiseAccuracy(PerformanceMeasurement):
    def __init__(self):
        self.sum = 0.0

    def add_measurement(self, outputs: torch.Tensor, labels: torch.Tensor):
        total = labels.nelement()
        predicted = outputs.data
        predicted = predicted.to('cpu')
        predicted_img = predicted.argmax(dim=1).numpy()

        labels_data = labels.data
        labels_data = labels_data.to('cpu')
        labels_data = labels_data.numpy().astype(np.int)
        if labels_data.shape == predicted_img.shape:
            corr = (predicted_img == labels_data)
            correct = np.sum(corr)
            self.sum += (correct / total)

    def write_step(self, writer: SummaryWriter, mini_batches: int, step: int):
        writer.add_scalar(tag=self.__class__.__name__, scalar_value=self.sum / mini_batches, global_step=step)
        self.sum = 0.0

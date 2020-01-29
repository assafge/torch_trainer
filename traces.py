import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import abc
from trainer_utils import plot_confusion_matrix


class Trace(metaclass=abc.ABCMeta):
    def __init__(self, writer: SummaryWriter, mini_batches: int):
        self.writer = writer
        self.mini_batches = mini_batches

    @abc.abstractmethod
    def add_measurement(self, predictions: torch.Tensor, labels: torch.Tensor):
        pass

    @abc.abstractmethod
    def write_epoch(self, epoch: int):
        pass

class StepTrace(Trace):
    def __init__(self, writer: SummaryWriter, mini_batches: int):
        super().__init__(writer, mini_batches)
        self.sum = 0.0

    def write_epoch(self, epoch: int):
        self.writer.add_scalar(tag=self.__class__.__name__, scalar_value=self.sum / self.mini_batches, global_step=epoch)
        self.sum = 0.0


class PixelWiseAccuracy(StepTrace):
    def add_measurement(self, predictions: torch.Tensor, labels: torch.Tensor):
        self.sum += (int(torch.sum(predictions.argmax(dim=1) == labels.data).to('cpu')) / labels.data.nelement()) * 100


class QuaziPixelWiseAccuracy(PixelWiseAccuracy):
    def add_measurement(self, predictions: torch.Tensor, labels: torch.Tensor):
        super().add_measurement(predictions, labels)
        super().add_measurement(predictions, labels + 1)
        super().add_measurement(predictions, labels - 1)


class ImageTrace(Trace):
    def __init__(self, writer: SummaryWriter, mini_batches: int):
        super().__init__(writer, mini_batches)
        self.did_wrote = False  # write every image once
        self.step = 0
        self.pred = None
        self.lbl = None
        self.dataformat = 'CHW'

    def add_measurement(self, predictions: torch.Tensor, labels: torch.Tensor):
        if not self.did_wrote:
            self.pred = (predictions[0].cpu().numpy() * 255).astype(np.uint8)
            self.lbl = (labels[0].cpu().numpy() * 255).astype(np.uint8)
            self.did_wrote = True

    def write_epoch(self, step):
        self.did_wrote = False
        self.writer.add_image('Predicted', self.pred, global_step=step, dataformats=self.dataformat)
        self.writer.add_image('GT', self.lbl, global_step=step, dataformats=self.dataformat)
        self.step = step

class DepthImageTrace(ImageTrace):
    def __init__(self, writer: SummaryWriter, mini_batches: int):
        super().__init__(writer, mini_batches)
        self.dataformat = 'HW'

class ClassificationImageTrace(ImageTrace):
    def __init__(self, writer: SummaryWriter, mini_batches: int):
        super().__init__(writer, mini_batches)
        self.dataformat = 'HW'

    def add_measurement(self, predictions: torch.Tensor, labels: torch.Tensor):
        if not self.did_wrote:
            classes = predictions[0].shape[0]
            self.pred = (predictions[0].argmax(dim=0).cpu().numpy() * (255 / classes)).astype(np.uint8)
            self.lbl = (labels[0].cpu().numpy() * (255 / classes)).astype(np.uint8)
            self.did_wrote = True

class ConfusionMatrix(ImageTrace):
    def add_measurement(self, outputs, labels):
        if not self.did_wrote:
            pred = outputs.argmax(dim=1).cpu().numpy().astype(np.uint8).ravel()
            ref = labels.cpu().numpy().astype(np.uint8).ravel()
            im = plot_confusion_matrix(ref, pred, [str(i) for i in range(16)])
            self.writer.add_image('confusion_mat', im, global_step=self.step, dataformats='HWC')
            self.did_wrote = True
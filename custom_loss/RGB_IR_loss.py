import torch
from abc import ABC, abstractmethod
from pytorch_msssim import MS_SSIM


class RGBIRConsistency:
    def __call__(self, inputs, preds):
        if preds.shape[1] == 6:
            return torch.nn.functional.smooth_l1_loss(inputs, ((preds[:, :3] + preds[:, 3:]) / 2))
        elif preds.shape[1] == 4:
            return torch.nn.functional.smooth_l1_loss(inputs, ((preds[:, :3] + preds[:, 3]) / (4/3)))


class MS_SSIM_Loss(MS_SSIM):
    def forward(self, preds, labels):
        return 1 - super().forward(preds, labels)


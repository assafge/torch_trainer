import torch


class Consistency(torch.nn.Module):
    def forward(self, preds, targets):
        return (preds[:3] + preds[3:])

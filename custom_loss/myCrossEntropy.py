import torch

#
# class WeightedPerPixelCrossEntropyLoss(torch.nn.L1Loss):
#     def __init__(self, weight, size_average=None, reduce=None, reduction='mean'):
#         super(WeightedPerPixelCrossEntropyLoss, self).__init__(size_average, reduce, reduction)
#         self.weights = weight

def WeightedPerPixelCrossEntropyLoss(inp, target, weights):
    total = torch.tensor(0, dtype=torch.long)
    diff = (torch.argmax(inp, dim=1) - target) ** 2
    z = torch.zeros(inp[0].shape, dtype=torch.long, device=inp.device)
    for ind, w in enumerate(weights):
        total += torch.sum(w * torch.where(target == ind, diff, z))
    loss = torch.tensor(data=total, dtype=torch.float, device=inp.device)
    loss = loss / (target.data.nelement() * 100)
    return loss



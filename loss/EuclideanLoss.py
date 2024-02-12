import torch
from torch import nn

class EuclideanLoss(nn.Module):
    def __init__(self):
        super(EuclideanLoss, self).__init__()

    def forward(self, x, y):
        loss = torch.sqrt((x - y) ** 2).sum()
        return loss
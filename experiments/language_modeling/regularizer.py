import torch
import torch.nn as nn

from torch import Tensor


class Regularizer(nn.Module):
    def __init__(self):
        super(self).__init__()


    def forward(self, mu: Tensor, logsigma: Tensor) -> Tensor:
        var = torch.exp(2 * logsigma)
        return 0.5 * (mu.pow(2) + var - 2*logsigma - 1).sum(dim=1).mean()

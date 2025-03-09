from typing import Callable, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

import numpy as np


class OldGenomicBottleNet(nn.Module):
    """
    Improved version of the variable-length g-net that uses a for-loop for 
    initialization.
    
    Args:
        layers (nn.ModuleList): ModuleList that contains all differentiable
            layers of the g-net.
        sizes (Sequence[int]): List of sizes for the g-net layers.
        output_scale (float): Scaling factor for the output of the g-net.
        activation_fn (Optional[Callable[[Tensor], Tensor]]): Activation 
            function for the hidden layers. Default is ReLU.

    Returns:
        Tensor: Prediction of the new weight.
    """
    layers: nn.ModuleList
    sizes: Sequence[int]
    output_scale: float

    def __init__(self, sizes: Sequence[int], output_scale: float,
                 activation_fn: Optional[Callable[[Tensor], Tensor]] = F.relu) -> None:
        super(GenomicBottleNet, self).__init__()
        self.output_scale = output_scale.detach()
        self.activation_fn = activation_fn
        self.sizes = sizes
        length = len(sizes) - 1
        self.layers = nn.ModuleList([nn.Linear(sizes[i], sizes[i+1], bias=True) 
                                    for i in range(length)])
        self.batchnorm = nn.BatchNorm1d(sizes[-2])

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers[:-1]:
            x = self.activation_fn(layer(x))
        x = self.batchnorm(x)
        x = self.layers[-1](x) # no non-linearity on the last layer
        # NOTE: Do not touch output norm! Carefully computed by hand...
        return x * torch.tensor(2.) * self.output_scale
                

GNetLayerTuple = Tuple[Tensor, Sequence[OldGenomicBottleNet], Sequence[int], float]

# Stochastic
class GenomicBottleNet(nn.Module):
    """
    Improved version of the variable-length g-net that uses a for-loop for 
    initialization.
    
    Args:
        layers (nn.ModuleList): ModuleList that contains all differentiable
            layers of the g-net.
        sizes (Sequence[int]): List of sizes for the g-net layers.
        output_scale (float): Scaling factor for the output of the g-net.
        activation_fn (Optional[Callable[[Tensor], Tensor]]): Activation 
            function for the hidden layers. Default is ReLU.

    Returns:
        Tensor: Prediction of the new weight.
    """
    layers: nn.ModuleList
    sizes: Sequence[int]
    output_scale: float

    def __init__(self, sizes: Sequence[int], output_scale: float,
                 activation_fn: Optional[Callable[[Tensor], Tensor]] = F.relu) -> None:
        super(GenomicBottleNet, self).__init__()
        self.output_scale = output_scale.detach()
        self.activation_fn = activation_fn
        self.sizes = sizes
        length = len(sizes) - 1
        self.layers = nn.ModuleList([nn.Linear(sizes[i], sizes[i+1], bias=True) 
                                    for i in range(length)])
        self.init_weights()
        self.layers[-1].bias.data = torch.tensor([0.0, np.log(0.02, dtype=np.float32)])
        
    def init_weights(self):
        for layer in self.layers[:-1]:
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.weight) # zeros or xavier?

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers[:-1]:
            x = self.activation_fn(layer(x))
        x = self.layers[-1](x) # no non-linearity on the last layer

        mu, logsigma = x[:, 0], x[:, 1]
        eps = torch.randn_like(logsigma).detach()
        return (mu + torch.exp(logsigma) * eps) * torch.tensor(2.) * self.output_scale


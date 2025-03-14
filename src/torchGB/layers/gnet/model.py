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
    model: nn.Sequential
    sizes: Sequence[int]
    output_scale: float

    def __init__(self, sizes: Sequence[int], output_scale: float,
                activation_fn: Optional[Callable[[Tensor], Tensor]] = nn.ReLU) -> None:
        super(GenomicBottleNet, self).__init__()
        self.output_scale = output_scale.detach()
        self.activation_fn = activation_fn
        self.sizes = sizes
        length = len(sizes) - 1 # no non-linearity on the last layer
        
        layer_list = []
        for i in range(length):
            layer_list.append(nn.Linear(sizes[i], sizes[i+1]))
            layer_list.append(activation_fn())
                
        layer_list.pop(-1)
            
        self.model = nn.Sequential(*layer_list)
        self.init_weights()
        self.model[-1].bias.data = torch.tensor([0.0, np.log(0.02, dtype=np.float32)])
        
    def init_weights(self):
        for layer in self.model:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def forward(self, x: Tensor) -> Tensor:
        # NOTE: Do not touch output norm! Carefully computed by hand...
        return self.model(x) * torch.tensor(2.) * self.output_scale
                

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
    model: nn.Sequential
    sizes: Sequence[int]
    output_scale: float

    def __init__(self, sizes: Sequence[int], output_scale: float,
                 activation_fn: Optional[Callable[[Tensor], Tensor]] = nn.ReLU) -> None:
        super(GenomicBottleNet, self).__init__()
        self.output_scale = output_scale.detach()
        self.activation_fn = activation_fn
        self.sizes = sizes
        length = len(sizes) - 1 # no non-linearity on the last layer
        
        layer_list = []
        for i in range(length):
            layer_list.append(nn.Linear(sizes[i], sizes[i+1]))
            layer_list.append(activation_fn())
                
        layer_list.pop(-1)
            
        self.model = nn.Sequential(*layer_list)
        self.init_weights()
        self.model[-1].bias.data = torch.tensor([0.0, np.log(0.02, dtype=np.float32)])
        
    def init_weights(self):
        for layer in self.model:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def forward(self, x: Tensor) -> Tensor:
        out = self.model(x)

        mu, logsigma = out[:, 0], out[:, 1]
        eps = torch.randn_like(logsigma).detach()
        return (mu + torch.exp(logsigma) * eps) * torch.tensor(2.) * self.output_scale
    
    
class Reshape(nn.Module):
    def __init__(self, *shape) -> None:
        super().__init__()
        self.shape = shape

    def forward(self, x: Tensor) -> Tensor:
        return x.view(x.size(0), *self.shape) # Preserve batch dimension
    
    
class FastGenomicBottleNet(nn.Module):
    """_summary_
    
    TODO: generalize the implementation?
    TODO: use this implementation for fast computation of a set of adjacent tiles

    Args:
        nn (_type_): _description_

    Returns:
        _type_: _description_
    """
    model: nn.Sequential
    sizes: Sequence[int]
    output_scale: float
    
    def __init__(self, num_tiles: int, sizes: Sequence[int], 
                 activation_fn: nn.Module = nn.ReLU) -> None:
        super().__init__()
        self.sizes = sizes
        layer_list = [nn.Conv1d(1, num_tiles*sizes[1], kernel_size=sizes[0])]
        layer_list.append(activation_fn())
        # NOTE: The `groups` argument in the following layers ensures that each 
        # tile has its own set of weights.
        layer_list.append(Reshape(num_tiles, sizes[1]))
        layer_list.append(nn.Conv1d(num_tiles, num_tiles, kernel_size=sizes[1], groups=num_tiles))
        self.model = nn.Sequential(*layer_list)
        self.output_scale = 1.0
        
    def init_weights(self):
        for layer in self.model:
            if isinstance(layer, nn.Conv2d):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
                    
    def forward(self, x: Tensor) -> Tensor:
        # NOTE: 
        return self.model(x) * torch.tensor(2.) * self.output_scale
        

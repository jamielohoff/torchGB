from typing import Sequence, Tuple
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .utils import make_row_col_encoding, get_tile_size, EncodingType



class GenomicBottleNet(nn.Module):
    """
    Improved version of the variable-length Gnet that uses a for-loop for 
    initialization. This is a more flexible version of the GDNx classes.

    Args:
        `layers` (nn.ModuleList): ModuleList that contains all differentiable
                                layers of the G-Net.
        `output_scale` (float): Scaling factor for the output of the G-Net.

    Returns:
        float: Prediction of the new weight.
    """
    layers: nn.ModuleList
    sizes: Sequence[int]
    output_scale: float
    
    def __init__(self, 
                sizes: Sequence[int], 
                output_scale: float) -> None:
        assert len(sizes) > 1, "List must have at least 3 entries!"
        super(GenomicBottleNet, self).__init__()
        self.output_scale = output_scale
        self.sizes = sizes
        length = len(sizes) - 1
        self.layers = nn.ModuleList([nn.Linear(sizes[i], sizes[i+1]) 
                                    for i in range(length)])

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers[:-1]:
            x = F.silu(layer(x))
        x = self.layers[-1](x) # no non-linearity on the last layer
        output_scale = self.output_scale if self.output_scale > 1e-8 else torch.tensor(1.).to(self.output_scale.device)
        return output_scale * x
    
    def init_weights(self, module: nn.Module) -> None:         
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0, std=np.sqrt(2./module.weight.shape[0]))
            nn.init.zeros_(module.bias)


def conv2d_gnet_layer(param_shape: Tuple[int, int], 
                    hidden_dim: int, 
                    output_scale: float, 
                    max_gnet_batch: int) -> Tuple[Tensor, GenomicBottleNet, Tuple[int, int]]:
    num_encoding_bits = np.ceil(np.log(param_shape)/np.log(2)).astype(np.uint16)
    num_encoding_bits[:2] = param_shape[:2]
    num_encoding_bits[np.where(num_encoding_bits == 0)] = 1
    encoding_type = (EncodingType.ONEHOT, 
                    EncodingType.ONEHOT, 
                    EncodingType.BINARY, 
                    EncodingType.BINARY)         
                  
    row_col_encoding = make_row_col_encoding(param_shape,
                                            encoding_type,
                                            num_encoding_bits)
    num_inputs = row_col_encoding.shape[1]
    
    gnet_sizes = (num_inputs, hidden_dim, hidden_dim//2, 1) 
    gnet = GenomicBottleNet(gnet_sizes, output_scale=output_scale)   
    return row_col_encoding, gnet


def default_gnet_layer(param_shape: Tuple[int, int], 
                        hidden_dim: int, 
                        output_scale: float, 
                        max_gnet_batch: int) -> Tuple[Tensor, GenomicBottleNet, Tuple[int, int]]:
    num_row_tiles, num_col_tiles, row_tile_size, col_tile_size = get_tile_size(param_shape, max_gnet_batch)
    
    num_encoding_bits = np.ceil(np.log(param_shape)/np.log(2)).astype(np.uint16)
    num_encoding_bits[np.where(num_encoding_bits == 0)] = 1
    encoding_type = (EncodingType.BINARY, 
                    EncodingType.BINARY)
    
    tile_shape = (row_tile_size, col_tile_size)
    row_col_encoding = make_row_col_encoding(tile_shape,
                                            encoding_type, 
                                            num_encoding_bits)
    num_inputs = row_col_encoding.shape[-1]

    gnet_sizes = (num_inputs, hidden_dim, 1)
    gnets = [GenomicBottleNet(gnet_sizes, output_scale) 
            for _ in range(num_row_tiles*num_col_tiles)]
    return row_col_encoding, gnets, (row_tile_size, col_tile_size)


def qkv_gnet_layer(param_shape: Tuple[int, int], 
                    hidden_dim: int, 
                    output_scale: float, 
                    max_gnet_batch: int) -> Tuple[Tensor, GenomicBottleNet, Tuple[int, int]]:
    # Subdivide the attention weight matrix in three similar parts Wq, Wk, Wv
    _param_shape = (param_shape[0] // 3, param_shape[1])
    num_row_tiles, num_col_tiles, row_tile_size, col_tile_size = get_tile_size(_param_shape, max_gnet_batch)
    
    # Treat 2D weight as fully connected
    num_encoding_bits = np.ceil(np.log(_param_shape)/np.log(2)).astype(np.uint16)
    num_encoding_bits[np.where(num_encoding_bits == 0)] = 1
    encoding_type = (EncodingType.BINARY, 
                    EncodingType.BINARY)
    
    tile_shape = (row_tile_size, col_tile_size)
    row_col_encoding = make_row_col_encoding(tile_shape,
                                            encoding_type, 
                                            num_encoding_bits)
    num_inputs = row_col_encoding.shape[-1]
 
    gnet_sizes = (num_inputs, hidden_dim, 1)
    gnets = [GenomicBottleNet(gnet_sizes, output_scale) 
            for _ in range(3*num_row_tiles*num_col_tiles)]
    
    return row_col_encoding, gnets, (row_tile_size, col_tile_size)
    
    
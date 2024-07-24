from typing import Sequence
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .utils import make_row_col_encoding, tile_matrix


NIL = 0 # No encoding
HOT = 1 # One-hot vector
BIN = 2 # Binary code
GRY = 3 # Gray code
LIN = 4 # Linear? code
RND = 5 # Random code


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
    output_scale: float
    
    def __init__(self, 
                sizes: Sequence[int], 
                output_scale: float) -> None:
        assert len(sizes) > 1, "List must have at least 3 entries!"
        super(GenomicBottleNet, self).__init__()
        self.output_scale = output_scale
        length = len(sizes) - 1
        self.layers = nn.ModuleList([nn.Linear(sizes[i], sizes[i+1]) 
                                    for i in range(length)])
        # self.apply(self.init_weights)

    def forward(self, x: Tensor) -> Tensor:
        # TODO Rescaling inputs such that they are in range [0, inf] with std 1
        
        for layer in self.layers:
            x = F.silu(layer(x))
        output_scale = self.output_scale if self.output_scale > 1e-8 else torch.tensor(1.).to(self.output_scale.device)
        return output_scale * x
    
    def init_weights(self, module: nn.Module) -> None:     
        # TODO weights should only be initialized with inputs in mind       
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0, std=np.sqrt(2./module.weight.shape[0]))
            nn.init.zeros_(module.bias)


def conv2d_gnet_layer(param_shape, hidden_dim, output_scale):
    encoding_bits = np.ceil(np.log(param_shape)/np.log(2)).astype("uint16")
    encoding_bits[:2] = param_shape[:2]
    encoding_bits[np.where(encoding_bits == 0)] = 1
    encoding_type = (HOT, HOT, BIN, BIN)         
                  
    row_col_encoding = make_row_col_encoding(encoding_type, param_shape, encoding_bits)
    num_inputs = row_col_encoding.shape[1]
    
    gnet_sizes = (num_inputs, hidden_dim, hidden_dim, hidden_dim//2, 1) 
    gnet = GenomicBottleNet(gnet_sizes, output_scale=output_scale)   
    return row_col_encoding, gnet
    
    
max_gnet_batch = 36_864

def default_gnet_layer(param_shape, hidden_dim, output_scale):
    row_size, col_size = param_shape
    
    numel = np.prod(param_shape)
    n = numel / max_gnet_batch
    num_row_tiles = np.sqrt(n * row_size / col_size)
    num_col_tiles = np.sqrt(n * col_size / row_size)
    
    row_tile_size = np.min([row_size, np.ceil(row_size/num_row_tiles)])
    col_tile_size = np.min([col_size, np.ceil(col_size/num_col_tiles)])
    
    # Treat 2D weight as fully connected
    encoding_bits = np.ceil(np.log(param_shape)/np.log(2)).astype("uint16")
    encoding_bits[np.where(encoding_bits == 0)] = 1
    encoding_type = (BIN, BIN)
    
    row_col_encoding = make_row_col_encoding(encoding_type, 
                                            param_shape, 
                                            encoding_bits)
    row_col_encoding = row_col_encoding.reshape(*param_shape, -1)
    num_inputs = row_col_encoding.shape[1]
    print(row_tile_size, col_tile_size)
    subdivided_row_col_encodings = tile_matrix(row_col_encoding, 
                                                row_tile_size, 
                                                col_tile_size)
    
    gnet_sizes = (num_inputs, hidden_dim, hidden_dim//2, 1)
    gnets = [GenomicBottleNet(gnet_sizes, output_scale=output_scale) 
            for _ in range(num_row_tiles*num_col_tiles)]
    return subdivided_row_col_encodings, gnets, (row_tile_size, col_tile_size)


def qkv_gnet_layer(param_shape, hidden_dim, output_scale):
    # Subdivide the attention weight matrix in three similar parts Wq, Wk, Wv
    param_shape = (param_shape[0]//3, param_shape[1])
    encodings_layers = [default_gnet_layer(param_shape, hidden_dim, output_scale) for _ in range(3)]
    return zip(*encodings_layers)
    
    
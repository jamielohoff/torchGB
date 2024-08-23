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
        `sizes` (Sequence[int]): List of sizes for the G-Net layers.
        `output_scale` (float): Scaling factor for the output of the G-Net.

    Returns:
        Tensor: Prediction of the new weight.
    """
    layers: nn.ModuleList
    sizes: Sequence[int]
    output_scale: float
    
    def __init__(self, 
                sizes: Sequence[int], 
                output_scale: float) -> None:
        assert len(sizes) > 1, "List must have at least 3 entries!"
        super(GenomicBottleNet, self).__init__()
        self.output_scale = output_scale.detach()
        self.sizes = sizes
        length = len(sizes) - 1
        self.layers = nn.ModuleList([nn.Linear(sizes[i], sizes[i+1]) 
                                    for i in range(length)])
        self.apply(self.init_weights)

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        x = self.layers[-1](x) # no non-linearity on the last layer
        return x # * self.output_scale
    
    def init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=1.25*0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias) # nn.init.normal_(module.bias, std=0.02) # 
                

GNetLayerTuple = Tuple[Tensor, Sequence[GenomicBottleNet], Sequence[int], float]


def conv2d_gnet_layer(param: Tensor, 
                    hidden_dim: int, 
                    max_gnet_batch: int) -> GNetLayerTuple:
    param_shape = param.data.shape          
    num_encoding_bits = np.ceil(np.log(param_shape)/np.log(2)).astype(np.uint16)
    num_encoding_bits[:2] = param_shape[:2]
    num_encoding_bits[np.where(num_encoding_bits == 0)] = 1
    encoding_type = (EncodingType.BINARY, 
                    EncodingType.BINARY, 
                    EncodingType.ONEHOT, 
                    EncodingType.ONEHOT)   

    tile_size = np.ceil(np.sqrt(max_gnet_batch//(param_shape[2]*param_shape[3]))).astype(np.int32)
    num_row_tiles = np.ceil(param_shape[0] / tile_size).astype(np.int32)
    num_col_tiles = np.ceil(param_shape[1] / tile_size).astype(np.int32)

    tile_shape = (tile_size, tile_size, param_shape[2], param_shape[3])  
    row_col_encoding = make_row_col_encoding(tile_shape,
                                            encoding_type,
                                            num_encoding_bits)
    num_inputs = row_col_encoding.shape[-1]
    
    # Normalizes the output to follow the initial parameter
    # distribution at initialization of the model       
    with torch.no_grad():
        output_scale = torch.std(param.data) 
    
    gnet_sizes = (num_inputs, hidden_dim, 1) 
    gnets = [GenomicBottleNet(gnet_sizes, output_scale) 
            for _ in range(num_row_tiles*num_col_tiles)]        
    return row_col_encoding, gnets, tile_shape, output_scale


def default_gnet_layer(param: Tensor, 
                        hidden_dim: int, 
                        max_gnet_batch: int) -> GNetLayerTuple:
    param_shape = param.data.shape
    num_row_tiles, num_col_tiles, row_tile_size, col_tile_size = get_tile_size(param_shape, 
                                                                                max_gnet_batch)
    
    num_encoding_bits = np.ceil(np.log(param_shape)/np.log(2)).astype(np.uint16)
    num_encoding_bits[np.where(num_encoding_bits == 0)] = 1
    encoding_type = (EncodingType.BINARY, 
                    EncodingType.BINARY)
    
    tile_shape = (row_tile_size, col_tile_size)
    row_col_encoding = make_row_col_encoding(tile_shape,
                                            encoding_type, 
                                            num_encoding_bits)
    num_inputs = row_col_encoding.shape[-1]
    
    # Normalizes the output to follow the initial parameter
    # distribution at initialization of the model       
    with torch.no_grad():
        output_scale = torch.std(param.data)    

    gnet_sizes = (num_inputs, hidden_dim, 1)
    gnets = [GenomicBottleNet(gnet_sizes, output_scale) 
            for _ in range(num_row_tiles*num_col_tiles)]     
    
    return row_col_encoding, gnets, tile_shape, output_scale


def square_default_gnet_layer(param: Tensor, 
                            hidden_dim: int,  
                            max_gnet_batch: int) -> GNetLayerTuple:
    param_shape = param.data.shape
    # Calculates the number of square tiles of size `max_gnet_batch` we need to
    # completely cover the weight matrix
    tile_size = np.ceil(np.sqrt(max_gnet_batch)).astype(np.int32)
    num_row_tiles = np.ceil(param_shape[0] / tile_size).astype(np.int32)
    num_col_tiles = np.ceil(param_shape[1] / tile_size).astype(np.int32)
    
    num_encoding_bits = np.ceil(np.log(param_shape)/np.log(2)).astype(np.uint16)
    num_encoding_bits[np.where(num_encoding_bits == 0)] = 1
    encoding_type = (EncodingType.BINARY, 
                    EncodingType.BINARY)
    
    tile_shape = (tile_size, tile_size)
    row_col_encoding = make_row_col_encoding(tile_shape,
                                            encoding_type, 
                                            num_encoding_bits)
    num_inputs = row_col_encoding.shape[-1]
    
    # Normalizes the output to follow the initial parameter
    # distribution at initialization of the model       
    with torch.no_grad():
        output_scale = torch.std(param.data)   

    gnet_sizes = (num_inputs, hidden_dim,  1)
    gnets = [GenomicBottleNet(gnet_sizes, output_scale) 
            for _ in range(num_row_tiles*num_col_tiles)]      
    
    return row_col_encoding, gnets, tile_shape, output_scale


def qkv_gnet_layer(param: Tensor, 
                    hidden_dim: int, 
                    max_gnet_batch: int) -> GNetLayerTuple:
    # Subdivide the attention weight matrix in three similar parts Wq, Wk, Wv
    param_shape = param.data.shape
    _param_shape = (param_shape[0] // 3, param_shape[1])
    num_row_tiles, num_col_tiles, row_tile_size, col_tile_size = get_tile_size(_param_shape, 
                                                                                max_gnet_batch)
    
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
    
    # Normalizes the output to follow the initial parameter
    # distribution at initialization of the model       
    with torch.no_grad():
        output_scale = torch.std(param.data)  
 
    gnet_sizes = (num_inputs, hidden_dim, 1)
    gnets = [GenomicBottleNet(gnet_sizes, output_scale) 
            for _ in range(3*num_row_tiles*num_col_tiles)]       
    
    return row_col_encoding, gnets, tile_shape, output_scale


def square_qkv_gnet_layer(param: Tensor, 
                    hidden_dim: int, 
                    max_gnet_batch: int) -> GNetLayerTuple:
    # Subdivide the attention weight matrix in three similar parts Wq, Wk, Wv
    param_shape = param.data.shape
    _param_shape = (param_shape[0] // 3, param_shape[1])
    # num_row_tiles, num_col_tiles, row_tile_size, col_tile_size = get_tile_size(_param_shape, 
    #                                                                             max_gnet_batch)
    tile_size = np.ceil(np.sqrt(max_gnet_batch)).astype(np.int32)
    num_row_tiles = np.ceil(_param_shape[0] / tile_size).astype(np.int32)
    num_col_tiles = np.ceil(_param_shape[1] / tile_size).astype(np.int32)
    
    # Treat 2D weight as fully connected
    num_encoding_bits = np.ceil(np.log(_param_shape)/np.log(2)).astype(np.uint16)
    num_encoding_bits[np.where(num_encoding_bits == 0)] = 1
    encoding_type = (EncodingType.BINARY, 
                    EncodingType.BINARY)
    
    tile_shape = (tile_size, tile_size)
    row_col_encoding = make_row_col_encoding(tile_shape,
                                            encoding_type, 
                                            num_encoding_bits)
    num_inputs = row_col_encoding.shape[-1]
    
    # Normalizes the output to follow the initial parameter
    # distribution at initialization of the model       
    with torch.no_grad():
        output_scale = torch.std(param.data)    
 
    gnet_sizes = (num_inputs, hidden_dim, 1)
    gnets = [GenomicBottleNet(gnet_sizes, output_scale) 
            for _ in range(3*num_row_tiles*num_col_tiles)]     
    
    return row_col_encoding, gnets, tile_shape, output_scale


def onedim_gnet_layer(param: Tensor, 
                        hidden_dim: int, 
                        max_gnet_batch: int) -> GNetLayerTuple:
    param_shape = param.data.shape
    
    num_encoding_bits = np.ceil(np.log(param_shape)/np.log(2)).astype(np.uint16)
    num_encoding_bits[np.where(num_encoding_bits == 0)] = 1
    encoding_type = (EncodingType.BINARY, 
                    EncodingType.BINARY)
    
    row_col_encoding = make_row_col_encoding(param_shape,
                                            encoding_type, 
                                            num_encoding_bits)
    num_inputs = row_col_encoding.shape[-1]
    
    # Normalizes the output to follow the initial parameter
    # distribution at initialization of the model       
    with torch.no_grad():
        output_scale = torch.std(param.data)    

    gnet_sizes = (num_inputs, hidden_dim, 1)
    gnets = [GenomicBottleNet(gnet_sizes, output_scale)]     
    
    return row_col_encoding, gnets, param_shape, output_scale
    

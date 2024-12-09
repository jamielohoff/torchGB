from typing import Callable, Optional, Sequence, Tuple
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .utils import (make_row_col_encoding, 
                    make_random_row_col_encoding,
                    get_tile_size, 
                    EncodingType)


ceil = lambda x: np.ceil(x).astype(np.int32)


class GenomicBottleNet(nn.Module):
    """
    Improved version of the variable-length Gnet that uses a for-loop for 
    initialization.
    
    Args:
        `layers` (nn.ModuleList): ModuleList that contains all differentiable
            layers of the G-Net.
        `sizes` (Sequence[int]): List of sizes for the G-Net layers.
        `output_scale` (float): Scaling factor for the output of the G-Net.
        `activation_fn` (Optional[Callable[[Tensor], Tensor]]): Activation 
            function for the hidden layers. Default is ReLU.

    Returns:
        Tensor: Prediction of the new weight.
    """
    layers: nn.ModuleList
    sizes: Sequence[int]
    output_scale: float
    
    def __init__(self, 
                sizes: Sequence[int], 
                output_scale: float,
                activation_fn: Optional[Callable[[Tensor], Tensor]] = F.tanh) -> None:
        assert len(sizes) > 1, "List must have at least 3 entries!"
        super(GenomicBottleNet, self).__init__()
        self.output_scale = output_scale.detach()
        self.activation_fn = activation_fn
        self.sizes = sizes
        length = len(sizes) - 1
        self.layers = nn.ModuleList([nn.Linear(sizes[i], sizes[i+1], bias=True) 
                                    for i in range(length)])
        self.apply(self.init_weights)

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers[:-1]:
            x = self.activation_fn(layer(x))
        x = self.layers[-1](x) # no non-linearity on the last layer
        return x * self.output_scale
    
    def init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            # nn.init.normal_(module.weight, std=1.35*0.02, mean=0.0)
            if module.bias is not None:
                nn.init.zeros_(module.bias) # nn.init.normal_(module.bias, std=0.02) # 
                

GNetLayerTuple = Tuple[Tensor, Sequence[GenomicBottleNet], Sequence[int], float]


def square_conv2d_gnet_layer(param: Tensor, 
                            hidden_dim: int, 
                            gnet_batchsize: int) -> GNetLayerTuple:   
    num_encoding_bits = ceil(np.log(param.shape)/np.log(2))
    num_encoding_bits[:2] = param.shape[:2]
    num_encoding_bits[np.where(num_encoding_bits == 0)] = 1
    encoding_type = (EncodingType.BINARY, EncodingType.BINARY, 
                    EncodingType.ONEHOT, EncodingType.ONEHOT)   

    kernel_size = param.shape[2]*param.shape[3]
    tile_size = ceil(np.sqrt(gnet_batchsize//kernel_size))
    num_row_tiles = ceil(param.shape[0]/tile_size)
    num_col_tiles = ceil(param.shape[1]/tile_size)

    tile_shape = (tile_size, tile_size, param.shape[2], param.shape[3])  
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
                        gnet_batchsize: int) -> GNetLayerTuple:
    num_encoding_bits = ceil(np.log(param.shape)/np.log(2))
    num_encoding_bits[np.where(num_encoding_bits == 0)] = 1
    encoding_type = (EncodingType.BINARY, EncodingType.BINARY)
    
    row_col_encoding = make_random_row_col_encoding(param.shape,
                                            encoding_type, 
                                            num_encoding_bits)
    num_inputs = row_col_encoding.shape[-1]
    
    # Normalizes the output to follow the initial parameter
    # distribution at initialization of the model       
    with torch.no_grad():
        output_scale = torch.std(param.data)    

    gnet_sizes = (num_inputs, hidden_dim, 1)
    gnets = [GenomicBottleNet(gnet_sizes, output_scale)]     
    
    return row_col_encoding, gnets, param.shape, output_scale


def square_default_gnet_layer(param: Tensor, 
                            hidden_dim: int,  
                            gnet_batchsize: int) -> GNetLayerTuple:
    # Calculates the number of square tiles of size `gnet_batchsize` we need to
    # completely cover the weight matrix
    gnet_batchsize = 1
    tile_size = ceil(np.sqrt(gnet_batchsize))
    tile_shape = (tile_size, tile_size)
    num_row_tiles = ceil(param.shape[0]/tile_size)
    num_col_tiles = ceil(param.shape[1]/tile_size)
    
    num_encoding_bits = ceil(np.log(param.shape)/np.log(2))
    num_encoding_bits[np.where(num_encoding_bits == 0)] = 1
    encoding_type = (EncodingType.BINARY, EncodingType.BINARY)
    
    
    row_col_encoding = make_row_col_encoding(tile_shape,
                                            encoding_type, 
                                            num_encoding_bits)
    num_inputs = row_col_encoding.shape[-1]
    
    # Normalizes the output to follow the initial parameter
    # distribution at initialization of the model       
    with torch.no_grad():
        output_scale = torch.tensor([1.]).to(param.data.device) # torch.std(param.data)   

    gnet_sizes = (num_inputs, 1)
    gnets = [GenomicBottleNet(gnet_sizes, output_scale) 
            for _ in range(num_row_tiles*num_col_tiles)]     
    
    return row_col_encoding, gnets, tile_shape, output_scale


def qkv_gnet_layer(param: Tensor, 
                    hidden_dim: int, 
                    gnet_batchsize: int) -> GNetLayerTuple:
    # Subdivide the attention weight matrix in three similar parts Wq, Wk, Wv
    _param_shape = (param.shape[0], param.shape[1]) # //3
    
    # Treat 2D weight as fully connected
    num_encoding_bits = ceil(np.log(_param_shape)/np.log(2))
    num_encoding_bits[np.where(num_encoding_bits == 0)] = 1
    encoding_type = (EncodingType.BINARY, EncodingType.BINARY)
    
    row_col_encoding = make_random_row_col_encoding(_param_shape,
                                                    encoding_type, 
                                                    num_encoding_bits)
    num_inputs = row_col_encoding.shape[-1]
    
    # Normalizes the output to follow the initial parameter
    # distribution at initialization of the model       
    with torch.no_grad():
        output_scale = torch.std(param.data)  
 
    gnet_sizes = (num_inputs, hidden_dim, 1)
    gnets = [GenomicBottleNet(gnet_sizes, output_scale)] # add 3*     
    
    return row_col_encoding, gnets, _param_shape, output_scale


def square_qkv_gnet_layer(param: Tensor, 
                    hidden_dim: int, 
                    gnet_batchsize: int) -> GNetLayerTuple:
    # Subdivide the attention weight matrix in three similar parts Wq, Wk, Wv
    _param_shape = (param.shape[0], param.shape[1]) # TODO add //3 again?

    tile_size = ceil(np.sqrt(gnet_batchsize))
    num_row_tiles = ceil(_param_shape[0]/tile_size)
    num_col_tiles = ceil(_param_shape[1]/tile_size)
    
    # Treat 2D weight as fully connected
    num_encoding_bits = ceil(np.log(_param_shape)/np.log(2))
    num_encoding_bits[np.where(num_encoding_bits == 0)] = 1
    encoding_type = (EncodingType.BINARY, EncodingType.BINARY)
    
    tile_shape = (tile_size, tile_size)
    row_col_encoding = make_random_row_col_encoding(tile_shape,
                                            encoding_type, 
                                            num_encoding_bits)
    num_inputs = row_col_encoding.shape[-1]
    
    # Normalizes the output to follow the initial parameter
    # distribution at initialization of the model       
    with torch.no_grad():
        output_scale = torch.std(param.data)    
 
    gnet_sizes = (num_inputs, hidden_dim, 1)
    gnets = [GenomicBottleNet(gnet_sizes, output_scale) 
            for _ in range(num_row_tiles*num_col_tiles)]     # add 3*
    
    return row_col_encoding, gnets, tile_shape, output_scale


def onedim_gnet_layer(param: Tensor, 
                        hidden_dim: int, 
                        gnet_batchsize: int) -> GNetLayerTuple:
    num_encoding_bits = ceil(np.log(param.shape)/np.log(2))
    num_encoding_bits[np.where(num_encoding_bits == 0)] = 1
    encoding_type = (EncodingType.BINARY,)
    
    row_col_encoding = make_random_row_col_encoding(param.shape,
                                            encoding_type, 
                                            num_encoding_bits)
    num_inputs = row_col_encoding.shape[-1]
    
    # Normalizes the output to follow the initial parameter
    # distribution at initialization of the model       
    with torch.no_grad():
        output_scale = torch.std(param.data)    

    gnet_sizes = (num_inputs, hidden_dim, 1)
    gnets = [GenomicBottleNet(gnet_sizes, output_scale)]     
    
    return row_col_encoding, gnets, param.shape, output_scale
    

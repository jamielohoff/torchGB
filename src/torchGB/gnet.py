from typing import Callable, Optional, Sequence, Tuple
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .utils import (make_row_col_encoding,
                    get_tile_size, 
                    EncodingType)


ceil = lambda x: np.ceil(x).astype(np.int32)


class GenomicBottleNet(nn.Module):
    """
    Improved version of the variable-length g-net that uses a `for`-loop for 
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
        self.batchnorm = nn.BatchNorm1d(sizes[-2])

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers[:-1]:
            x = self.activation_fn(layer(x))
        x = self.batchnorm(x)
        x = self.layers[-1](x) # no non-linearity on the last layer
        # NOTE: Do not touch output norm! Carfully computed by hand...
        return x * torch.tensor(2.) * self.output_scale
                

GNetLayerTuple = Tuple[Tensor, Sequence[GenomicBottleNet], Sequence[int], float]


def square_conv2d_gnet_layer(param: Tensor, 
                            hidden_dim: int, 
                            gnet_batchsize: int) -> GNetLayerTuple:   
    """_summary_

    Args:
        param (Tensor): _description_
        hidden_dim (int): _description_
        gnet_batchsize (int): _description_

    Returns:
        GNetLayerTuple: _description_
    """
    kernel_size = param.shape[2]*param.shape[3]

    tile_size = ceil(np.sqrt(gnet_batchsize//kernel_size))
    num_row_tiles = ceil(param.shape[0]/tile_size)
    num_col_tiles = ceil(param.shape[1]/tile_size)

    tile_shape = (tile_size, tile_size, param.shape[2], param.shape[3])
    
    num_encoding_bits = ceil(np.log(tile_shape)/np.log(2))
    num_encoding_bits[2:] = tile_shape[2:]
    num_encoding_bits[np.where(num_encoding_bits == 0)] = 1
    encoding_type = (EncodingType.BINARY, EncodingType.BINARY, 
                    EncodingType.ONEHOT, EncodingType.ONEHOT)   

    
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
                            gnet_batchsize: int) -> GNetLayerTuple:
    """
    Calculates the number of square tiles of size `gnet_batchsize` we need to
    completely cover the weight matrix. This function is usually used to initialize
    dense layers of multi-layer perceptrons. It automatically subdivides the weight
    matrix into square tiles the contain at most `gnet_batchsize` weights. 
    It also automatically computes the encoding of every single position of the 
    a weight in the square tiles. Since the size of all tiles is the same, we 
    reuse this encoding.

    Args:
        param (Tensor): Parameter, i.e. weight matrix that we wish to compress/
            predict using a genomic network.
        hidden_dim (int): Size of the hidden dimension of the g-net(s) that predict
            the weight matrix.
        gnet_batchsize (int): Maximum size of a single square tile.

    Returns:
        GNetLayerTuple: A tuple that contains the encoding of the weight positions,
            the genomic networks, the shape of a single tile and the scale of the outputs.
    """
    tile_size = ceil(np.sqrt(gnet_batchsize))
    tile_shape = (tile_size, tile_size)
    num_row_tiles = ceil(param.shape[0]/tile_size)
    num_col_tiles = ceil(param.shape[1]/tile_size)
    
    num_encoding_bits = ceil(np.log([tile_size, tile_size])/np.log(2))
    num_encoding_bits[np.where(num_encoding_bits == 0)] = 1
    encoding_type = (EncodingType.BINARY, EncodingType.BINARY)
    
    
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


def square_qkv_gnet_layer(param: Tensor, 
                    hidden_dim: int, 
                    gnet_batchsize: int) -> GNetLayerTuple:
    """_summary_
    TODO: reimplement this with the separation of the q,k, and v part of the matrix
    because otherwise we might get weird correlations...

    Args:
        param (Tensor): _description_
        hidden_dim (int): _description_
        gnet_batchsize (int): _description_

    Returns:
        GNetLayerTuple: _description_
    """
    # Subdivide the attention weight matrix in three similar parts Wq, Wk, Wv
    tile_size = ceil(np.sqrt(gnet_batchsize))
    tile_shape = (tile_size, tile_size)
    num_row_tiles = ceil(param.shape[0]/tile_size//3)
    num_col_tiles = ceil(param.shape[1]/tile_size)
    
    # Treat 2D weight as fully connected
    num_encoding_bits = ceil(np.log([tile_size, tile_size])/np.log(2))
    num_encoding_bits[np.where(num_encoding_bits == 0)] = 1
    encoding_type = (EncodingType.BINARY, EncodingType.BINARY)
    
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
            for _ in range(3*num_row_tiles*num_col_tiles)]     # add 3*
    
    return row_col_encoding, gnets, tile_shape, output_scale

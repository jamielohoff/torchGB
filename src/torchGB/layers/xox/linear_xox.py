from typing import Sequence

import numpy as np

import torch
from torch import Tensor

from .model import XOXLayer
from ...utils import ceil, crop_matrix, build_matrix

# TODO: Fix documentation

def linear_xox_layer(param: Tensor, hidden_dim: int, gnet_batchsize: int):
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
            predict using a g-net.
        hidden_dim (int): Size of the hidden dimension of the g-nets that predict
            the weight matrix.
        gnet_batchsize (int): Maximum size of a single square tile.

    Returns:
        GNetLayerTuple: A tuple that contains the encoding of the weight positions,
            the g-net, the shape of a single tile and the scale of the outputs.
    """
    tile_size = ceil(np.sqrt(gnet_batchsize))
    tile_shape = (tile_size, tile_size)
    num_row_tiles = ceil(param.shape[0]/tile_size)
    num_col_tiles = ceil(param.shape[1]/tile_size)

    row_col_encoding = torch.zeros((gnet_batchsize, 1))
    
    # Normalizes the output to follow the initial parameter
    # distribution at initialization of the model       
    with torch.no_grad():
        output_scale = torch.std(param.data)   

    gnet_sizes = tile_shape
    gnets = [XOXLayer(*gnet_sizes, num_genes=hidden_dim) 
             for _ in range(num_row_tiles*num_col_tiles)]     
    
    return row_col_encoding, gnets, tile_shape, output_scale


def init_linear_xox(pname: str, param: Tensor, hidden_dim: int,
                    gnet_batchsize: int):
    """
    Initializes a GenomicBottleNet (g-net) for a linear layer.

    Args:
        pname (str): Name of the parameter. If "weight" is in the name,
            this function will initialize the linear g-net.
        param (Tensor): Parameter, i.e. weight matrix that we wish to compress/
            predict using a g-net.
        hidden_dim (int): Size of the hidden dimension of the g-nets that predict
            the weight matrix.
        gnet_batchsize (int): Maximum size of a single square tile.

    Returns:
        GNetLayerTuple: A tuple that contains the encoding of the weight positions,
            the g-net, the shape of a single tile and the scale of the outputs.
    """
    if "weight" in pname:
        return linear_xox_layer(param, hidden_dim, gnet_batchsize)
    else:
        return None


def build_linear_xox_output(name: str, param: Tensor, weights: Tensor, 
                            tile_shape: Sequence[int]) -> Tensor:
    """
    Builds the output of a linear layer using a GenomicBottleNet (g-net).

    Args:
        name (str): Name of the parameter.
        param (Tensor): Parameter, i.e. weight matrix that we wish to compress/
            predict using a g-net.
        weights (Tensor): Weights used in the computation.
        tile_shape: Shape of each tile.

    Returns:
        Tensor: Output of the linear layer computed using the g-net.
    """
    num_row_tiles = ceil(param.shape[0]/tile_shape[0])
    num_col_tiles = ceil(param.shape[1]/tile_shape[1])

    shape = (num_row_tiles*tile_shape[0], num_col_tiles*tile_shape[1])

    new_weights = build_matrix(weights, shape)
    new_weights = crop_matrix(new_weights, param.shape)
    return new_weights


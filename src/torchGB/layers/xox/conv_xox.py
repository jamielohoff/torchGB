from typing import Sequence

import numpy as np

import torch
from torch import Tensor

from .model import XOXLayer
from ...utils import ceil, cut_matrix, build_4d_kernel

# TODO: write docstrings
# TODO: fix scalings and input sizes

def conv2d_xox_layer(param: Tensor, hidden_dim: int, gnet_batchsize: int):
    """
    Creates a GenomicBottleNet (g-net) layer for 2D convolution.

    Args:
        param (Tensor): The kernel of the convolution.
        hidden_dim (int): The number of hidden units in each g-net.
        gnet_batchsize (int): The batch size used in the g-net.

    Returns:
        GnetLayerTuple: A tuple containing the row-col encoding, a list of g-net 
        instances, and the tile shape.
    """
    kernel_size = param.shape[2]*param.shape[3]

    # Calculate the number of tiles for efficient computation
    tile_size = ceil(np.sqrt(gnet_batchsize//kernel_size))
    num_row_tiles = ceil(param.shape[0]/tile_size)
    num_col_tiles = ceil(param.shape[1]/tile_size)

    # Define the tile shape and encoding type
    tile_shape = (tile_size, tile_size, param.shape[2], param.shape[3])

    # Create the row-col encoding
    row_col_encoding = torch.zeros((gnet_batchsize, 1))

    # Normalize the output to follow the initial parameter distribution at initialization of the model
    with torch.no_grad():
        output_scale = torch.std(param.data)

    gnet_sizes = (tile_size*param.shape[2], tile_size*param.shape[3])
    gnets = [XOXLayer(*gnet_sizes, num_genes=hidden_dim)
            for _ in range(num_row_tiles*num_col_tiles)]
    return row_col_encoding, gnets, tile_shape, output_scale


def init_conv2d_xox(pname: str, param: Tensor, hidden_dim: int,
                    gnet_batchsize: int):
    """
    Initializes a GenomicBottleNet (g-net) for a 2D convolutional operation.

    Args:
        pname (str): The name of the parameter.
        param (Tensor): The kernel of the convolution.
        hidden_dim (int): The number of hidden units in each g-net.
        gnet_batchsize (int): The batch size used in the g-net.

    Returns:
        tuple: A tuple containing the row-col encoding, a list of g-net instances,
            and the tile shape. If pname is 'weight', it returns the result
            of conv2d_gnet_layer. Otherwise, it returns None.
    """
    if "weight" in pname:
        return conv2d_xox_layer(param, hidden_dim, gnet_batchsize)

    else:
        return None


def build_conv2d_xox_output(name: str, param: Tensor, weights: Tensor,
                             tile_shape: Sequence[int]) -> Tensor:
    """
    Builds the output of a 2D convolutional operation using a GenomicBottleNet (g-net).

    Args:
        name (str): The name of the parameter.
        param (Tensor): The kernel of the convolution.
        weights (Tensor): The weights of the convolution.
        tile_shape (Sequence[int]): The shape of each tile in the convolution.

    Returns:
        Tensor: The output of the 2D convolutional operation using g-net.
    """
    num_row_tiles = ceil(param.shape[0]/tile_shape[0])
    num_col_tiles = ceil(param.shape[1]/tile_shape[1])

    # Define the shape of the output
    shape = (num_row_tiles*tile_shape[0], 
             num_col_tiles*tile_shape[1],
             param.shape[2], param.shape[3])

    # Build the 4D kernel and cut it to match the input shape
    new_weights = build_4d_kernel(weights, shape)
    new_weights = cut_matrix(new_weights, param.shape)
    return new_weights


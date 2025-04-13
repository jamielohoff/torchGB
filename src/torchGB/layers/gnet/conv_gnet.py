from typing import Sequence

import numpy as np

import torch
from torch import Tensor

from .model import GenomicBottleNet, GNetLayerTuple
from ...utils import EncodingType, make_row_col_encoding, ceil, crop_matrix, build_4d_kernel


def conv2d_gnet_layer(model: GenomicBottleNet, param: Tensor, hidden_dim: int, 
                      gnet_batchsize: int) -> GNetLayerTuple:
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

    # Calculate the number of bits needed for encoding
    num_encoding_bits = ceil(np.log(tile_shape)/np.log(2))
    num_encoding_bits[2:] = tile_shape[2:]
    num_encoding_bits[np.where(num_encoding_bits == 0)] = 1

    # Define the encoding type (binary, one-hot)
    encoding_type = (EncodingType.BINARY, EncodingType.BINARY,
                     EncodingType.ONEHOT, EncodingType.ONEHOT)

    # Create the row-col encoding
    row_col_encoding = make_row_col_encoding(tile_shape, encoding_type,
                                             num_encoding_bits)
    num_inputs = row_col_encoding.shape[-1]

    # Normalize the output to follow the initial parameter distribution at initialization of the model
    with torch.no_grad():
        output_scale = torch.std(param.data)

    output_size = 1 # if isinstance(model, GenomicBottleNet) else 2
    gnet_sizes = (num_inputs, hidden_dim, output_size)
    gnets = [model(gnet_sizes, output_scale)
            for _ in range(num_row_tiles*num_col_tiles)]
    return row_col_encoding, gnets, tile_shape, output_scale


def init_conv2d_gnet(model: GenomicBottleNet, pname: str, param: Tensor, 
                     hidden_dim: int, gnet_batchsize: int) -> GNetLayerTuple:
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
        return conv2d_gnet_layer(model, param, hidden_dim, gnet_batchsize)

    else:
        return None


def build_conv2d_gnet_output(name: str, param: Tensor, weights: Tensor,
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
    shape = (num_row_tiles*tile_shape[0], num_col_tiles*tile_shape[1],
             param.shape[2], param.shape[3])

    # Build the 4D kernel and crop it to match the input shape
    new_weights = build_4d_kernel(weights, shape)
    new_weights = crop_matrix(new_weights, param.shape)
    return new_weights

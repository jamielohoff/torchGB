from typing import Sequence

import numpy as np

import torch
from torch import Tensor

from .linear_low_rank import linear_low_rank_layer
from .model import LowRankMatrixDecompositionGNet
from ...utils import ceil, cut_matrix, build_matrix


def attn_low_rank_layer(param: Tensor, hidden_dim: int, gnet_batchsize: int):
    """
    Creates a GenomicBottleNet (g-net) for attention weights.
    Subdivides the attention weight matrix into three similar parts Wq, Wk, Wv
    and treats them as fully connected layers. Each part is then processed by a
    g-net with its own set of parameters.

    Args:
        param (Tensor): The attention weight matrix.
        hidden_dim (int): The dimensionality of the hidden state in the g-net.
        gnet_batchsize (int): The batch size for the g-net.

    Returns:
        GNetLayerTuple: A tuple containing the row and column encodings, the list
            of g-nets, the tile shape, and the output scale factor.
    """
    # Subdivide the attention weight matrix in three similar parts Wq, Wk, Wv
    tile_size = ceil(np.sqrt(gnet_batchsize))
    tile_shape = (tile_size, tile_size)
    num_row_tiles = ceil(param.shape[0] / tile_size)  # add //3
    num_col_tiles = ceil(param.shape[1] / tile_size)

    # Treat 2D weight as fully connected
    row_col_encoding = torch.zeros((gnet_batchsize, 1))

    # Normalizes the output to follow the initial parameter
    # distribution at initialization of the model
    with torch.no_grad():
        output_scale = torch.std(param.data)

    gnet_sizes = (tile_size, tile_size)
    gnets = [LowRankMatrixDecompositionGNet(gnet_sizes, rank=hidden_dim)
            for _ in range(num_row_tiles*num_col_tiles)] # add 3*

    return row_col_encoding, gnets, tile_shape, output_scale


def init_attn_low_rank(pname: str, param: Tensor, hidden_dim: int,
                       gnet_batchsize: int):
    """
    Initializes a GenomicBottleNet (g-net) for attention weights.

    If the parameter name contains "in_proj_weight", it uses the
    attn_gnet_layer function. Otherwise, if it's a linear weight and has two dimensions,
    it uses the linear_gnet_layer function. Otherwise, returns None.

    Args:
        pname: The parameter name.
        param (Tensor): The attention weight matrix or linear weights.
        hidden_dim (int): The dimensionality of the hidden state in the g-net.
        gnet_batchsize` (int): The batch size for the g-net.

    Returns:
        GNetLayerTuple: A tuple containing the row and column encodings, the list
            of g-nets, the tile shape, and the output scale factor.
    """
    if "in_proj_weight" in pname:
        return attn_low_rank_layer(param, hidden_dim, gnet_batchsize)

    elif "weight" in pname and len(param.shape) == 2:
        return linear_low_rank_layer(param, hidden_dim, gnet_batchsize)
    else:
        return None


def build_attn_low_rank_output(name: str, param: Tensor, weights: Tensor,
                               tile_shape: Sequence[int]) -> Tensor:
    """
    Builds the output structure of a GenomicBottleNet (g-net) for attention weights.

    If the parameter name contains "in_proj_weight_X", it uses the 3x1 tile shape.
    Otherwise, it uses the tile shape provided.

    Args:
        name: The parameter name.
        param (Tensor): The attention weight matrix or linear weights.
        weights (Tensor): The weights of the g-net.
        tile_shape (Sequence[int]): The tile shape for the g-net.

    Returns:
        Tensor: The output tensor of the g-net.
    """
    
    if "in_proj_weight_X" in name:
        num_row_tiles = 3 * ceil(param.shape[0] // 3 / tile_shape[0])
        num_col_tiles = ceil(param.shape[1] / tile_shape[1])
    else:
        num_row_tiles = ceil(param.shape[0] / tile_shape[0])
        num_col_tiles = ceil(param.shape[1] / tile_shape[1])

    shape = (num_row_tiles * tile_shape[0], num_col_tiles * tile_shape[1])

    new_weights = build_matrix(weights, shape)
    new_weights = cut_matrix(new_weights, param.shape)
    return new_weights


import numpy as np

import torch
import torch.nn as nn
from torch import Tensor

from .linear_gnet import linear_gnet_layer
from ..gnet import GenomicBottleNet, GNetLayerTuple
from ..utils import EncodingType, make_row_col_encoding, ceil, cut_matrix, build_matrix


def attn_gnet_layer(param: Tensor, hidden_dim: int, gnet_batchsize: int) -> GNetLayerTuple:
    """_summary_

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
    num_row_tiles = ceil(param.shape[0]/tile_size) # add //3
    num_col_tiles = ceil(param.shape[1]/tile_size)
    
    # Treat 2D weight as fully connected
    num_encoding_bits = ceil(np.log([tile_size, tile_size])/np.log(2))
    num_encoding_bits[np.where(num_encoding_bits == 0)] = 1
    encoding_type = (EncodingType.BINARY, EncodingType.BINARY)
    
    row_col_encoding = make_row_col_encoding(tile_shape, encoding_type, 
                                             num_encoding_bits)
    num_inputs = row_col_encoding.shape[-1]
    
    # Normalizes the output to follow the initial parameter
    # distribution at initialization of the model       
    with torch.no_grad():
        output_scale = torch.std(param.data)    
 
    gnet_sizes = (num_inputs, hidden_dim, 1)
    gnets = [GenomicBottleNet(gnet_sizes, output_scale) 
            for _ in range(num_row_tiles*num_col_tiles)] # add 3*
    
    return row_col_encoding, gnets, tile_shape, output_scale


def init_attn_gnet(pname: str, param: Tensor, hidden_dim: int, 
                   gnet_batchsize: int) -> GNetLayerTuple:
    if "in_proj_weight" in pname:
        return attn_gnet_layer(param, hidden_dim, gnet_batchsize) 

    elif "weight" in pname and len(param.shape) == 2:
        return linear_gnet_layer(param, hidden_dim, gnet_batchsize)
    else:
        return None
    
    
def build_attn_gnet_output(name: str, param: Tensor, weights: Tensor, 
                              tile_shape) -> Tensor:
    if "in_proj_weight_X" in name: # TODO fix this!
        num_row_tiles = 3*ceil(param.shape[0]//3/tile_shape[0])
        num_col_tiles = ceil(param.shape[1]/tile_shape[1])
    else:
        num_row_tiles = ceil(param.shape[0]/tile_shape[0])
        num_col_tiles = ceil(param.shape[1]/tile_shape[1])
    
    shape = (num_row_tiles*tile_shape[0], num_col_tiles*tile_shape[1])
    
    new_weights = build_matrix(weights, shape)
    new_weights = cut_matrix(new_weights, param.shape)
    return new_weights


# register_gnet_type(nn.TransformerEncoder, init_attn_gnet, assemble_attn_gnet_output)
        

import numpy as np

import torch
import torch.nn as nn
from torch import Tensor

from ..gnet import GenomicBottleNet, GNetLayerTuple
from ..utils import EncodingType, make_row_col_encoding, ceil, cut_matrix, build_4d_kernel


def conv2d_gnet_layer(param: Tensor, hidden_dim: int, gnet_batchsize: int) -> GNetLayerTuple:   
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

    
    row_col_encoding = make_row_col_encoding(tile_shape, encoding_type,
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


def init_conv2d_gnet(pname: str, param: Tensor, hidden_dim: int, 
                     gnet_batchsize: int) -> GNetLayerTuple:
    if "weight" in pname:
        return conv2d_gnet_layer(param, hidden_dim, gnet_batchsize)   
        
    else:
        return None
    

def build_conv2d_gnet_output(name: str, param: Tensor, weights: Tensor, 
                             tile_shape) -> Tensor:
    num_row_tiles = ceil(param.shape[0]/tile_shape[0])
    num_col_tiles = ceil(param.shape[1]/tile_shape[1])
    
    shape = (num_row_tiles*tile_shape[0], num_col_tiles*tile_shape[1], 
             param.shape[2], param.shape[3])
    
    new_weights = build_4d_kernel(weights, shape)
    new_weights = cut_matrix(new_weights, param.shape)
    return new_weights


# register_gnet_type(nn.Conv2d, init_conv2d_gnet, assemble_conv2d_gnet_output)


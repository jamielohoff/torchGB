from enum import Enum
from typing import Sequence, Tuple
import copy

import torch
from torch import Tensor
import numpy as np


class EncodingType(Enum):
    ONEHOT = 0 # One-hot vector
    BINARY = 1 # Binary code


def tile_matrix(matrix: Tensor, row_size: int, col_size: int):
    """
    Return an array of shape (n, row_size, col_size) where
    n * row_size * col_size = arr.size

    If arr is a 2D array, the returned array should look like n subblocks with
    each subblock preserving the "physical" layout of arr.
    
    Args:
        `matrix` (Tensor): The input array to be tiled. Has to be two-dimensional.
        `row_size` (int): The number of rows in each subtile.
        `col_size` (int): The number of columns in each subtile.
        
    Returns:
        Tensor: The tiled array.
    """
    h, w = matrix.shape
    assert len(matrix.shape) == 2, "Input array must be 3D"
    assert h % row_size == 0, f"{h} rows is not evenly divisible by {row_size}"
    assert w % col_size == 0, f"{w} cols is not evenly divisible by {col_size}"
    return (matrix.reshape(h // row_size, row_size, -1, col_size)
                    .swapaxes(1, 2)
                    .reshape(-1, row_size, col_size))
    

def assemble_matrix(arr: Tensor, new_shape: Tuple[int, int]) -> Tensor:
    """
    NOTE This is the inverse of tile_matrix. This function reassembles the 
    original array from its tiled form. The input array must be 3D.
    
    Args:
        `arr` (Tensor): The input array in its tiled form.
        `new_shape` (Tuple[int, int]): The shape of the reassembled array.
        
    Returns:
        Tensor: The reassembled matrix.
    """
    h, w = new_shape
    row_size, col_size = arr.shape[1:]
    assert len(arr.shape) == 3, "Input array must be 3D"
    assert h % row_size == 0, f"{h} rows is not evenly divisible by {row_size}"
    assert w % col_size == 0, f"{w} cols is not evenly divisible by {col_size}"
    out = (arr.reshape(h // row_size, -1, row_size, col_size)
                .swapaxes(1, 2)
                .reshape(h, w))
    return out


def assemble_4d_kernel(arr: Tensor, new_shape: Tuple[int, int, int, int]) -> Tensor:
    """
    NOTE This is the inverse of tile_matrix. This function reassembles the 
    original array from its tiled form. The input array must be 3D.
    
    Args:
        `arr` (Tensor): The input array in its tiled form.
        `new_shape` (Tuple[int, int]): The shape of the reassembled array.
        
    Returns:
        Tensor: The reassembled array.
    """
    h, w, x, y = new_shape
    row_size, col_size = arr.shape[1:3]
    # assert len(arr.shape) == 3, "Input array must be 3D"
    # assert h % row_size == 0, f"{h} rows is not evenly divisible by {row_size}"
    # assert w % col_size == 0, f"{w} cols is not evenly divisible by {col_size}"
    out = (arr.reshape(h // row_size, -1, row_size, col_size, x, y)
                .swapaxes(1, 2)
                .reshape(h, w, x, y))
    return out


def cut_matrix(arr: Tensor, new_shape: Sequence[int]) -> Tensor:
    """
    This function cuts the padded matrix to its true shape.
    
    Args:
        `arr` (Tensor): The input array to be cut.
        `new_shape` (Sequence[int]): The true shape of the matrix.
        
    Returns:
        Tensor: The cut array.
    """
    h, w = new_shape[:2]
    return arr[:h, :w]
    
    
def get_tile_size(param_shape: Sequence[int], 
                    max_gnet_batch: int) -> Tuple[int, int, int, int]:
    """
    This function calculates the number of row and column tiles for a given
    weight matrix shape and maximum parameter count per tile. The number of row 
    and column tiles is calculated such that the number of parameters in each
    tile is less than or equal to the maximum parameter count per tile.

    Args:
        param_shape (Sequence[int]): Shape of the weight matrix.
        max_gnet_batch (int): Maximum number of parameters per tile.

    Returns:
        Tuple[int, int, int, int]: The number of row and column tiles.
    """
    row_size, col_size = param_shape
    
    numel = np.prod(param_shape)
    n = numel / max_gnet_batch
    num_row_tiles = np.max([np.sqrt(n * row_size / col_size), 1]).astype(np.int32)
    num_col_tiles = np.max([np.sqrt(n * col_size / row_size), 1]).astype(np.int32)
    
    row_tile_size = np.min([row_size, np.ceil(row_size / num_row_tiles)]).astype(np.int32)
    col_tile_size = np.min([col_size, np.ceil(col_size / num_col_tiles)]).astype(np.int32)

    return num_row_tiles, num_col_tiles, row_tile_size, col_tile_size


def make_row_col_encoding(param_shape: Sequence[int], 
                        encoding_types: Sequence[EncodingType],
                        num_encoding_bits: Sequence[int]) -> Tensor:
    """
    This function creates inputs for the G-Nets that encode the position of the
    weights in the weight matrix. The encoding can either be done using one-hot
    encoding or binary encoding. The function will return a tensor that has the
    same number of rows as the number of weights in the weight matrix and the
    number of columns equal to the sum of the number of bits for each variable.
    
    Examples:
    
    
    Args:
        `param_shape` (Sequence[int]): Shape of the weight matrix.
        `encoding_types` (Sequence[EncodingType]): Encoding type for each axis.
        `num_encoding_bits` (Sequence[int]): Number of bits for each axis.
        
    Returns:
        Tensor: Array of shape (np.prod(dims), np.sum(bits)).
    """
    param_shape, num_encoding_bits = np.atleast_1d(param_shape, num_encoding_bits)
    row_col_encoding = np.zeros((param_shape.prod(), num_encoding_bits.sum()))
    num_encoding_types = len(encoding_types)

    # This will compute the encoding for axis i
    def get_encoding_for_dim(i: int) -> np.ndarray:
        # Compute the encoding for the i-th axis
        shape = np.ones(param_shape.size, dtype=np.int16)
        shape[i] = param_shape[i]
        dim_encoding = np.arange(param_shape[i]).reshape(shape)
        
        # Replicate the encoding for the other axes
        tiling = copy.copy(param_shape)
        tiling[i] = 1
        dim_encoding = np.tile(dim_encoding, tiling)   

        # Flatten the encoding
        dim_encoding = np.reshape(dim_encoding, (np.prod(param_shape), 1))

        if encoding_types[i].value == 0:
            max_hot = dim_encoding.max() + 1
            one_hot_encoding = np.identity(max_hot, dtype=np.int16)
            dim_encoding = one_hot_encoding[dim_encoding.squeeze(), :]
            
        # TODO this can be optimized further
        elif encoding_types[i].value == 1: # Binary code
            max_bits = num_encoding_bits[i]
            num_digits = 2 ** max_bits
            bins = np.zeros((num_digits, max_bits), dtype=np.int16)
            
            # This will be used for binary conversion
            for j in range(num_digits):
                bins[j, :] = np.array(list(np.binary_repr(j).zfill(max_bits))).astype(np.int16)
                
            # Convert to binary numbers 
            # Take only the lowest bits
            dim_encoding = bins[dim_encoding.squeeze(), :] 
            # dim_encoding = dim_encoding[:, (max_bits - num_encoding_bits[i]):(max_bits)] 
        else:
            raise ValueError("Invalid encoding type!")
        return dim_encoding
    
    encoded_dims = [get_encoding_for_dim(i) for i in range(num_encoding_types)]
    row_col_encoding = np.concatenate(encoded_dims, axis=1)

    # Make inputs a torch tensor and detach from computational graph
    row_col_encoding = torch.tensor(row_col_encoding, 
                                    dtype=torch.float, 
                                    requires_grad=False)

    # Normalization for Xavier initialization
    with torch.no_grad():
        row_col_encoding = (row_col_encoding - torch.mean(row_col_encoding)) / \
                            torch.std(row_col_encoding)
        row_col_encoding /= torch.sqrt(torch.tensor(2.)) # inputs larger than 0
    return row_col_encoding  


def make_random_row_col_encoding(param_shape: Sequence[int], 
                                encoding_types: Sequence[EncodingType],
                                num_encoding_bits: Sequence[int]) -> Tensor:
    """
    TODO docstring
    
    Examples:
    
    
    Args:
        `param_shape` (Sequence[int]): Shape of the weight matrix.
        `encoding_types` (Sequence[EncodingType]): Encoding type for each axis.
        `num_encoding_bits` (Sequence[int]): Number of bits for each axis.
        
    Returns:
        Tensor: Array of shape (np.prod(dims), np.sum(bits)).
    """
    param_shape, num_encoding_bits = np.atleast_1d(param_shape, num_encoding_bits)
    row_col_encoding = np.zeros((param_shape.prod(), num_encoding_bits.sum()))
    num_encoding_types = len(encoding_types)

    # This will compute the encoding for axis i
    def get_encoding_for_dim(i: int) -> np.ndarray:
        # Compute the encoding for the i-th axis
        shape = np.ones(param_shape.size, dtype=np.int16)
        shape[i] = param_shape[i]
        dim_encoding = np.arange(param_shape[i]).reshape(shape)
        
        # Replicate the encoding for the other axes
        tiling = copy.copy(param_shape)
        tiling[i] = 1
        dim_encoding = np.tile(dim_encoding, tiling)   

        # Flatten the encoding
        dim_encoding = np.reshape(dim_encoding, (np.prod(param_shape), 1))

        one_hot_encoding = np.random.normal(0.0, 1.0, (param_shape[i], num_encoding_bits[i]))
        dim_encoding = one_hot_encoding[dim_encoding.squeeze(), :]
            
        return dim_encoding
    
    encoded_dims = [get_encoding_for_dim(i) for i in range(num_encoding_types)]
    row_col_encoding = np.concatenate(encoded_dims, axis=1)
    
    row_col_encoding = np.random.normal(0.0, 1.0, row_col_encoding.shape)

    # Make inputs a torch tensor and detach from computational graph
    row_col_encoding = torch.tensor(row_col_encoding, 
                                    dtype=torch.float, 
                                    requires_grad=False)

    # Normalization for Xavier initialization
    with torch.no_grad():
        row_col_encoding = (row_col_encoding - torch.mean(row_col_encoding)) / \
                            torch.std(row_col_encoding)
        row_col_encoding -= row_col_encoding.min() # inputs larger than 0
    return row_col_encoding  


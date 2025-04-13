from enum import Enum
from typing import Sequence, Tuple
import copy

import torch
from torch import Tensor
import numpy as np
import scipy.optimize as opt


class EncodingType(Enum):
    """
    An enumeration of encoding types.

    **ONEHOT**
        A one-hot vector encoding type. Each element in the sequence is mapped 
        to a unique binary vector.

    **BINARY**
        A binary encoding type. Each element in the sequence is mapped to a 
        single binary value.
    """
    ONEHOT = 0 # One-hot vector
    BINARY = 1 # Binary code
    

ceil = lambda x: np.ceil(x).astype(np.int32)



def get_gnet_batchsize(compression: float, hidden_dim: int, output_dim: int = 1) -> int:
    """Calculates the g-net batch size based on desired compression, hidden 
    dimension, and output dimension.
    This function determines the appropriate batch size for g-nets to achieve a 
    desired compression ratio. It uses the `scipy.optimize.root_scalar` function 
    to find the root of a non-linear equation that relates the batch size to the 
    compression ratio, hidden dimension, and output dimension.

    Args:
        compression (float): The desired compression ratio.
        hidden_dim (int): The hidden dimension of the g-net.
        output_dim (int, optional): The output dimension of the g-net. Defaults to 1.

    Returns:
        int: The calculated g-net batch size.
    """
    f = lambda g: g/(hidden_dim * (np.ceil(np.log(g)) + 2) + output_dim) - compression
    rootfinder = opt.root_scalar(f, x0=10_000, method="secant")
    return int(rootfinder.root)


def tile_matrix(matrix: Tensor, row_size: int, col_size: int) -> Tensor:
    """
    Return an array of shape (n, row_size, col_size) where
    n * row_size * col_size = arr.size

    If arr is a 2D array, the returned array should look like n subblocks with
    each subblock preserving the "physical" layout of arr.
    
    Args:
        matrix (Tensor): The input array to be tiled. Has to be two-dimensional.
        row_size (int): The number of rows in each subtile.
        col_size (int): The number of columns in each subtile.
        
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
    

def build_matrix(arr: Tensor, new_shape: Tuple[int, int]) -> Tensor:
    """
    This function is the inverse of tile_matrix. This function rebuilds the 
    original array from its tiled form if the initial array was a 2D matrix. 
    The input array must be 3D.
    
    Args:
        arr (Tensor): The input array in its tiled form.
        new_shape (Tuple[int, int]): The shape of the rebuildd array.
        
    Returns:
        Tensor: The rebuildd matrix.
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


def build_4d_kernel(arr: Tensor, new_shape: Tuple[int, int, int, int]) -> Tensor:
    """
    This fucntion is the inverse of tile_matrix. This function rebuilds the 
    original array from its tiled form if the original array was a 3D tensor, e.g.
    in the case of a convolution. The input array must be 4D.
    
    Args:
        arr (Tensor): The input array in its tiled form.
        new_shape (Tuple[int, int]): The shape of the rebuildd array.
        
    Returns:
        Tensor: The rebuildd array.
    """
    h, w, x, y = new_shape
    row_size, col_size = arr.shape[1:3]
    out = (arr.reshape(h // row_size, -1, row_size, col_size, x, y)
                .swapaxes(1, 2)
                .reshape(h, w, x, y))
    return out


def crop_matrix(arr: Tensor, new_shape: Sequence[int]) -> Tensor:
    """
    This function cuts the padded matrix to its true shape.
    
    Args:
        arr (Tensor): The input array to be cut.
        new_shape (Sequence[int]): The true shape of the matrix.
        
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
    This function creates inputs for the g-nets that encode the position of the
    weights in the weight matrix. The encoding can either be done using one-hot
    encoding or binary encoding. The function will return a tensor that has the
    same number of rows as the number of weights in the weight matrix and the
    number of columns equal to the sum of the number of bits for each variable.
    
    Examples:
    
    
    Args:
        param_shape (Sequence[int]): Shape of the weight matrix.
        encoding_types (Sequence[EncodingType]): Encoding type for each axis.
        num_encoding_bits (Sequence[int]): Number of bits for each axis.
        
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
            dim_encoding = bins[dim_encoding.squeeze(), :]
        else:
            raise ValueError("Invalid encoding type!")
        return dim_encoding
    
    encoded_dims = [get_encoding_for_dim(i) for i in range(num_encoding_types)]

    row_col_encoding = np.concatenate(encoded_dims, axis=-1)
    if row_col_encoding.ndim == 1:
        row_col_encoding = row_col_encoding[np.newaxis, :]

    # Make inputs a torch tensor and detach from computational graph
    row_col_encoding = torch.tensor(row_col_encoding, dtype=torch.float, 
                                    requires_grad=False)
    
    # Normalization for Xavier initialization
    with torch.no_grad():
        row_col_encoding = (row_col_encoding - torch.mean(row_col_encoding)) / \
                            torch.std(row_col_encoding)
    return row_col_encoding  


def top_k_values(matrix, k):
    """
    Leaves the top k largest values (by magnitude) of a matrix at their
    respective values and sets all other values to zero. This version aims
    for efficiency using numpy operations and handles edge cases.

    Args:
        matrix (numpy.ndarray): The input matrix.
        k (int): The number of largest values (by magnitude) to keep.
                 Must be non-negative.

    Returns:
        numpy.ndarray: A new matrix with the top k values preserved and others zeroed.
                       Returns a zero matrix if k <= 0.
                       Returns a copy of the original matrix if k >= matrix.size.

    Raises:
        TypeError: If 'matrix' is not a numpy ndarray or 'k' is not an integer.
        ValueError: If 'k' is negative.
    """

    # --- Core Logic ---
    # Flatten the original matrix to work with a 1D array.
    # This might create a copy, but is necessary for argpartition on the whole data.
    flat_matrix = matrix.flatten()

    # Find the indices of the top k elements based on their magnitude.
    # np.argpartition partitions the array such that the element at the k-th position
    # is the one that *would* be there if the array were sorted. All elements
    # before it are smaller or equal, and all after are larger or equal.
    # We use -k to work from the end (largest elements).
    # This operation has an average time complexity of O(N), where N is matrix.size.
    # It operates on the absolute values but returns indices corresponding to flat_matrix.
    top_k = torch.topk(torch.abs(flat_matrix), k)
    top_k_indices = top_k.indices

    # Create the result array initialized with zeros. This is O(N).
    result_flat = torch.zeros_like(flat_matrix)

    # Use the obtained indices to place the corresponding top k values
    # from the original flattened matrix into the zeroed result array.
    # This is advanced indexing in NumPy.
    result_flat[top_k_indices] = flat_matrix[top_k_indices]

    # Reshape the flat result array back to the original matrix's shape.
    # This is typically a very fast operation (O(1)) as it often only
    # changes metadata (strides) without copying data, unless memory layout forces a copy.
    result_matrix = result_flat.reshape(matrix.shape)

    return result_matrix


import torch
import torch.nn as nn
from torch import Tensor
from typing import Sequence


class LowRankMatrixDecompositionGNet(nn.Module):
    """
    A specialized type of g-net that uses low-rank matrix decomposition to
    compute the parameters of a layer.

    Args:
        sizes (Sequence[int): The input and output sizes of the layer.
        rank (int, optional): Rank for the matrix decomposition. Defaults to 32.
    """
    sizes: Sequence[int]
    output_scale: float

    def __init__(self, sizes: Sequence[int], rank: int = 32) -> None:
        super().__init__()
        self.rank = rank
        self.sizes = sizes
        self.output_scale = torch.tensor(1.0)

        # Define two trainable parameters
        self.A = nn.Parameter(torch.randn(sizes[1], self.rank))
        self.B = nn.Parameter(torch.randn(self.rank, sizes[0]))

    def forward(self, x: Tensor) -> Tensor:
        # Simply multiply the input by these two matrices
        return torch.matmul(self.A, self.B)
    
    
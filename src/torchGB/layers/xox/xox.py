from typing import Sequence

import torch
import torch.nn as nn
from torch import Tensor

import numpy as np


def std(array: Tensor) -> Tensor:
    return max(array.std().item(), 1e-9)


def calculate_reference_scaling(weights, init_fn=torch.nn.init.kaiming_normal_):
    # NOTE: Not sure if Kaiming is the best init! Rather use Xavier...
    ref_weights = torch.empty(weights.size())
    init_fn(ref_weights)
    return std(ref_weights) / std(weights)


def XOYGenerateWeights(X, O, Y) -> Tensor:
  return torch.matmul(torch.matmul(Y, O), X.T)


class XOXLayer(nn.Module):
    """Implements a linear layer with a low-rank factorization through a central matrix 'O'.

    This layer performs a linear transformation using a factorization of the weight matrix
    into three smaller matrices: X_input, O, and X_output.  The forward pass calculates
    output = Scaling_input * X_output @ O @ X_input.T @ input.  This structure aims to reduce
    the number of parameters compared to a standard linear layer, especially when the
    dimension of 'O' (num_genes) is much smaller than the input and output dimensions.
    Args:
        num_input (int): The dimensionality of the input.
        num_output (int): The dimensionality of the output.
        num_genes (int): The dimensionality of the central matrix 'O', controlling the rank of the factorization.
                          This acts as a bottleneck, hence the "Genomic" nomenclature referencing the idea of a
                          compressed genetic representation.
    """
    sizes: Sequence[int]
    output_scale: float
    
    def __init__(self, num_input: int, num_output: int, num_genes: int) -> None:
        super().__init__()
        
        self.sizes = (num_input, num_genes, num_output)
        self.output_scale = torch.tensor(1.0)
        
        self.O_mat = torch.nn.Parameter((1.0 / num_genes) * torch.randn(num_genes, num_genes))
        norm = (1.0 / np.sqrt(3 * num_genes))
        self.X_input = torch.nn.Parameter(norm * torch.randn(np.prod(num_input), num_genes))
        self.X_output = torch.nn.Parameter(norm * torch.randn(np.prod(num_output), num_genes))

        # Calculate a scaling factor based on the initialization scheme. This helps normalize
        # the output variance and likely improves training stability. The `calculate_reference_scaling`
        # function attempts to match the standard deviation of the initialized weights to that
        # of a Kaiming He initialized layer, a common initialization strategy for ReLU networks.
        
        # NOTE: this implementation is unnecessary complicated. Since we know the
        # stddevs of X, O, Y we could replace this with a more efficient solution
        # that directly computes the scaling factor.
        self.Scaling_input = nn.Parameter(torch.tensor(calculate_reference_scaling(XOYGenerateWeights(self.X_input,self.O_mat,self.X_output))))

    def forward(self, x: Tensor) -> Tensor:
        # Note: The current implementation performs the multiplication as  X_output @ O @ X_input.T
        # which doesn't seem correct for a standard linear layer replacement unless x is already flattened.
        #  A typical linear layer implementation would expect x to be (batch_size, input_dim) and perform
        # something like (x @ W.T) + b.  The .T in the current implementation suggests the expected
        # input might be (input_dim, batch_size).  This should be clarified and potentially corrected
        # to make the layer truly drop-in compatible. Also a bias term is missing.  See `TODO` in the original code.
        return self.Scaling_input * self.X_output @ self.O_mat @ self.X_input.T
    
    
import torch
import torch.nn as nn
from torch import Tensor

import numpy as np


def std(array: Tensor) -> Tensor:
    return max(array.std().item(), 1e-9)


def calculate_reference_scaling(weights, init_fn=torch.nn.init.kaiming_normal_):
    ref_weights = torch.empty(weights.size())
    init_fn(ref_weights)
    return std(ref_weights) / std(weights)


def XOYGenerateWeights(X, O, Y, Scaling=1.) -> Tensor:
  return torch.matmul(torch.matmul(Y, O), X.T)


class XOXLayer(nn.Module):
    """_summary_

    Args:
        nn (_type_): _description_
    """
    def __init__(self, num_input, num_output, num_genes):
        super().__init__()
        self.O_mat = torch.nn.Parameter((1.0/num_genes)*torch.randn(num_genes,num_genes))
        self.X_input = torch.nn.Parameter((1.0/np.sqrt(3*num_genes))*torch.randn(np.prod(num_input),num_genes))
        self.X_output = torch.nn.Parameter((1.0/np.sqrt(3*num_genes))*torch.randn(np.prod(num_output),num_genes))
        
        self.Scaling_input = nn.Parameter(torch.tensor(calculate_reference_scaling(XOYGenerateWeights(self.X_input,self.O_mat,self.X_output))))

    def forward(self, x: Tensor):
        # Base implementation of the XOX layer differs from what we need!
        # TODO make it compatible with our model
        return x @ (self.Scaling_input * self.X_output @ self.O_mat @ self.X_input.T).T
    
    
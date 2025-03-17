import unittest

import torch
from torch import Tensor
import numpy as np

from torchGB.utils import build_matrix


class TestBuildMatrix(unittest.TestCase):

    def test_build_matrix(self):
        # Test case 1: 2x2 matrix tiled into 1x1 tiles
        matrix = torch.arange(4).reshape(2, 2)
        tiled_matrix = matrix.reshape(2, 1, 2, 1).swapaxes(1, 2).reshape(4, 1, 1)
        reassembled_matrix = build_matrix(tiled_matrix, (2, 2))
        self.assertTrue(torch.equal(matrix, reassembled_matrix))

        # Test case 2: 4x4 matrix tiled into 2x2 tiles
        matrix = torch.arange(16).reshape(4, 4)
        tiled_matrix = matrix.reshape(2, 2, 2, 2).swapaxes(1, 2).reshape(4, 2, 2)
        reassembled_matrix = build_matrix(tiled_matrix, (4, 4))
        self.assertTrue(torch.equal(matrix, reassembled_matrix))

        # Test case 3: 6x6 matrix tiled into 3x2 tiles
        matrix = torch.arange(36).reshape(6, 6)
        tiled_matrix = matrix.reshape(2, 3, 3, 2).swapaxes(1,2).reshape(6, 3, 2)
        reassembled_matrix = build_matrix(tiled_matrix, (6, 6))
        self.assertTrue(torch.equal(matrix, reassembled_matrix))
        
        # Test case 4: Check for error if input is not 3D
        matrix = torch.arange(4).reshape(2, 2)
        with self.assertRaisesRegex(AssertionError, "Input array must be 3D"):
            build_matrix(matrix, (2, 2))

        # Test case 5: Check for error if dimensions are not divisible
        matrix = torch.arange(16).reshape(4, 2, 2)
        with self.assertRaisesRegex(AssertionError, "4 rows is not evenly divisible by 3"):
            build_matrix(matrix, (4, 8))
        
        # Test case 6: Explicitly handcoded matrix
        matrix = torch.tensor([[0, 0, 1, 1],
                               [0, 0, 1, 1],
                               [2, 2, 3, 3],
                               [2, 2, 3, 3]])
        print(matrix)
        tiled_matrix = torch.tile(torch.arange(4).reshape((-1, 1, 1)), (1, 2, 2))
        print(tiled_matrix)
        reassembled_matrix = build_matrix(tiled_matrix, (4, 4))
        print(reassembled_matrix)
        self.assertTrue(torch.equal(matrix, reassembled_matrix))
            

if __name__ == "__main__":
    unittest.main()
            
            
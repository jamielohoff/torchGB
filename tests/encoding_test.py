import unittest
import numpy as np
import torch

from torchGB.utils import make_row_col_encoding, EncodingType

class TestBinaryEncoding(unittest.TestCase):
    def test_encoding(self):
        encoding_types = (EncodingType.BINARY, EncodingType.BINARY)
        shape = (4, 4)
        
        num_encoding_bits = np.ceil(np.log(shape)/np.log(2)).astype(np.uint16)
        num_encoding_bits[np.where(num_encoding_bits == 0)] = 1
        
        encoding = make_row_col_encoding(shape, encoding_types, num_encoding_bits)
        print(encoding)
        true_encoding = torch.Tensor(
            [[-0.9922, -0.9922, -0.9922, -0.9922],
            [-0.9922, -0.9922, -0.9922,  0.9922],
            [-0.9922, -0.9922,  0.9922, -0.9922],
            [-0.9922, -0.9922,  0.9922,  0.9922],
            [-0.9922,  0.9922, -0.9922, -0.9922],
            [-0.9922,  0.9922, -0.9922,  0.9922],
            [-0.9922,  0.9922,  0.9922, -0.9922],
            [-0.9922,  0.9922,  0.9922,  0.9922],
            [ 0.9922, -0.9922, -0.9922, -0.9922],
            [ 0.9922, -0.9922, -0.9922,  0.9922],
            [ 0.9922, -0.9922,  0.9922, -0.9922],
            [ 0.9922, -0.9922,  0.9922,  0.9922],
            [ 0.9922,  0.9922, -0.9922, -0.9922],
            [ 0.9922,  0.9922, -0.9922,  0.9922],
            [ 0.9922,  0.9922,  0.9922, -0.9922],
            [ 0.9922,  0.9922,  0.9922,  0.9922]]
        )
        self.assertTrue((true_encoding-encoding).sum() < 1e-7)


if __name__ == "__main__":
    unittest.main()


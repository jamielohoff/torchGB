import unittest

import torch
from torchGB.layers.gnet.model import FastGenomicBottleneck

class TestFastGenomicBottleneck(unittest.TestCase):
    def test_fast_genomic_bottleneck(self):
        input_encoding_size = 13
        num_params = 72
        num_tiles = 10
        model = FastGenomicBottleneck(num_tiles=num_tiles, sizes=[input_encoding_size, 32, 1])
        data = torch.randn(num_params, 1, input_encoding_size) # predict 72 parameters for 10 tiles
        output = model(data)
        self.assertEqual(output.shape, (num_params, num_tiles, 1))
    
if __name__ == '__main__':
    unittest.main()
  
    
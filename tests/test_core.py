import unittest
import torch
import torch.nn as nn
import torch.distributed as dist

from src.torchGB.core import GenomicBottleneck, register_gnet_type
from src.torchGB.gnet import GenomicBottleNet

class MockModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(10, 20)
        self.linear2 = nn.Linear(20, 30)
        self.conv1 = nn.Conv2d(3, 6, 3)


class TestGenomicBottleneck(unittest.TestCase):

    def setUp(self):
        self.model = MockModule()
        self.num_batches = 10
        self.hidden_dim = 32
        self.lr = 0.001
        self.gnet_batchsize = 10000

    def test_init(self):
        gnet = GenomicBottleneck(self.model, self.num_batches, self.hidden_dim, self.lr, self.gnet_batchsize)
        self.assertEqual(gnet.lr, self.lr)
        self.assertEqual(gnet.num_batches, self.num_batches)
        self.assertIs(gnet.model, self.model)
        self.assertIsInstance(gnet.gnetdict, dict)

    def test_get_num_params_gnet(self):
        gnet = GenomicBottleneck(self.model, self.num_batches, self.hidden_dim, self.lr, self.gnet_batchsize)
        num_params = gnet.get_num_params_gnet()
        self.assertGreaterEqual(num_params, 0)

    def test_get_num_params_no_gnet(self):
        gnet = GenomicBottleneck(self.model, self.num_batches, self.hidden_dim, self.lr, self.gnet_batchsize)
        num_params = gnet.get_num_params_no_gnet()
        self.assertGreaterEqual(num_params, 0)

    def test_compression(self):
        gnet = GenomicBottleneck(self.model, self.num_batches, self.hidden_dim, self.lr, self.gnet_batchsize)
        compression = gnet.compression()
        self.assertGreaterEqual(compression, 0)


    def test_register_gnet_type(self):

        def mock_init(pname, param, hidden_dim, gnet_batchsize):
            return torch.randn(10, 10), [GenomicBottleNet([2,10,1], output_scale=1.)], (10,10), 1., 1.

        def mock_build(name, param, new_weights, tile_shape):
            return new_weights.sum(0)

        register_gnet_type(nn.Linear, mock_init, mock_build)
        gnet = GenomicBottleneck(self.model, self.num_batches, self.hidden_dim, self.lr, self.gnet_batchsize)
        self.assertIn(nn.Linear, gnet.gnetdict)



if __name__ == '__main__':
    unittest.main()

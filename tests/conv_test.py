import os
import unittest
import numpy as np
import torch

import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from torchGB.core import GenomicBottleneck


class TestBinaryEncoding(unittest.TestCase):
    def test_conv2d_encoding(self):
        class TestModel(torch.nn.Module):
            conv1: torch.nn.Conv2d
            conv2: torch.nn.Conv2d
            def __init__(self):
                super(TestModel, self).__init__()
                self.conv1 = torch.nn.Conv2d(3, 8, 5)
                self.conv2 = torch.nn.Conv2d(8, 16, 5)
                
            def forward(self, x):
                x = F.relu(self.conv1(x))
                return F.relu(self.conv2(x))

        model = TestModel()
        model = model.to("cuda")
        
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        ddp_model = DDP(model, device_ids=[rank], output_device=rank)
        gnets = GenomicBottleneck(ddp_model, 1)
        
        gnets.predict_weights(ddp_model)


if __name__ == "__main__":
    unittest.main()
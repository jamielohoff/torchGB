import os
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from torchvision import datasets, transforms

from torchGB import GenomicBottleneck

### How to launch this script ##################################################
# To launch this script, you should use the following command:
# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 mlp_example.py --gpus 0,1,2,3 --seed 8888 --language en --batchsize 56
# This launches the script on 4 GPUs. It is highly recommended to use multiple
# GPUs, as it is much faster to train the g-nets on multiple GPUs.
# This example should work with a single GPU none-the-less.

parser = argparse.ArgumentParser()

parser.add_argument("--gpus", type=str, default="0", help="Which GPUs to use.")

parser.add_argument("--seed", type=int, default=0, help="Random seed of the experiment.")

parser.add_argument("--batchsize", type=int, default=0, help="Batchsize for the experiment.")

args = parser.parse_args()

# Setup of global variables
torch.manual_seed(args.seed)
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

EPOCHS = 10
BATCHSIZE = args.batchsize
LOG_INTERVAL = 100

train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transforms.ToTensor())
test_dataset = datasets.MNIST(root="./data", train=False, transform=transforms.ToTensor())
train_loader = DataLoader(train_dataset, batch_size=BATCHSIZE)
test_loader = DataLoader(test_dataset, batch_size=BATCHSIZE)

# Multiprocessing setup
dist.init_process_group(backend="nccl")
rank = dist.get_rank()
world_size = dist.get_world_size()
print(f"Rank: {rank}, World Size: {world_size}")

# Define the model and wrap it into a distributed model
model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 256),
    nn.ReLU(),
    nn.Linear(256, 64),
    nn.ReLU(),
    nn.Linear(64, 10)
).to(rank)
# Wrap the model into a distributed model
model = DDP(model).to(rank)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Create g-nets/hypernetworks for all linear layers
gnets = GenomicBottleneck(model)


# Training function
def train(model: nn.Module, gnets: GenomicBottleneck):
    gnets.train()
    model.train()
    
    pbar = tqdm(train_loader)
    for data, targets in pbar:
        # Move data to GPU with rank `rank`
        data = data.reshape(data.shape[0], -1).to(rank)
        targets = targets.to(rank)
        
        # Zero out the gradients before backpropagation
        optimizer.zero_grad()
        gnets.zero_grad()
        
        # Implicitly updates the model weights!
        gnets.predict_weights() 

        output = model(data)
        loss = criterion(output, targets)

        # Backpropagate the gradients to the g-nets/hypernetworks
        loss.backward()
        gnets.backward()
        
        # Update the g-nets/hypernetworks with the gradients 
        optimizer.step()
        gnets.step()

        loss.item()
        pbar.set_description(f"Loss: {loss.item():.2f}")
        
                                       
# Evaluation function
def evaluate(model: nn.Module):
    model.eval() # turn on evaluation mode
    total_loss = 0
    acc = 0

    with torch.no_grad():
        for data, targets in test_loader:
            data = data.reshape(data.shape[0], -1).to(rank)
            targets = targets.to(rank)
            
            output = model(data)
            total_loss += criterion(output, targets).item()
            acc += (output.argmax(dim=1) == targets).sum().item()
            
    print(f"Test loss: {total_loss/len(test_loader):.2f}, "
          f"Test accuracy: {acc/len(test_loader):.2f}")


for e in range(EPOCHS):
    train(model, gnets)
    evaluate(model, test_loader)


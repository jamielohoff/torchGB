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

# ----------------------------- #
#        Argument Parser        #
# ----------------------------- #
parser = argparse.ArgumentParser()

parser.add_argument("--gpus", type=str, default="0", help="Which GPUs to use.")
parser.add_argument("--seed", type=int, default=0, help="Random seed of the experiment.")
parser.add_argument("--batchsize", type=int, default=256, help="Batch size for the experiment.")

args = parser.parse_args()

# ----------------------------- #
#         Setup GPU and Seeds   #
# ----------------------------- #
torch.manual_seed(args.seed)
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

# ----------------------------- #
#       Experiment Settings     #
# ----------------------------- #
EPOCHS = 10
BATCHSIZE = args.batchsize

# ----------------------------- #
#        Load Datasets          #
# ----------------------------- #
train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transforms.ToTensor())
test_dataset = datasets.MNIST(root="./data", train=False, transform=transforms.ToTensor())
train_loader = DataLoader(train_dataset, batch_size=BATCHSIZE)
test_loader = DataLoader(test_dataset, batch_size=BATCHSIZE)

# ----------------------------- #
#     Distributed Setup         #
# ----------------------------- #
dist.init_process_group(backend="nccl")
rank = dist.get_rank()
world_size = dist.get_world_size()
print(f"Rank: {rank}, World Size: {world_size}")

# ----------------------------- #
#      Define the MLP Model     #
# ----------------------------- #
model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 256),
    nn.ReLU(),
    nn.Linear(256, 64),
    nn.ReLU(),
    nn.Linear(64, 10)
).to(rank)

# Wrap the model for Distributed Data Parallel
model = DDP(model).to(rank)

# ----------------------------- #
#       Loss and Optimizer      #
# ----------------------------- #
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# ----------------------------- #
#     Initialize Genomic Bottleneck with XOX Encoding
# ----------------------------- #
gnets = GenomicBottleneck(
    model,
    hypernet_type="xox",  # <--- Use XOX encoding
)

# ----------------------------- #
#     Function to Print Xs      #
# ----------------------------- #
def print_xox_matrices(gnets, title):
    print(f"\n--- {title} ---")
    if rank != 0:
        print('rank bad!')
        return
    
    for name, gnet_layer in gnets.gnetdict.items():
        if gnet_layer.gnets:
            for gnet in gnet_layer.gnets:
                if hasattr(gnet, 'X_input') and hasattr(gnet, 'X_output'):
                    print(f"Layer: {name}")
                    print(f"X_input:\n{gnet.X_input}")
                    print(f"X_output:\n{gnet.X_output}\n")

# Print Initial Xs
print_xox_matrices(gnets, "Initial X Matrices")

# ----------------------------- #
#          Training Loop        #
# ----------------------------- #
def train(model: nn.Module, gnets: GenomicBottleneck):
    gnets.train()
    model.train()

    pbar = tqdm(train_loader) if rank == 0 else train_loader
    for data, targets in pbar:
        # Move data to GPU
        data = data.reshape(data.shape[0], -1).to(rank)
        targets = targets.to(rank)

        # Zero gradients
        optimizer.zero_grad()
        gnets.zero_grad()

        # Predict and update weights using XOX encoding
        gnets.predict_weights()

        # Forward pass
        output = model(data)
        loss = criterion(output, targets)

        # Backward pass
        loss.backward()
        gnets.backward()

        # Optimizer step
        optimizer.step()
        gnets.step()

        if rank == 0:
            pbar.set_description(f"Loss: {loss.item()/BATCHSIZE:.2f}")

# ----------------------------- #
#         Evaluation Loop       #
# ----------------------------- #
def evaluate(model: nn.Module):
    model.eval()
    total_loss = 0.
    acc = 0.

    with torch.no_grad():
        for data, targets in test_loader:
            data = data.reshape(data.shape[0], -1).to(rank)
            targets = targets.to(rank)

            output = model(data)
            total_loss += criterion(output, targets).item()
            acc += (output.argmax(dim=1) == targets).sum().item()

    norm = len(test_loader) * BATCHSIZE
    if rank == 0:
        print(f"Test loss: {total_loss/norm:.2f}, Test accuracy: {acc/norm:.2f}")

# ----------------------------- #
#        Run Training Loop      #
# ----------------------------- #
for e in range(EPOCHS):
    train(model, gnets)
    evaluate(model)

# Print Final Xs
print_xox_matrices(gnets, "Final X Matrices")

# Cleanup after multi-GPU training
dist.destroy_process_group()
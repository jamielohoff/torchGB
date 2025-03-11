import os
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from torchvision import datasets, transforms

## To Run: python experiments/examples/mlp_no_gnet.py --gpu 0 --seed 8888 --batchsize 256

# ----------------------------- #
#        Argument Parser        #
# ----------------------------- #
parser = argparse.ArgumentParser()

parser.add_argument("--gpu", type=int, default=0, help="GPU index to use.")
parser.add_argument("--seed", type=int, default=0, help="Random seed of the experiment.")
parser.add_argument("--batchsize", type=int, default=256, help="Batch size for the experiment.")

args = parser.parse_args()

# ----------------------------- #
#         Setup Device          #
# ----------------------------- #
device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
torch.manual_seed(args.seed)

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
train_loader = DataLoader(train_dataset, batch_size=BATCHSIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCHSIZE)

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
).to(device)

# ----------------------------- #
#       Loss and Optimizer      #
# ----------------------------- #
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# ----------------------------- #
#          Training Loop        #
# ----------------------------- #
def train():
    model.train()
    pbar = tqdm(train_loader)
    for data, targets in pbar:
        data = data.reshape(data.shape[0], -1).to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, targets)
        loss.backward()
        optimizer.step()

        pbar.set_description(f"Loss: {loss.item():.4f}")

# ----------------------------- #
#         Evaluation Loop       #
# ----------------------------- #
def evaluate():
    model.eval()
    total_loss = 0.0
    acc = 0.0

    with torch.no_grad():
        for data, targets in test_loader:
            data = data.reshape(data.shape[0], -1).to(device)
            targets = targets.to(device)

            output = model(data)
            total_loss += criterion(output, targets).item()
            acc += (output.argmax(dim=1) == targets).sum().item()

    total = len(test_loader.dataset)
    print(f"Test loss: {total_loss / total:.4f}, Test accuracy: {acc / total:.4f}")

# ----------------------------- #
#        Run Training Loop      #
# ----------------------------- #
for e in range(EPOCHS):
    print(f"\nEpoch {e+1}/{EPOCHS}")
    train()
    evaluate()
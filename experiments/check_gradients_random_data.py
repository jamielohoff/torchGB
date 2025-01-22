import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.parallel import DistributedDataParallel as DDP

from torchGB import GenomicBottleneck


SEED = 42
torch.manual_seed(SEED)

# Multiprocessing setup
dist.init_process_group(backend="nccl")
rank = dist.get_rank()
world_size = dist.get_world_size()

# Hyperparameters
input_size = 4  # Number of input features
hidden_size = 4  # Size of the hidden layer
output_size = 2  # Number of output classes (adjust as needed)
batch_size = 16
learning_rate = 0.01
num_epochs = 10

# Generate arbitrary data
num_samples = 100
X = torch.rand(num_samples, input_size)  # Random input data
y = torch.randint(0, output_size, (num_samples,))  # Random target labels

# Create DataLoader
dataset = TensorDataset(X, y)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Define the model
class FeedForwardNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(FeedForwardNet, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        return self.layers(x)

model = FeedForwardNet(input_size, hidden_size, output_size).to(rank)
model = DDP(model, device_ids=[rank], output_device=rank)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

# Initialize Genomic Bottleneck
gnets = GenomicBottleneck(model, batch_size, ignore_layers=["bias"], lr=learning_rate)


def difference_per_batch(old_pnet, true_pnet_update, gnet_update):
    true_dp = []
    gnet_dp = []
    for op, tp, gp in zip(old_pnet[1:-1], true_pnet_update[1:-1], gnet_update[1:-1]):
        true_dp.append((op-tp).flatten())
        gnet_dp.append((op-gp).flatten())
        
    return true_dp, gnet_dp
    

# Training loop
for epoch in range(num_epochs):
    model.train()
    old_pnet = []
    true_pnet_update = []
    gnet_update = []
    layer_idx = 0
    for inputs, labels in data_loader:
        inputs, labels = inputs.to(rank), labels.to(rank)

        # Zeroing grads and predicting model weights
        optimizer.zero_grad()
        gnets.zero_grad()
        old_pnet.append(model.module.layers[layer_idx].weight.data)
        
        gnets.predict_weights() # implicitly updates the model weights!
        # gnet_update.append(model.module.layers[layer_idx].weight.data)

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        loss.backward()
        gnets.backward()

        optimizer.step()
        true_pnet_update.append(model.module.layers[layer_idx].weight.data)
        gnets.step()
        
        gnets.predict_weights() # implicitly updates the model weights!
        gnet_update.append(model.module.layers[layer_idx].weight.data)
        
    true_dp, gnet_dp = difference_per_batch(old_pnet, true_pnet_update, gnet_update)
    true_dp = torch.stack(true_dp).t().cpu().numpy()
    gnet_dp = torch.stack(gnet_dp).t().cpu().numpy()

    print(true_dp[:10, :])
    print(gnet_dp[:10, :])
    sns.heatmap(true_dp)
    plt.savefig("true_pnet_update.png")
    plt.clf()
    sns.heatmap(gnet_dp)
    plt.savefig("gnet_pnet_update.png")
    break

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")
    



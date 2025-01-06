import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.parallel import DistributedDataParallel as DDP

from core import GenomicBottleneck

# Multiprocessing setup
dist.init_process_group(backend="nccl")
rank = dist.get_rank()
world_size = dist.get_world_size()

# Hyperparameters
input_size = 4  # Number of input features
hidden_size = 4  # Size of the hidden layer
output_size = 4  # Number of output classes (adjust as needed)
batch_size = 10000
learning_rate = 0.001
num_epochs = 10

# Generate arbitrary data
IN_SZ = 60;
KER_SZ = 30;
num_samples = 100000
X = torch.rand(num_samples, input_size, IN_SZ, IN_SZ)  # Random input data
y = torch.randint(0, output_size, (num_samples,))  # Random target labels

# Create DataLoader
dataset = TensorDataset(X, y)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Define the model
class FeedForwardNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(FeedForwardNet, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(input_size, hidden_size, KER_SZ, padding='valid'),
            nn.Sigmoid(),
            nn.AvgPool2d(IN_SZ + 1 - KER_SZ)
        )
        self.fc = nn.Linear(hidden_size, output_size)
        self.apply(self.init_weights)

    def forward(self, x):
        x = self.layers(x)
        x = torch.flatten(x, 1)
        return self.fc(x)
    
    def init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=1e-2, mean=0.0)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

model = FeedForwardNet(input_size, hidden_size, output_size).to(rank)
model = DDP(model, device_ids=[rank], output_device=rank)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Initialize Genomic Bottleneck
gnets = GenomicBottleneck(model, batch_size, ignore_layers=["bias"], lr=learning_rate)


def difference_per_batch(old_pnet, true_pnet_update, gnet_update):
    true_dp = []
    gnet_dp = []
    for op, tp, gp in zip(old_pnet, true_pnet_update, gnet_update):
        true_dp.append((op-tp).flatten())
        gnet_dp.append((op-gp).flatten())
        
    return true_dp, gnet_dp
    

# Training loop
for epoch in range(num_epochs):
    print(f'Epoch: {epoch:d}')
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
        
        gnets.predict_weights() # implicitly updates the model weights!
        old_pnet.append(model.module.layers[layer_idx].weight.data.detach().cpu().numpy())

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        loss.backward()
        gnets.backward()
        
        optimizer.step()
        true_pnet_update.append(model.module.layers[layer_idx].weight.data.detach().cpu().numpy())
        gnets.step()
        
        gnets.predict_weights() # implicitly updates the model weights!
        gnet_update.append(model.module.layers[layer_idx].weight.data.detach().cpu().numpy())
        
    true_dp, gnet_dp = difference_per_batch(old_pnet, true_pnet_update, gnet_update)
    true_dp = np.stack(true_dp).T
    gnet_dp = np.stack(gnet_dp).T
    
    cc = np.corrcoef(true_dp.flatten(), gnet_dp.flatten())[0][1]
    print(f'Correlation: {cc:f}')
    cc_batch = np.corrcoef(true_dp.T, gnet_dp.T)
    cc_batch = np.diag(cc_batch[true_dp.shape[1]:, :])
    cc_batch = np.mean(cc_batch)
    print(f'Average in-batch correlation: {cc_batch:f}')
    
    mask = np.multiply(np.abs(true_dp.flatten()) > 0, np.abs(gnet_dp.flatten()) > 0)
    ix = np.where(mask == True)[0]

    ratios = np.divide(np.abs(true_dp.flatten()[ix]), np.abs(gnet_dp.flatten()[ix]))
    ratio = np.mean(ratios)
    ratio_std = np.std(ratios)
    pc_unmasked = np.sum(1-mask) / len(true_dp.flatten())
    print(f'Fraction unmasked: {pc_unmasked:f}')
    print(f'Ratio: {ratio:f} +- {ratio_std:f}')
    print(ratios)
    
    plt.clf()
    sns.heatmap(true_dp)
    plt.savefig("true_pnet_update.png")
    plt.clf()
    sns.heatmap(gnet_dp/3*2)
    plt.savefig("gnet_pnet_update.png")

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

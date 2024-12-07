import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.parallel import DistributedDataParallel as DDP

from torchGB import GenomicBottleneck


SEED = 42

# Multiprocessing setup
dist.init_process_group(backend="nccl")
rank = dist.get_rank()
world_size = dist.get_world_size()

# Device configuration
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(SEED)

# Hyperparameters
input_size = 10  # Number of input features
hidden_size = 32  # Size of the hidden layer
output_size = 2  # Number of output classes (adjust as needed)
batch_size = 16
learning_rate = 0.01
num_epochs = 20

# Generate arbitrary data
num_samples = 1000
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

model = FeedForwardNet(input_size, hidden_size, output_size)
model = DDP(model, device_ids=[rank], output_device=rank)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

# Initialize Genomic Bottleneck
gnets = GenomicBottleneck(model, batch_size, ignore_layers=["bias"])

# Training loop
for epoch in range(num_epochs):
    model.train()

    gnets.zero_grad()
    gnets.predict_weights(model) # implicitly updates the model weights!
    for inputs, labels in data_loader:
        inputs, labels = inputs, labels

        # Zeroing grads and predicting model weights
        optimizer.zero_grad()
        gnets.zero_grad()
        gnets.predict_weights(model) # implicitly updates the model weights!

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        loss.backward()
        gnets.backward()

        optimizer.step()
        gnets.step()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")


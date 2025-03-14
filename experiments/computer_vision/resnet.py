import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import os
import torchGB.core_fast as gn

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

def trainE2E(model, rank, train_loader, optimizer, GNets):
    criterion = torch.nn.CrossEntropyLoss()
    correct = 0
    
    model.train()
    GNets.train() 
    
    for batch_idx, (inputs, labels) in enumerate(train_loader):
        # Move input and label tensors to the device
        inputs = inputs.to(rank)
        labels = labels.to(rank)
        
        # Zero out the optimizer
        optimizer.zero_grad()
        GNets.zero_grad()

        # Forward pass
        GNets.predict_weights()
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()
        GNets.backward()
        optimizer.step()
        GNets.step()

        pred = outputs.argmax(dim=1, keepdim=True)
        correct += pred.eq(labels.view_as(pred)).sum().item()
        
    train_acc = torch.tensor(correct / 1281167).to(rank)
    dist.barrier()
    dist.all_reduce(train_acc, op=dist.ReduceOp.SUM)
    train_acc = train_acc.cpu().item()
    dist.barrier()
    
    if rank == 0:
        print(f'Training accuracy: {train_acc:.4f}')

    return train_acc

def testResNet(model, rank, val_loader):
    correct = 0
    # model.eval()
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(val_loader):
            # Move input and label tensors to the device
            inputs = inputs.to(rank)
            labels = labels.to(rank)
        
            outputs = model(inputs)
            pred = outputs.argmax(dim=1, keepdim=True)
            correct += pred.eq(labels.view_as(pred)).sum().item()

        val_acc = torch.tensor(correct / 50000).to(rank)
        dist.barrier()
        dist.all_reduce(val_acc, op=dist.ReduceOp.SUM)
        val_acc = val_acc.cpu().item()
        dist.barrier()
        
        if rank == 0:
            print(f'Validation accuracy: {val_acc:.4f}')
    
    return val_acc

def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    dist.init_process_group(backend="nccl")
    
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    correct_max = 0
    max_epochs = 1000

    CorrectTrain = np.zeros(max_epochs)
    CorrectGDN = np.zeros(max_epochs)
  
    # Set hyperparameters
    batch_size = int(np.ceil(750 / world_size))
    
    # Initialize transformations for data augmentation
    transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    transform_val = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Load the ImageNet Object Localization Challenge dataset
    train_dataset = torchvision.datasets.ImageNet(
        root='/home/sergey/IMAGENET',
        split='train',
        transform=transform
    )
    
    val_dataset = torchvision.datasets.ImageNet(
        root='/home/sergey/IMAGENET',
        split='val',
        transform=transform_val
    )
    
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    train_loader_ResNet = torch.utils.data.DataLoader(train_dataset,
                                                      batch_size=batch_size,
                                                      shuffle=False,
                                                      sampler=train_sampler,
                                                      num_workers=8,
                                                      prefetch_factor=4,
                                                      pin_memory=True)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank)
    val_loader_ResNet = torch.utils.data.DataLoader(val_dataset,
                                                    batch_size=batch_size,
                                                    shuffle=False,
                                                    sampler=val_sampler,
                                                    num_workers=8,
                                                    prefetch_factor=4,
                                                    pin_memory=True)
    
    # Load the ResNet18 model
    model_ResNet = torchvision.models.resnet18(weights=None)
    for m in model_ResNet.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            m.track_running_stats=False
            m.affine=False
    
    # Set the model to run on the device
    model_ResNet = model_ResNet.to(rank)
    model_ResNet = DDP(model_ResNet, device_ids=[rank], output_device=rank)
    GNets = gn.GenomicBottleneck(model_ResNet, ignore_layers=["fc.weight"])#, "fc.bias"])
      
    # Define the optimizer
    optimizer_ResNet = torch.optim.Adam(model_ResNet.parameters())
   
    # Training  
    for epoch in range(max_epochs):
        if rank == 0:
            print(f'Epoch: {epoch}')
        
        dist.barrier()
        train_sampler.set_epoch(epoch)
        val_sampler.set_epoch(epoch)

        CorrectTrain[epoch] = trainE2E(model_ResNet, rank, train_loader_ResNet, optimizer_ResNet, GNets)
        GNets.predict_weights()
        CorrectGDN[epoch] = testResNet(model_ResNet, rank, val_loader_ResNet)

        if rank == 0:
            np.save('accuracy_train.npy', CorrectTrain)
            np.save('accuracy_test.npy', CorrectGDN)

            # Saving the model
            correct = CorrectGDN[epoch]
            if correct > correct_max:
                # GNets.save('model_gnet.pt', model_ResNet, optimizer_ResNet)
                torch.save(model_ResNet.state_dict(), 'model_pnet.pt')
                correct_max = correct

    print("done")
    dist.destroy_process_group()
    return model_ResNet

if __name__ == '__main__':
    model_ResNet = main()
    
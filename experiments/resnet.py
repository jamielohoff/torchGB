"""
    007_ImageNET_automatic_1.py

    This is a demo file for geneNets2_0
    This file shows how to encode ImageNET via the bottleneck by automatically setting up
    G-net networks. In this case the last two layers of the network are changed using 
    updateLayer function to yield one-hot vector as an output. The biases for the last
    layer are not learned but directly encoded in the weights of a one-layer g-net
    
    questions -- ask Alex (akula@cshl.edu)
    
"""

import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import os
# import lamb
import torchGB.core as gn
# from torch.profiler import profile, ProfilerActivity

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
# from datasets.distributed import split_dataset_by_node
from torch.utils.data.distributed import DistributedSampler

def trainE2E(model, rank, train_loader, optimizer, GNets):
    
    # A readout to compare pNet & gNet updates

    readout_size = 100    
    num_batches = len(train_loader)
    update_readout = np.zeros([3, num_batches, readout_size])
    
    # print('Training ----------------------------------')

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
        
        GNets.predict_weights()
        # print('ololololo')
        # dist.barrier()
        # Forward pass
        outputs = model(inputs)
        # print('yeyeyeyeye')
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()
        GNets.backward()

        # update_readout[0, batch_idx, :] = model.module.layer4[1].conv2.weight.cpu().detach().numpy()[: readout_size, 0, 0, 0]
        # optimizer.step()
        # update_readout[1, batch_idx, :] = model.module.layer4[1].conv2.weight.cpu().detach().numpy()[: readout_size, 0, 0, 0]
        # GNets.step()
        # GNets.predict_weights()
        # update_readout[2, batch_idx, :] = model.module.layer4[1].conv2.weight.cpu().detach().numpy()[: readout_size, 0, 0, 0]
        
        update_readout[0, batch_idx, :] = model.module.fc.weight.cpu().detach().numpy()[: readout_size, 0]
        optimizer.step()
        update_readout[1, batch_idx, :] = model.module.fc.weight.cpu().detach().numpy()[: readout_size, 0]
        GNets.step()
        GNets.predict_weights()
        update_readout[2, batch_idx, :] = model.module.fc.weight.cpu().detach().numpy()[: readout_size, 0]
        
        # scheduler.step(epoch_overall + batch_idx / len(train_loader))
        
        pred = outputs.argmax(dim=1, keepdim=True)
        correct += pred.eq(labels.view_as(pred)).sum().item()
        
        # if batch_idx == 4:
        #     break
        
    # train_acc = correct / len(train_loader.dataset)
    train_acc = torch.tensor(correct / 1281167).to(rank)
    dist.barrier()
    dist.all_reduce(train_acc, op=dist.ReduceOp.SUM)
    train_acc = train_acc.cpu().item()
    dist.barrier()
    
    if rank == 0:
        print(f'Epoch , Training accuracy: {train_acc:.4f}')

    return train_acc, update_readout

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
            
            # if batch_idx == 4:
            #     break
    
        # val_acc = correct / len(val_loader.dataset)
        val_acc = torch.tensor(correct / 50000).to(rank)
        dist.barrier()
        dist.all_reduce(val_acc, op=dist.ReduceOp.SUM)
        val_acc = val_acc.cpu().item()
        dist.barrier()
        
        if rank == 0:
            print(f'Validation accuracy: {val_acc:.4f}')
    
    return val_acc

def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    
    # Set device
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # os.environ["MASTER_ADDR"] = 'localhost'
    # os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group(backend="nccl")
    
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    correct_max = 0
    max_epochs = 1

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
    
    # Parallelize training across multiple GPUs
    # model_ResNet = torch.nn.DataParallel(model_ResNet)
    
    # Set the model to run on the device
    model_ResNet = model_ResNet.to(rank)
    model_ResNet = DDP(model_ResNet, device_ids=[rank], output_device=rank)

    GNets = gn.GenomicBottleneck(model_ResNet, ignore_layers=["fc.weight", "fc.bias"])

    # GNets.updateParameter('fc.weight', (gn.BIN, gn.HOT),[512, 1000],[9, 1000], numGDNLayers=-1) 
    # GNets.updateParameter('fc.bias', (gn.HOT),[1000],[1000], numGDNLayers = -1)
    # TODO: add IGNORE here

    # Parallelize training across multiple GPUs
    # GNets.parallelize()
            
    # GNets.to(device)
    # GNets.displayGNets()
      
    # Define the optimizer
    optimizer_ResNet = torch.optim.SGD(model_ResNet.parameters(), lr=0.001)
    # optimizer_ResNet = lamb.Lamb(model_ResNet.parameters())
    # scheduler_ResNet = CosineAnnealingWarmRestarts(optimizer_ResNet, 10)
        
    # Training  
    for epoch in range(max_epochs):
        
        dist.barrier()
        train_sampler.set_epoch(epoch)
        val_sampler.set_epoch(epoch)
        
        # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        # CorrectTrain[epoch] = trainE2E(model_ResNet, device, train_loader_ResNet, optimizer_ResNet, None, epoch, GNets)
        CorrectTrain[epoch], update_readout = trainE2E(model_ResNet, rank, train_loader_ResNet, optimizer_ResNet, GNets)
        #     print('Profiling completed\n')
        #     dist.barrier()
        # if rank == 0:
        #     prof.export_chrome_trace("trace_1000.json")
        #     print('Profile exported\n')
            
        # print(GNets.numberOfParameters(model_ResNet))
        # print(GNets.gnetParameters())
        # print(GNets.compression(model_ResNet))
        # print(GNets.correlations().transpose())
                
        # Testing the new model      
        # with torch.no_grad():
        #     GNets.predict_weights(model_ResNet) #just to make sure
        # TODO: eventually restore it
        CorrectGDN[epoch] = testResNet(model_ResNet, rank, val_loader_ResNet)

        if rank == 0:
            np.save('correct_train_lastuncompressed.npy', CorrectTrain)
            np.save('correct_test_gdn_lastuncompressed.npy', CorrectGDN)
            np.save(f'update_readout_epoch_{epoch:d}.npy', update_readout)
            
            # # Plotting the progress
            # x = np.arange(1, max_epochs + 1)
            # plt.semilogx(x, CorrectTrain, label = "Training") 
            # plt.semilogx(x, CorrectGDN, label = "Testing - compressed") 
            # plt.legend()
            # plt.title('ImageNET')
            # plt.xlabel('Epoch')
            # plt.ylabel('Accuracy')
            # plt.show()
            
            # print(model_ResNet.layer2[0].conv2.weight)
            
            # Saving the model
            correct = CorrectGDN[epoch]
            if correct > correct_max:
                # GNets.save('007_ResNet_lastuncompressed.pt', model_ResNet, optimizer_ResNet)
                torch.save(model_ResNet.state_dict(), '007_ResNet_pNet_lastuncompressed.pt')
                correct_max = correct

    print("done")
    dist.destroy_process_group()
    return model_ResNet

if __name__ == '__main__':
    model_ResNet = main()
    
    
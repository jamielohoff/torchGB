import os
import time
import yaml
import git 

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchdata.stateful_dataloader import StatefulDataLoader

from datasets.distributed import split_dataset_by_node


def load_config(file: str) -> dict:
    with open(file) as stream:
        try:
            conf = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            
    return conf


def commit_to_experiments_branch(project_root: str):
    # Wait for your turn to access the repository
    while os.path.exists(os.path.join(project_root, ".git", "index.lock")):
        print("Waiting for the index.lock file to be released.")
        time.sleep(5)
    
    # Open the repository
    repo = git.Repo(project_root)    
    
    # Get the experiments branch
    experiments_branch = repo.branches["experiments"]
    
    try:       
        # Switch to the main branch
        repo.git.checkout("main")  
        
        if repo.is_dirty(untracked_files=True): 
            # Add all changes to the staging area
            repo.git.add(all=True)

            # Commit the changes to the experiments branch
            repo.git.commit(message="Auto-commit to `experiment` branch.")

        # Checkout the experiments branch
        repo.git.checkout("experiments")
        
        # Accept incoming changes on the new branch
        repo.git.merge("main", X="theirs")
            
        # Push the changes to the remote repository
        repo.remote().push(experiments_branch)
        
    except Exception as e:
        print(f"Error: {e}")
        print("An error occurred while committing to the experiments branch.")
        
    # Get the commit hash values
    commit_hash = repo.head.commit.hexsha
    print(f"Commit hash: {commit_hash}")
    
    # Checkout the  branch
    repo.git.checkout("main")
        
    return commit_hash


# Creating dataloaders
def get_dataloader(dataset, 
                    rank: int, 
                    world_size: int, 
                    batchsize: int,
                    num_workers: int = 8, 
                    prefetch_factor: int = 4, 
                    stateful: bool = False) -> DataLoader:
    node_data = split_dataset_by_node(dataset, rank=rank, world_size=world_size)
    loader = StatefulDataLoader if stateful else DataLoader
    return loader(node_data, 
                pin_memory=True,
                batch_size=batchsize,
                num_workers=num_workers,
                prefetch_factor=prefetch_factor)


def angle_between_tensors(tensor1, tensor2):
  """
  Computes the angle between two pytorch tensors.

  Args:
    tensor1: The first pytorch tensor.
    tensor2: The second pytorch tensor.

  Returns:
    The angle between the two tensors in radians.
  """
  return torch.atan2(torch.sum(tensor1 * tensor2), torch.sum(tensor1) * torch.sum(tensor2))


def cosine_similarity(tensor1, tensor2):
  """
  Computes the cosine similarity between two PyTorch tensors.

  Args:
    tensor1: The first pytorch tensor.
    tensor2: The second pytorch tensor.

  Returns:
    The cosine similarity between the two tensors.
  """
  return torch.sum(tensor1 * tensor2) / (torch.norm(tensor1) * torch.norm(tensor2))

def save_model(logger, file_path: str, seed: int, model: nn.Module, 
               optimizer: optim.Optimizer, scheduler):
    logger.debug(f"Saving model weights, optimizers and seed under {file_path}.")
    param_dict = {"seed": seed,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict()}
    torch.save(param_dict, file_path)


def load_model(logger, file_path: str, rank: int, seed: int, model: nn.Module, 
               optimizer: optim.Optimizer, scheduler):
    assert os.path.exists(file_path), f"File {file_path} does not exist."
    logger.debug(f"Loading model weights, optimizer and seed from {file_path}.")
    
    map_loc = torch.device(rank)

    # Load the model weights
    state_dict = torch.load(file_path, map_location=map_loc)
    
    seed = int(state_dict["seed"])
    torch.manual_seed(seed)
    model.load_state_dict(state_dict["model"])
    optimizer.load_state_dict(state_dict["optimizer"])
    scheduler.load_state_dict(state_dict["scheduler"])
    
    
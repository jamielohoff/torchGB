import os
import time
import yaml
import git 

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


def load_model_layers(state_dict, model, optimizer, layer_names):
    model_dict = state_dict["model"]
    optim_dict = state_dict["optimizer"]
    new_model_dict = model.state_dict()
    new_optim_dict = optimizer.state_dict()
    for key in model_dict.keys():
        load_layer = any([layer_name in key for layer_name in layer_names])
        if key in new_model_dict.keys() and load_layer:
            new_model_dict[key] = model_dict[key]
            new_optim_dict[key] = optim_dict[key]
    model.load_state_dict(new_model_dict)
    optimizer.load_state_dict(new_optim_dict)
    
    
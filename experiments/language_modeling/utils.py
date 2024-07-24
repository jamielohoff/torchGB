from torch.utils.data import DataLoader
from torchdata.stateful_dataloader import StatefulDataLoader

from datasets.distributed import split_dataset_by_node


# Creating dataloaders
def get_dataloader(dataset, 
                    rank: int, 
                    world_size: int, 
                    num_workers: int = 8, 
                    prefetch_factor: int = 4, 
                    batchsize: int = 16, 
                    stateful: bool = False) -> DataLoader:
    node_data = split_dataset_by_node(dataset, rank=rank, world_size=world_size)
    loader = StatefulDataLoader if stateful else DataLoader
    return loader(node_data, 
                pin_memory=True,
                batch_size=batchsize,
                num_workers=num_workers,
                prefetch_factor=prefetch_factor)
    
    
# val_num_words = sum(len(line["text"].split(" ")) for line in oscar_dataset["validation"])
# test_num_words = sum(len(line["text"].split(" ")) for line in oscar_dataset["test"])
# print(f"Number of words in val: {val_num_words}\n"
#       f"Number of words in test: {test_num_words}")


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
    
    
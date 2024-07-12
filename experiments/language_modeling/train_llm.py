import os
import sys
import copy
import time
import argparse

from tqdm import tqdm
import wandb
import numpy as np 

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP

from datasets import load_dataset
from datasets.distributed import split_dataset_by_node
from transformers import AutoTokenizer

from torchGB import GenomicBottleneck
from _transformer import GPT, predict_sequence, generate_square_subsequent_mask
from config import load_config

parser = argparse.ArgumentParser()

parser.add_argument("--gpus", type=str, default="0", help="Which GPUs to use.")

parser.add_argument("--seed", type=int, default=0, help="Random seed of the experiment.")

parser.add_argument("--name", type=str, default="Test", help="Name of the experiment.")

parser.add_argument("--language", type=str, default="en", help="Which language to use.")

parser.add_argument("--batchsize", type=int, default=16, help="Training batchsize.")

parser.add_argument("--compression_size", type=int, default=64, 
                    help="Size of the hidden layer of the genomic bottleneck MLPs.")

parser.add_argument("--load_gnets", type=str, default=None, 
                    help="Path to the gnet weights.")

parser.add_argument("--load_model", type=str, default=None,
                    help="Path to the model weights.")

parser.add_argument("--wandb", type=str, default="run", help="Wandb mode.")

parser.add_argument("--wandb_id", type=str, default=None,
                    help="Wandb id to resume a crashed run.")

parser.add_argument("--disable_gnets", action="store_false",
                    help="Use genomic bottleneck compression?")

parser.add_argument("--init_with_gnets", action="store_false",
                    help="Initialize the model with the weights predicted by the gnets.")

parser.add_argument("--checkpoint_model", action="store_false",
                    help="Whether to store the model weights and optimizer.")

parser.add_argument("--ignore_layers", type=str, default="",
                    help="Which layers to ignore when compressing with the genomic bottleneck.")

parser.add_argument("--prompt", type=str, default="Hello World!",
                    help="Test prompt for LM.")

parser.add_argument("--transfer_layers", type=str, default="",
                    help="Which layers to transfer from the loaded model.")

args = parser.parse_args()


# Setup of global variables
torch.manual_seed(args.seed)
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
os.environ["TOKENIZERS_PARALLELISM"] = "true"


# Multiprocessing setup
dist.init_process_group(backend="nccl")
rank = dist.get_rank()
world_size = dist.get_world_size()
print("Rank:", rank, "World Size:", world_size)


# Initialize experiment hyperparameters
experiment_config = load_config("experiment_config.yaml")
COMPRESSION_LAYER_SIZE = args.compression_size
EPOCHS = experiment_config["epochs"]
BATCHSIZE = args.batchsize
SEQ_LEN = experiment_config["model"]["seq_len"]
LOG_INTERVAL = experiment_config["log_interval"]
VAL_INTERVAL = experiment_config["val_interval"]


# Initialize the data directory
base_config = load_config("base_config.yaml")
prefix = base_config["data_dirs"]["prefix"]
data_dir = prefix + base_config["data_dirs"][args.language]
print("Data directory:", data_dir)
cache_dir = base_config["cache_dir"]
print("Cache directory:", cache_dir)


# Determine mode of the experiment and set name
init_with_gnets = not args.init_with_gnets
enable_gnets = args.disable_gnets if not init_with_gnets else False
if init_with_gnets:
    print(f"Initializing with G-Net weights: {args.load_gnets}")
    assert args.load_gnets is not None, "Please enter a path to the weights for the G-Nets."
experiment_name = "GBT_" + args.name if enable_gnets else args.name
print("Starting experiment", experiment_name)


# Load and create datasets
train_dataset = load_dataset("arrow", 
                            data_dir=data_dir,
                            cache_dir=cache_dir, 
                            split="train[:94%]")

val_dataset = load_dataset("arrow", 
                            data_dir=data_dir,
                            cache_dir=cache_dir, 
                            split="train[94%:95%]")

test_dataset = load_dataset("arrow", 
                            data_dir=data_dir, 
                            cache_dir=cache_dir,
                            split="train[95%:]")


# val_num_words = sum(len(line["text"].split(" ")) for line in oscar_dataset["validation"])
# test_num_words = sum(len(line["text"].split(" ")) for line in oscar_dataset["test"])
# print(f"Number of words in val: {val_num_words}\n"
#       f"Number of words in test: {test_num_words}")


# Initialize the tokenizer
tokenizer_path = os.path.join(base_config["tokenizer_path"], "oscar_2301_" + args.language)
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
tokenizer.pad_token = tokenizer.eos_token


# Initialize the model and optimizers
model = GPT(**experiment_config["model"]).to(rank)
model = DDP(model, device_ids=[rank], output_device=rank)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=experiment_config["lr"])
scheduler = None


# Dealing with custom model loading
if args.load_model is not None:
    if args.transfer_layers == "":
        try:
            model.load_state_dict(torch.load(args.load_model, map_location=torch.device(rank))["model"])
            optimizer.load_state_dict(torch.load(args.load_model, map_location=torch.device(rank))["optimizer"])
            print("Loaded model weights from", args.load_model)
        except:
            raise f"No model existing under {args.load_model}"
    else:
        try:
            model_dict = torch.load(args.load_model, map_location=torch.device(rank))["model"]
            optim_dict = torch.load(args.load_model, map_location=torch.device(rank))["optimizer"]
            new_model_dict = model.state_dict()
            new_optim_dict = optimizer.state_dict()
            for key in model_dict.keys():
                load_layer = any([layer_name in key for layer_name in args.transfer_layers.split(",")])
                if key in new_model_dict.keys():
                    new_model_dict[key] = model_dict[key]
                    new_optim_dict[key] = optim_dict[key]
            model.load_state_dict(new_model_dict)
            optimizer.load_state_dict(new_optim_dict)
            print("Loaded model weights from", args.load_model, "for weights", args.transfer_layers)
        except:
            raise f"No model existing under {args.load_model}"
        print("Using scheduler...")
        scheduler = optim.lr_scheduler.LinearLR(optimizer, 1e-2, 1., 2000)
else:
    print("Using scheduler...")
    scheduler = optim.lr_scheduler.LinearLR(optimizer, 1e-2, 1., 2000)


# Creating dataloaders
def get_dataloader(dataset, rank, world_size, num_workers=1, prefetch_factor=1):
    node_data = split_dataset_by_node(dataset, rank=rank, world_size=world_size)
    dataloader = DataLoader(node_data, 
                            pin_memory=True,
                            batch_size=BATCHSIZE,
                            num_workers=num_workers,
                            prefetch_factor=prefetch_factor)
    return dataloader

train_loader = get_dataloader(train_dataset, rank, world_size, num_workers=8, prefetch_factor=2)
val_loader = get_dataloader(val_dataset, rank, world_size, num_workers=4, prefetch_factor=2)
test_loader = get_dataloader(test_dataset, rank, world_size, num_workers=4, prefetch_factor=2)


# Other stuff that is required
best_val_loss = float("inf")
best_model = None
src_mask = generate_square_subsequent_mask(SEQ_LEN).to(rank)

def train(model: nn.Module, gnets: GenomicBottleneck) -> None:
    if enable_gnets: gnets.train()
    total_loss = 0.
    start_time = time.time()
    pbar = enumerate(tqdm(train_loader))
    for batch, data in pbar:
        model.train()
        data = torch.stack(data["input_ids"]).t().to(rank)
        optimizer.zero_grad()

        if enable_gnets:
            gnets.zero_grad()
            gnets.predict_weights(model)

        output = model(data[:, :-1], src_mask)
        loss = criterion(output.view(-1, tokenizer.vocab_size), data[:, 1:].reshape(-1))

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 0.25)
        if enable_gnets: gnets.backward(model)
        
        dist.barrier()
        optimizer.step()
        if scheduler is not None: scheduler.step()
        if enable_gnets: gnets.step()
        
        total_loss += loss.item()
        # Logging
        if batch % LOG_INTERVAL == 0 and batch > 0:
            ms_per_batch = (time.time() - start_time) * 1000 / LOG_INTERVAL
            cur_loss = torch.tensor([total_loss / LOG_INTERVAL]).to(rank)
            dist.barrier()
            dist.all_reduce(cur_loss, op=dist.ReduceOp.AVG)
            cur_loss = cur_loss.cpu().item()
            ppl = np.exp(cur_loss)
            predicted_seq = predict_sequence(args.prompt, tokenizer, model, rank, seq_size=32)
            if rank == 0:
                run.log({"train_loss": cur_loss, "train ppl": ppl})
                print("-" * 89)
                print(predicted_seq)
                print("-" * 89)
                print(f"| epoch {epoch:3d} | {batch:5d} batches | "
                        f"ms/batch {ms_per_batch:5.2f} | "
                        f"loss {cur_loss:5.2f} | ppl {ppl:8.2f}")
            dist.barrier()
            
            total_loss = 0
            start_time = time.time()
        
        # Validation
        if batch % VAL_INTERVAL == 0 and batch > 0:
            val_loss = evaluate(model, val_loader)
            dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)
            val_ppl = np.exp(val_loss.cpu().item()) # use Word-level PPL

            print("-" * 89)
            print("Validation of joint P-Net and G-Net training:")
            print(f"valid loss {float(val_loss):5.2f} | valid ppl {val_ppl:8.2f}")
            if rank == 0:
                run.log({"validation_loss": val_loss, "val ppl": val_ppl})

            global best_val_loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                global best_model
                best_model = copy.deepcopy(model).cpu()
                
                pth = base_config["save_gnets"]
                fname = experiment_name + "_gnets_" + args.language + ".pth"
                fname = os.path.join(pth, fname)
                
                if enable_gnets:
                    print("Saving G-Net weights under", fname)
                    gnets.save(fname) 
                
                if args.checkpoint_model and rank == 0:
                    fname = fname.replace("gnets", "model")
                    print("Saving model weights under", fname)
                    param_dict = {"model": model.state_dict(),
                                "optimizer": optimizer.state_dict()}
                    torch.save(param_dict, fname)
                else:
                    time.sleep(1)
                

def evaluate(model: nn.Module, eval_loader: DataLoader) -> float:
    model.eval() # turn on evaluation mode
    total_loss, norm = 0, 0
    global src_mask
    with torch.no_grad():
        start_time = time.time()
        for data in eval_loader:
            data = torch.stack(data["input_ids"]).t().to(rank)
            seq_len = data.size(1)
            output = model(data[:, :-1], src_mask[:seq_len, :seq_len])
            output_flat = output.view(-1, tokenizer.vocab_size)
            total_loss += criterion(output_flat, data[:, 1:].reshape(-1)).item()
            norm += 1
    print("Validation time:", time.time() - start_time)
    return torch.tensor([total_loss / norm]).to(rank)


# Loop over epochs. Save the model if the validation loss is the best we've seen so far.
ignore_layers = ["encoder.weight", "decoder.weight", "norm", "bias"]
ignore_layers += [f".{l}." for l in args.ignore_layers.split(",")]
print("Ignoring layers:", ignore_layers)
gnets = GenomicBottleneck(model, 
                        COMPRESSION_LAYER_SIZE, 
                        lr=experiment_config["gnets_lr"],
                        ignore_layers=ignore_layers) if (enable_gnets or init_with_gnets) else None


# Load G-Net weights if applicable and predict the weights
if args.load_gnets is not None:
    try:
        print("Loading G-Net weights from", args.load_gnets)
        gnets.load(args.load_gnets)
        print("Predicting weights...")
        with torch.no_grad():
            gnets.predict_weights(model)
    except:
        raise f"No G-Net existing under {args.load_gnets}"


# Delete the G-Nets after initialization if we only initialize the model with them
if init_with_gnets:
    print("Deleting G-Nets...")
    gnets = None
    enable_gnets = False


# Calculate model num_params and compression factor if applicable
num_params = sum(p.numel() for p in model.parameters())
print("Number of model parameters:", num_params)   
compression_factor = gnets.compression(model) if enable_gnets else 1.0
print("G-Net compression:", compression_factor)  


# Initialize the run config
run_config = {"batchsize": BATCHSIZE,
                "language": args.language,
                "compression_layer_size": COMPRESSION_LAYER_SIZE,
                "compression_factor": compression_factor,
                "num_params": num_params,
                "ignored layers": args.ignore_layers,
                "seed": args.seed,
                **experiment_config,
                **base_config}


# Compute initial validation loss
print("Computing initial validation loss...")
val_loss = evaluate(model, val_loader)
dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)
val_ppl = np.exp(val_loss.cpu().item()) # use Word-level PPL
print("Validation loss:", val_loss, "Validation PPL:", val_ppl)


# Initialize Wandb on rank 0
if rank == 0:
    wandb.login(key="local-84c6642fa82dc63629ceacdcf326632140a7a899", 
                host="https://wandb.fz-juelich.de")
    run_name = experiment_name + "_" + args.language
    run = wandb.init(entity="ja-lohoff", 
                    project="GenomicBottleneck", 
                    group="oscar", 
                    config=run_config, 
                    mode=args.wandb,
                    dir=base_config["wandb_path"],
                    id=args.wandb_id,
                    resume="allow",
                    name=run_name)

    run.log({"validation_loss": val_loss, "val ppl": val_ppl})
    

# Actual training loop
for epoch in range(1, EPOCHS + 1):
    dist.barrier()
    print("New epoch...")
    train(model, gnets)
      
  
# Evaluate the best model on the test dataset
test_loss = evaluate(best_model.to(rank), test_loader) 
test_ppl = np.exp(test_loss) # word-level PPL
print("=" * 89)
print(f"| End of training | test loss {test_loss:5.2f} | test ppl {test_ppl:8.2f}")
print("=" * 89)
run.finish()
dist.destroy_process_group()


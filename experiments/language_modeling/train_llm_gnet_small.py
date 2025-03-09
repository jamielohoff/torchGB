import os
import sys
import time
import argparse
from functools import partial

from loguru import logger
from tqdm import tqdm
import wandb

import numpy as np 

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from datasets import load_dataset
from transformers import AutoTokenizer

from torchGB import GenomicBottleneck
from _transformer import GPT, generate_square_subsequent_mask, predict_sequence
from utils import get_dataloader, load_model_layers, load_config, commit_to_experiments_branch

parser = argparse.ArgumentParser()

parser.add_argument("--gpus", type=str, default="0", help="Which GPUs to use.")

parser.add_argument("--seed", type=int, default=0, help="Random seed of the experiment.")

parser.add_argument("--name", type=str, default="Test", help="Name of the experiment.")

parser.add_argument("--language", type=str, default="en", help="Which language to use.")

parser.add_argument("--batchsize", type=int, default=16, help="Training batchsize.")

parser.add_argument("--load_gnets", type=str, default=None, 
                    help="Path to the gnet weights.")

parser.add_argument("--load_model", type=str, default=None,
                    help="Path to the model weights.")

parser.add_argument("--checkpoint", action="store_true",
                    help="Whether to store the model weights and optimizer.")

parser.add_argument("--ignore_layers", type=str, default="",
                    help="Which layers to ignore when compressing with the genomic bottleneck.")

parser.add_argument("--transfer_layers", type=str, default="",
                    help="Which layers to transfer from the loaded model.")

parser.add_argument("--no_commit", action="store_false", 
                    help="Whether to create a commit that belongs to the experiment.")

parser.add_argument("--log_level", type=str, default="INFO", help="Set log level.")

parser.add_argument("--load_dataloader_state", action="store_true",
                    help="Whether to load the dataset state from a checkpoint.")

parser.add_argument("--scheduler", action="store_true",
                    help="Whether to use a warmup schedule or not.")

parser.add_argument("--config", type=str, default="experiment_config.yml",
                    help="Experiment configuration.")

parser.add_argument("--wandb", type=str, default="online",
                    help="Experiment configuration.")

args = parser.parse_args()


# Setup of logging
logger.remove()
logger.add(sys.stderr, level=args.log_level)


# Setup of global variables
torch.manual_seed(args.seed)
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus


# Multiprocessing setup
dist.init_process_group(backend="nccl")
rank = dist.get_rank()
world_size = dist.get_world_size()
logger.debug(f"Rank: {rank}, World Size: {world_size}")


# Initialize experiment hyperparameters
experiment_config = load_config("config/" + args.config)
EPOCHS = experiment_config["epochs"]
BATCHSIZE = args.batchsize
SEQ_LEN = experiment_config["model"]["seq_len"]
VOCAB_SIZE = experiment_config["model"]["vocab_size"]
LOG_INTERVAL = experiment_config["log_interval"]
VAL_INTERVAL = experiment_config["val_interval"]
SEED = args.seed
global_step = 0


# Initialize the data directory
base_config = load_config("config/base_config.yml")
prefix = base_config["data_dirs"]["prefix"]
data_dir = prefix + base_config["data_dirs"][args.language]
logger.debug(f"Data directory: {data_dir}")
cache_dir = base_config["dirs"]["cache"]
logger.debug(f"Cache directory: {cache_dir}")


# Commit the current codebase to the experiments branch
if rank == 0 and args.no_commit:
    project_root = base_config["dirs"]["project_root"]
    logger.info(f"Committing {project_root} on branch `experiments`")
    commit_hash = commit_to_experiments_branch(project_root)
else:
    commit_hash = "placeholder_hash"
    

# Set the model and gnet checkpoint paths
experiment_name = "GBT_" + args.name
fname = experiment_name + "_gnets_" + args.language + ".pth"
GNET_CHCKPT_PATH  = os.path.join(base_config["dirs"]["save_gnets"], fname)
fname = experiment_name + "_model_" + args.language + ".pth"
MODEL_CHCKPT_PATH  = os.path.join(base_config["dirs"]["save_model"], fname)


# Check if the files already exist
if rank == 0 and args.checkpoint:
    if os.path.isfile(GNET_CHCKPT_PATH):
        logger.warning(f"File {GNET_CHCKPT_PATH} already exists.")
    if os.path.isfile(MODEL_CHCKPT_PATH):
        logger.warning(f"File {MODEL_CHCKPT_PATH} already exists.")


# Load and create datasets
train_dataset = load_dataset("wikimedia/wikipedia", "20231101." + args.language, 
                             cache_dir=cache_dir, split="train[:95%]")

val_dataset = load_dataset("wikimedia/wikipedia", "20231101." + args.language, 
                           cache_dir=cache_dir, split="train[95%:96%]")

val_dataset = val_dataset.take(experiment_config["val_dataset_len"]//world_size)

test_dataset = load_dataset("wikimedia/wikipedia", "20231101." + args.language, 
                            cache_dir=cache_dir, split="train[95%:96%]")

# Initialize the tokenizer
tokenizer_path = os.path.join(base_config["dirs"]["tokenizer"], "wikipedia_" + args.language)
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
logger.debug(f"Vocab size: {tokenizer.vocab_size}")


def tokenize_function(element):
    outputs = tokenizer(element["text"], truncation=True, max_length=SEQ_LEN+1,
                        return_overflowing_tokens=True, return_length=True)
    
    input_batch = []
    for length, input_ids in zip(outputs["length"], outputs["input_ids"]):
        if length == SEQ_LEN+1: # Drops all sentences that are not long enough...
            input_batch.append(input_ids)
    return {"input_ids": input_batch}


rm_cols = train_dataset.column_names
tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True, 
                                            num_proc=16, remove_columns=rm_cols)
tokenized_train_dataset = tokenized_train_dataset.shuffle(seed=SEED)

tokenized_val_dataset = val_dataset.map(tokenize_function, batched=True, 
                                        num_proc=16, remove_columns=rm_cols)

tokenized_test_dataset = test_dataset.map(tokenize_function, batched=True, 
                                          num_proc=16, remove_columns=rm_cols)


num_batches = len(tokenized_train_dataset)*EPOCHS//(BATCHSIZE*world_size) + 1
logger.debug(f"Number of batches in train dataset: {num_batches}")

train_loader = get_dataloader(tokenized_train_dataset, rank, world_size, 
                              BATCHSIZE, stateful=True)

# Initialize the model and optimizers
model = GPT(**experiment_config["model"]).to(rank)
model = DDP(model, device_ids=[rank], output_device=rank)
gnets = GenomicBottleneck(model, **experiment_config["gnets"])

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=experiment_config["lr"])
scheduler = None

if args.scheduler:
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, 
                                              max_lr=experiment_config["lr"], 
                                              pct_start=0.2, div_factor=250,
                                              final_div_factor=1000,
                                              total_steps=num_batches)


# Dealing with custom model loading
# TODO outsource this into an additional lib!
if args.load_model is not None:
    assert os.path.exists(args.load_model), f"File {args.load_model} does not exist."
    map_loc = torch.device(rank)
    cpu_loc = torch.device("cpu")
    
    # Load dataset state if applicable
    if args.load_dataloader_state:
        state_dict = torch.load(args.load_model, map_location=cpu_loc)
        SEED = state_dict["seed"]
        
        tokenized_train_dataset.shuffle(seed=SEED)
        train_loader = get_dataloader(tokenized_train_dataset, rank, world_size, 
                                      BATCHSIZE, stateful=True)
        train_loader.load_state_dict(state_dict["dataloader"])
        
        logger.info(f"Loaded dataloader state from {args.load_model} "
                    f"with random seed {SEED}")

    # Load the model weights
    state_dict = torch.load(args.load_model, map_location=map_loc)
    if args.transfer_layers == "":
        model.load_state_dict(state_dict["model"])
        optimizer.load_state_dict(state_dict["optimizer"])
        logger.debug(f"Loaded model weights from {args.load_model}")
    else:
        layer_names = args.transfer_layers.split(",")
        load_model_layers(state_dict, model, optimizer, layer_names)
        logger.debug(f"Loaded model weights from {args.load_model} "
                     f"for weights {args.transfer_layers}")


# Which layers to ignore when assigning gnets
ignore_layers = [f".{l}." for l in args.ignore_layers.split(",")]
experiment_config["gnets"]["ignore_layers"] += ignore_layers


# Load g-net weights if applicable and predict the weights
if args.load_gnets is not None:
    assert os.path.exists(args.load_gnets), f"File {args.load_gnets} does not exist."
    logger.debug(f"Loading g-net weights from {args.load_gnets}.")
    gnets.load(args.load_gnets)
    logger.debug("Predicting weights...")
    with torch.no_grad():
        gnets.predict_weights()


# Calculate model num_params and compression factor if applicable
num_params = sum(p.numel() for p in model.parameters())
compression_factor = gnets.compression()


# Other stuff that is required
best_val_loss = float("inf")
mask = generate_square_subsequent_mask(SEQ_LEN-1).to(rank)


# Training function
def train(model: nn.Module, gnets: GenomicBottleneck) -> None:
    gnets.train()
    total_loss = 0
    start_time = time.time()
    
    for data in tqdm(train_loader):
        model.train()
        global global_step
        global_step += 1
        data = torch.stack(data["input_ids"]).t().to(rank)
        
        # Zeroing out the gradients in the p-net and g-net optimizers
        optimizer.zero_grad()
        gnets.zero_grad()
        gnets.predict_weights() # implicitly updates the model weights!
        
        tokens = data[:, :SEQ_LEN-1]
        output = model(tokens, mask)
        
        # Auto-regressive, unsupervised loss
        logits = output.view(-1, VOCAB_SIZE)
        labels = data[:, 1:SEQ_LEN].reshape(-1)
        loss = criterion(logits, labels)

        # Backpropagate the error through the p-net and then through the g-nets
        loss.backward()
        gnets.backward()
        
        nn.utils.clip_grad_norm_(model.parameters(), 0.25) # what value to use here?
        
        # Do a gradient-descent step with the p-nets and then the g-nets
        optimizer.step() # Need to still update the parameters that have no g-nets attached!
        if scheduler is not None: scheduler.step()
        gnets.step()

        total_loss += loss.item()
        # Logging
        if global_step % LOG_INTERVAL == 0:
            ms_per_batch = (time.time() - start_time) * 1000 / LOG_INTERVAL
            train_loss = torch.tensor([total_loss / LOG_INTERVAL]).to(rank)
            dist.barrier()
            dist.all_reduce(train_loss, op=dist.ReduceOp.AVG)
            train_loss = train_loss.cpu().item()
            train_ppl = np.exp(train_loss)

            if rank == 0:
                logger.debug(f"epoch {epoch:3d} | {global_step:5d} batches | "
                             f"ms/batch {ms_per_batch:5.2f} | "
                             f"loss {train_loss:5.2f} | ppl {train_ppl:8.2f}")
                run.log({"train_loss": train_loss, "train ppl": train_ppl})
            dist.barrier()
            
            total_loss = 0
            start_time = time.time()

        # Validation
        if global_step % VAL_INTERVAL == 0:
            val_loss = evaluate(model, val_loader)
            dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)
            val_ppl = np.exp(val_loss.cpu().item()) # use Word-level PPL
               
            test_seq = predict_sequence("I seek glory", tokenizer, model, rank)
            
            if rank == 0:
                logger.info("Output sequence: " + test_seq)
                logger.info(f"validation loss {float(val_loss):5.2f} | "
                            f"validation ppl {val_ppl:8.2f}")
                run.log({"validation_loss": val_loss, "val ppl": val_ppl})
                
            global best_val_loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
   
                if args.checkpoint:
                    logger.debug(f"Saving g-net weights under {GNET_CHCKPT_PATH}.")
                    gnets.save(GNET_CHCKPT_PATH) 
                
                    if rank == 0:
                        logger.debug(f"Saving model weights, optimizer,"
                                     f"seed and dataset state under {MODEL_CHCKPT_PATH}.")
                        param_dict = {"seed": SEED,
                                      "model": model.state_dict(),
                                      "optimizer": optimizer.state_dict(),
                                      "dataloader": train_loader.state_dict()}
                        torch.save(param_dict, MODEL_CHCKPT_PATH)
                    else:
                        # Put other processes to sleep while logging...good night!
                        time.sleep(1)


# Evaluation function
def evaluate(model: nn.Module, eval_loader) -> torch.Tensor:
    model.eval() # turn on evaluation mode
    total_loss, norm = 0, 0
    global mask
    with torch.no_grad():
        start_time = time.time()
        for data in eval_loader:
            data = torch.stack(data["input_ids"]).t().to(rank)
            seq_len = min(SEQ_LEN, data.size(1))
            
            tokens = data[:, :seq_len-1]
            _mask = mask[:seq_len-1, :seq_len-1]
            output = model(tokens, _mask)
            
            logits = output.view(-1, VOCAB_SIZE)
            labels = data[:, 1:seq_len].reshape(-1)
            total_loss += criterion(logits, labels).item()
            norm += 1
    logger.info(f"Validation time: {time.time() - start_time}")
    return torch.tensor([total_loss / norm]).to(rank)


# Compute initial validation loss
val_loader = get_dataloader(tokenized_val_dataset, rank, world_size, BATCHSIZE)
val_loss = evaluate(model, val_loader)
dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)
val_ppl = np.exp(val_loss.cpu().item()) # use Word-level PPL
val_loader = get_dataloader(tokenized_val_dataset, rank, world_size, BATCHSIZE)


# Initialize metrics logging on rank 0
if rank == 0:
    logger.info(f"Number of model parameters: {num_params}")   
    logger.info(f"g-net compression: {compression_factor}") 
    logger.info(f"validation loss {float(val_loss):5.2f} | "
                f"validation ppl {val_ppl:8.2f}")
    
    run_config = {"commit_hash": commit_hash,
                  "batchsize": BATCHSIZE,
                  "language": args.language,
                  "compression_factor": compression_factor,
                  "num_params": num_params,
                  "ignored layers": args.ignore_layers,
                  "seed": args.seed,
                  **experiment_config,
                  **base_config}
    
    wandb.login(key=base_config["accounts"]["wandb_key"], 
                host=base_config["accounts"]["wandb_address"])
    
    run_name = experiment_name + "_" + args.language
    log_dir = os.path.join(base_config["dirs"]["log"], run_name)
    
    run = wandb.init(entity=base_config["accounts"]["wandb_user"], 
                     project="GenomicBottleneck", 
                     group="wikipedia", 
                     config=run_config, 
                     mode=args.wandb,
                     dir=base_config["dirs"]["wandb"],
                     name=run_name)

    run.log({"validation_loss": val_loss, "val ppl": val_ppl})


loader = partial(get_dataloader, tokenized_train_dataset, rank, world_size, BATCHSIZE)

# Actual training loop
for epoch in range(EPOCHS):
    dist.barrier()
    train(model, gnets)
    SEED += 1
    
    tokenized_train_dataset = tokenized_train_dataset.shuffle(seed=SEED)
    tokenized_val_dataset = tokenized_val_dataset.shuffle(seed=SEED)
    
    train_loader = loader(stateful=True)
    val_loader = loader()


# Evaluate the best model on the test dataset
# tokenized_test_dataset = tokenized_test_dataset.shuffle(seed=SEED)
# test_loader = get_dataloader(tokenized_test_dataset, rank, world_size, 4*BATCHSIZE)
# model.load_state_dict(torch.load(args.load_model, map_location=torch.device(rank))["model"])

# test_loss = evaluate(model.to(rank), test_loader) 
# test_ppl = np.exp(test_loss) # word-level PPL
# logger.info(f"End of training | test loss {test_loss:5.2f} | test ppl {test_ppl:8.2f}")
dist.destroy_process_group()
run.finish()


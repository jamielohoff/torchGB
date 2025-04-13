import os
import sys
import time
import argparse

from loguru import logger
import wandb

import numpy as np 

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR

from torchGB import GenomicBottleneck, FastGenomicBottleneck
from _transformer import GPT, generate_square_subsequent_mask
from utils import load_model_layers, load_config, commit_to_experiments_branch

from gradient_utils import cosine_similarity, angle_between_tensors

parser = argparse.ArgumentParser()

parser.add_argument("--gpus", type=str, default="0", help="Which GPUs to use.")

parser.add_argument("--seed", type=int, default=0, help="Random seed of the experiment.")

parser.add_argument("--name", type=str, default="Test", help="Name of the experiment.")

parser.add_argument("--batchsize", type=int, default=12, help="Training batchsize.")

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

parser.add_argument("--config", type=str, default="owt_config.yml",
                    help="Experiment configuration.")

parser.add_argument("--wandb", type=str, default="online",
                    help="Experiment configuration.")

args = parser.parse_args()


# Setup of logging
logger.remove()
logger.add(sys.stderr, level=args.log_level)


# Setup of global variables
torch.manual_seed(args.seed)


# Multiprocessing setup
dist.init_process_group(backend="nccl")
rank = dist.get_rank()
local_rank = int(os.environ["LOCAL_RANK"])
world_size = dist.get_world_size()
logger.debug(f"Rank: {rank}, World Size: {world_size}")


# Initialize experiment hyperparameters
experiment_config = load_config("config/" + args.config)
LR = experiment_config["lr"]
BATCHSIZE = args.batchsize
SEQ_LEN = experiment_config["model"]["seq_len"]
VOCAB_SIZE = experiment_config["model"]["vocab_size"]
LOG_INTERVAL = experiment_config["log_interval"]
VAL_INTERVAL = experiment_config["val_interval"]
SEED = args.seed
MAX_ITERS = experiment_config["max_iters"]
WARMUP_ITERS = experiment_config["warmup_iters"]
EVAL_ITERS = experiment_config["val_iters"]
GRADIENT_ACCUMULATION_STEPS = experiment_config["gradient_accumulation_steps"]

TOTAL_NUM_TOKENS = SEQ_LEN * BATCHSIZE * GRADIENT_ACCUMULATION_STEPS * MAX_ITERS / 1e9
TOKENS_PER_STEP = BATCHSIZE * GRADIENT_ACCUMULATION_STEPS * world_size * SEQ_LEN

# Initialize the data directory
base_config = load_config("config/base_config.yml")
prefix = base_config["data_dirs"]["prefix"]
data_dir = os.path.join("/Users/grieser/Projects/bla", "data", "openwebtext")
cache_dir = base_config["dirs"]["cache"]

if rank == 0:
    logger.debug(f"Total number of tokens in training: {TOTAL_NUM_TOKENS:.2f}B")
    logger.debug(f"Tokens per step: {TOKENS_PER_STEP}")
    logger.debug(f"Data directory: {data_dir}")
    logger.debug(f"Cache directory: {cache_dir}")


# Commit the current codebase to the experiments branch
if rank == 0 and args.no_commit:
    project_root = base_config["dirs"]["project_root"]
    logger.info(f"Committing {project_root} on branch `experiments`")
    commit_hash = commit_to_experiments_branch(project_root)
else:
    commit_hash = "placeholder_hash"
    

# Set the model and gnet checkpoint paths
experiment_name = "GBT_OWT_" + args.name
fname = experiment_name + "_gnets_" + ".pth"
GNET_CHCKPT_PATH  = os.path.join(base_config["dirs"]["save_gnets"], fname)
fname = experiment_name + "_model_" + ".pth"
MODEL_CHCKPT_PATH  = os.path.join(base_config["dirs"]["save_model"], fname)


# Check if the files already exist
if rank == 0 and args.checkpoint:
    if os.path.isfile(GNET_CHCKPT_PATH):
        logger.warning(f"File {GNET_CHCKPT_PATH} already exists.")
    if os.path.isfile(MODEL_CHCKPT_PATH):
        logger.warning(f"File {MODEL_CHCKPT_PATH} already exists.")


# Load and create datasets
train_data = np.memmap(os.path.join(data_dir, "train.bin"), dtype=np.uint16, mode="r")
val_data = np.memmap(os.path.join(data_dir, "val.bin"), dtype=np.uint16, mode="r")


# TODO: This needs a stateful dataloader because otherwise we may oversample stuff...
def get_batch(split: str):
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - SEQ_LEN, (BATCHSIZE,))
    x = torch.stack([torch.from_numpy((data[i:i+SEQ_LEN]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+SEQ_LEN]).astype(np.int64)) for i in ix])
    
    # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
    x, y = x.pin_memory().to(local_rank, non_blocking=True), y.pin_memory().to(local_rank, non_blocking=True)
    return x, y


# Initialize the model and optimizers
model = GPT(**experiment_config["model"]).to(local_rank)
model = DDP(model, device_ids=[local_rank], output_device=local_rank)
# gnets = GenomicBottleneck(model, **experiment_config["gnets"])

# Communicate local rank to all other ranks
global_local_rank = torch.tensor([rank, local_rank], dtype=torch.int32).to(local_rank)
local_rank_tensor = torch.zeros(2*world_size, dtype=torch.int32).to(local_rank)

dist.all_gather_into_tensor(local_rank_tensor, global_local_rank)
dist.barrier()

# Assemble local ranks into a dictionary
local_rank_dict = {}
for i in range(world_size):
    idx = local_rank_tensor[2*i].cpu().item()
    local_rank_dict[idx] = local_rank_tensor[2*i+1].cpu().item()

if rank == 0:
    logger.debug(f"Local Rank Dictionary: {local_rank_dict}")

# Setting up the loss and optimizers
criterion = nn.CrossEntropyLoss()
# optimizer = optim.AdamW(model.parameters(), lr=LR, fused=True, weight_decay=0.1)
optimizer = optim.SGD(model.parameters(), lr=LR, fused=True)
scheduler = None


# NOTE: for larger models, the scheduling should be used!
# TODO apply scheduler to the gnet as well!

def make_scheduler(opt, lr):
    linear_increase = LinearLR(opt, start_factor=0.1,end_factor=1.,
                               total_iters=WARMUP_ITERS)
    cosine_annealing = CosineAnnealingLR(opt, eta_min=lr*0.1,
                                         T_max=MAX_ITERS-WARMUP_ITERS)
    scheduler = SequentialLR(opt, [linear_increase, cosine_annealing],
                             milestones=[WARMUP_ITERS])
    return scheduler


if args.scheduler:
    scheduler = make_scheduler(optimizer, LR)
    gnets = GenomicBottleneck(model, local_rank_dict, scheduler=make_scheduler, **experiment_config["gnets"])
else:
    gnets = GenomicBottleneck(model, local_rank_dict, **experiment_config["gnets"])

# Dealing with custom model loading
# TODO outsource this into an additional lib!
if args.load_model is not None:
    assert os.path.exists(args.load_model), f"File {args.load_model} does not exist."
    map_loc = torch.device(local_rank)
    cpu_loc = torch.device("cpu")
    
    # Load dataset state if applicable
    if args.load_dataloader_state:
        state_dict = torch.load(args.load_model, map_location=cpu_loc)
        SEED = state_dict["seed"]
        
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
best_val_loss = 1e9
mask = generate_square_subsequent_mask(SEQ_LEN).to(local_rank)
layer = 0

# Training function
def train(model: nn.Module, gnets: GenomicBottleneck) -> None:
    global mask
    gnets.train()
    total_loss = 0
    start_time = time.time()
    
    new_params = model.module.transformer_encoder.layers[layer].self_attn.in_proj_weight.data.clone()
    old_params = model.module.transformer_encoder.layers[layer].self_attn.in_proj_weight.data.clone()
    pnet_grad = torch.ones_like(new_params)
    
    for iter in range(1, MAX_ITERS+1):
        model.train()
        
        # Zeroing out the gradients in the p-net and g-net optimizers
        optimizer.zero_grad(set_to_none=True)
        gnets.zero_grad()
        gnets.predict_weights() # implicitly updates the model weights!
        
        new_params = model.module.transformer_encoder.layers[layer].self_attn.in_proj_weight.data.clone()
        dp = old_params - new_params
        if rank == 0:
            cos_sim = cosine_similarity(dp, pnet_grad)
            run.log({"cos_sim": cos_sim})
            logger.debug(f"Cosine similarity: {cos_sim}")
        
        for accum_step in range(GRADIENT_ACCUMULATION_STEPS):
            model.require_backward_grad_sync = (accum_step == GRADIENT_ACCUMULATION_STEPS - 1)
            
            tokens, next_tokens = get_batch("train")
            
            output = model(tokens, mask)
            logits = output.view(-1, VOCAB_SIZE)
            
            # Auto-regressive, unsupervised loss
            loss = criterion(logits, next_tokens.flatten())
            loss /= GRADIENT_ACCUMULATION_STEPS

            # Backpropagate the error through the p-net and then through the g-nets
            loss.backward()
            
        # Access gradient of p-net weights
        old_params = model.module.transformer_encoder.layers[layer].self_attn.in_proj_weight.data.clone()
        pnet_grad = model.module.transformer_encoder.layers[layer].self_attn.in_proj_weight.grad.data
            
        gnets.backward()
        
        nn.utils.clip_grad_norm_(model.parameters(), 1.0) # what value to use here?
        
        # Do a gradient-descent step with the p-nets and then the g-nets
        optimizer.step() # Need to still update the parameters that have no g-nets attached!
        if scheduler is not None: scheduler.step()
        gnets.step()

        total_loss += GRADIENT_ACCUMULATION_STEPS*loss.item()
        # Logging
        if iter % LOG_INTERVAL == 0:
            ms_per_batch = (time.time() - start_time) * 1000 / (LOG_INTERVAL*GRADIENT_ACCUMULATION_STEPS)
            train_loss = torch.tensor([total_loss / LOG_INTERVAL]).to(local_rank)
            dist.barrier()
            dist.all_reduce(train_loss, op=dist.ReduceOp.AVG)
            train_loss = train_loss.cpu().item()
            train_ppl = np.exp(train_loss)

            if rank == 0:
                logger.debug(f"iter {iter:3d} | "
                             f"ms/batch {ms_per_batch:5.2f} | "
                             f"loss {train_loss:5.2f} | ppl {train_ppl:8.2f}")
                run.log({"train_loss": train_loss, "train ppl": train_ppl})
            dist.barrier()
            
            total_loss = 0.
            start_time = time.time()

        # Validation
        if iter % VAL_INTERVAL == 0:
            val_loss = evaluate(model)
            dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)
            val_ppl = np.exp(val_loss.cpu().item()) # use Word-level PPL
            
            if rank == 0:
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
                                      "iter": iter,
                                      "model": model.state_dict(),
                                      "optimizer": optimizer.state_dict()}
                        torch.save(param_dict, MODEL_CHCKPT_PATH)
                    else:
                        # Put other processes to sleep while logging...good night!
                        time.sleep(1)


# Evaluation function
@torch.no_grad()
def evaluate(model: nn.Module) -> torch.Tensor:
    model.eval() # turn on evaluation mode
    total_loss, norm = 0, 0
    global mask
    start_time = time.time()
    for iter in range(EVAL_ITERS):
        tokens, next_tokens = get_batch("val")
        
        output = model(tokens, mask)
        logits = output.view(-1, VOCAB_SIZE)
        total_loss += criterion(logits, next_tokens.flatten()).item()
        norm += 1
    logger.info(f"Validation time: {time.time() - start_time}")
    return torch.tensor([total_loss / norm]).to(local_rank)


# Compute initial validation loss
val_loss = evaluate(model)
dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)
val_ppl = np.exp(val_loss.cpu().item()) # use Word-level PPL


# Initialize metrics logging on rank 0
if rank == 0:
    logger.info(f"Number of model parameters: {num_params}")   
    logger.info(f"g-net compression: {compression_factor}") 
    logger.info(f"validation loss {float(val_loss):5.2f} | "
                f"validation ppl {val_ppl:8.2f}")
    
    run_config = {"commit_hash": commit_hash,
                  "batchsize": BATCHSIZE,
                  "compression_factor": compression_factor,
                  "num_params": num_params,
                  "ignored layers": args.ignore_layers,
                  "seed": args.seed,
                  **experiment_config,
                  **base_config}
    
    wandb.login(key=base_config["accounts"]["wandb_key"], 
                host=base_config["accounts"]["wandb_address"])
    
    run_name = experiment_name
    log_dir = os.path.join(base_config["dirs"]["log"], run_name)
    
    run = wandb.init(entity=base_config["accounts"]["wandb_user"], 
                     project="GenomicBottleneck", 
                     group="wikipedia", 
                     config=run_config, 
                     mode=args.wandb,
                     dir=base_config["dirs"]["wandb"],
                     name=run_name)

    run.log({"validation_loss": val_loss, "val ppl": val_ppl})


dist.barrier()
train(model, gnets)

dist.destroy_process_group()
run.finish()


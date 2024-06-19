import os
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

from datasets import load_dataset, DatasetDict
from datasets.distributed import split_dataset_by_node
from transformers import AutoTokenizer, DataCollatorForLanguageModeling

from torchGB import GenomicBottleneck
from _transformer import LanguageModel, predict_sequence, generate_square_subsequent_mask


parser = argparse.ArgumentParser()

parser.add_argument("--name", type=str, 
                    default="Test", help="Name of the experiment.")

parser.add_argument("--gpus", type=str, 
                    default="0", help="Which GPUS to use.")

parser.add_argument("--dataset", type=str, 
                    default="oscar2301", help="Name of the language dataset.")

parser.add_argument("--language", type=str, 
                    default="en", help="Which language to use.")

parser.add_argument("--epochs", type=int,
                    default=50, help="Number of training epochs.")

parser.add_argument("--batchsize", type=int,
                    default=64, help="Training batchsize.")

parser.add_argument("--seq_len", type=int, 
                    default=256, help="Context length of the transformer.")

parser.add_argument("--log_interval", type=int, 
                    default=100, help="Logging every n batches.")

parser.add_argument("--val_interval", type=int, 
                    default=5000, help="Validation every n batches.")

parser.add_argument("--compression_size", type=int, 
                    default=64, help="Size of the hidden layer of the genomic bottleneck MLPs.")

parser.add_argument("--load_gnets", type=str, 
                    default=None, 
                    help="Path to the gnet weights.")

parser.add_argument("--wandb", type=str,
                    default="run", help="Wandb mode.")

parser.add_argument("--disable_gnets", action="store_false",
                    help="Use genomic bottleneck compression?")

parser.add_argument("--init_with_gnets", action="store_false",
                    help="Initialize the model with the weights predicted by the gnets.")

parser.add_argument("--checkpoint_model", action="store_false",
                    help="Whether to store the model weights and optimizer.")

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

COMPRESSION_LAYER_SIZE = args.compression_size
EPOCHS = args.epochs
LR = 2e-4
BATCHSIZE = args.batchsize
EVAL_BATCHSIZE = args.batchsize
SEQ_LEN = args.seq_len+1
LOG_INTERVAL = args.log_interval
init_with_gnets = not args.init_with_gnets
enable_gnets = args.disable_gnets if not init_with_gnets else False
if init_with_gnets:
    print(f"Initializing with gnets weights:{args.load_gnets}")
    assert args.load_gnets is not None, "Please enter a path to the weights for the gnets."


experiment_name = "GBT_" + args.name if enable_gnets else args.name
best_val_loss = float("inf")
best_model = None
   
with open("./token", "r") as f:
    token = f.read()

oscar_dataset = load_dataset("oscar-corpus/OSCAR-2301", 
                            language=args.language, 
                            split="train", 
                            token=token, 
                            trust_remote_code=True,
                            streaming=True)

if args.language == "en":
    language_identifier = ""
elif args.language == "de":
    language_identifier = "-german"
elif args.language == "es":
    language_identifier = "-spanish"

tokenizer = AutoTokenizer.from_pretrained("gpt2", do_lower_case=False)
tokenizer.pad_token = tokenizer.eos_token

M = 2 # TODO do something about this hardcoding and validation
test_dataset = oscar_dataset.take(M*1024)
validation_dataset = oscar_dataset.take(M*1024)
# gather everyone if you want to have a single DatasetDict
oscar_dataset = DatasetDict({"train": oscar_dataset,
                            "test": test_dataset,
                            "validation": validation_dataset})


# val_num_words = sum(len(line["text"].split(" ")) for line in oscar_dataset["validation"])
# test_num_words = sum(len(line["text"].split(" ")) for line in oscar_dataset["test"])
# print(f"Number of words in val: {val_num_words}\n"
#       f"Number of words in test: {test_num_words}")


def tokenize_function(element):
    outputs = tokenizer(element["text"],
                        truncation=True,
                        max_length=SEQ_LEN,
                        return_overflowing_tokens=True,
                        return_length=True)
    input_batch = []
    for length, input_ids in zip(outputs["length"], outputs["input_ids"]):
        if length == SEQ_LEN: # Drops all sentences that are not long enough...
            input_batch.append(input_ids)
    return {"input_ids": input_batch}


# Multiprocessing setup
dist.init_process_group(backend="nccl")
rank = dist.get_rank()
world_size = dist.get_world_size()


rm_cols = oscar_dataset["train"].column_names
train_data = oscar_dataset["train"].map(tokenize_function, 
                                        batched=True, 
                                        remove_columns=rm_cols)
train_data = split_dataset_by_node(train_data, rank=rank, world_size=world_size)
val_data = oscar_dataset["validation"].map(tokenize_function, 
                                            batched=True, 
                                            remove_columns=rm_cols)
val_data = split_dataset_by_node(val_data, rank=rank, world_size=world_size)
test_data = oscar_dataset["test"].map(tokenize_function, 
                                        batched=True, 
                                        remove_columns=rm_cols)
test_data = split_dataset_by_node(test_data, rank=rank, world_size=world_size)


num_tokens = tokenizer.vocab_size # size of vocabulary
embedding_dim = 256 # embedding dimension
hidden_dim = 512 # dimension of the feedforward network model in nn.TransformerEncoder
num_layers = 12 # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
num_heads = 8 # number of heads in nn.MultiheadAttention
dropout = 0.1 # dropout probability
model = LanguageModel(num_tokens, embedding_dim, num_heads, hidden_dim, num_layers, dropout).to(rank)
model = DDP(model, device_ids=[rank], output_device=rank)


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)
# scheduler = optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.975)


train_loader = DataLoader(train_data, 
                        pin_memory=True,
                        batch_size=BATCHSIZE,
                        collate_fn=DataCollatorForLanguageModeling(tokenizer, mlm=False))
val_loader = DataLoader(val_data,
                        pin_memory=True, 
                        batch_size=BATCHSIZE, 
                        collate_fn=DataCollatorForLanguageModeling(tokenizer, mlm=False))
test_loader = DataLoader(test_data, 
                        pin_memory=True,
                        batch_size=BATCHSIZE, 
                        collate_fn=DataCollatorForLanguageModeling(tokenizer, mlm=False))


src_mask = generate_square_subsequent_mask(SEQ_LEN-1).to(rank)
def train(model: nn.Module, gnets: GenomicBottleneck) -> None:
    if enable_gnets: gnets.train()
    total_loss = 0.
    start_time = time.time()
    pbar = enumerate(tqdm(train_loader))
    for batch, data in pbar:
        model.train()
        data = data["input_ids"] .to(rank)
        # print(torch.max(data), num_tokens)
        optimizer.zero_grad()
        # This probably fucks up the training a bit because the weights change
        # slightly which messes with the optimizer momentum, it is propably 
        # responsible for the bumps in the loss curve
        # print("rank", rank, "\nbefore", model.module.transformer_encoder.layers[7].self_attn.in_proj_weight)
        if enable_gnets:
            gnets.zero_grad()
            gnets.predict_weights(model)
        # print("rank", rank, "\nafter", model.module.transformer_encoder.layers[7].self_attn.in_proj_weight)
        output = model(data[:, :-1], src_mask)
        loss = criterion(output.view(-1, num_tokens), data[:, 1:].reshape(-1))

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 0.25)
        if enable_gnets: gnets.backward(model)
        
        optimizer.step()
        if enable_gnets: gnets.step()
        total_loss += loss.item()
        if batch % LOG_INTERVAL == 0 and batch > 0:
            ms_per_batch = (time.time() - start_time) * 1000 / LOG_INTERVAL
            cur_loss = torch.tensor([total_loss / LOG_INTERVAL]).to(rank)
            dist.all_reduce(cur_loss, op=dist.ReduceOp.AVG)
            cur_loss = cur_loss.cpu().item()
            ppl = np.exp(cur_loss)
            if rank == 0:
                wandb.log({"train_loss": cur_loss, 
                            "train ppl": ppl})
            print("-" * 89)
            predict_sequence("Who invented the car?", tokenizer, model, rank, seq_size=32)
            print("-" * 89)
            
            print(f"| epoch {epoch:3d} | {batch:5d} batches | "
                  f"ms/batch {ms_per_batch:5.2f} | "
                  f"loss {cur_loss:5.2f} | ppl {ppl:8.2f}")
            
            total_loss = 0
            start_time = time.time()
        
        if batch % args.val_interval == 0 and batch > 0:
            print(gnets.state_dict())
            # We do not train on the entire dataset per epoch...
            val_loss = evaluate(model, val_loader)
            dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)
            val_ppl = np.exp(val_loss.cpu().item()) # Word-level PPL -- use 
            elapsed = time.time() - epoch_start_time

            print("-" * 89)
            print("After joint P-Net and G-Net training:")
            print(f"| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | "
                f"valid loss {float(val_loss):5.2f} | valid ppl {val_ppl:8.2f}")
            if rank == 0:
                wandb.log({"validation_loss": val_loss, 
                            "val ppl": val_ppl})

            global best_val_loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                global best_model
                best_model = copy.deepcopy(model).cpu()
                
                if enable_gnets:
                    print("Saving gnet weights...")
                    fname = os.path.join(os.getcwd(), "weights/")
                    fname += experiment_name + "_gnets_" + "_" + args.dataset
                    fname += "_"  + args.language + ".pth"
                    gnets.save(fname) 
                    if args.checkpoint_model:
                        print("Saving model weights...")
                        param_dict = {"model": model.state_dict(),
                                    "optimizer": optimizer.state_dict()}
                        torch.save(param_dict, fname.replace("gnets", "model"))


def evaluate(model: nn.Module, eval_loader: DataLoader) -> float:
    model.eval() # turn on evaluation mode
    total_loss, norm = 0, 0
    global src_mask
    with torch.no_grad():
        for data in eval_loader:
            data = data["input_ids"].to(rank)
            seq_len = data.size(1)
            output = model(data[:, :-1], src_mask[:seq_len, :seq_len])
            output_flat = output.view(-1, num_tokens)
            total_loss += criterion(output_flat, data[:, 1:].reshape(-1)).item()
            norm += 1
    return torch.tensor([total_loss / norm]).to(rank)


# Loop over epochs. Save the model if the validation loss is the best
# we've seen so far. Adjust the learning rate after each epoch.
ignore_layers = ["encoder.weight", "decoder.weight", "norm", "bias"]
gnets = GenomicBottleneck(model, COMPRESSION_LAYER_SIZE, ignore_layers=ignore_layers) if (enable_gnets or init_with_gnets) else None

if args.load_gnets is not None:
    gnets.load(args.load_gnets)
    print("Loaded gnet weights from", args.load_gnets)
    
if init_with_gnets:
    gnets.predict_weights(model, rank)
    print("Initial performance with gnet init:")
    val_ppl = evaluate(model, val_loader)
    test_ppl = evaluate(model, test_loader)
    print(f"Validation ppl: {np.exp(val_ppl)}\n"
          f"Test ppl: {np.exp(test_ppl)}")


num_params = sum(p.numel() for p in model.parameters())
print("Number of model parameters:", num_params)   


compression_factor = gnets.compression(model) if enable_gnets else 1.0
print("G-Net compression:", compression_factor)  
run_config = {"epochs": EPOCHS,
                "lr": LR,
                "batchsize": BATCHSIZE,
                "seq_len": SEQ_LEN,
                "vocab_size": num_tokens,
                "embedding_dim": embedding_dim,
                "hidden_dim": hidden_dim,
                "num_layers": num_layers,
                "num_head": num_heads,
                "dropout": dropout,
                "log_interval": LOG_INTERVAL,
                "language": args.language,
                "compression_layer_size": COMPRESSION_LAYER_SIZE,
                "compression_factor": compression_factor,
                "num_params": num_params,}

if rank == 0:
    wandb.login(key="local-84c6642fa82dc63629ceacdcf326632140a7a899", 
                host="https://wandb.fz-juelich.de")
    wandb.init(entity="ja-lohoff", project="GenomicBottleneck", 
                group="oscar", config=run_config, mode=args.wandb)
    wandb.run.name = experiment_name + "_" + args.dataset + "_" + args.language


for epoch in range(1, EPOCHS + 1):
    epoch_start_time = time.time()
    train_data.set_epoch(epoch)
    val_data.set_epoch(epoch)
    train(model, gnets)
    # scheduler.step()
      
  
# Evaluate the best model on the test dataset
test_loss = evaluate(best_model, test_loader) 
test_ppl = np.exp(test_loss) # word-level PPL
print("=" * 89)
print(f"| End of training | test loss {test_loss:5.2f} | "
      f"test ppl {test_ppl:8.2f}")
print("=" * 89)
dist.destroy_process_group()


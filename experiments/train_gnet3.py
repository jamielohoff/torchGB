import os
import copy
import time
import argparse
from typing import Tuple
from tqdm import tqdm
import wandb

import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader

from datasets import load_dataset
from transformers import AutoTokenizer

import numpy as np 
import torch.optim as optim
import scipy.io as sio

import geneNets_2_3_17 as gn
from utils import generate_square_subsequent_mask
from experiments._transformer import LanguageModel, predict_sequence


parser = argparse.ArgumentParser()

parser.add_argument("--name", type=str, 
                    default="Test", help="Name of the experiment.")

parser.add_argument("--dataset", type=str, 
                    default="wiki103", help="Name of the language dataset.")

parser.add_argument("--epochs", type=int,
                    default=50, help="Number of training epochs.")

parser.add_argument("--batchsize", type=int,
                    default=96, help="Training batchsize.")

parser.add_argument("--seq_len", type=int, 
                    default=256, help="Context length of the transformer.")

parser.add_argument("--log_interval", type=int, 
                    default=100, help="Logging every n batches.")

parser.add_argument("--compression_size", type=int, 
                    default=64, help="Size of the hidden layer of the genomic bottleneck MLPs.")

parser.add_argument("--config_path", type=str, 
                    default=os.path.join(os.getcwd(), "config"), 
                    help="Path to the directory containing the configuration files.")

parser.add_argument("--wandb", type=str,
                    default="run", help="Wandb mode.")

parser.add_argument("--disable_gnets", action='store_false',
                    help="Use genomic bottleneck compression?")

args = parser.parse_args()


COMPRESSION_LAYER_SIZE = args.compression_size
EPOCHS = args.epochs
LR = 2e-4
BATCHSIZE = args.batchsize
EVAL_BATCHSIZE = args.batchsize
SEQ_LEN = args.seq_len
LOG_INTERVAL = args.log_interval
enable_gnets = args.disable_gnets


wiki_dataset = load_dataset("wikitext", "wikitext-103-raw-v1")
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased", do_lower_case=False)


def tokenize_function(element):
    outputs = tokenizer(
        element["text"],
        # truncation=True,
        # padding="max_length",
        # max_length=SEQ_LEN,
        # return_overflowing_tokens=True,
        return_length=True
    )
    input_batch = []
    for length, input_ids in zip(outputs["length"], outputs["input_ids"]):
        # if length == SEQ_LEN: # Drops all sentences that are not long enough...
            input_batch.append(input_ids)
    return {"input_ids": input_batch}


val_num_words = sum(len(line.split(" ")) for line in wiki_dataset["validation"]["text"])
test_num_words = sum(len(line.split(" ")) for line in wiki_dataset["test"]["text"])
print(f"Number of words in val: {val_num_words}\n"
      f"Number of words in test: {test_num_words}")


train_data = wiki_dataset["train"].map(tokenize_function, 
                                        batched=True, 
                                        num_proc=8, 
                                        remove_columns=wiki_dataset["train"].column_names)

if os.path.isfile("wikitext103_train.pt"):
    train_dataset = torch.load("wikitext103_train.pt")
else:
    train_dataset = torch.tensor([index for line in train_data["input_ids"] for index in line], dtype=torch.long)
    torch.save(train_dataset, "wikitext103_train.pt")


train_size = (train_dataset.size(0) // SEQ_LEN) * SEQ_LEN
train_dataset = train_dataset[:train_size].reshape(-1, SEQ_LEN)


val_data = wiki_dataset["validation"].map(tokenize_function, 
                                            batched=True, 
                                            num_proc=8, 
                                            remove_columns=wiki_dataset["validation"].column_names)

if os.path.isfile("wikitext103_validation.pt"):
    val_dataset = torch.load("wikitext103_validation.pt")
else:
    val_dataset = torch.tensor([index for line in val_data["input_ids"] for index in line], dtype=torch.long)
    torch.save(val_dataset, "wikitext103_validation.pt")


val_num_tokens = val_dataset.numel()
val_size = (val_dataset.size(0) // SEQ_LEN) * SEQ_LEN
val_dataset = val_dataset[:val_size].reshape(-1, SEQ_LEN)


test_data = wiki_dataset["test"].map(tokenize_function, 
                                    batched=True, 
                                    num_proc=8, 
                                    remove_columns=wiki_dataset["test"].column_names)

if os.path.isfile("wikitext103_test.pt"):
    test_dataset = torch.load("wikitext103_test.pt")
else:
    test_dataset = torch.tensor([index for line in test_data["input_ids"] for index in line], dtype=torch.long)
    torch.save(train_dataset, "wikitext103_test.pt")

test_num_tokens = test_dataset.numel()
test_size = (test_dataset.size(0) // SEQ_LEN) * SEQ_LEN
test_dataset = test_dataset[:test_size].reshape(-1, SEQ_LEN)

print(f"Number of tokens in val: {val_num_tokens}\n"
        f"Number of tokens in test: {test_num_tokens}")


ntokens = tokenizer.vocab_size  # size of vocabulary
emsize = 256  # embedding dimension
d_hid = 256  # dimension of the feedforward network model in nn.TransformerEncoder
nlayers = 8  # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
nhead = 8  # number of heads in nn.MultiheadAttention
dropout = 0.1  # dropout probability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LanguageModel(ntokens, emsize, nhead, d_hid, nlayers, dropout).to(device)


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)
# scheduler = optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.975)


def collate_fn(source) -> Tuple[Tensor, Tensor]:
    """
    Args:
        source: Tensor, shape [full_seq_len, batch_size]
        i: int
    Returns:
        tuple (data, target), where data has shape [seq_len, batch_size] and
        target has shape [seq_len * batch_size]
    """
    data = torch.stack([sample[:-1] for sample in source]).t()
    target = torch.stack([sample[1:] for sample in source]).t().reshape(-1)
    return data, target


train_loader = DataLoader(train_dataset, 
                        batch_size=BATCHSIZE, 
                        shuffle=True, 
                        collate_fn=collate_fn,
                        drop_last=True)
val_loader = DataLoader(val_dataset, 
                        batch_size=EVAL_BATCHSIZE, 
                        shuffle=True, 
                        collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, 
                        batch_size=EVAL_BATCHSIZE, 
                        shuffle=True, 
                        collate_fn=collate_fn)


def train(model: nn.Module, GNets) -> None:
    model.train()
    if enable_gnets: GNets.train()
    
    total_loss = 0.
    start_time = time.time()
    src_mask = generate_square_subsequent_mask(SEQ_LEN-1).to(device)

    pbar = enumerate(tqdm(train_loader))
    for batch, (data, targets) in pbar:
        data = data.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        # print(GNets.gnet[0].layers[0].weight.data)
        # This probably fucks up the training a bit because the weights change
        # slightly which messes with the optimizer momentum, it is propably 
        # responsible for the bumps in the loss curve
        if enable_gnets:
            GNets.zero_grad()
            GNets.snatchWeights(model, device)
            
        output = model(data, src_mask)
        loss = criterion(output.view(-1, ntokens), targets)

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 0.25)
        if enable_gnets: GNets.backward(model)
        
        optimizer.step()
        if enable_gnets: GNets.step()
        total_loss += loss.item()
        if batch % LOG_INTERVAL == 0 and batch > 0:
            ms_per_batch = (time.time() - start_time) * 1000 / LOG_INTERVAL
            cur_loss = total_loss / LOG_INTERVAL
            ppl = np.exp(cur_loss)
            
            wandb.log({"train_loss": cur_loss, 
                        "train ppl": ppl})
            
            print(f'| epoch {epoch:3d} | {batch:5d} batches | '
                  f'ms/batch {ms_per_batch:5.2f} | '
                  f'loss {cur_loss:5.2f} | ppl {ppl:8.2f}')
            
            total_loss = 0
            start_time = time.time()
            print('-' * 89)
            predict_sequence("cold spring harbor", tokenizer, model, device, seq_size=32)
            print('-' * 89)


def evaluate(model: nn.Module, eval_loader: DataLoader) -> float:
    model.eval() # turn on evaluation mode
    total_loss = 0.
    src_mask = generate_square_subsequent_mask(SEQ_LEN-1).to(device)
    with torch.no_grad():
        for data, targets in eval_loader:
            data = data.to(device)
            targets = targets.to(device)
            seq_len = data.size(0)
            if seq_len != SEQ_LEN:
                src_mask = src_mask[:seq_len, :seq_len]
            output = model(data, src_mask)
            output_flat = output.view(-1, ntokens)
            total_loss += criterion(output_flat, targets).item()
    return total_loss / (len(eval_loader))


######################################################################
# Loop over epochs. Save the model if the validation loss is the best
# we've seen so far. Adjust the learning rate after each epoch.

ignore_layers = ["encoder.weight", "decoder.weight", "norm"]
GNets = gn.GNetList(model, COMPRESSION_LAYER_SIZE, ignore_layers=ignore_layers) if enable_gnets else None

if enable_gnets:
    compression_factor = GNets.compression(model)
    num_params = GNets.numberOfParameters(model)
    print("GNet compression:", GNets.compression(model))
    print("Number of parameters:", num_params)
else:
    compression_factor = 1.0
    num_params = sum(p.numel() for p in model.parameters())
    print("Number of parameters:", num_params)


run_config = {"epochs": EPOCHS,
                "lr": LR,
                "batchsize": BATCHSIZE,
                "seq_len": SEQ_LEN,
                "ntokens": ntokens,
                "emsize": emsize,
                "d_hid": d_hid,
                "nlayers": nlayers,
                "nhead": nhead,
                "dropout": dropout,
                "log_interval": LOG_INTERVAL,
                "compression_layer_size": COMPRESSION_LAYER_SIZE,
                "compression_factor": compression_factor,
                "num_params": num_params,}

wandb.login(key="local-84c6642fa82dc63629ceacdcf326632140a7a899", 
            host="https://wandb.fz-juelich.de")
wandb.init(entity="ja-lohoff", project="GenomicBottleneck", 
           group="wiki", config=run_config, mode=args.wandb)
experiment_name = "GBT_" + args.name if enable_gnets else args.name
wandb.run.name = experiment_name + "_" + args.dataset


if enable_gnets: GNets.to(device)
best_val_loss = float('inf')
best_model = None

for epoch in range(1, EPOCHS + 1):
    epoch_start_time = time.time()
    train(model, GNets)
    val_loss = evaluate(model, val_loader)
    val_ppl = np.exp(val_loss * val_num_tokens / val_num_words) # Word-level PPL -- use 
    
    elapsed = time.time() - epoch_start_time

    print('-' * 89)
    print('After joint p-net and g-net training:')
    print(f'| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | '
          f'valid loss {val_loss:5.2f} | valid ppl {val_ppl:8.2f}')
    
    wandb.log({"validation_loss": val_loss, 
                "val ppl": val_ppl})
        
    if enable_gnets:    
        print("Compression factor", GNets.compression(model))
        print("Layerwise parameter correlations", GNets.correlations().transpose())
        
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model = copy.deepcopy(model)

    # scheduler.step()
        
        
######################################################################
# Evaluate the best model on the test dataset
# -------------------------------------------
#

test_loss = evaluate(best_model, test_loader) 
test_ppl = np.exp(test_loss * test_num_tokens / test_num_words) # word-level PPL
print("Testing...")
print('=' * 89)
print(f'| End of training | test loss {test_loss:5.2f} | '
      f'test ppl {test_ppl:8.2f}')
print('=' * 89)


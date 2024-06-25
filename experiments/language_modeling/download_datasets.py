import argparse

import numpy as np

from datasets import load_dataset
from transformers import AutoTokenizer


parser = argparse.ArgumentParser()

parser.add_argument("--language", type=str, default="en", help="Which language to use.")

parser.add_argument("--single_file_size", type=int, default=647, 
                    help="The size of each file in megabytes.")

parser.add_argument("--corpus_size", type=int, default=15, 
                    help="The desired size of the subcorpus in GB.")

parser.add_argument("--total_num_files", type=int, default=2256,
                    help="The total number of files in the dataset.")

parser.add_argument("--vocab_size", type=int, 
                    default=50257, help="Size of the vocabulary.")

parser.add_argument("--download_all", action="store_true")

parser.add_argument("--small", action="store_true")

args = parser.parse_args()

gpt2_tokenizer = AutoTokenizer.from_pretrained("gpt2")
num_files = args.corpus_size * 1024 // args.single_file_size + 1

with open("./token", "r") as f:
    token = f.read()
    
idxs = np.random.choice(np.arange(args.total_num_files)+1, size=num_files)

if args.download_all:
    file_list = np.arange(args.total_num_files)+1
    
l = args.language
file_list = [f"hf://datasets/oscar-corpus/OSCAR-2301/{l}_meta/{l}_meta_part_{i}.jsonl.zst" for i in idxs]

if args.small:
    file_list = [f"hf://datasets/oscar-corpus/OSCAR-2301/{l}_meta/{l}_meta.jsonl.zst"]

print(file_list)

oscar_dataset = load_dataset("text", 
                            name="OSCAR-2301_" + l,
                            trust_remote_code=True,
                            token=token, 
                            split="train",
                            data_files=file_list, 
                            cache_dir="/Data/pgi-15/lohoff/hf_cache",
                            num_proc=32)


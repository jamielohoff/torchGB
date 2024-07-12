import os
import argparse

from datasets import load_dataset
from transformers import AutoTokenizer

from config import load_config


parser = argparse.ArgumentParser()

parser.add_argument("--target_path", type=str, default=".", 
                    help="Where to store the tokenized datset.")

parser.add_argument("--source_path", type=str, default=".", 
                    help="Where to store the tokenized datset.")

parser.add_argument("--language", type=str, default="en", 
                    help="Which language to use.")

parser.add_argument("--seq_len", type=int, default=512, 
                    help="Context length to tokenize for.")

args = parser.parse_args()


# Initialize the data directory
base_config = load_config("base_config.yaml")
data_dir = args.source_path
print("Data directory:", data_dir)


# Initialize the tokenizer
tokenizer_path = os.path.join(base_config["tokenizer_path"], "oscar_2301_" + args.language)
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
tokenizer.pad_token = tokenizer.eos_token


def tokenize_function(element):
    outputs = tokenizer(element["text"],
                        truncation=True,
                        max_length=args.seq_len+1,
                        return_overflowing_tokens=True,
                        return_length=True)
    input_batch = []
    for length, input_ids in zip(outputs["length"], outputs["input_ids"]):
        if length == args.seq_len+1: # Drops all sentences that are not long enough...
            input_batch.append(input_ids)
    return {"input_ids": input_batch}


dataset = load_dataset("arrow", 
                        data_dir=data_dir, 
                        split="train",
                        num_proc=16)


rm_cols = dataset.column_names
tokenized_dataset = dataset.map(tokenize_function, 
                                batched=True, 
                                remove_columns=rm_cols)


new_path = os.path.join(args.target_path, args.language)
print("Saving tokenized dataset to:", new_path)
os.makedirs(new_path, exist_ok=False)
tokenized_dataset.save_to_disk(new_path)


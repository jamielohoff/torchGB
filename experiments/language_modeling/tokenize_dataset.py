import os
import argparse

from datasets import load_dataset, DatasetDict
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


train_dataset = load_dataset("arrow", 
                        data_dir=data_dir, 
                        split="train[:94%]",
                        num_proc=16)

val_dataset = load_dataset("arrow", 
                        data_dir=data_dir, 
                        split="train[94%:95%]",
                        num_proc=16)

test_dataset = load_dataset("arrow", 
                            data_dir=data_dir, 
                            split="train[95%:]",
                            num_proc=16)

rm_cols = train_dataset.column_names
tokenized_train_dataset = train_dataset.map(tokenize_function, 
                                            batched=True, 
                                            remove_columns=rm_cols)

tokenized_val_dataset = val_dataset.map(tokenize_function, 
                                        batched=True, 
                                        remove_columns=rm_cols)

tokenized_test_dataset = test_dataset.map(tokenize_function, 
                                        batched=True, 
                                        remove_columns=rm_cols)

tokenized_dataset = DatasetDict({"train": tokenized_train_dataset,
                                "validation": tokenized_val_dataset,
                                "test": tokenized_test_dataset})

new_path = os.path.join(args.target_path, args.language)
print("Saving tokenized dataset to:", new_path)
os.makedirs(new_path, exist_ok=False)
tokenized_dataset.save_to_disk(new_path)


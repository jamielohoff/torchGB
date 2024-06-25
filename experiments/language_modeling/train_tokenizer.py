import argparse

from datasets import load_dataset
from transformers import AutoTokenizer

parser = argparse.ArgumentParser()
parser.add_argument("--language", type=str, 
                    default="en", help="What language to use.")
parser.add_argument("--vocab_size", type=int, 
                    default=50257, help="Size of the vocabulary.")
parser.add_argument("--path", type=str, help="Path to data.")
args = parser.parse_args()

gpt2_tokenizer = AutoTokenizer.from_pretrained("gpt2")

with open("./token", "r") as f:
    token = f.read()

# data_dir="/Data/pgi-15/lohoff/hf_cache/text/default-b680ec0519013fd8/0.0.0/96636a050ef51804b84abbfd4f4ad440e01153c24b86293eb5c3b300a41f9101",

oscar_dataset = load_dataset("arrow",
                            data_dir=args.path,
                            split="train")

def batch_iterator(batch_size=1000):
    batch = []
    for example in oscar_dataset:
        batch.append(example["text"])
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:  # yield last batch
        yield batch
        
tokenizer = gpt2_tokenizer.train_new_from_iterator(batch_iterator(), 
                                                    vocab_size=args.vocab_size)


tokenizer.save_pretrained("tokenizers/oscar_2301_" + args.language)

# Load as transformers tokenizer

# from transformers import PreTrainedTokenizerFast
# tk_tokenizer = Tokenizer.from_file("mybart/tokenizer.json")
# tokenizer = PreTrainedTokenizerFast(tokenizer_object=tk_tokenizer)


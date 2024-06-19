import argparse

from datasets import load_dataset
from transformers import AutoTokenizer

parser = argparse.ArgumentParser()
parser.add_argument("--language", type=str, 
                    default="en", help="What language to use.")
parser.add_argument("--vocab_size", type=int, 
                    default=50265, help="Size of the vocabulary.")
args = parser.parse_args()

gpt2_tokenizer = AutoTokenizer.from_pretrained("gpt2")

with open("./token", "r") as f:
    token = f.read()

oscar_dataset = load_dataset("oscar-corpus/OSCAR-2301", 
                            language=args.language, 
                            split="train", 
                            token=token, 
                            trust_remote_code=True,
                            streaming=True)

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


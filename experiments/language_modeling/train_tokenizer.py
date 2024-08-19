import argparse

from datasets import load_dataset
from transformers import AutoTokenizer, PreTrainedTokenizerFast
from tokenizers import SentencePieceBPETokenizer

from utils import load_config

parser = argparse.ArgumentParser()
parser.add_argument("--language", type=str, 
                    default="en", help="What language to use.")
parser.add_argument("--vocab_size", type=int, 
                    default=50257, help="Size of the vocabulary.")
parser.add_argument("--path", type=str, help="Path to data.")

eos_token = "<eos>"
unk_token = "<unk>"  # token for unknown words
pad_token = "<pad>"  # token for padding
args = parser.parse_args()

# Initialize the data directory
base_config = load_config("config/base_config.yml")
cache_dir = base_config["cache_dir"]

gpt2_tokenizer = AutoTokenizer.from_pretrained("gpt2")
special_tokens = [eos_token, unk_token, pad_token]
print(special_tokens)

with open("./token", "r") as f:
    token = f.read()

# data_dir="/Data/pgi-15/lohoff/hf_cache/text/default-b680ec0519013fd8/0.0.0/96636a050ef51804b84abbfd4f4ad440e01153c24b86293eb5c3b300a41f9101",

# oscar_dataset = load_dataset("arrow",
#                             data_dir=args.path,
#                             split="train")

wiki_dataset = load_dataset("wikimedia/wikipedia", "20231101." + args.language, 
                            cache_dir=cache_dir, 
                            split="train[:95%]")


spbpe_tokenizer = SentencePieceBPETokenizer()


def batch_iterator(batch_size=5000):
    batch = []
    for example in wiki_dataset:
        batch.append(example["text"])
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:  # yield last batch
        yield batch
        
# tokenizer = gpt2_tokenizer.train_new_from_iterator(batch_iterator(), 
#                                                     vocab_size=args.vocab_size)


spbpe_tokenizer.train_from_iterator(
    batch_iterator(),
    vocab_size=args.vocab_size,
    min_frequency=5,
    show_progress=True,
    limit_alphabet=500,
    special_tokens=special_tokens,
)

tokenizer = PreTrainedTokenizerFast(tokenizer_object=spbpe_tokenizer,
                                    eos_token=eos_token,
                                    unk_token=unk_token,
                                    pad_token=pad_token)
tokenizer.save_pretrained("tokenizers/wikipedia_" + args.language)

# Load as transformers tokenizer

# from transformers import PreTrainedTokenizerFast
# tk_tokenizer = Tokenizer.from_file("mybart/tokenizer.json")
# tokenizer = PreTrainedTokenizerFast(tokenizer_object=tk_tokenizer)


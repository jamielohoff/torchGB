import numpy as np

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer, Transformer


def generate_square_subsequent_mask(sz: int):
    """Generates an upper-triangular matrix of -inf, with zeros on diag."""
    return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)


class PositionalEncoding(nn.Module):
    """
    `PositionalEncoding` module injects some information about the relative or 
    absolute position of the tokens in the sequence. The positional encodings 
    have the same dimension as the embeddings so that the two can be summed. 
    Here, we use ``sine`` and ``cosine`` functions of different frequencies.

    Args:
        - d_model (int): the number of expected features in the input (required).
        - dropout (float): the dropout value (default=0.1).
        - max_len (int): the maximum length of the input sequences (default=5000).
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class LanguageModel(nn.Module):
    def __init__(self, 
                num_tokens: int, 
                embedding_dim: int, 
                num_heads: int, 
                hidden_dim: int,
                num_layers: int, 
                dropout: float = 0.1) -> None:
        super().__init__()
        self.pos_encoder = PositionalEncoding(embedding_dim, dropout)
        encoder_layers = TransformerEncoderLayer(embedding_dim, 
                                                num_heads, 
                                                hidden_dim, 
                                                dropout,
                                                norm_first=False,
                                                batch_first=True, 
                                                activation=F.gelu)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers)
        self.encoder = nn.Embedding(num_tokens, embedding_dim)
        self.embedding_dim = embedding_dim
        self.decoder = nn.Linear(embedding_dim, num_tokens, bias=False)
        
        self.apply(self.init_weights)
        self.decoder.weight = self.encoder.weight # Tie weights
        
    def init_weights(self, module: nn.Module) -> None:
        # if isinstance(module, nn.Embedding):
        #     nn.init.xavier_normal_(module.weight)
            
        if isinstance(module, nn.TransformerEncoderLayer):
            nn.init.xavier_normal_(module.self_attn.in_proj_weight)
            nn.init.zeros_(module.self_attn.in_proj_bias)
            nn.init.xavier_normal_(module.self_attn.out_proj.weight)
            nn.init.zeros_(module.self_attn.out_proj.bias)
            
            nn.init.xavier_normal_(module.linear1.weight)
            nn.init.zeros_(module.linear1.bias)
            nn.init.xavier_normal_(module.linear2.weight)
            nn.init.zeros_(module.linear2.bias)
            
        elif isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            
        elif isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)
        
    def forward(self, src: Tensor, src_mask: Tensor) -> Tensor:
        """
        Args:
            src: Tensor, shape [seq_len, batch_size]
            src_mask: Tensor, shape [seq_len, seq_len]
        Returns:
            output Tensor of shape [seq_len, batch_size, ntoken]
        """
        src = self.encoder(src) # * np.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask)
        output = self.decoder(output)
        return output
    

def generate_sequence(model, sequence, device, seq_size=32):       
    sequence = sequence.unsqueeze(0)
    src_mask = generate_square_subsequent_mask(seq_size+sequence.size(1))
    generate_step = 0
    while generate_step < seq_size:
        _src_mask = src_mask[:sequence.size(1), :sequence.size(1)].to(device)
        output_word = torch.argmax(model(sequence, _src_mask)[-1, :], dim=1)[-1:]
        output_word = output_word.unsqueeze(0)
        sequence = torch.cat((sequence, output_word), dim=1)
        generate_step += 1
    sequence = sequence.squeeze(0)
    return sequence


def predict_sequence(sentence, tokenizer, model, device, seq_size=10):
    print(f"Source: {sentence}")
    input_ids = torch.tensor(tokenizer.encode(sentence), dtype=torch.long).to(device)
    generated_sequence = generate_sequence(model, input_ids, device, seq_size)
    print(f"Result: {tokenizer.decode(generated_sequence)}")


# helper Module to convert tensor of input indices into corresponding tensor of token embeddings
class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens: Tensor):
        return self.embedding(tokens.long()) * np.sqrt(self.emb_size)


# Seq2Seq Network
class Seq2SeqTransformer(nn.Module):
    def __init__(self,
                 num_encoder_layers: int,
                 num_decoder_layers: int,
                 emb_size: int,
                 nhead: int,
                 src_vocab_size: int,
                 tgt_vocab_size: int,
                 dim_feedforward: int = 512,
                 dropout: float = 0.1):
        super(Seq2SeqTransformer, self).__init__()
        self.transformer = Transformer(d_model=emb_size,
                                       nhead=nhead,
                                       num_encoder_layers=num_encoder_layers,
                                       num_decoder_layers=num_decoder_layers,
                                       dim_feedforward=dim_feedforward,
                                       dropout=dropout)
        self.generator = nn.Linear(emb_size, tgt_vocab_size)
        self.src_tok_emb = TokenEmbedding(src_vocab_size, emb_size)
        self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, emb_size)
        self.positional_encoding = PositionalEncoding(
            emb_size, dropout=dropout)

    def forward(self,
                src: Tensor,
                trg: Tensor,
                src_mask: Tensor,
                tgt_mask: Tensor,
                src_padding_mask: Tensor,
                tgt_padding_mask: Tensor,
                memory_key_padding_mask: Tensor):
        src_emb = self.positional_encoding(self.src_tok_emb(src))
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(trg))
        outs = self.transformer(src_emb, tgt_emb, src_mask, tgt_mask, None,
                                src_padding_mask, tgt_padding_mask, memory_key_padding_mask)
        return self.generator(outs)

    def encode(self, src: Tensor, src_mask: Tensor):
        return self.transformer.encoder(self.positional_encoding(
                            self.src_tok_emb(src)), src_mask)

    def decode(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor):
        return self.transformer.decoder(self.positional_encoding(
                          self.tgt_tok_emb(tgt)), memory,
                          tgt_mask)
    
    
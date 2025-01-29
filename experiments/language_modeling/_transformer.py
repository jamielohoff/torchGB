import numpy as np

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer, Transformer


def generate_square_subsequent_mask(size: int) -> Tensor:
    """
    Generates an upper-triangular matrix of `-inf`, with zeros on diagonal. This
    mask is then used for autoregressive generation of text in the GPT-2 model.
    
    Args:
        `size` (int): Size of the square matrix.
        
    Returns
        Tensor: Upper-triangular matrix of `-inf` values.
    """
    return torch.triu(torch.ones(size, size) * float("-inf"), diagonal=1)


class PositionalEncoding(nn.Module):
    """
    `PositionalEncoding` module injects some information about the relative or 
    absolute position of the tokens in the sequence. The positional encodings 
    have the same dimension as the embeddings so that the two can be summed. 
    Here, we use ``sine`` and ``cosine`` functions of different frequencies.

    Args:
        `d_model` (int): the number of expected features in the input (required).
        `dropout` (float): the dropout value (default=0.1).
        `max_len` (int): the maximum length of the input sequences (default=5000).
        
    Returns:
        Tensor: 
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class GPT(nn.Module):
    """
    Simple implementation of a GPT/GPT2 model with weight
    tying and positional encoding. Weight tying significantly reduces
    the used numner of parameters in the model, while the positional encoding is
    used instead of the learned positional embeddings to be able to transfer
    weights and use G-Nets across different language tasks.

    Args:
        `seq_len` (int): Context window size.
        `vocab_size` (int): Size of the vocabulary.
        `embedding_dim` (int): Dimension of the Q,K,V embeddings.
        `num_heads` (int): Number of attention heads.
        `ff_dim` (int): Dimension of the feed-forward MLP.
        `num_layers` (int): Number of transformer layers.
        `dropout` (float): Dropout rate.
        `tie_weights` (bool): Whether to tie the weights of the decoder layer 
            and the embedding layer.
    """
    def __init__(self, seq_len: int = 512, vocab_size: int = 50257, 
                embedding_dim: int = 768, num_heads: int = 12, 
                ff_dim: int = 3072, num_layers: int = 12, dropout: float = 0.1,
                tie_weights: bool = True) -> None:
        super().__init__()
        self.pos_encoder = PositionalEncoding(embedding_dim, dropout)
        self.dropout = nn.Dropout(dropout)
        encoder_layers = TransformerEncoderLayer(embedding_dim, num_heads, 
                                                 ff_dim, dropout, norm_first=False,
                                                 batch_first=True, activation=F.gelu)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers)
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding_dim = torch.tensor([embedding_dim])
        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.decoder = nn.Linear(embedding_dim, vocab_size, bias=False)
        
        self.apply(self.init_weights)
        if tie_weights:
            self.decoder.weight = self.embedding.weight # Tie weights
        
    def init_weights(self, module: nn.Module) -> None:
        """
        GPT-2 init as described in paper

        Args:
            module (nn.Module): _description_
        """
        if isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)
            
        elif isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.normal_(module.bias, std=0.02)
            
        elif isinstance(module, nn.TransformerEncoderLayer):
            nn.init.normal_(module.self_attn.in_proj_weight, std=0.02)
            if module.self_attn.in_proj_bias is not None:
                nn.init.normal_(module.self_attn.in_proj_bias, std=0.02)
            nn.init.normal_(module.self_attn.out_proj.weight, std=0.02)
            if module.self_attn.out_proj.bias is not None:
                nn.init.normal_(module.self_attn.out_proj.bias, std=0.02)
            
            nn.init.normal_(module.linear1.weight, std=0.02)
            if module.linear1.bias is not None:
                nn.init.normal_(module.linear1.bias, std=0.02)
            nn.init.normal_(module.linear2.weight, std=0.02)
            if module.linear2.bias is not None:
                nn.init.normal_(module.linear2.bias, std=0.02) 

        
    def forward(self, src: Tensor, src_mask: Tensor) -> Tensor:
        """
        Args:
            `src`: Tensor, shape [seq_len, batch_size]
            `src_mask`: Tensor, shape [seq_len, seq_len]
        Returns:
            output Tensor of shape [seq_len, batch_size, ntoken]
        """
        src = self.embedding(src) * torch.sqrt(self.embedding_dim).to(src.device)
        src = self.pos_encoder(src)
        src = self.transformer_encoder(src, src_mask)
        src = self.layer_norm(src)
        return self.decoder(src)
    

def generate_sequence(model: nn.Module, sequence: Tensor, device: torch.device, 
                      seq_size: int = 32, k: int = 20, 
                      temperature: float = 0.9) -> Tensor:       
    """_summary_

    Args:
        model (nn.Module): _description_
        sequence (Tensor): _description_
        device (torch.device): _description_
        seq_size (int, optional): _description_. Defaults to 32.
        k (int, optional): _description_. Defaults to 20.
        temperature (float, optional): _description_. Defaults to 0.9.

    Returns:
        Tensor: _description_
    """
    sequence = sequence.unsqueeze(0)
    src_mask = generate_square_subsequent_mask(seq_size+sequence.size(1))
    for _ in range(seq_size):
        _src_mask = src_mask[:sequence.size(1), :sequence.size(1)].to(device)
        with torch.no_grad():
            logits = model(sequence, _src_mask)[0, -1, :]
            topk = torch.topk(logits, k)
            topk_probs = F.softmax(topk.values / temperature, dim=0)
            output_word = topk.indices[torch.multinomial(topk_probs, 1)]
            output_word = output_word.unsqueeze(0)
        
        sequence = torch.cat((sequence, output_word), dim=1)
    sequence = sequence.squeeze(0)
    return sequence


def predict_sequence(sentence, tokenizer, model, device, seq_size=32) -> str:
    input_ids = torch.tensor(tokenizer.encode(sentence), dtype=torch.long).to(device)
    generated_sequence = generate_sequence(model, input_ids, device, seq_size)
    return f"Result: {tokenizer.decode(generated_sequence)}"


# helper Module to convert tensor of input indices into corresponding tensor of token embeddings
class TokenEmbedding(nn.Module):
    """_summary_

    Args:
        nn (_type_): _description_
    """
    def __init__(self, vocab_size: int, emb_size) -> None:
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens: Tensor) -> Tensor:
        return self.embedding(tokens.long()) * np.sqrt(self.emb_size)


# TODO: document (and use?) Seq2Seq Network
class Seq2SeqTransformer(nn.Module):
    def __init__(self, num_encoder_layers: int, num_decoder_layers: int,
                 emb_size: int, nhead: int, src_vocab_size: int, tgt_vocab_size: int,
                 dim_feedforward: int = 512, dropout: float = 0.1) -> None:
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
        self.positional_encoding = PositionalEncoding(emb_size, dropout=dropout)

    def forward(self, src: Tensor, trg: Tensor, src_mask: Tensor, 
                tgt_mask: Tensor, src_padding_mask: Tensor,
                tgt_padding_mask: Tensor, memory_key_padding_mask: Tensor) -> Tensor:
        src_emb = self.positional_encoding(self.src_tok_emb(src))
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(trg))
        outs = self.transformer(src_emb, tgt_emb, src_mask, tgt_mask, None,
                                src_padding_mask, tgt_padding_mask, memory_key_padding_mask)
        return self.generator(outs)

    def encode(self, src: Tensor, src_mask: Tensor) -> Tensor:
        return self.transformer.encoder(self.positional_encoding(
                            self.src_tok_emb(src)), src_mask)

    def decode(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor) -> Tensor:
        pos_enc = self.positional_encoding(self.tgt_tok_emb(tgt))
        return self.transformer.decoder(pos_enc, memory, tgt_mask)
    
    
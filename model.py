"""
Model architectures for the LLaDA project.

Contains the implementation of the bidirectional LLaDA model and the
causal autoregressive baseline model.
"""

import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """
    Standard positional encoding for Transformer models.
    Adds sine/cosine embeddings to the input tensor to provide positional information.
    """
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, seq_len, embedding_dim]
        Returns:
            torch.Tensor: Output tensor with positional encoding added.
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class LLaDAModel(nn.Module):
    """
    LLaDA: A Transformer Encoder-based model for masked language modeling.
    This model uses bidirectional attention to predict masked tokens.
    """
    def __init__(self, config):
        super(LLaDAModel, self).__init__()
        self.embedding_dim = config.EMBEDDING_DIM
        self.embedding = nn.Embedding(config.VOCAB_SIZE, config.EMBEDDING_DIM)
        self.pos_encoding = PositionalEncoding(config.EMBEDDING_DIM, dropout=config.DROPOUT, max_len=5000)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.EMBEDDING_DIM, 
            nhead=config.NUM_HEADS, 
            dim_feedforward=config.EMBEDDING_DIM * 4, # Standard practice
            batch_first=True,
            dropout=config.DROPOUT
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.NUM_LAYERS)
        self.output_layer = nn.Linear(config.EMBEDDING_DIM, config.VOCAB_SIZE)

    def forward(self, src):
        """
        Forward pass for the LLaDA model.

        Args:
            src (torch.Tensor): Input tensor of token IDs (shape: [batch, seq_len]).

        Returns:
            torch.Tensor: Logits over the vocabulary for each position (shape: [batch, seq_len, vocab_size]).
        """
        src_emb = self.embedding(src) * math.sqrt(self.embedding_dim)
        src_pos = self.pos_encoding(src_emb)
        output = self.transformer_encoder(src_pos) # No causal mask needed for bidirectional attention
        logits = self.output_layer(output)
        return logits

class AutoregressiveModel(nn.Module):
    """
    A baseline Transformer model for autoregressive language modeling.
    This model uses causal (masked) attention to predict the next token.
    """
    def __init__(self, config):
        super(AutoregressiveModel, self).__init__()
        self.embedding_dim = config.EMBEDDING_DIM
        self.embedding = nn.Embedding(config.VOCAB_SIZE, config.EMBEDDING_DIM)
        self.pos_encoding = PositionalEncoding(config.EMBEDDING_DIM, dropout=config.DROPOUT, max_len=5000)

        # Using a standard TransformerEncoder with a causal mask is common and efficient
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.EMBEDDING_DIM,
            nhead=config.NUM_HEADS,
            dim_feedforward=config.EMBEDDING_DIM * 4,
            batch_first=True,
            dropout=config.DROPOUT
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.NUM_LAYERS)
        self.output_layer = nn.Linear(config.EMBEDDING_DIM, config.VOCAB_SIZE)
        self.device = config.DEVICE

    @staticmethod
    def _generate_causal_mask(size, device):
        """
        Generates a square causal mask for autoregressive models.
        The masked positions are filled with float('-inf').
        Unmasked positions are filled with float(0.0).
        """
        mask = (torch.triu(torch.ones(size, size)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask.to(device)

    def forward(self, src):
        """
        Forward pass for the autoregressive model.

        Args:
            src (torch.Tensor): Input tensor of token IDs (shape: [batch, seq_len]).

        Returns:
            torch.Tensor: Logits over the vocabulary for each position (shape: [batch, seq_len, vocab_size]).
        """
        mask = self._generate_causal_mask(src.size(1), self.device)
        src_emb = self.embedding(src) * math.sqrt(self.embedding_dim)
        src_pos = self.pos_encoding(src_emb)
        output = self.transformer_encoder(src_pos, mask=mask)
        logits = self.output_layer(output)
        return logits

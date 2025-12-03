import math

import torch
import torch.nn as nn


class SimpleTransformer(nn.Module):
    def __init__(self, d_vocab, d_model, n_heads, layers, context_length):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.embedding = nn.Embedding(d_vocab, d_model)
        self.pos_embedding = SinusoidalPositionalEmbedding(
            d_model, max_len=context_length
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, d_vocab)
        self.attention_mask = nn.Transformer.generate_square_subsequent_mask(
            context_length
        )

    def forward(self, x):
        context_length = x.size(1)
        pos_ids = (
            torch.arange(context_length, device=x.device).unsqueeze(0).expand_as(x)
        )

        x = self.embedding(x)
        pos_emb = self.pos_embedding(x)
        x += pos_emb
        mask = self.attention_mask[:context_length, :context_length].to(x.device)
        x = self.transformer(x, mask=mask, is_causal=True)
        x = self.ln_f(x)
        return self.lm_head(x)

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class SinusoidalPositionalEmbedding(nn.Module):
    """Sinusoidal positional embeddings from 'Attention is All You Need'"""

    def __init__(self, d_model, max_len=2048):
        super().__init__()

        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Register as buffer (not a parameter, but part of state)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, context_length, d_model)
        Returns:
            Positional embeddings of shape (batch_size, context_length, d_model)
        """
        return self.pe[: x.size(1), :]

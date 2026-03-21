"""
FT-Transformer: Feature Tokenizer + Transformer for tabular data.

Reference: Gorishniy et al. (2021), "Revisiting Deep Learning Models for Tabular Data"
           https://arxiv.org/abs/2106.11959

Architecture
------------
1. Feature Tokenizer  — embeds each numerical/categorical feature as a learned token
2. Transformer blocks — self-attention over the token sequence
3. [CLS] head         — classification/regression MLP on the [CLS] token

Workstream: Implement FT-Transformer for XBRL financial data.

TODO:
  - Implement FeatureTokenizer (numerical linear projection + [CLS] token)
  - Implement TransformerLayer with pre-LN and FFN
  - Implement FTTransformer assembling tokenizer + N transformer blocks + head
  - Add gradient checkpointing for large feature counts
"""

from __future__ import annotations

import torch
import torch.nn as nn


class FeatureTokenizer(nn.Module):
    """Embed each scalar feature as a d_token-dimensional vector.

    Numerical features: x_i -> W_i * x_i + b_i  (per-feature linear)
    A learnable [CLS] token is prepended to the sequence.
    """

    def __init__(self, n_features: int, d_token: int) -> None:
        super().__init__()
        self.n_features = n_features
        self.d_token = d_token
        # Per-feature weight and bias
        self.weight = nn.Parameter(torch.empty(n_features, d_token))
        self.bias = nn.Parameter(torch.zeros(n_features, d_token))
        self.cls_token = nn.Parameter(torch.empty(1, 1, d_token))
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        nn.init.normal_(self.cls_token, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, n_features) -> tokens: (B, n_features + 1, d_token)"""
        # (B, n_features, d_token)
        tokens = x.unsqueeze(-1) * self.weight.unsqueeze(0) + self.bias.unsqueeze(0)
        cls = self.cls_token.expand(x.size(0), -1, -1)
        return torch.cat([cls, tokens], dim=1)


class FTTransformer(nn.Module):
    """Full FT-Transformer encoder.

    Parameters
    ----------
    n_features:   number of input scalar features
    d_token:      token embedding dimension
    n_heads:      number of attention heads
    n_layers:     number of Transformer blocks
    d_ffn_factor: FFN hidden dim = d_token * d_ffn_factor
    dropout:      attention + FFN dropout rate
    n_outputs:    number of output classes (set to 1 for binary with BCEWithLogits)
    """

    def __init__(
        self,
        n_features: int,
        d_token: int = 192,
        n_heads: int = 8,
        n_layers: int = 3,
        d_ffn_factor: int = 4,
        dropout: float = 0.0,
        n_outputs: int = 1,
    ) -> None:
        super().__init__()
        self.tokenizer = FeatureTokenizer(n_features, d_token)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_token,
            nhead=n_heads,
            dim_feedforward=d_token * d_ffn_factor,
            dropout=dropout,
            batch_first=True,
            norm_first=True,  # pre-LN as in the paper
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_token),
            nn.ReLU(),
            nn.Linear(d_token, n_outputs),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, n_features) -> logits: (B, n_outputs)"""
        tokens = self.tokenizer(x)              # (B, n_feat+1, d_token)
        encoded = self.transformer(tokens)      # (B, n_feat+1, d_token)
        cls_repr = encoded[:, 0]                # (B, d_token)
        return self.head(cls_repr)              # (B, n_outputs)

    def get_representation(self, x: torch.Tensor) -> torch.Tensor:
        """Return [CLS] embedding without the classification head."""
        tokens = self.tokenizer(x)
        encoded = self.transformer(tokens)
        return encoded[:, 0]

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
"""

from __future__ import annotations

import torch
import torch.nn as nn


class FeatureTokenizer(nn.Module):
    """Embed each scalar feature as a d_token-dimensional vector.

    Numerical features: x_i -> W_i * x_i + b_i  (per-feature linear)
    Categorical features: looked up via per-feature nn.Embedding tables.
    A learnable [CLS] token is prepended to the sequence.
    """

    def __init__(
        self,
        n_features: int,
        d_token: int,
        n_cat_features: int = 0,
        cat_cardinalities: list[int] | None = None,
    ) -> None:
        super().__init__()
        self.n_features = n_features
        self.d_token = d_token
        self.n_cat_features = n_cat_features
        # Per-feature weight and bias for numerical features
        self.weight = nn.Parameter(torch.empty(n_features, d_token))
        self.bias = nn.Parameter(torch.zeros(n_features, d_token))
        self.cls_token = nn.Parameter(torch.empty(1, 1, d_token))
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        nn.init.normal_(self.cls_token, std=0.02)

        # Categorical embeddings
        if n_cat_features > 0:
            if cat_cardinalities is None or len(cat_cardinalities) != n_cat_features:
                raise ValueError(
                    f"cat_cardinalities must have length {n_cat_features}, "
                    f"got {cat_cardinalities}"
                )
            self.cat_embeddings = nn.ModuleList(
                [nn.Embedding(card, d_token) for card in cat_cardinalities]
            )
        else:
            self.cat_embeddings = nn.ModuleList()

    def forward(
        self,
        x_num: torch.Tensor,
        x_cat: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x_num: (B, n_features) — numerical features.
            x_cat: (B, n_cat_features) long tensor — categorical features.
                   Optional; ignored when n_cat_features == 0.

        Returns:
            tokens: (B, n_features + n_cat_features + 1, d_token)
        """
        # (B, n_features, d_token)
        tokens = x_num.unsqueeze(-1) * self.weight.unsqueeze(0) + self.bias.unsqueeze(0)

        # Append categorical tokens
        if x_cat is not None and self.n_cat_features > 0:
            cat_tokens = torch.stack(
                [emb(x_cat[:, i]) for i, emb in enumerate(self.cat_embeddings)],
                dim=1,
            )  # (B, n_cat, d_token)
            tokens = torch.cat([tokens, cat_tokens], dim=1)

        cls = self.cls_token.expand(x_num.size(0), -1, -1)
        return torch.cat([cls, tokens], dim=1)


class FTTransformer(nn.Module):
    """Full FT-Transformer encoder.

    Parameters
    ----------
    n_features:       number of input numerical features
    d_token:          token embedding dimension
    n_heads:          number of attention heads
    n_layers:         number of Transformer blocks
    d_ffn_factor:     FFN hidden dim = d_token * d_ffn_factor
    dropout:          attention + FFN dropout rate
    n_outputs:        number of output classes (set to 1 for binary with BCEWithLogits)
    n_cat_features:   number of categorical input features (default 0)
    cat_cardinalities: cardinality of each categorical feature (e.g. [12] for SIC sector)
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
        n_cat_features: int = 0,
        cat_cardinalities: list[int] | None = None,
    ) -> None:
        super().__init__()
        self.tokenizer = FeatureTokenizer(
            n_features, d_token, n_cat_features, cat_cardinalities,
        )
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

    def forward(
        self,
        x: torch.Tensor,
        x_cat: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """x: (B, n_features) -> logits: (B, n_outputs)"""
        tokens = self.tokenizer(x, x_cat)        # (B, n_feat+n_cat+1, d_token)
        encoded = self.transformer(tokens)        # (B, n_feat+n_cat+1, d_token)
        cls_repr = encoded[:, 0]                  # (B, d_token)
        return self.head(cls_repr)                # (B, n_outputs)

    def get_representation(
        self,
        x: torch.Tensor,
        x_cat: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return [CLS] embedding without the classification head."""
        tokens = self.tokenizer(x, x_cat)
        encoded = self.transformer(tokens)
        return encoded[:, 0]

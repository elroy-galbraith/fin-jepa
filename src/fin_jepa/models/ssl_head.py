"""
Self-supervised pretraining: Masked Feature Reconstruction head.

Workstream: Self-supervised pretraining experiment (masked feature reconstruction).

Approach
--------
Given an input feature vector x:
  1. Randomly mask a fraction of features -> x_masked
  2. Run x_masked through FT-Transformer encoder -> repr
  3. A reconstruction MLP predicts the masked feature values from repr
  4. MSE loss on masked positions only

This is analogous to BERT masked language modeling, applied to tabular features.

TODO:
  - Implement masking strategy (random, block, feature-group-aware)
  - Implement ReconstructionHead MLP
  - Implement MaskedFeatureSSL wrapper combining encoder + head
  - Define pretraining loss (MSE on numerical, CE on categorical)
"""

from __future__ import annotations

import torch
import torch.nn as nn

from fin_jepa.models.ft_transformer import FTTransformer


class MaskedFeatureSSL(nn.Module):
    """Wrapper for SSL pretraining via masked feature reconstruction.

    Parameters
    ----------
    encoder:      FTTransformer backbone
    mask_ratio:   fraction of features to mask during pretraining
    """

    def __init__(self, encoder: FTTransformer, mask_ratio: float = 0.15) -> None:
        super().__init__()
        self.encoder = encoder
        self.mask_ratio = mask_ratio
        d_token = encoder.tokenizer.d_token
        n_features = encoder.tokenizer.n_features
        self.reconstruction_head = nn.Sequential(
            nn.Linear(d_token, d_token * 2),
            nn.GELU(),
            nn.Linear(d_token * 2, n_features),
        )

    def forward(
        self,
        x: torch.Tensor,
        x_cat: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x:     (B, n_features) numerical features
        x_cat: (B, n_cat_features) long tensor — categorical features (optional)

        Returns
        -------
        loss:        scalar reconstruction MSE (on numerical features only)
        x_hat:       (B, n_features) predicted values
        mask:        (B, n_features) bool mask (True = masked)
        """
        mask = torch.rand_like(x) < self.mask_ratio
        x_masked = x.clone()
        x_masked[mask] = 0.0  # replace masked features with zero

        repr_ = self.encoder.get_representation(x_masked, x_cat)  # (B, d_token)
        x_hat = self.reconstruction_head(repr_)                    # (B, n_features)

        loss = ((x_hat - x) ** 2 * mask.float()).sum() / mask.float().sum().clamp(min=1)
        return loss, x_hat, mask

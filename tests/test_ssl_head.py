"""Unit tests for SSL masked feature reconstruction head."""

import torch

from fin_jepa.models.ft_transformer import FTTransformer
from fin_jepa.models.ssl_head import MaskedFeatureSSL


def test_ssl_loss_scalar():
    B, F = 8, 30
    encoder = FTTransformer(n_features=F, d_token=64, n_heads=4, n_layers=2)
    ssl = MaskedFeatureSSL(encoder, mask_ratio=0.15)
    x = torch.randn(B, F)
    loss, x_hat, mask = ssl(x)
    assert loss.ndim == 0       # scalar
    assert x_hat.shape == (B, F)
    assert mask.shape == (B, F)
    assert mask.dtype == torch.bool

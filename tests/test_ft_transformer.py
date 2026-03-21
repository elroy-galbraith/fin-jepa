"""Unit tests for FT-Transformer."""

import torch
import pytest

from fin_jepa.models.ft_transformer import FeatureTokenizer, FTTransformer


def test_feature_tokenizer_shape():
    B, F, D = 4, 20, 64
    tokenizer = FeatureTokenizer(n_features=F, d_token=D)
    x = torch.randn(B, F)
    tokens = tokenizer(x)
    # Expect B x (F+1) x D  ([CLS] + F feature tokens)
    assert tokens.shape == (B, F + 1, D)


def test_ft_transformer_forward():
    B, F = 4, 20
    model = FTTransformer(n_features=F, d_token=64, n_heads=4, n_layers=2, n_outputs=1)
    x = torch.randn(B, F)
    logits = model(x)
    assert logits.shape == (B, 1)


def test_ft_transformer_representation():
    B, F, D = 4, 20, 64
    model = FTTransformer(n_features=F, d_token=D, n_heads=4, n_layers=2)
    x = torch.randn(B, F)
    repr_ = model.get_representation(x)
    assert repr_.shape == (B, D)

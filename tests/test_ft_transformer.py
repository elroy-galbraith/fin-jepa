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


# ---------------------------------------------------------------------------
# Categorical embedding tests (ATS-173)
# ---------------------------------------------------------------------------

def test_feature_tokenizer_with_categoricals():
    B, F, D = 4, 20, 64
    n_cat = 1
    tokenizer = FeatureTokenizer(
        n_features=F, d_token=D, n_cat_features=n_cat, cat_cardinalities=[12],
    )
    x_num = torch.randn(B, F)
    x_cat = torch.randint(0, 12, (B, n_cat))
    tokens = tokenizer(x_num, x_cat)
    # [CLS] + F numerical + 1 categorical = F + 2
    assert tokens.shape == (B, F + n_cat + 1, D)


def test_feature_tokenizer_no_cat_backward_compat():
    """Without categorical args, FeatureTokenizer behaves identically to before."""
    B, F, D = 4, 20, 64
    tokenizer = FeatureTokenizer(n_features=F, d_token=D)
    x = torch.randn(B, F)
    tokens = tokenizer(x)  # x_cat defaults to None
    assert tokens.shape == (B, F + 1, D)


def test_ft_transformer_with_categoricals():
    B, F = 4, 20
    model = FTTransformer(
        n_features=F, d_token=64, n_heads=4, n_layers=2, n_outputs=1,
        n_cat_features=1, cat_cardinalities=[12],
    )
    x_num = torch.randn(B, F)
    x_cat = torch.randint(0, 12, (B, 1))
    logits = model(x_num, x_cat)
    assert logits.shape == (B, 1)


def test_ft_transformer_repr_with_categoricals():
    B, F, D = 4, 20, 64
    model = FTTransformer(
        n_features=F, d_token=D, n_heads=4, n_layers=2,
        n_cat_features=1, cat_cardinalities=[12],
    )
    x_num = torch.randn(B, F)
    x_cat = torch.randint(0, 12, (B, 1))
    repr_ = model.get_representation(x_num, x_cat)
    assert repr_.shape == (B, D)


def test_feature_tokenizer_bad_cardinalities_raises():
    with pytest.raises(ValueError, match="cat_cardinalities"):
        FeatureTokenizer(n_features=10, d_token=64, n_cat_features=2, cat_cardinalities=[12])

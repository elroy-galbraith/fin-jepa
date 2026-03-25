"""Tests for fin_jepa.training.dataset."""

import numpy as np
import pandas as pd
import pytest
import torch

from fin_jepa.training.dataset import TabularDataset, make_dataloader


class TestTabularDataset:
    def test_shapes_with_labels(self):
        features = np.random.randn(20, 5).astype(np.float32)
        labels = np.random.randint(0, 2, size=20).astype(np.float32)
        ds = TabularDataset(features, labels)

        assert len(ds) == 20
        x, y = ds[0]
        assert x.shape == (5,)
        assert y.shape == ()
        assert x.dtype == torch.float32

    def test_shapes_without_labels(self):
        features = np.random.randn(10, 3).astype(np.float32)
        ds = TabularDataset(features, labels=None)

        assert len(ds) == 10
        (x,) = ds[0]
        assert x.shape == (3,)

    def test_nan_features_replaced(self):
        features = np.array([[1.0, np.nan, 3.0], [np.nan, 2.0, np.nan]])
        ds = TabularDataset(features)

        (x,) = ds[0]
        assert x[1].item() == 0.0
        (x,) = ds[1]
        assert x[0].item() == 0.0
        assert x[2].item() == 0.0

    def test_shapes_with_categoricals_and_labels(self):
        features = np.random.randn(10, 5).astype(np.float32)
        labels = np.random.randint(0, 2, size=10).astype(np.float32)
        cat_features = np.random.randint(0, 12, size=(10, 1)).astype(np.int64)
        ds = TabularDataset(features, labels, cat_features)

        assert len(ds) == 10
        x, x_cat, y = ds[0]
        assert x.shape == (5,)
        assert x_cat.shape == (1,)
        assert x_cat.dtype == torch.int64
        assert y.shape == ()

    def test_shapes_with_categoricals_no_labels(self):
        features = np.random.randn(10, 5).astype(np.float32)
        cat_features = np.random.randint(0, 12, size=(10, 1)).astype(np.int64)
        ds = TabularDataset(features, labels=None, cat_features=cat_features)

        x, x_cat = ds[0]
        assert x.shape == (5,)
        assert x_cat.shape == (1,)


class TestMakeDataloader:
    def _make_df(self):
        return pd.DataFrame({
            "f1": [1.0, 2.0, 3.0, 4.0, 5.0],
            "f2": [10.0, 20.0, 30.0, 40.0, 50.0],
            "label": pd.array([1, 0, pd.NA, 1, pd.NA], dtype="Int8"),
        })

    def test_filters_nan_labels(self):
        df = self._make_df()
        loader = make_dataloader(df, ["f1", "f2"], "label", batch_size=10, shuffle=False)
        xs, ys = next(iter(loader))
        # Only 3 rows have non-null labels
        assert xs.shape == (3, 2)
        assert ys.shape == (3,)

    def test_no_label_col(self):
        df = self._make_df()
        loader = make_dataloader(df, ["f1", "f2"], label_col=None, batch_size=10, shuffle=False)
        (xs,) = next(iter(loader))
        # All 5 rows kept
        assert xs.shape == (5, 2)

    def test_batch_iteration(self):
        df = pd.DataFrame({
            "f1": np.arange(100, dtype=np.float32),
            "label": np.random.randint(0, 2, size=100).astype(np.float32),
        })
        loader = make_dataloader(df, ["f1"], "label", batch_size=32, shuffle=False)
        total = sum(x.shape[0] for x, _ in loader)
        assert total == 100

    def test_with_cat_feature_cols(self):
        df = pd.DataFrame({
            "f1": [1.0, 2.0, 3.0],
            "sector_idx": [0, 5, 11],
            "label": [1.0, 0.0, 1.0],
        })
        loader = make_dataloader(
            df, ["f1"], "label", batch_size=10, shuffle=False,
            cat_feature_cols=["sector_idx"],
        )
        x, x_cat, y = next(iter(loader))
        assert x.shape == (3, 1)
        assert x_cat.shape == (3, 1)
        assert x_cat.dtype == torch.int64
        assert y.shape == (3,)

    def test_no_cat_cols_backward_compat(self):
        df = pd.DataFrame({
            "f1": [1.0, 2.0],
            "label": [1.0, 0.0],
        })
        loader = make_dataloader(df, ["f1"], "label", batch_size=10, shuffle=False)
        x, y = next(iter(loader))
        assert x.shape == (2, 1)
        assert y.shape == (2,)

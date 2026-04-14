"""PyTorch Dataset and DataLoader utilities for tabular financial data."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


class TabularDataset(Dataset):
    """Wraps NumPy feature (and optional label/categorical) arrays as a PyTorch Dataset.

    Residual NaN values in features are replaced with 0.0 at construction
    time as a safety net (the FeatureScaler should already have imputed them).
    """

    def __init__(
        self,
        features: np.ndarray,
        labels: np.ndarray | None = None,
        cat_features: np.ndarray | None = None,
    ) -> None:
        self.features = np.nan_to_num(features.astype(np.float32), nan=0.0)
        if labels is not None:
            self.labels = labels.astype(np.float32)
        else:
            self.labels = None
        if cat_features is not None:
            self.cat_features = cat_features.astype(np.int64)
        else:
            self.cat_features = None

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, ...]:
        x = torch.from_numpy(self.features[idx])
        if self.cat_features is not None:
            x_cat = torch.from_numpy(self.cat_features[idx])
            if self.labels is not None:
                y = torch.tensor(self.labels[idx], dtype=torch.float32)
                return x, x_cat, y
            return x, x_cat
        if self.labels is not None:
            y = torch.tensor(self.labels[idx], dtype=torch.float32)
            return x, y
        return (x,)


def make_dataloader(
    df: pd.DataFrame,
    feature_cols: list[str],
    label_col: str | None = None,
    batch_size: int = 256,
    shuffle: bool = True,
    cat_feature_cols: list[str] | None = None,
    seed: int | None = None,
) -> DataLoader:
    """Build a DataLoader from a DataFrame, filtering NaN labels.

    Parameters
    ----------
    df:
        Source DataFrame containing feature and (optionally) label columns.
    feature_cols:
        Column names to use as model input features (continuous/numerical).
    label_col:
        Column name for the binary label.  Rows where this column is NaN /
        ``pd.NA`` are excluded.  Pass ``None`` for unsupervised (SSL) usage.
    batch_size:
        Mini-batch size.
    shuffle:
        Whether to shuffle the data each epoch.
    cat_feature_cols:
        Column names for categorical features (integer-coded).  These are
        extracted as int64 arrays and passed separately to the model via
        ``nn.Embedding``.  Pass ``None`` or ``[]`` when there are no
        categorical features.
    seed:
        When provided and *shuffle* is True, creates a pinned
        ``torch.Generator`` so that shuffle order is deterministic
        regardless of global RNG state.

    Returns
    -------
    DataLoader wrapping a :class:`TabularDataset`.
    """
    work = df
    labels: np.ndarray | None = None

    if label_col is not None:
        # Filter rows with non-null labels (handles nullable Int8 / float NaN)
        mask = work[label_col].notna()
        work = work.loc[mask]
        labels = work[label_col].to_numpy(dtype=np.float32, na_value=np.nan)

    features = work[feature_cols].to_numpy(dtype=np.float32)

    cat_features: np.ndarray | None = None
    if cat_feature_cols:
        cat_features = work[cat_feature_cols].to_numpy(dtype=np.int64)

    ds = TabularDataset(features, labels, cat_features)
    generator = None
    if shuffle and seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, generator=generator)

"""PyTorch Dataset and DataLoader utilities for tabular financial data."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


class TabularDataset(Dataset):
    """Wraps NumPy feature (and optional label) arrays as a PyTorch Dataset.

    Residual NaN values in features are replaced with 0.0 at construction
    time as a safety net (the FeatureScaler should already have imputed them).
    """

    def __init__(
        self,
        features: np.ndarray,
        labels: np.ndarray | None = None,
    ) -> None:
        self.features = np.nan_to_num(features.astype(np.float32), nan=0.0)
        if labels is not None:
            self.labels = labels.astype(np.float32)
        else:
            self.labels = None

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, ...]:
        x = torch.from_numpy(self.features[idx])
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
) -> DataLoader:
    """Build a DataLoader from a DataFrame, filtering NaN labels.

    Parameters
    ----------
    df:
        Source DataFrame containing feature and (optionally) label columns.
    feature_cols:
        Column names to use as model input features.
    label_col:
        Column name for the binary label.  Rows where this column is NaN /
        ``pd.NA`` are excluded.  Pass ``None`` for unsupervised (SSL) usage.
    batch_size:
        Mini-batch size.
    shuffle:
        Whether to shuffle the data each epoch.

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
    ds = TabularDataset(features, labels)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

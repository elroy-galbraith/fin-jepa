"""
Reproducible train/validation/test splits.

Workstream: Define reproducible train/test splits and publish data specifications.

Design decisions
----------------
- Time-based splits only (no random shuffle) to prevent look-ahead bias
- Splits defined by fiscal period end date cutoffs
- Split spec is version-controlled as a YAML file in configs/

TODO:
  - Define canonical cutoff dates (e.g. train <2018, val 2018-2020, test >2020)
  - Ensure no entity appears in both train and test (entity-aware split)
  - Write split manifest to results/study0/data_spec.yaml
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass
class SplitConfig:
    train_end: str        # e.g. "2017-12-31"
    val_end: str          # e.g. "2019-12-31"
    test_end: str         # e.g. "2023-12-31"
    date_col: str = "period_end"


def make_splits(df: pd.DataFrame, config: SplitConfig) -> dict[str, pd.DataFrame]:
    """Return {'train': ..., 'val': ..., 'test': ...} DataFrames."""
    dates = pd.to_datetime(df[config.date_col])
    return {
        "train": df[dates <= config.train_end],
        "val": df[(dates > config.train_end) & (dates <= config.val_end)],
        "test": df[(dates > config.val_end) & (dates <= config.test_end)],
    }

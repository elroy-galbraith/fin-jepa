"""
Reproducible train/validation/test splits.

Workstream: Define reproducible train/test splits and publish data specifications.

Design decisions
----------------
- Time-based splits only (no random shuffle) to prevent look-ahead bias
- Splits defined by fiscal period end date cutoffs
- Split spec is version-controlled as a YAML file in configs/
- Canonical cutoffs: train ≤2017-12-31, val 2018–2019, test 2020–2023
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass
class SplitConfig:
    train_end: str        # e.g. "2017-12-31"
    val_end: str          # e.g. "2019-12-31"
    test_end: str         # e.g. "2023-12-31"
    date_col: str = "period_end"


@dataclass
class RollingSplitConfig:
    """Configuration for expanding-window walk-forward splits.

    Each fold has an expanding training set (from the start of data up to
    that fold's train_end), a fixed-length validation window, and a
    fixed-length test window.
    """
    first_train_end: str       # e.g. "2014-12-31" — earliest train cutoff
    val_window_years: int      # e.g. 1
    test_window_years: int     # e.g. 2
    step_years: int            # e.g. 1 — how far to advance each fold
    last_test_end: str         # e.g. "2023-12-31" — latest data boundary
    date_col: str = "period_end"


def make_splits(df: pd.DataFrame, config: SplitConfig) -> dict[str, pd.DataFrame]:
    """Return {'train': ..., 'val': ..., 'test': ...} DataFrames."""
    dates = pd.to_datetime(df[config.date_col])
    return {
        "train": df[dates <= config.train_end],
        "val": df[(dates > config.train_end) & (dates <= config.val_end)],
        "test": df[(dates > config.val_end) & (dates <= config.test_end)],
    }


def make_rolling_splits(
    df: pd.DataFrame,
    config: RollingSplitConfig,
) -> list[dict[str, pd.DataFrame]]:
    """Generate expanding-window walk-forward splits for robustness checks.

    Each fold advances by ``step_years``.  Training is always expanding
    (from the start of the data up to the fold's train cutoff).

    Returns a list of ``{'train': ..., 'val': ..., 'test': ...}`` dicts,
    one per fold.
    """
    dates = pd.to_datetime(df[config.date_col])
    train_end = pd.Timestamp(config.first_train_end)
    last_test = pd.Timestamp(config.last_test_end)
    step = pd.DateOffset(years=config.step_years)
    val_offset = pd.DateOffset(years=config.val_window_years)
    test_offset = pd.DateOffset(years=config.test_window_years)

    folds: list[dict[str, pd.DataFrame]] = []
    while True:
        val_end = train_end + val_offset
        test_end = val_end + test_offset
        if test_end > last_test:
            break
        folds.append({
            "train": df[dates <= train_end],
            "val": df[(dates > train_end) & (dates <= val_end)],
            "test": df[(dates > val_end) & (dates <= test_end)],
        })
        train_end = train_end + step

    return folds


def describe_splits(
    splits: dict[str, pd.DataFrame] | list[dict[str, pd.DataFrame]],
    date_col: str = "period_end",
    cik_col: str = "cik",
) -> dict[str, Any]:
    """Compute summary statistics for one split dict or a list of fold dicts.

    Returns a dict with row counts, date ranges, and unique-company counts
    per split partition.
    """
    if isinstance(splits, list):
        return {
            "n_folds": len(splits),
            "folds": [describe_splits(fold, date_col, cik_col) for fold in splits],
        }

    summary: dict[str, Any] = {}
    for name, part in splits.items():
        dates = pd.to_datetime(part[date_col]) if date_col in part.columns else None
        info: dict[str, Any] = {"n_rows": len(part)}
        if cik_col in part.columns:
            info["n_companies"] = int(part[cik_col].nunique())
        if dates is not None and len(part) > 0:
            info["date_min"] = str(dates.min().date())
            info["date_max"] = str(dates.max().date())
        summary[name] = info
    return summary

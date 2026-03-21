"""Unit tests for data splitting utilities."""

import pandas as pd
import pytest

from fin_jepa.data.splits import SplitConfig, make_splits


def test_make_splits_no_overlap():
    df = pd.DataFrame({
        "period_end": pd.date_range("2010-01-01", periods=100, freq="QE"),
        "value": range(100),
    })
    config = SplitConfig(train_end="2017-12-31", val_end="2019-12-31", test_end="2023-12-31")
    splits = make_splits(df, config)
    assert set(splits.keys()) == {"train", "val", "test"}
    total = sum(len(v) for v in splits.values())
    assert total == len(df)
    # No overlap
    train_idx = set(splits["train"].index)
    val_idx = set(splits["val"].index)
    test_idx = set(splits["test"].index)
    assert train_idx.isdisjoint(val_idx)
    assert val_idx.isdisjoint(test_idx)

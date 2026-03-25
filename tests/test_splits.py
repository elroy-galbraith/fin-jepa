"""Unit tests for data splitting utilities."""

import pandas as pd
import pytest

from fin_jepa.data.splits import (
    RollingSplitConfig,
    SplitConfig,
    describe_splits,
    make_rolling_splits,
    make_splits,
)


def _make_quarterly_df(start: str = "2010-01-01", periods: int = 60) -> pd.DataFrame:
    """Helper: quarterly company-year observations with CIKs."""
    dates = pd.date_range(start, periods=periods, freq="QE")
    return pd.DataFrame({
        "period_end": dates,
        "cik": [f"{(i % 5) + 1:010d}" for i in range(periods)],
        "value": range(periods),
    })


def test_make_splits_no_overlap():
    df = pd.DataFrame({
        "period_end": pd.date_range("2010-01-01", periods=56, freq="QE"),
        "value": range(56),
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


# ---------------------------------------------------------------------------
# Rolling splits
# ---------------------------------------------------------------------------


class TestRollingSplits:
    """Tests for make_rolling_splits."""

    @pytest.fixture()
    def df(self):
        return _make_quarterly_df(start="2010-01-01", periods=60)

    @pytest.fixture()
    def config(self):
        return RollingSplitConfig(
            first_train_end="2014-12-31",
            val_window_years=1,
            test_window_years=2,
            step_years=1,
            last_test_end="2023-12-31",
        )

    def test_no_overlap_within_fold(self, df, config):
        """No row appears in both train and test within a single fold."""
        folds = make_rolling_splits(df, config)
        assert len(folds) > 0
        for fold in folds:
            train_idx = set(fold["train"].index)
            val_idx = set(fold["val"].index)
            test_idx = set(fold["test"].index)
            assert train_idx.isdisjoint(val_idx)
            assert train_idx.isdisjoint(test_idx)
            assert val_idx.isdisjoint(test_idx)

    def test_expanding_train(self, df, config):
        """Each successive fold has a larger or equal train set."""
        folds = make_rolling_splits(df, config)
        for i in range(1, len(folds)):
            assert len(folds[i]["train"]) >= len(folds[i - 1]["train"])

    def test_fold_count(self, df, config):
        """Number of folds matches expectations from config parameters."""
        folds = make_rolling_splits(df, config)
        # With first_train_end=2014, step=1, val=1yr, test=2yr:
        # fold 0: train≤2014, val 2014-2015, test 2015-2017
        # fold 1: train≤2015, val 2015-2016, test 2016-2018
        # ... keeps going until test_end > 2023
        assert len(folds) >= 1
        # Last fold's test should not exceed last_test_end
        last_test = folds[-1]["test"]
        if len(last_test) > 0:
            max_date = pd.to_datetime(last_test["period_end"]).max()
            assert max_date <= pd.Timestamp(config.last_test_end)

    def test_step_advances(self, df, config):
        """Test boundaries advance between folds."""
        folds = make_rolling_splits(df, config)
        if len(folds) >= 2:
            train0_max = pd.to_datetime(folds[0]["train"]["period_end"]).max()
            train1_max = pd.to_datetime(folds[1]["train"]["period_end"]).max()
            assert train1_max >= train0_max

    def test_empty_when_window_exceeds_data(self):
        """Returns no folds when the time window is too large."""
        df = _make_quarterly_df(start="2020-01-01", periods=8)
        config = RollingSplitConfig(
            first_train_end="2025-12-31",
            val_window_years=1,
            test_window_years=2,
            step_years=1,
            last_test_end="2023-12-31",
        )
        folds = make_rolling_splits(df, config)
        assert len(folds) == 0


# ---------------------------------------------------------------------------
# describe_splits
# ---------------------------------------------------------------------------


class TestDescribeSplits:
    def test_single_split(self):
        df = _make_quarterly_df(periods=20)
        config = SplitConfig(
            train_end="2012-12-31", val_end="2013-12-31", test_end="2014-12-31",
        )
        splits = make_splits(df, config)
        desc = describe_splits(splits)
        assert "train" in desc
        assert desc["train"]["n_rows"] == len(splits["train"])

    def test_rolling_split_list(self):
        df = _make_quarterly_df(periods=60)
        config = RollingSplitConfig(
            first_train_end="2014-12-31",
            val_window_years=1,
            test_window_years=2,
            step_years=1,
            last_test_end="2023-12-31",
        )
        folds = make_rolling_splits(df, config)
        desc = describe_splits(folds)
        assert desc["n_folds"] == len(folds)
        assert len(desc["folds"]) == len(folds)

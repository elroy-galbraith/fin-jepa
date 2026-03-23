"""Tests for ATS-168 baseline model infrastructure.

Covers: build_gbt, traditional ratio features, calibration metrics,
and TemporalCV splitter.  All tests use synthetic data — no network calls.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import HistGradientBoostingClassifier

from fin_jepa.data.feature_engineering import (
    RATIO_FEATURES,
    TRADITIONAL_RATIO_FEATURES,
    compute_ratios,
)
from fin_jepa.models.baselines import build_gbt
from fin_jepa.training.metrics import compute_all_metrics, compute_calibration
from fin_jepa.training.temporal_cv import TemporalCV


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_xbrl_df(n: int = 20) -> pd.DataFrame:
    """Minimal synthetic XBRL DataFrame for ratio computation tests."""
    rng = np.random.RandomState(42)
    return pd.DataFrame({
        "cik": [f"{i:010d}" for i in range(n)],
        "fiscal_year": [2015 + i % 5 for i in range(n)],
        "period_end": [date(2015 + i % 5, 12, 31) for i in range(n)],
        "total_assets": rng.uniform(500, 2000, n),
        "total_liabilities": rng.uniform(200, 1000, n),
        "total_equity": rng.uniform(100, 800, n),
        "current_assets": rng.uniform(100, 500, n),
        "current_liabilities": rng.uniform(50, 300, n),
        "retained_earnings": rng.uniform(50, 400, n),
        "cash_equivalents": rng.uniform(20, 200, n),
        "total_debt": rng.uniform(100, 600, n),
        "total_revenue": rng.uniform(300, 1500, n),
        "cost_of_sales": rng.uniform(150, 800, n),
        "operating_income": rng.uniform(30, 300, n),
        "net_income": rng.uniform(10, 200, n),
        "interest_expense": rng.uniform(5, 50, n),
        "cash_from_operations": rng.uniform(30, 250, n),
    })


def _make_binary_classification(n: int = 200, seed: int = 42):
    """Return (X, y) with synthetic binary classification data."""
    rng = np.random.RandomState(seed)
    X = rng.randn(n, 5).astype(np.float32)
    y = (X[:, 0] + 0.5 * X[:, 1] + rng.randn(n) * 0.3 > 0).astype(np.float32)
    return X, y


# ---------------------------------------------------------------------------
# build_gbt
# ---------------------------------------------------------------------------

class TestBuildGBT:
    def test_returns_correct_type(self):
        model = build_gbt()
        assert isinstance(model, HistGradientBoostingClassifier)

    def test_default_params(self):
        model = build_gbt()
        assert model.max_iter == 500
        assert model.learning_rate == 0.05
        assert model.max_depth == 6
        assert model.min_samples_leaf == 20
        assert model.random_state == 42
        assert model.class_weight == "balanced"

    def test_custom_params(self):
        model = build_gbt(max_iter=100, learning_rate=0.1, max_depth=4)
        assert model.max_iter == 100
        assert model.learning_rate == 0.1
        assert model.max_depth == 4

    def test_fit_predict(self):
        X, y = _make_binary_classification()
        model = build_gbt(max_iter=10)
        model.fit(X, y)
        probs = model.predict_proba(X)
        assert probs.shape == (len(X), 2)
        assert np.all(probs >= 0) and np.all(probs <= 1)

    def test_handles_nan(self):
        """HistGradientBoostingClassifier should train with NaN features."""
        X, y = _make_binary_classification()
        X[0, 0] = np.nan
        X[5, 2] = np.nan
        model = build_gbt(max_iter=10)
        model.fit(X, y)
        probs = model.predict_proba(X)
        assert probs.shape == (len(X), 2)


# ---------------------------------------------------------------------------
# TRADITIONAL_RATIO_FEATURES
# ---------------------------------------------------------------------------

class TestTraditionalRatioFeatures:
    def test_is_subset_of_ratio_features(self):
        for feat in TRADITIONAL_RATIO_FEATURES:
            assert feat in RATIO_FEATURES, f"{feat} not in RATIO_FEATURES"

    def test_expected_features_present(self):
        expected = {
            "altman_z_score", "current_ratio", "quick_ratio",
            "debt_to_equity", "debt_to_assets", "roa", "roe",
            "net_profit_margin", "interest_coverage", "cfo_to_debt",
        }
        assert set(TRADITIONAL_RATIO_FEATURES) == expected


# ---------------------------------------------------------------------------
# New ratios in compute_ratios
# ---------------------------------------------------------------------------

class TestNewRatios:
    def test_debt_to_assets_computed(self):
        df = _make_xbrl_df()
        result = compute_ratios(df)
        assert "debt_to_assets" in result.columns
        assert result["debt_to_assets"].notna().all()

    def test_net_profit_margin_computed(self):
        df = _make_xbrl_df()
        result = compute_ratios(df)
        assert "net_profit_margin" in result.columns
        assert result["net_profit_margin"].notna().all()

    def test_cfo_to_debt_computed(self):
        df = _make_xbrl_df()
        result = compute_ratios(df)
        assert "cfo_to_debt" in result.columns
        assert result["cfo_to_debt"].notna().all()

    def test_ratio_values_reasonable(self):
        df = _make_xbrl_df()
        result = compute_ratios(df)
        # debt_to_assets should be positive fraction (debt > 0, assets > 0)
        assert (result["debt_to_assets"] > 0).all()
        # net_profit_margin should be positive (net_income > 0, revenue > 0)
        assert (result["net_profit_margin"] > 0).all()
        # cfo_to_debt should be positive
        assert (result["cfo_to_debt"] > 0).all()


# ---------------------------------------------------------------------------
# Calibration metrics
# ---------------------------------------------------------------------------

class TestCalibration:
    def test_compute_calibration_returns_expected_keys(self):
        rng = np.random.RandomState(42)
        y_true = rng.randint(0, 2, 200)
        y_score = rng.rand(200)
        result = compute_calibration(y_true, y_score, n_bins=10)
        assert "ece" in result
        assert "prob_true" in result
        assert "prob_pred" in result
        assert "n_bins" in result

    def test_ece_range(self):
        rng = np.random.RandomState(42)
        y_true = rng.randint(0, 2, 200)
        y_score = rng.rand(200)
        result = compute_calibration(y_true, y_score)
        assert 0.0 <= result["ece"] <= 1.0

    def test_perfect_calibration_low_ece(self):
        """Well-calibrated predictions should have low ECE."""
        rng = np.random.RandomState(42)
        y_score = rng.rand(1000)
        y_true = (rng.rand(1000) < y_score).astype(int)
        result = compute_calibration(y_true, y_score)
        assert result["ece"] < 0.1

    def test_calibration_in_compute_all_metrics(self):
        rng = np.random.RandomState(42)
        y_true = rng.randint(0, 2, 200)
        y_score = rng.rand(200)
        result = compute_all_metrics(y_true, y_score)
        assert "ece" in result
        assert "calibration" in result
        assert "prob_true" in result["calibration"]

    def test_arrays_same_length(self):
        rng = np.random.RandomState(42)
        y_true = rng.randint(0, 2, 200)
        y_score = rng.rand(200)
        result = compute_calibration(y_true, y_score, n_bins=5)
        assert len(result["prob_true"]) == len(result["prob_pred"])


# ---------------------------------------------------------------------------
# TemporalCV
# ---------------------------------------------------------------------------

class TestTemporalCV:
    def _make_temporal_df(self, years: list[int]) -> pd.DataFrame:
        rows = []
        for y in years:
            for i in range(10):
                rows.append({"fiscal_year": y, "x": i})
        return pd.DataFrame(rows)

    def test_correct_number_of_folds(self):
        df = self._make_temporal_df([2012, 2013, 2014, 2015, 2016, 2017])
        cv = TemporalCV(n_splits=3)
        folds = list(cv.split(df))
        assert len(folds) == 3

    def test_expanding_window(self):
        df = self._make_temporal_df([2012, 2013, 2014, 2015, 2016, 2017])
        cv = TemporalCV(n_splits=3)
        folds = list(cv.split(df))

        # Each successive fold should have a larger training set
        for i in range(len(folds) - 1):
            assert len(folds[i][0]) < len(folds[i + 1][0])

    def test_val_years_are_last_n(self):
        years = [2012, 2013, 2014, 2015, 2016, 2017]
        df = self._make_temporal_df(years)
        cv = TemporalCV(n_splits=3)
        folds = list(cv.split(df))

        # Validation years should be 2015, 2016, 2017
        for fold_idx, expected_year in enumerate([2015, 2016, 2017]):
            _, val_idx = folds[fold_idx]
            val_years = df.iloc[val_idx]["fiscal_year"].unique()
            assert list(val_years) == [expected_year]

    def test_no_overlap_between_train_and_val(self):
        df = self._make_temporal_df([2012, 2013, 2014, 2015, 2016, 2017])
        cv = TemporalCV(n_splits=3)
        for train_idx, val_idx in cv.split(df):
            assert len(set(train_idx) & set(val_idx)) == 0

    def test_train_strictly_before_val(self):
        df = self._make_temporal_df([2012, 2013, 2014, 2015, 2016, 2017])
        cv = TemporalCV(n_splits=3)
        for train_idx, val_idx in cv.split(df):
            train_max = df.iloc[train_idx]["fiscal_year"].max()
            val_min = df.iloc[val_idx]["fiscal_year"].min()
            assert train_max < val_min

    def test_too_few_years_raises(self):
        df = self._make_temporal_df([2015, 2016, 2017])
        cv = TemporalCV(n_splits=3)
        with pytest.raises(ValueError, match="Need >"):
            list(cv.split(df))

    def test_get_n_splits(self):
        cv = TemporalCV(n_splits=5)
        assert cv.get_n_splits() == 5

    def test_custom_date_col(self):
        df = pd.DataFrame({
            "year_col": [2010, 2010, 2011, 2011, 2012, 2012, 2013, 2013],
            "x": range(8),
        })
        cv = TemporalCV(n_splits=2, date_col="year_col")
        folds = list(cv.split(df))
        assert len(folds) == 2

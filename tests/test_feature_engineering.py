"""Tests for fin_jepa.data.feature_engineering.

All tests use synthetic DataFrames — no real data or network calls.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from fin_jepa.data.feature_engineering import (
    RATIO_FEATURES,
    RAW_FEATURES,
    YOY_BASE_FEATURES,
    YOY_FEATURES,
    FeatureConfig,
    FeatureScaler,
    _safe_div,
    add_missingness_flags,
    apply_scaler,
    build_feature_matrix,
    compute_ratios,
    compute_yoy_changes,
    coverage_report,
    fit_scaler,
    prune_low_coverage,
)
from fin_jepa.data.splits import SplitConfig


# ---------------------------------------------------------------------------
# Helpers: synthetic data
# ---------------------------------------------------------------------------

def _make_xbrl_df(
    n_companies: int = 3,
    n_years: int = 4,
    start_year: int = 2015,
    inject_nans: bool = False,
) -> pd.DataFrame:
    """Create a synthetic XBRL-pipeline-style DataFrame.

    Each company gets *n_years* consecutive fiscal years with plausible
    values so that ratio computations work.
    """
    rows = []
    for i in range(n_companies):
        cik = f"{(i + 1):010d}"
        for j in range(n_years):
            fy = start_year + j
            rows.append({
                "cik": cik,
                "ticker": f"TKR{i}",
                "fiscal_year": fy,
                "period_end": date(fy, 12, 31),
                "filed_date": date(fy + 1, 2, 15),
                # Balance sheet
                "total_assets": 1000.0 + i * 100 + j * 50,
                "total_liabilities": 600.0 + i * 50 + j * 20,
                "total_equity": 400.0 + i * 50 + j * 30,
                "current_assets": 300.0 + j * 10,
                "current_liabilities": 200.0 + j * 5,
                "retained_earnings": 150.0 + j * 20,
                "cash_equivalents": 80.0 + j * 5,
                "total_debt": 250.0 + j * 10,
                # Income statement
                "total_revenue": 500.0 + i * 30 + j * 25,
                "cost_of_sales": 300.0 + i * 15 + j * 10,
                "operating_income": 100.0 + i * 10 + j * 5,
                "net_income": 60.0 + i * 8 + j * 3,
                "interest_expense": 20.0 + j * 1,
                # Cash flow
                "cash_from_operations": 80.0 + j * 5,
                "cash_from_investing": -40.0 - j * 2,
                "cash_from_financing": -30.0 - j * 3,
            })
    df = pd.DataFrame(rows)

    if inject_nans:
        # Sprinkle NaN into a few cells
        df.loc[0, "interest_expense"] = np.nan
        df.loc[1, "retained_earnings"] = np.nan
        df.loc[2, "cost_of_sales"] = np.nan

    return df


# ---------------------------------------------------------------------------
# TestSafeDiv
# ---------------------------------------------------------------------------

class TestSafeDiv:

    def test_normal_division(self):
        num = pd.Series([10.0, 20.0, 30.0])
        den = pd.Series([2.0, 5.0, 10.0])
        result = _safe_div(num, den)
        expected = pd.Series([5.0, 4.0, 3.0])
        pd.testing.assert_series_equal(result, expected)

    def test_zero_denominator_returns_nan(self):
        num = pd.Series([10.0, 20.0])
        den = pd.Series([0.0, 5.0])
        result = _safe_div(num, den)
        assert np.isnan(result.iloc[0])
        assert result.iloc[1] == pytest.approx(4.0)

    def test_nan_propagation(self):
        num = pd.Series([np.nan, 10.0])
        den = pd.Series([5.0, np.nan])
        result = _safe_div(num, den)
        assert np.isnan(result.iloc[0])
        assert np.isnan(result.iloc[1])

    def test_near_zero_denominator(self):
        num = pd.Series([10.0])
        den = pd.Series([1e-12])
        result = _safe_div(num, den)
        assert np.isnan(result.iloc[0])


# ---------------------------------------------------------------------------
# TestComputeRatios
# ---------------------------------------------------------------------------

class TestComputeRatios:

    def test_all_ratios_present(self):
        df = _make_xbrl_df(n_companies=1, n_years=1)
        result = compute_ratios(df)
        for ratio in RATIO_FEATURES:
            assert ratio in result.columns, f"Missing ratio: {ratio}"

    def test_debt_to_equity_values(self):
        df = _make_xbrl_df(n_companies=1, n_years=1)
        result = compute_ratios(df)
        expected = df["total_debt"].iloc[0] / df["total_equity"].iloc[0]
        assert result["debt_to_equity"].iloc[0] == pytest.approx(expected)

    def test_current_ratio_values(self):
        df = _make_xbrl_df(n_companies=1, n_years=1)
        result = compute_ratios(df)
        expected = df["current_assets"].iloc[0] / df["current_liabilities"].iloc[0]
        assert result["current_ratio"].iloc[0] == pytest.approx(expected)

    def test_roa_values(self):
        df = _make_xbrl_df(n_companies=1, n_years=1)
        result = compute_ratios(df)
        expected = df["net_income"].iloc[0] / df["total_assets"].iloc[0]
        assert result["roa"].iloc[0] == pytest.approx(expected)

    def test_gross_margin_values(self):
        df = _make_xbrl_df(n_companies=1, n_years=1)
        result = compute_ratios(df)
        rev = df["total_revenue"].iloc[0]
        cogs = df["cost_of_sales"].iloc[0]
        expected = (rev - cogs) / rev
        assert result["gross_margin"].iloc[0] == pytest.approx(expected)

    def test_altman_z_score_components(self):
        df = _make_xbrl_df(n_companies=1, n_years=1)
        result = compute_ratios(df)
        r = df.iloc[0]
        wc = r["current_assets"] - r["current_liabilities"]
        z = (
            1.2 * (wc / r["total_assets"])
            + 1.4 * (r["retained_earnings"] / r["total_assets"])
            + 3.3 * (r["operating_income"] / r["total_assets"])
            + 0.6 * (r["total_equity"] / r["total_liabilities"])
            + 1.0 * (r["total_revenue"] / r["total_assets"])
        )
        assert result["altman_z_score"].iloc[0] == pytest.approx(z)

    def test_zero_equity_gives_nan_roe(self):
        df = _make_xbrl_df(n_companies=1, n_years=1)
        df["total_equity"] = 0.0
        result = compute_ratios(df)
        assert np.isnan(result["roe"].iloc[0])
        assert np.isnan(result["debt_to_equity"].iloc[0])

    def test_missing_input_gives_nan_ratio(self):
        df = _make_xbrl_df(n_companies=1, n_years=1)
        df.loc[0, "interest_expense"] = np.nan
        result = compute_ratios(df)
        assert np.isnan(result["interest_coverage"].iloc[0])

    def test_missing_column_gives_nan(self):
        df = _make_xbrl_df(n_companies=1, n_years=1)
        df = df.drop(columns=["interest_expense"])
        result = compute_ratios(df)
        assert np.isnan(result["interest_coverage"].iloc[0])

    def test_returns_copy(self):
        df = _make_xbrl_df(n_companies=1, n_years=1)
        result = compute_ratios(df)
        assert result is not df


# ---------------------------------------------------------------------------
# TestComputeYoYChanges
# ---------------------------------------------------------------------------

class TestComputeYoYChanges:

    def test_basic_yoy(self):
        df = _make_xbrl_df(n_companies=1, n_years=2)
        result = compute_yoy_changes(df, ["total_revenue"])
        # First year should be NaN
        assert np.isnan(result["total_revenue_yoy"].iloc[0])
        # Second year should have the correct change
        rev0 = df.iloc[0]["total_revenue"]
        rev1 = df.iloc[1]["total_revenue"]
        expected = (rev1 - rev0) / abs(rev0)
        assert result["total_revenue_yoy"].iloc[1] == pytest.approx(expected)

    def test_negative_base(self):
        df = _make_xbrl_df(n_companies=1, n_years=2)
        df["net_income"] = [-100.0, -50.0]
        result = compute_yoy_changes(df, ["net_income"])
        # (-50 - (-100)) / |-100| = 50/100 = 0.5
        assert result["net_income_yoy"].iloc[1] == pytest.approx(0.5)

    def test_zero_prior_gives_nan(self):
        df = _make_xbrl_df(n_companies=1, n_years=2)
        df["total_revenue"] = [0.0, 100.0]
        result = compute_yoy_changes(df, ["total_revenue"])
        assert np.isnan(result["total_revenue_yoy"].iloc[1])

    def test_multiple_companies(self):
        df = _make_xbrl_df(n_companies=2, n_years=2)
        result = compute_yoy_changes(df, ["total_assets"])
        # Each company's first year should be NaN
        for cik in df["cik"].unique():
            mask = result["cik"] == cik
            group = result[mask].sort_values("fiscal_year")
            assert np.isnan(group["total_assets_yoy"].iloc[0])
            assert not np.isnan(group["total_assets_yoy"].iloc[1])

    def test_column_naming(self):
        df = _make_xbrl_df(n_companies=1, n_years=2)
        result = compute_yoy_changes(df, ["total_revenue", "net_income"])
        assert "total_revenue_yoy" in result.columns
        assert "net_income_yoy" in result.columns

    def test_missing_feature_column(self):
        df = _make_xbrl_df(n_companies=1, n_years=2)
        result = compute_yoy_changes(df, ["nonexistent_col"])
        assert "nonexistent_col_yoy" in result.columns
        assert result["nonexistent_col_yoy"].isna().all()

    def test_default_features(self):
        df = _make_xbrl_df(n_companies=1, n_years=2)
        result = compute_yoy_changes(df)
        for col in YOY_BASE_FEATURES:
            assert f"{col}_yoy" in result.columns


# ---------------------------------------------------------------------------
# TestCoverageReport
# ---------------------------------------------------------------------------

class TestCoverageReport:

    def test_full_coverage(self):
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
        report = coverage_report(df, ["a", "b"])
        assert report["a"] == pytest.approx(1.0)
        assert report["b"] == pytest.approx(1.0)

    def test_partial_coverage(self):
        df = pd.DataFrame({"a": [1.0, np.nan, 3.0]})
        report = coverage_report(df, ["a"])
        assert report["a"] == pytest.approx(2 / 3)

    def test_all_null(self):
        df = pd.DataFrame({"a": [np.nan, np.nan]})
        report = coverage_report(df, ["a"])
        assert report["a"] == pytest.approx(0.0)

    def test_empty_dataframe(self):
        df = pd.DataFrame({"a": pd.Series([], dtype=float)})
        report = coverage_report(df, ["a"])
        assert report["a"] == 0.0


# ---------------------------------------------------------------------------
# TestPruneLowCoverage
# ---------------------------------------------------------------------------

class TestPruneLowCoverage:

    def test_drops_low_coverage_column(self):
        df = pd.DataFrame({
            "cik": ["001", "002", "003"],
            "a": [1.0, np.nan, np.nan],  # 33% coverage
            "b": [1.0, 2.0, 3.0],        # 100% coverage
        })
        pruned, kept = prune_low_coverage(df, ["a", "b"], threshold=0.50)
        assert "a" not in pruned.columns
        assert "b" in pruned.columns
        assert "a" not in kept
        assert "b" in kept

    def test_keeps_high_coverage_column(self):
        df = pd.DataFrame({
            "a": [1.0, 2.0, np.nan],  # 67% coverage
        })
        pruned, kept = prune_low_coverage(df, ["a"], threshold=0.50)
        assert "a" in pruned.columns
        assert "a" in kept

    def test_identifier_columns_not_dropped(self):
        df = pd.DataFrame({
            "cik": ["001", "002"],
            "fiscal_year": [2020, 2021],
            "a": [np.nan, np.nan],  # 0% coverage
        })
        pruned, kept = prune_low_coverage(df, ["a"], threshold=0.50)
        assert "cik" in pruned.columns
        assert "fiscal_year" in pruned.columns

    def test_returns_kept_columns_list(self):
        df = pd.DataFrame({
            "a": [1.0, 2.0],
            "b": [np.nan, np.nan],
            "c": [1.0, np.nan],
        })
        _, kept = prune_low_coverage(df, ["a", "b", "c"], threshold=0.50)
        assert "a" in kept
        assert "b" not in kept
        assert "c" in kept


# ---------------------------------------------------------------------------
# TestAddMissingnessFlags
# ---------------------------------------------------------------------------

class TestAddMissingnessFlags:

    def test_flag_columns_created(self):
        df = pd.DataFrame({"a": [1.0, np.nan], "b": [np.nan, 2.0]})
        result, flags = add_missingness_flags(df, ["a", "b"])
        assert "a_missing" in result.columns
        assert "b_missing" in result.columns
        assert flags == ["a_missing", "b_missing"]

    def test_flag_values(self):
        df = pd.DataFrame({"a": [1.0, np.nan, 3.0]})
        result, _ = add_missingness_flags(df, ["a"])
        expected = pd.Series([0, 1, 0], dtype=np.int8, name="a_missing")
        pd.testing.assert_series_equal(result["a_missing"], expected)

    def test_flag_column_naming(self):
        df = pd.DataFrame({"total_assets": [1.0], "roa": [0.5]})
        _, flags = add_missingness_flags(df, ["total_assets", "roa"])
        assert flags == ["total_assets_missing", "roa_missing"]

    def test_returns_copy(self):
        df = pd.DataFrame({"a": [1.0]})
        result, _ = add_missingness_flags(df, ["a"])
        assert result is not df


# ---------------------------------------------------------------------------
# TestFeatureScaler
# ---------------------------------------------------------------------------

class TestFeatureScaler:

    def _make_train_df(self, n: int = 200) -> pd.DataFrame:
        """Create a training DataFrame with known distributions."""
        rng = np.random.RandomState(42)
        return pd.DataFrame({
            "a": rng.normal(100, 20, n),
            "b": rng.exponential(50, n),
            "c": rng.uniform(-10, 10, n),
        })

    def test_fit_stores_state(self):
        df = self._make_train_df()
        scaler = FeatureScaler()
        scaler.fit(df, ["a", "b", "c"])
        assert scaler._fitted is True
        assert "a" in scaler._medians
        assert "a" in scaler._percentiles

    def test_transform_before_fit_raises(self):
        scaler = FeatureScaler()
        df = self._make_train_df()
        with pytest.raises(RuntimeError, match="not been fitted"):
            scaler.transform(df)

    def test_transform_winsorizes(self):
        df = self._make_train_df(500)
        scaler = FeatureScaler(winsorize_limits=(0.05, 0.95))
        scaler.fit(df, ["a", "b", "c"])
        result = scaler.transform(df)
        # After winsorisation + quantile transform, no extreme outliers
        # The quantile transform maps to normal, so values should be bounded
        for col in ["a", "b", "c"]:
            assert result[col].max() < 10  # normal dist, bounded
            assert result[col].min() > -10

    def test_quantile_output_approximately_normal(self):
        df = self._make_train_df(1000)
        scaler = FeatureScaler(method="quantile")
        result = scaler.fit_transform(df, ["a", "b", "c"])
        for col in ["a", "b", "c"]:
            assert abs(result[col].mean()) < 0.2
            assert abs(result[col].std() - 1.0) < 0.2

    def test_zscore_method(self):
        df = self._make_train_df(500)
        scaler = FeatureScaler(method="zscore")
        result = scaler.fit_transform(df, ["a", "b", "c"])
        for col in ["a", "b", "c"]:
            assert abs(result[col].mean()) < 0.1
            assert abs(result[col].std() - 1.0) < 0.15

    def test_median_imputation(self):
        df = self._make_train_df(100)
        df.loc[0, "a"] = np.nan
        df.loc[1, "b"] = np.nan
        scaler = FeatureScaler(method="zscore")
        scaler.fit(df, ["a", "b", "c"])
        result = scaler.transform(df)
        # After imputation, no NaN values in feature columns
        assert not result["a"].isna().any()
        assert not result["b"].isna().any()

    def test_train_only_statistics(self):
        train = pd.DataFrame({"a": [10.0, 20.0, 30.0, 40.0, 50.0]})
        val = pd.DataFrame({"a": [100.0, 200.0]})
        scaler = FeatureScaler(method="zscore")
        scaler.fit(train, ["a"])
        # Train median should be 30.0
        assert scaler._medians["a"] == pytest.approx(30.0)
        # Val values should be transformed using train stats, not val stats
        result = scaler.transform(val)
        assert result["a"].iloc[0] != pytest.approx(0.0)  # not centered on val mean

    def test_missingness_flags_not_scaled(self):
        df = pd.DataFrame({
            "a": [1.0, 2.0, 3.0, 4.0, 5.0],
            "a_missing": np.int8([0, 0, 1, 0, 0]),
        })
        scaler = FeatureScaler(method="zscore")
        scaler.fit(df, ["a"])  # Only fit on "a", not "a_missing"
        result = scaler.transform(df)
        # a_missing should be unchanged
        pd.testing.assert_series_equal(
            result["a_missing"], df["a_missing"], check_names=True
        )

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError, match="Unknown normalization method"):
            scaler = FeatureScaler(method="invalid")
            scaler.fit(pd.DataFrame({"a": [1.0, 2.0]}), ["a"])

    def test_fit_transform_same_as_fit_then_transform(self):
        df = self._make_train_df(100)
        scaler1 = FeatureScaler(method="zscore")
        result1 = scaler1.fit_transform(df, ["a", "b", "c"])

        scaler2 = FeatureScaler(method="zscore")
        scaler2.fit(df, ["a", "b", "c"])
        result2 = scaler2.transform(df)

        pd.testing.assert_frame_equal(result1, result2)


# ---------------------------------------------------------------------------
# TestBuildFeatureMatrix
# ---------------------------------------------------------------------------

class TestBuildFeatureMatrix:

    def test_end_to_end(self):
        df = _make_xbrl_df(n_companies=3, n_years=4, start_year=2015)
        splits, scaler, features = build_feature_matrix(df)
        assert "train" in splits
        assert isinstance(scaler, FeatureScaler)
        assert scaler._fitted
        assert len(features) > 0

    def test_with_splits(self):
        df = _make_xbrl_df(n_companies=3, n_years=6, start_year=2015)
        split_cfg = SplitConfig(
            train_end="2017-12-31",
            val_end="2019-12-31",
            test_end="2023-12-31",
        )
        splits, scaler, features = build_feature_matrix(df, split_config=split_cfg)
        assert "train" in splits
        assert "val" in splits
        assert "test" in splits
        # All splits should have the same columns
        assert list(splits["train"].columns) == list(splits["val"].columns)
        assert list(splits["train"].columns) == list(splits["test"].columns)

    def test_feature_count_all_enabled(self):
        df = _make_xbrl_df(n_companies=3, n_years=4, start_year=2015)
        config = FeatureConfig(
            use_raw=True,
            use_ratios=True,
            use_yoy=True,
            use_missingness_flags=True,
            coverage_threshold=0.0,  # keep everything
        )
        _, _, features = build_feature_matrix(df, feature_config=config)
        # 16 raw + 9 ratios + 5 yoy = 30 numeric features
        # + 30 missingness flags = 60 total
        n_numeric = len(RAW_FEATURES) + len(RATIO_FEATURES) + len(YOY_FEATURES)
        expected = n_numeric * 2  # numeric + flags
        assert len(features) == expected

    def test_config_disables_ratios(self):
        df = _make_xbrl_df(n_companies=2, n_years=2)
        config = FeatureConfig(
            use_ratios=False,
            use_yoy=False,
            use_missingness_flags=False,
            coverage_threshold=0.0,
        )
        _, _, features = build_feature_matrix(df, feature_config=config)
        for ratio in RATIO_FEATURES:
            assert ratio not in features
        for yoy in YOY_FEATURES:
            assert yoy not in features

    def test_config_disables_yoy(self):
        df = _make_xbrl_df(n_companies=2, n_years=2)
        config = FeatureConfig(
            use_yoy=False,
            use_missingness_flags=False,
            coverage_threshold=0.0,
        )
        _, _, features = build_feature_matrix(df, feature_config=config)
        for yoy in YOY_FEATURES:
            assert yoy not in features
        # Ratios should still be present
        for ratio in RATIO_FEATURES:
            assert ratio in features

    def test_no_data_leakage(self):
        # Create data where train and val have different distributions
        df = _make_xbrl_df(n_companies=2, n_years=6, start_year=2015)
        # Make val period have very different values
        val_mask = df["fiscal_year"].isin([2018, 2019])
        df.loc[val_mask, "total_assets"] = 99999.0

        split_cfg = SplitConfig(
            train_end="2017-12-31",
            val_end="2019-12-31",
            test_end="2023-12-31",
        )
        splits, scaler, _ = build_feature_matrix(df, split_config=split_cfg)

        # Scaler medians should come from train only (not influenced by val)
        train_only = df[df["fiscal_year"] <= 2017]
        assert scaler._medians["total_assets"] == pytest.approx(
            float(train_only["total_assets"].median())
        )

    def test_no_nan_in_output(self):
        df = _make_xbrl_df(n_companies=3, n_years=4, inject_nans=True)
        config = FeatureConfig(coverage_threshold=0.0)
        splits, _, features = build_feature_matrix(df, feature_config=config)
        # After imputation, numeric features should have no NaN
        numeric_feats = [f for f in features if not f.endswith("_missing")]
        for key, split_df in splits.items():
            for col in numeric_feats:
                if col in split_df.columns:
                    assert not split_df[col].isna().any(), (
                        f"NaN found in {key}[{col}]"
                    )


# ---------------------------------------------------------------------------
# TestBackwardCompatibility
# ---------------------------------------------------------------------------

class TestBackwardCompatibility:

    def test_fit_scaler_returns_dict(self):
        df = pd.DataFrame({
            "a": [1.0, 2.0, 3.0, 4.0, 5.0],
            "b": [10.0, 20.0, 30.0, 40.0, 50.0],
        })
        result = fit_scaler(df, ["a", "b"])
        assert isinstance(result, dict)
        assert "_scaler" in result
        assert "feature_cols" in result

    def test_apply_scaler_transforms(self):
        train = pd.DataFrame({
            "a": [1.0, 2.0, 3.0, 4.0, 5.0],
            "b": [10.0, 20.0, 30.0, 40.0, 50.0],
        })
        scaler_dict = fit_scaler(train, ["a", "b"])

        test = pd.DataFrame({
            "a": [2.5, 3.5],
            "b": [25.0, 35.0],
        })
        result = apply_scaler(test, scaler_dict, ["a", "b"])
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2
        assert "a" in result.columns
        assert "b" in result.columns

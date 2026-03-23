"""Feature engineering: financial ratios, YoY changes, and normalisation.

Transforms raw XBRL line items into a model-ready feature matrix for both
baseline models (logistic regression, XGBoost) and the FT-Transformer.

Pipeline steps (executed by :func:`build_feature_matrix`):
  1. Compute standard financial ratios from raw XBRL columns
  2. Compute year-over-year percentage changes for key items
  3. Prune features below a coverage threshold (fit on train split)
  4. Add binary missingness indicator flags
  5. Fit a robust normalisation scaler on the train split
  6. Transform all splits (train / val / test)

Design decisions:
  - Quantile normalisation (rank -> normal) is default because financial
    ratios have extreme tails that break simple z-scoring.
  - Winsorisation at 1st/99th percentile as a safety net before transform.
  - Scaler is always fit on the train split only to prevent look-ahead bias.
  - Missingness flags are binary and NOT normalised.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import QuantileTransformer

from fin_jepa.data.xbrl_pipeline import FEATURE_NAMES as RAW_FEATURES
from fin_jepa.data.splits import SplitConfig, make_splits

logger = logging.getLogger(__name__)

# ── Feature name constants ───────────────────────────────────────────────

RATIO_FEATURES = [
    "debt_to_equity",
    "current_ratio",
    "quick_ratio",
    "roa",
    "roe",
    "gross_margin",
    "operating_margin",
    "interest_coverage",
    "altman_z_score",
]

YOY_BASE_FEATURES = [
    "total_revenue",
    "net_income",
    "total_assets",
    "total_debt",
    "cash_from_operations",
]

YOY_FEATURES = [f"{col}_yoy" for col in YOY_BASE_FEATURES]

ID_COLUMNS = ["cik", "ticker", "fiscal_year", "period_end", "filed_date"]


# ── Configuration ────────────────────────────────────────────────────────

@dataclass
class FeatureConfig:
    """Configuration for the feature engineering pipeline.

    Mirrors the ``features:`` block in ``configs/study0/benchmark.yaml``.
    """

    use_raw: bool = True
    use_ratios: bool = True
    use_yoy: bool = True
    use_missingness_flags: bool = True
    coverage_threshold: float = 0.50
    winsorize_limits: tuple[float, float] = (0.01, 0.99)
    normalization_method: str = "quantile"  # "quantile" | "zscore"
    median_impute: bool = True


# ── Helpers ──────────────────────────────────────────────────────────────

def _safe_div(
    num: pd.Series,
    den: pd.Series,
    min_abs: float = 1e-8,
) -> pd.Series:
    """Element-wise division that returns NaN where *den* is zero or near-zero."""
    safe_den = den.where(den.abs() >= min_abs, np.nan)
    return num / safe_den


def _col_or_nan(df: pd.DataFrame, col: str) -> pd.Series:
    """Return column *col* from *df*, or a NaN series if the column is absent."""
    if col in df.columns:
        return df[col]
    return pd.Series(np.nan, index=df.index, name=col)


# ── Ratio computation ───────────────────────────────────────────────────

def compute_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """Derive financial ratio features from raw XBRL columns.

    Appends 9 ratio columns to *df* (returns a copy). If a required input
    column is absent or contains NaN, the corresponding ratio is NaN for
    those rows.

    Ratios computed:
      1. debt_to_equity         = total_debt / total_equity
      2. current_ratio          = current_assets / current_liabilities
      3. quick_ratio            = cash_equivalents / current_liabilities
      4. roa                    = net_income / total_assets
      5. roe                    = net_income / total_equity
      6. gross_margin           = (total_revenue - cost_of_sales) / total_revenue
      7. operating_margin       = operating_income / total_revenue
      8. interest_coverage      = operating_income / interest_expense
      9. altman_z_score         = Altman (1968) 5-component Z-score
    """
    out = df.copy()

    total_assets = _col_or_nan(out, "total_assets")
    total_liabilities = _col_or_nan(out, "total_liabilities")
    total_equity = _col_or_nan(out, "total_equity")
    current_assets = _col_or_nan(out, "current_assets")
    current_liabilities = _col_or_nan(out, "current_liabilities")
    retained_earnings = _col_or_nan(out, "retained_earnings")
    cash_equivalents = _col_or_nan(out, "cash_equivalents")
    total_debt = _col_or_nan(out, "total_debt")
    total_revenue = _col_or_nan(out, "total_revenue")
    cost_of_sales = _col_or_nan(out, "cost_of_sales")
    operating_income = _col_or_nan(out, "operating_income")
    net_income = _col_or_nan(out, "net_income")
    interest_expense = _col_or_nan(out, "interest_expense")

    out["debt_to_equity"] = _safe_div(total_debt, total_equity)
    out["current_ratio"] = _safe_div(current_assets, current_liabilities)
    out["quick_ratio"] = _safe_div(cash_equivalents, current_liabilities)
    out["roa"] = _safe_div(net_income, total_assets)
    out["roe"] = _safe_div(net_income, total_equity)
    out["gross_margin"] = _safe_div(total_revenue - cost_of_sales, total_revenue)
    out["operating_margin"] = _safe_div(operating_income, total_revenue)
    out["interest_coverage"] = _safe_div(operating_income, interest_expense)

    # Altman Z-score: Z = 1.2*X1 + 1.4*X2 + 3.3*X3 + 0.6*X4 + 1.0*X5
    working_capital = current_assets - current_liabilities
    x1 = _safe_div(working_capital, total_assets)
    x2 = _safe_div(retained_earnings, total_assets)
    x3 = _safe_div(operating_income, total_assets)
    x4 = _safe_div(total_equity, total_liabilities)
    x5 = _safe_div(total_revenue, total_assets)
    out["altman_z_score"] = 1.2 * x1 + 1.4 * x2 + 3.3 * x3 + 0.6 * x4 + 1.0 * x5

    return out


# ── Year-over-year changes ──────────────────────────────────────────────

def compute_yoy_changes(
    df: pd.DataFrame,
    features: list[str] | None = None,
) -> pd.DataFrame:
    """Compute per-company year-over-year percentage changes.

    For each feature, the YoY change is::

        (current - prior) / |prior|

    Using the absolute value of the prior handles negative base values
    (e.g. negative net_income going to positive). The first observation
    per CIK is NaN (no prior year).

    New columns are named ``{feature}_yoy``.
    """
    if features is None:
        features = YOY_BASE_FEATURES

    out = df.copy()
    # Ensure sorted for correct shift
    out = out.sort_values(["cik", "fiscal_year"]).reset_index(drop=True)

    for col in features:
        if col not in out.columns:
            out[f"{col}_yoy"] = np.nan
            continue

        prior = out.groupby("cik")[col].shift(1)
        abs_prior = prior.abs()
        # Replace zero priors with NaN to avoid inf
        abs_prior = abs_prior.where(abs_prior > 0, np.nan)
        change = (out[col] - prior) / abs_prior
        out[f"{col}_yoy"] = change

    return out


# ── Coverage and pruning ────────────────────────────────────────────────

def coverage_report(
    df: pd.DataFrame,
    feature_cols: list[str],
) -> dict[str, float]:
    """Return ``{column: fraction_non_null}`` for each feature column."""
    n = len(df)
    if n == 0:
        return {col: 0.0 for col in feature_cols}
    return {
        col: float(df[col].notna().sum() / n)
        for col in feature_cols
        if col in df.columns
    }


def prune_low_coverage(
    df: pd.DataFrame,
    feature_cols: list[str],
    threshold: float = 0.50,
) -> tuple[pd.DataFrame, list[str]]:
    """Drop feature columns whose non-null fraction is below *threshold*.

    Identifier columns (cik, fiscal_year, etc.) are never dropped.

    Returns:
        Tuple of ``(pruned_df, kept_feature_cols)``.
    """
    report = coverage_report(df, feature_cols)
    kept: list[str] = []
    dropped: list[str] = []

    for col in feature_cols:
        cov = report.get(col, 0.0)
        if cov >= threshold:
            kept.append(col)
        else:
            dropped.append(col)

    if dropped:
        logger.info(
            "Pruned %d low-coverage features (threshold=%.0f%%): %s",
            len(dropped),
            threshold * 100,
            dropped,
        )

    # Keep all non-feature columns + kept features
    cols_to_keep = [c for c in df.columns if c not in feature_cols or c in kept]
    return df[cols_to_keep].copy(), kept


# ── Missingness flags ───────────────────────────────────────────────────

def add_missingness_flags(
    df: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[pd.DataFrame, list[str]]:
    """Add binary ``{col}_missing`` indicator columns.

    Returns:
        Tuple of ``(df_with_flags, flag_column_names)``.
    """
    out = df.copy()
    flag_cols: list[str] = []
    for col in feature_cols:
        if col in out.columns:
            flag_name = f"{col}_missing"
            out[flag_name] = out[col].isna().astype(np.int8)
            flag_cols.append(flag_name)
    return out, flag_cols


# ── Feature scaler ──────────────────────────────────────────────────────

class FeatureScaler:
    """Robust normalisation pipeline for financial features.

    Steps applied during :meth:`transform`:
      1. Median imputation (using train-split medians)
      2. Winsorisation (clip to train-split percentile bounds)
      3. Quantile normalisation or z-score standardisation

    The scaler is always **fit on the train split only**.
    """

    def __init__(
        self,
        winsorize_limits: tuple[float, float] = (0.01, 0.99),
        method: str = "quantile",
        median_impute: bool = True,
    ) -> None:
        self.winsorize_limits = winsorize_limits
        self.method = method
        self.median_impute = median_impute
        self._fitted = False
        self._feature_cols: list[str] = []
        self._medians: dict[str, float] = {}
        self._percentiles: dict[str, tuple[float, float]] = {}
        # For quantile method
        self._qt: QuantileTransformer | None = None
        # For zscore method
        self._mean: dict[str, float] = {}
        self._std: dict[str, float] = {}

    def fit(
        self,
        train_df: pd.DataFrame,
        feature_cols: list[str],
    ) -> FeatureScaler:
        """Fit normalisation statistics from the training split."""
        self._feature_cols = list(feature_cols)

        # Compute per-column train-split medians for imputation
        for col in feature_cols:
            if col in train_df.columns:
                med = train_df[col].median()
                self._medians[col] = float(med) if pd.notna(med) else 0.0
            else:
                self._medians[col] = 0.0

        # Impute for fitting purposes
        cols_present = [c for c in feature_cols if c in train_df.columns]
        work = train_df[cols_present].copy()
        for col in cols_present:
            work[col] = work[col].fillna(self._medians[col])

        # Compute winsorisation bounds
        lo_q, hi_q = self.winsorize_limits
        for col in cols_present:
            lo_val = float(work[col].quantile(lo_q))
            hi_val = float(work[col].quantile(hi_q))
            self._percentiles[col] = (lo_val, hi_val)

        # Winsorise before fitting the transform
        for col in cols_present:
            lo_val, hi_val = self._percentiles[col]
            work[col] = work[col].clip(lower=lo_val, upper=hi_val)

        if self.method == "quantile":
            n_samples = len(work)
            n_quantiles = min(1000, n_samples)
            if n_quantiles < 2:
                n_quantiles = 2
            self._qt = QuantileTransformer(
                output_distribution="normal",
                n_quantiles=n_quantiles,
            )
            self._qt.fit(work.values)
        elif self.method == "zscore":
            for col in cols_present:
                self._mean[col] = float(work[col].mean())
                std = float(work[col].std())
                self._std[col] = std if std > 1e-12 else 1.0
        else:
            raise ValueError(f"Unknown normalization method: {self.method!r}")

        self._fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply the fitted normalisation to *df*."""
        if not self._fitted:
            raise RuntimeError("FeatureScaler has not been fitted yet.")

        out = df.copy()
        cols_present = [c for c in self._feature_cols if c in out.columns]

        # 1. Median imputation
        if self.median_impute:
            for col in cols_present:
                out[col] = out[col].fillna(self._medians.get(col, 0.0))

        # 2. Winsorisation
        for col in cols_present:
            lo_val, hi_val = self._percentiles.get(col, (-np.inf, np.inf))
            out[col] = out[col].clip(lower=lo_val, upper=hi_val)

        # 3. Normalisation
        if self.method == "quantile" and self._qt is not None:
            arr = out[cols_present].values
            transformed = self._qt.transform(arr)
            out[cols_present] = transformed
        elif self.method == "zscore":
            for col in cols_present:
                out[col] = (out[col] - self._mean[col]) / self._std[col]

        return out

    def fit_transform(
        self,
        train_df: pd.DataFrame,
        feature_cols: list[str],
    ) -> pd.DataFrame:
        """Fit on *train_df* and return the transformed training data."""
        self.fit(train_df, feature_cols)
        return self.transform(train_df)


# ── Backward-compatible wrappers ─────────────────────────────────────────

def fit_scaler(
    train: pd.DataFrame,
    feature_cols: list[str],
) -> dict[str, Any]:
    """Fit mean/std scaler on train split. Returns scaler state dict.

    This is a backward-compatible wrapper around :class:`FeatureScaler`.
    """
    scaler = FeatureScaler()
    scaler.fit(train, feature_cols)
    return {"_scaler": scaler, "feature_cols": feature_cols}


def apply_scaler(
    df: pd.DataFrame,
    scaler: dict[str, Any],
    feature_cols: list[str],
) -> pd.DataFrame:
    """Apply pre-fit scaler to *df*.

    This is a backward-compatible wrapper around :class:`FeatureScaler`.
    """
    return scaler["_scaler"].transform(df)


# ── Top-level orchestrator ──────────────────────────────────────────────

def build_feature_matrix(
    xbrl_df: pd.DataFrame,
    split_config: SplitConfig | None = None,
    feature_config: FeatureConfig | None = None,
) -> tuple[dict[str, pd.DataFrame], FeatureScaler, list[str]]:
    """Build the full feature matrix from raw XBRL data.

    Steps:
      1. Compute financial ratios (optional, controlled by config)
      2. Compute year-over-year changes (optional)
      3. Split into train/val/test
      4. Prune low-coverage features (based on train split)
      5. Add missingness flags (optional)
      6. Fit :class:`FeatureScaler` on train, transform all splits

    Args:
        xbrl_df: Raw XBRL features DataFrame (output of the XBRL pipeline).
        split_config: Time-based split configuration. If None, all data is
            treated as a single "train" split.
        feature_config: Feature engineering configuration. Uses defaults if
            None.

    Returns:
        Tuple of ``(split_dfs, scaler, final_feature_cols)`` where:
        - *split_dfs* maps ``"train"``/``"val"``/``"test"`` to DataFrames
        - *scaler* is the fitted :class:`FeatureScaler`
        - *final_feature_cols* lists all feature columns (for model input)
    """
    if feature_config is None:
        feature_config = FeatureConfig()

    df = xbrl_df.copy()

    # 1. Compute ratios
    if feature_config.use_ratios:
        df = compute_ratios(df)

    # 2. Compute YoY changes
    if feature_config.use_yoy:
        df = compute_yoy_changes(df)

    # Assemble candidate feature columns
    candidate_features: list[str] = []
    if feature_config.use_raw:
        candidate_features.extend(
            col for col in RAW_FEATURES if col in df.columns
        )
    if feature_config.use_ratios:
        candidate_features.extend(
            col for col in RATIO_FEATURES if col in df.columns
        )
    if feature_config.use_yoy:
        candidate_features.extend(
            col for col in YOY_FEATURES if col in df.columns
        )

    # 3. Split
    if split_config is not None:
        splits = make_splits(df, split_config)
    else:
        splits = {"train": df}

    train_df = splits["train"]

    # 4. Coverage-based pruning (decision on train split)
    _, kept_features = prune_low_coverage(
        train_df, candidate_features, feature_config.coverage_threshold
    )

    # Apply same column selection to all splits
    for key in splits:
        cols_to_keep = [
            c for c in splits[key].columns
            if c not in candidate_features or c in kept_features
        ]
        splits[key] = splits[key][cols_to_keep].copy()

    # 5. Missingness flags
    flag_cols: list[str] = []
    if feature_config.use_missingness_flags:
        for key in splits:
            splits[key], fcs = add_missingness_flags(splits[key], kept_features)
            if not flag_cols:
                flag_cols = fcs

    # Numeric features to normalise (excludes binary flags)
    numeric_features = kept_features

    # 6. Fit scaler on train, transform all splits
    scaler = FeatureScaler(
        winsorize_limits=feature_config.winsorize_limits,
        method=feature_config.normalization_method,
        median_impute=feature_config.median_impute,
    )
    scaler.fit(splits["train"], numeric_features)

    for key in splits:
        splits[key] = scaler.transform(splits[key])

    final_feature_cols = kept_features + flag_cols

    logger.info(
        "Feature matrix built: %d features (%d numeric + %d flags), "
        "splits: %s",
        len(final_feature_cols),
        len(numeric_features),
        len(flag_cols),
        {k: len(v) for k, v in splits.items()},
    )

    return splits, scaler, final_feature_cols

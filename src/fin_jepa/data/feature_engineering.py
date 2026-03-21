"""
Feature engineering: traditional financial ratios and XBRL feature normalization.

Used by baseline models (logistic regression, XGBoost) and as input to FT-Transformer.

TODO:
  - Compute standard ratios: leverage, liquidity, profitability, coverage
  - Winsorize and standardize features per train-set statistics
  - Handle missing values (median imputation + missingness indicator flags)
  - Produce ratio feature spec document
"""

from __future__ import annotations

import numpy as np
import pandas as pd


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


def compute_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """Derive ratio features from raw XBRL columns in *df*."""
    raise NotImplementedError("Implement ratio computation.")


def fit_scaler(train: pd.DataFrame, feature_cols: list[str]) -> dict:
    """Fit mean/std scaler on train split. Returns scaler state dict."""
    raise NotImplementedError


def apply_scaler(df: pd.DataFrame, scaler: dict, feature_cols: list[str]) -> pd.DataFrame:
    """Apply pre-fit scaler to *df*."""
    raise NotImplementedError

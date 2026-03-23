"""
Baseline models: XGBoost, Logistic Regression, and GBT wrappers.

Workstream: Run FT-Transformer vs. baselines benchmark.

Provides a common sklearn-compatible interface so all models can be
evaluated through the same evaluation harness.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


def build_logistic_regression(
    C: float = 1.0,
    max_iter: int = 1000,
    class_weight: str = "balanced",
    **kwargs: Any,
) -> Pipeline:
    """Return a sklearn Pipeline: StandardScaler -> L2 Logistic Regression."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            C=C,
            max_iter=max_iter,
            class_weight=class_weight,
            solver="lbfgs",
            **kwargs,
        )),
    ])


def build_xgboost(
    n_estimators: int = 500,
    learning_rate: float = 0.05,
    max_depth: int = 6,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    scale_pos_weight: float = 1.0,
    random_state: int = 42,
    **kwargs: Any,
) -> XGBClassifier:
    """Return a configured XGBClassifier."""
    return XGBClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        scale_pos_weight=scale_pos_weight,
        random_state=random_state,
        eval_metric="auc",
        use_label_encoder=False,
        **kwargs,
    )


def build_gbt(
    max_iter: int = 500,
    learning_rate: float = 0.05,
    max_depth: int | None = 6,
    min_samples_leaf: int = 20,
    random_state: int = 42,
    class_weight: str = "balanced",
    **kwargs: Any,
) -> HistGradientBoostingClassifier:
    """Return a configured HistGradientBoostingClassifier.

    Uses sklearn's histogram-based GBT which natively handles NaN,
    making it suitable for raw XBRL features with minimal preprocessing.
    """
    return HistGradientBoostingClassifier(
        max_iter=max_iter,
        learning_rate=learning_rate,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
        class_weight=class_weight,
        **kwargs,
    )

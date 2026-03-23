"""
Evaluation metrics for the Study 0 benchmark.

Primary metric: AUROC (area under ROC curve)
Secondary metrics: AUPRC, F1 at optimal threshold, Brier score

For the go/no-go gate: FT-Transformer must beat XGBoost on >=3 of 5 outcomes
by AUROC margin >= 0.01 on the held-out test set.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    roc_auc_score,
)


def compute_calibration(
    y_true: np.ndarray,
    y_score: np.ndarray,
    n_bins: int = 10,
) -> dict[str, Any]:
    """Compute calibration curve data and expected calibration error (ECE).

    Returns
    -------
    dict with keys:
        ece        : float — expected calibration error (lower is better)
        prob_true  : list[float] — fraction of positives per bin
        prob_pred  : list[float] — mean predicted probability per bin
        n_bins     : int — number of bins requested
    """
    prob_true, prob_pred = calibration_curve(
        y_true, y_score, n_bins=n_bins, strategy="uniform",
    )

    # ECE: weighted average of |prob_true - prob_pred| across non-empty bins.
    # calibration_curve only returns non-empty bins, so we compute bin counts
    # to weight properly.
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_score, bin_edges[1:-1])
    bin_counts = np.array([
        np.sum(bin_indices == i) for i in range(n_bins)
    ])
    nonempty = bin_counts > 0
    nonempty_counts = bin_counts[nonempty]
    n_samples = len(y_true)

    ece = float(
        np.sum(nonempty_counts * np.abs(prob_true - prob_pred)) / n_samples
    )

    return {
        "ece": ece,
        "prob_true": prob_true.tolist(),
        "prob_pred": prob_pred.tolist(),
        "n_bins": n_bins,
    }


def compute_all_metrics(y_true: np.ndarray, y_score: np.ndarray) -> dict[str, float]:
    """Compute AUROC, AUPRC, Brier score, and F1 at Youden threshold.

    Returns a dict of NaN values if only one class is present in *y_true*.
    """
    # Guard against single-class test sets (e.g. rare outcomes with 0 positives)
    if len(np.unique(y_true)) < 2:
        return {
            "auroc": float("nan"),
            "auprc": float("nan"),
            "brier": float("nan"),
            "f1_youden": float("nan"),
            "youden_threshold": float("nan"),
            "ece": float("nan"),
            "calibration": {"ece": float("nan"), "prob_true": [], "prob_pred": [], "n_bins": 10},
        }

    from sklearn.metrics import roc_curve

    auroc = roc_auc_score(y_true, y_score)
    auprc = average_precision_score(y_true, y_score)
    brier = brier_score_loss(y_true, y_score)

    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    youden_idx = np.argmax(tpr - fpr)
    best_thresh = float(thresholds[youden_idx])
    y_pred = (y_score >= best_thresh).astype(int)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    cal = compute_calibration(y_true, y_score)

    return {
        "auroc": float(auroc),
        "auprc": float(auprc),
        "brier": float(brier),
        "f1_youden": float(f1),
        "youden_threshold": best_thresh,
        "ece": cal["ece"],
        "calibration": cal,
    }


def go_no_go_gate(
    ft_results: dict[str, dict[str, float]],
    xgb_results: dict[str, dict[str, float]],
    outcomes: list[str],
    margin: float = 0.01,
) -> tuple[bool, int, dict]:
    """
    Evaluate the Study 0 go/no-go gate.

    FT-Transformer passes if it beats XGBoost on >=3/5 outcomes by >= *margin* AUROC.

    Returns
    -------
    passed:    True if gate is passed
    n_wins:    number of outcomes where FT-Transformer wins
    detail:    per-outcome comparison dict
    """
    wins = 0
    detail = {}
    for outcome in outcomes:
        ft_auc = ft_results[outcome]["auroc"]
        xgb_auc = xgb_results[outcome]["auroc"]
        win = (ft_auc - xgb_auc) >= margin
        wins += int(win)
        detail[outcome] = {"ft_auroc": ft_auc, "xgb_auroc": xgb_auc, "win": win}
    return wins >= 3, wins, detail

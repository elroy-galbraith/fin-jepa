"""
Evaluation metrics for the Study 0 benchmark.

Primary metric: AUROC (area under ROC curve)
Secondary metrics: AUPRC, F1 at optimal threshold, Brier score

For the go/no-go gate: FT-Transformer must beat XGBoost on >=3 of 5 outcomes
by AUROC margin >= 0.01 on the held-out test set.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    roc_auc_score,
)


def compute_all_metrics(y_true: np.ndarray, y_score: np.ndarray) -> dict[str, float]:
    """Compute AUROC, AUPRC, Brier score, and F1 at Youden threshold."""
    from sklearn.metrics import roc_curve

    auroc = roc_auc_score(y_true, y_score)
    auprc = average_precision_score(y_true, y_score)
    brier = brier_score_loss(y_true, y_score)

    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    youden_idx = np.argmax(tpr - fpr)
    best_thresh = float(thresholds[youden_idx])
    y_pred = (y_score >= best_thresh).astype(int)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    return {
        "auroc": float(auroc),
        "auprc": float(auprc),
        "brier": float(brier),
        "f1_youden": float(f1),
        "youden_threshold": best_thresh,
    }


def go_no_go_gate(
    ft_results: dict[str, float],
    xgb_results: dict[str, float],
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

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


def compute_sector_stratified_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    sector_ids: np.ndarray,
    sector_names: list[str] | None = None,
    min_samples: int = 30,
) -> dict[str, dict[str, float]]:
    """Compute AUROC and AUPRC per Fama-French 12 sector.

    Parameters
    ----------
    y_true : binary labels
    y_score : predicted probabilities
    sector_ids : integer sector indices (0–11)
    sector_names : optional list mapping index → display name
    min_samples : minimum samples with both classes to compute metrics

    Returns
    -------
    Dict mapping sector name to ``{auroc, auprc, n_samples, n_positive}``.
    Sectors with fewer than *min_samples* or only one class get NaN metrics.
    """
    if sector_names is None:
        from fin_jepa.data.sector_map import FF12_SECTORS
        sector_names = FF12_SECTORS

    results: dict[str, dict[str, float]] = {}
    for idx, name in enumerate(sector_names):
        mask = sector_ids == idx
        n = int(mask.sum())
        n_pos = int(y_true[mask].sum()) if n > 0 else 0
        if n < min_samples or n_pos == 0 or n_pos == n:
            results[name] = {
                "auroc": float("nan"),
                "auprc": float("nan"),
                "n_samples": n,
                "n_positive": n_pos,
            }
        else:
            results[name] = {
                "auroc": float(roc_auc_score(y_true[mask], y_score[mask])),
                "auprc": float(average_precision_score(y_true[mask], y_score[mask])),
                "n_samples": n,
                "n_positive": n_pos,
            }
    return results


def hanley_mcneil_se(auroc: float, n_pos: int, n_neg: int) -> float:
    """Compute the Hanley-McNeil standard error for an AUROC estimate.

    Parameters
    ----------
    auroc : float
        The observed AUROC (A).
    n_pos : int
        Number of positive examples in the evaluation set.
    n_neg : int
        Number of negative examples in the evaluation set.

    Returns
    -------
    float — standard error of the AUROC estimate.

    References
    ----------
    Hanley & McNeil (1982), "The Meaning and Use of the Area under a Receiver
    Operating Characteristic (ROC) Curve", Radiology 143(1):29-36.
    """
    A = auroc
    Q1 = A / (2.0 - A)
    Q2 = 2.0 * A ** 2 / (1.0 + A)
    var = (
        A * (1.0 - A)
        + (n_pos - 1) * (Q1 - A ** 2)
        + (n_neg - 1) * (Q2 - A ** 2)
    ) / (n_pos * n_neg)
    return float(np.sqrt(max(var, 0.0)))


def bootstrap_auroc_ci(
    y_true: np.ndarray,
    y_score_a: np.ndarray,
    y_score_b: np.ndarray | None = None,
    n_bootstrap: int = 2000,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict[str, Any]:
    """Bootstrap confidence interval for AUROC (single model or pairwise difference).

    When *y_score_b* is None, returns a CI for ``auroc(y_score_a)``.
    When *y_score_b* is provided, returns a CI for ``auroc(y_score_a) - auroc(y_score_b)``
    using paired bootstrap (same resample indices for both models).

    Parameters
    ----------
    y_true : array of binary labels
    y_score_a : predicted probabilities for model A
    y_score_b : predicted probabilities for model B (optional, for pairwise delta)
    n_bootstrap : number of bootstrap replicates
    alpha : significance level (0.05 → 95% CI)
    seed : random seed

    Returns
    -------
    dict with keys:
        estimate  : point estimate (AUROC or delta AUROC)
        ci_lower  : lower bound of (1-alpha) CI
        ci_upper  : upper bound of (1-alpha) CI
        significant : bool — True if CI does not contain 0 (pairwise only)
    """
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(seed)
    n = len(y_true)

    auc_a = float(roc_auc_score(y_true, y_score_a)) if len(np.unique(y_true)) >= 2 else float("nan")

    if y_score_b is None:
        estimate = auc_a
        boot_stats = []
        for _ in range(n_bootstrap):
            idx = rng.integers(0, n, size=n)
            yt = y_true[idx]
            if len(np.unique(yt)) < 2:
                continue
            boot_stats.append(float(roc_auc_score(yt, y_score_a[idx])))
        boot_arr = np.array(boot_stats)
        ci_lower = float(np.percentile(boot_arr, 100 * alpha / 2))
        ci_upper = float(np.percentile(boot_arr, 100 * (1 - alpha / 2)))
        return {"estimate": estimate, "ci_lower": ci_lower, "ci_upper": ci_upper}

    auc_b = float(roc_auc_score(y_true, y_score_b)) if len(np.unique(y_true)) >= 2 else float("nan")
    estimate = auc_a - auc_b

    boot_deltas = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        yt = y_true[idx]
        if len(np.unique(yt)) < 2:
            continue
        da = float(roc_auc_score(yt, y_score_a[idx]))
        db = float(roc_auc_score(yt, y_score_b[idx]))
        boot_deltas.append(da - db)

    deltas_arr = np.array(boot_deltas)
    ci_lower = float(np.percentile(deltas_arr, 100 * alpha / 2))
    ci_upper = float(np.percentile(deltas_arr, 100 * (1 - alpha / 2)))
    significant = bool(ci_lower > 0 or ci_upper < 0)

    return {
        "estimate": estimate,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "significant": significant,
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

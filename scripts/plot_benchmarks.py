#!/usr/bin/env python
"""Regenerate the default-baseline benchmark figures from ``final_benchmark.json``.

Writes three figures to ``results/study0/figures/``:
  * ``baseline_auroc.png``            — test AUROC by model x outcome (5 models)
  * ``final_benchmark_all_models.png``— test AUROC by model x outcome (6 models)
  * ``bootstrap_ci_forest_plot.png``  — paired bootstrap CIs (FT minus default XGB)

These are functional regenerations of the original notebook figures (the originals
were never committed); they read the committed result JSON, so they reproduce the
same numbers. Style is not guaranteed pixel-identical to the originals.

Usage::

    python scripts/plot_benchmarks.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "results" / "study0" / "final_benchmark.json"
FIGDIR = ROOT / "results" / "study0" / "figures"

OUTCOMES = ["stock_decline", "earnings_restate", "sec_enforcement", "bankruptcy"]
MODELS = [
    ("lr_trad", "LR (trad.)"),
    ("lr_full", "LR (full)"),
    ("xgboost", "XGBoost"),
    ("gbt_raw", "GBT (raw)"),
    ("ft_scratch", "FT-Trans. (scratch)"),
    ("ft_ssl", "FT-Trans. (SSL)"),
]


def _auroc(d: dict, model: str, oc: str):
    model_data = d.get(model)
    if not isinstance(model_data, dict):
        return None
    v = model_data.get(oc)
    return v.get("auroc") if isinstance(v, dict) else None


def _grouped_auroc(d: dict, models: list[tuple[str, str]], out: Path, title: str) -> None:
    x = np.arange(len(OUTCOMES))
    w = 0.8 / len(models)
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    for i, (key, label) in enumerate(models):
        vals = [(_auroc(d, key, oc) or np.nan) for oc in OUTCOMES]
        ax.bar(x + i * w - 0.4 + w / 2, vals, w, label=label)
    ax.axhline(0.5, color="0.6", lw=0.9, ls=":")
    ax.set_xticks(x)
    ax.set_xticklabels([oc.replace("_", "\n") for oc in OUTCOMES])
    ax.set_ylabel("Test AUROC")
    ax.set_ylim(0.35, 0.75)
    ax.set_title(title)
    ax.legend(frameon=False, ncol=3, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    ax.grid(axis="y", color="0.92", lw=0.7)
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def _bootstrap_forest(d: dict, out: Path) -> None:
    bt = d.get("bootstrap")
    if not isinstance(bt, dict):
        bt = {}
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ys, labels = [], []
    y = 0
    for oc in OUTCOMES:
        for variant, lab, mk, col in [
            ("scratch_vs_xgb", "scratch", "o", "#2a7"),
            ("ssl_vs_xgb", "SSL", "s", "#d90"),
        ]:
            oc_data = bt.get(oc)
            v = oc_data.get(variant) if isinstance(oc_data, dict) else None
            if not isinstance(v, dict):
                continue
            ax.plot([v["ci_low"], v["ci_high"]], [y, y], color=col, lw=2)
            ax.plot(v["mean"], y, mk, color=col, ms=7,
                    label=lab if oc == OUTCOMES[0] else None)
            ys.append(y)
            labels.append(f"{oc}  ({lab})")
            y += 1
        y += 0.6
    ax.axvline(0.0, color="0.4", lw=1.0)
    ax.axvline(0.01, color="0.6", lw=0.9, ls=":")
    ax.set_yticks(ys)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel(r"AUROC difference (FT $-$ default XGBoost), 95% bootstrap CI")
    ax.set_title("Paired bootstrap CIs vs.\\ default XGBoost")
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"wrote {out}")


def main() -> None:
    d = json.loads(SRC.read_text())
    FIGDIR.mkdir(parents=True, exist_ok=True)
    _grouped_auroc(d, MODELS[:5], FIGDIR / "baseline_auroc.png",
                   "Baseline benchmark: test AUROC by model and outcome")
    _grouped_auroc(d, MODELS, FIGDIR / "final_benchmark_all_models.png",
                   "Final benchmark (default baselines): test AUROC, six models")
    _bootstrap_forest(d, FIGDIR / "bootstrap_ci_forest_plot.png")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Plot the walk-forward FT-Transformer minus XGBoost AUROC delta per fold.

Reads ``results/study0/corrected/walk_forward_results.json`` and writes
``results/study0/figures/walk_forward_delta.png``: one line per outcome showing
the per-fold (FT $-$ XGBoost) AUROC difference across the seven expanding-window
folds, with the COVID-era folds shaded and the gate margin marked.

Usage::

    python scripts/plot_walk_forward.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results" / "study0" / "corrected" / "walk_forward_results.json"
OUT = ROOT / "results" / "study0" / "figures" / "walk_forward_delta.png"

# Display order + labels (most stable first), matching the paper's narrative.
OUTCOMES = [
    ("earnings_restate", "earnings_restate"),
    ("bankruptcy", "bankruptcy"),
    ("stock_decline", "stock_decline"),
    ("sec_enforcement", "sec_enforcement"),
]


def main() -> None:
    data = json.loads(RESULTS.read_text())
    folds = data["walk_forward_folds"]

    # x labels = test window per fold, e.g. "2016-17"
    def _label(fold: dict) -> str:
        lbl = fold.get("label", "")
        # label like "train<=2014 -> test 2016-2017"
        if "test" in lbl:
            span = lbl.split("test", 1)[1].strip()
            yrs = span.replace("–", "-").split("-")
            if len(yrs) >= 2:
                return f"{yrs[0][-4:]}–{yrs[1][-2:]}"
        return str(fold.get("fold", ""))

    xs = list(range(len(folds)))
    xlabels = [_label(f) for f in folds]

    # COVID folds = test windows that include 2020 (folds whose label spans 2019-2020 or 2020-2021)
    covid_idx = [i for i, lbl in enumerate(xlabels) if lbl.startswith(("2019", "2020"))]

    fig, ax = plt.subplots(figsize=(7.2, 4.2))

    # Shade COVID-era folds
    for ci in covid_idx:
        ax.axvspan(ci - 0.5, ci + 0.5, color="#f2c14e", alpha=0.18, zorder=0)

    ax.axhline(0.0, color="0.4", lw=1.0, ls="-", zorder=1)
    ax.axhline(0.01, color="0.6", lw=0.9, ls=":", zorder=1)

    markers = ["o", "s", "^", "D"]
    for (key, label), mk in zip(OUTCOMES, markers):
        ys = []
        for f in folds:
            m = f.get("outcomes", {}).get(key)
            ys.append(None if m is None else m["ft_auroc"] - m["xgb_auroc"])
        ax.plot(xs, ys, marker=mk, lw=1.8, ms=6, label=label)

    ax.set_xticks(xs)
    ax.set_xticklabels(xlabels, rotation=0)
    ax.set_xlabel("Walk-forward test window")
    ax.set_ylabel(r"AUROC difference (FT $-$ XGBoost)")
    ax.set_title("Walk-forward: FT-Transformer advantage over XGBoost by fold")
    ax.legend(frameon=False, ncol=2, fontsize=9, loc="upper right")
    ax.grid(axis="y", color="0.9", lw=0.7)
    ax.margins(x=0.03)

    # annotate the COVID shading once
    if covid_idx:
        ax.text(
            covid_idx[0] - 0.5, ax.get_ylim()[1], " COVID", va="top", ha="left",
            fontsize=8, color="#9a7d2e",
        )

    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=200)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()

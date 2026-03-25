"""
Data specification generator for Study 0.

Produces a machine-readable YAML and a human-readable Markdown document
describing the full experimental setup: company universe, feature pipeline,
label definitions, split specification, and per-split statistics.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from fin_jepa.data.sector_map import FF12_SECTORS
from fin_jepa.data.splits import (
    RollingSplitConfig,
    SplitConfig,
    describe_splits,
)
from fin_jepa.data.xbrl_pipeline import FEATURE_NAMES as RAW_FEATURES
from fin_jepa.data.feature_engineering import RATIO_FEATURES, YOY_FEATURES


def generate_data_spec(
    splits: dict[str, pd.DataFrame],
    split_config: SplitConfig | None = None,
    rolling_config: RollingSplitConfig | None = None,
    universe_df: pd.DataFrame | None = None,
    label_provenance: dict | None = None,
    output_path: Path | str | None = None,
) -> dict[str, Any]:
    """Generate a machine-readable data specification.

    Parameters
    ----------
    splits : static split dict (train/val/test DataFrames)
    split_config : the SplitConfig used to produce *splits*
    rolling_config : optional RollingSplitConfig for robustness checks
    universe_df : company-level DataFrame (optional, for universe stats)
    label_provenance : provenance dict from label database (optional)
    output_path : directory to write data_spec.yaml and data_spec.md

    Returns
    -------
    dict with keys: universe, features, labels, splits, per_split_stats,
    reproducibility, generated_at
    """
    spec: dict[str, Any] = {"generated_at": datetime.now(timezone.utc).isoformat()}

    # ── Universe ──────────────────────────────────────────────────────────
    universe_info: dict[str, Any] = {
        "source": "SEC EDGAR 10-K filings",
        "sample_window": "2012–2024",
        "inclusion_criteria": "≥1 10-K filing in sample window",
        "survivorship_bias_free": True,
        "sectors": FF12_SECTORS,
    }
    if universe_df is not None:
        universe_info["n_companies"] = int(universe_df["cik"].nunique())
        if "sector" in universe_df.columns:
            universe_info["sector_distribution"] = (
                universe_df["sector"].value_counts().to_dict()
            )
    spec["universe"] = universe_info

    # ── Features ──────────────────────────────────────────────────────────
    spec["features"] = {
        "raw_xbrl": {"count": len(RAW_FEATURES), "names": list(RAW_FEATURES)},
        "ratios": {"count": len(RATIO_FEATURES), "names": list(RATIO_FEATURES)},
        "yoy_changes": {"count": len(YOY_FEATURES), "names": list(YOY_FEATURES)},
        "other": ["is_first_year", "sector_idx"],
        "total_numeric": len(RAW_FEATURES) + len(RATIO_FEATURES) + len(YOY_FEATURES) + 1,
        "missingness_flags": True,
        "normalization": "quantile (QuantileTransformer, output_distribution=normal)",
        "winsorization": "1st/99th percentile",
        "coverage_threshold": 0.50,
    }

    # ── Labels ────────────────────────────────────────────────────────────
    labels_info: dict[str, Any] = {
        "outcomes": [
            "stock_decline", "earnings_restate", "audit_qualification",
            "sec_enforcement", "bankruptcy",
        ],
        "horizon_days": 365,
        "nan_semantics": "unavailable (not equivalent to no event)",
        "delisting_treatment": "delisted companies treated as stock_decline=1",
    }
    if label_provenance:
        labels_info["provenance"] = label_provenance
    spec["labels"] = labels_info

    # ── Splits ────────────────────────────────────────────────────────────
    splits_info: dict[str, Any] = {"method": "temporal (fiscal period end date)"}
    if split_config:
        splits_info["static"] = {
            "train_end": split_config.train_end,
            "val_end": split_config.val_end,
            "test_end": split_config.test_end,
        }
    if rolling_config:
        splits_info["rolling"] = {
            "first_train_end": rolling_config.first_train_end,
            "val_window_years": rolling_config.val_window_years,
            "test_window_years": rolling_config.test_window_years,
            "step_years": rolling_config.step_years,
            "last_test_end": rolling_config.last_test_end,
        }
    spec["splits"] = splits_info

    # ── Per-split statistics ──────────────────────────────────────────────
    spec["per_split_stats"] = describe_splits(splits)

    # ── Reproducibility ───────────────────────────────────────────────────
    spec["reproducibility"] = {
        "random_seed": 42,
        "scaler_fit_on": "train split only",
        "label_look_ahead": "forward-looking window only",
        "edgar_data": "cached and deterministic",
    }

    # ── Write outputs ─────────────────────────────────────────────────────
    if output_path is not None:
        out = Path(output_path)
        out.mkdir(parents=True, exist_ok=True)

        # YAML
        with open(out / "data_spec.yaml", "w", encoding="utf-8") as fh:
            yaml.dump(spec, fh, default_flow_style=False, sort_keys=False)

        # Markdown
        _write_markdown(spec, out / "data_spec.md")

    return spec


def _write_markdown(spec: dict[str, Any], path: Path) -> None:
    """Render the spec dict as a human-readable Markdown document."""
    lines: list[str] = []
    lines.append("# Study 0 — Data Specification\n")
    lines.append(f"*Generated: {spec['generated_at']}*\n")

    # Universe
    u = spec["universe"]
    lines.append("## Company Universe\n")
    lines.append(f"- **Source:** {u['source']}")
    lines.append(f"- **Sample window:** {u['sample_window']}")
    lines.append(f"- **Inclusion:** {u['inclusion_criteria']}")
    lines.append(f"- **Survivorship-bias-free:** {u['survivorship_bias_free']}")
    if "n_companies" in u:
        lines.append(f"- **Companies:** {u['n_companies']:,}")
    lines.append("")

    # Features
    f = spec["features"]
    lines.append("## Feature Pipeline\n")
    lines.append(f"| Group | Count |")
    lines.append(f"|-------|-------|")
    lines.append(f"| Raw XBRL | {f['raw_xbrl']['count']} |")
    lines.append(f"| Financial ratios | {f['ratios']['count']} |")
    lines.append(f"| YoY changes | {f['yoy_changes']['count']} |")
    lines.append(f"| Other (is_first_year, sector_idx) | 2 |")
    lines.append(f"| **Total numeric** | **{f['total_numeric']}** |")
    lines.append(f"\n- Missingness flags: yes (one per numeric feature)")
    lines.append(f"- Normalization: {f['normalization']}")
    lines.append(f"- Winsorization: {f['winsorization']}")
    lines.append(f"- Coverage threshold: {f['coverage_threshold']}")
    lines.append("")

    # Labels
    lb = spec["labels"]
    lines.append("## Distress Labels\n")
    lines.append(f"- **Outcomes:** {', '.join(lb['outcomes'])}")
    lines.append(f"- **Horizon:** {lb['horizon_days']} days (forward-looking from period end)")
    lines.append(f"- **NaN semantics:** {lb['nan_semantics']}")
    lines.append(f"- **Delisting:** {lb['delisting_treatment']}")
    lines.append("")

    # Splits
    s = spec["splits"]
    lines.append("## Split Specification\n")
    lines.append(f"- **Method:** {s['method']}")
    if "static" in s:
        st = s["static"]
        lines.append(f"- **Train:** period_end ≤ {st['train_end']}")
        lines.append(f"- **Validation:** {st['train_end']} < period_end ≤ {st['val_end']}")
        lines.append(f"- **Test:** {st['val_end']} < period_end ≤ {st['test_end']}")
    if "rolling" in s:
        r = s["rolling"]
        lines.append(f"\n### Rolling Splits (Robustness)")
        lines.append(f"- First train cutoff: {r['first_train_end']}")
        lines.append(f"- Val window: {r['val_window_years']} year(s)")
        lines.append(f"- Test window: {r['test_window_years']} year(s)")
        lines.append(f"- Step: {r['step_years']} year(s)")
        lines.append(f"- Last test end: {r['last_test_end']}")
    lines.append("")

    # Per-split stats
    ps = spec["per_split_stats"]
    lines.append("## Per-Split Statistics\n")
    for part_name, info in ps.items():
        if isinstance(info, dict):
            parts = [f"{k}={v}" for k, v in info.items()]
            lines.append(f"- **{part_name}:** {', '.join(parts)}")
    lines.append("")

    # Reproducibility
    rp = spec["reproducibility"]
    lines.append("## Reproducibility\n")
    for k, v in rp.items():
        lines.append(f"- **{k}:** {v}")
    lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")

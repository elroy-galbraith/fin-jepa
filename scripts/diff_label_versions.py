#!/usr/bin/env python
"""Diff a label column between two versions of the label database.

Designed for the Phase 6 HF dataset re-publish: given the pre-fix and
post-fix ``label_database.parquet`` files (or any two label parquets),
report how many rows flipped ``0 → 1``, ``1 → 0``, or stayed the same,
with a small sample of flipped rows for spot-checking.

Usage
-----
    python scripts/diff_label_versions.py \
        --old data/processed/label_database_v1.0.parquet \
        --new data/processed/label_database.parquet \
        --column earnings_restate \
        --output-dir results/study0
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

logger = logging.getLogger("diff_label_versions")


def _normalize_key(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["cik"] = df["cik"].astype(str).str.strip().str.zfill(10)
    df["period_end"] = pd.to_datetime(df["period_end"]).dt.date
    return df


def diff_column(
    old: pd.DataFrame,
    new: pd.DataFrame,
    column: str,
) -> tuple[dict, pd.DataFrame]:
    """Inner-join on (cik, period_end) and report how *column* changed.

    Returns:
        ``(summary, flipped_rows)`` — counts and the subset of rows where
        the value actually changed.
    """
    if column not in old.columns:
        raise KeyError(f"old DataFrame missing column '{column}'")
    if column not in new.columns:
        raise KeyError(f"new DataFrame missing column '{column}'")

    old_k = _normalize_key(old)[["cik", "period_end", column]].rename(
        columns={column: f"{column}_old"}
    )
    new_k = _normalize_key(new)[["cik", "period_end", column]].rename(
        columns={column: f"{column}_new"}
    )

    inner = old_k.merge(new_k, on=["cik", "period_end"], how="inner")
    outer = old_k.merge(new_k, on=["cik", "period_end"], how="outer", indicator=True)

    old_col = inner[f"{column}_old"]
    new_col = inner[f"{column}_new"]

    flipped_0_to_1 = ((old_col == 0) & (new_col == 1)).sum()
    flipped_1_to_0 = ((old_col == 1) & (new_col == 0)).sum()
    unchanged_pos = ((old_col == 1) & (new_col == 1)).sum()
    unchanged_neg = ((old_col == 0) & (new_col == 0)).sum()
    became_nan = (old_col.notna() & new_col.isna()).sum()
    became_known = (old_col.isna() & new_col.notna()).sum()

    summary = {
        "column": column,
        "joined_rows": int(len(inner)),
        "flipped_0_to_1": int(flipped_0_to_1),
        "flipped_1_to_0": int(flipped_1_to_0),
        "unchanged_pos": int(unchanged_pos),
        "unchanged_neg": int(unchanged_neg),
        "became_nan": int(became_nan),
        "became_known": int(became_known),
        "new_rows": int((outer["_merge"] == "right_only").sum()),
        "dropped_rows": int((outer["_merge"] == "left_only").sum()),
        "old_positive_rate": (
            float((old_col == 1).sum() / old_col.notna().sum())
            if old_col.notna().any() else None
        ),
        "new_positive_rate": (
            float((new_col == 1).sum() / new_col.notna().sum())
            if new_col.notna().any() else None
        ),
    }

    flipped_mask = (old_col != new_col) & (old_col.notna() | new_col.notna())
    flipped_rows = inner.loc[flipped_mask].copy()
    return summary, flipped_rows


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else "")
    p.add_argument("--old", type=Path, required=True, help="Old label parquet.")
    p.add_argument("--new", type=Path, required=True, help="New label parquet.")
    p.add_argument("--column", default="earnings_restate",
                   help="Label column to diff (default: earnings_restate).")
    p.add_argument("--output-dir", type=Path, default=Path("results/study0"))
    p.add_argument("--sample-n", type=int, default=50,
                   help="Flipped rows to sample into the spot-check CSV.")
    p.add_argument("--log-level", default="INFO")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    )

    old = pd.read_parquet(args.old)
    new = pd.read_parquet(args.new)
    logger.info("Loaded old (%d rows) and new (%d rows)", len(old), len(new))

    summary, flipped = diff_column(old, new, args.column)
    logger.info("Diff: %s", json.dumps(summary, indent=2))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_json = args.output_dir / f"{args.column}_diff.json"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if not flipped.empty:
        sample = flipped.sample(n=min(args.sample_n, len(flipped)), random_state=0)
        sample.to_csv(args.output_dir / f"{args.column}_flipped_sample.csv", index=False)

    logger.info("Wrote %s", out_json)
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python
"""Audit the divergence between EDGAR 10-K/A and XBRL amendment signals.

Classifies every XBRL-detected amendment (i.e. every ``(cik, period_end)``
that has >=2 10-K filings in the Company Facts API) into one of five
buckets, then writes the counts to ``results/study0/restatement_audit.json``
plus a small CSV sample per bucket for spot-checking.

Buckets
-------
* ``both`` — ``labels.earnings_restate == 1`` already ✔
* ``edgar_misses_by_window`` — a matching ``10-K/A`` exists in the EDGAR
  index but ``filed_date > period_end + 365d`` (old horizon misses it)
* ``edgar_misses_by_form`` — an amendment exists under a variant form
  (``NT 10-K/A``, ``10-KT/A``, …) but not exact ``10-K/A``
* ``xbrl_only`` — no corresponding amendment in the EDGAR index at all
* ``label_only`` — a ``10-K/A`` exists in the index but XBRL has a single
  entry (original + amendment did not update the XBRL facts)

Usage
-----
    python scripts/audit_earnings_restate.py
    python scripts/audit_earnings_restate.py --raw-dir data/raw \
        --processed-dir data/processed \
        --output-dir results/study0
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

logger = logging.getLogger("audit_earnings_restate")

EDGAR_AMENDMENT_FORMS_STRICT = ("10-K/A",)
EDGAR_AMENDMENT_FORMS_LOOSE = ("10-K/A", "10-KT/A", "NT 10-K/A")
WINDOW_DAYS_NARROW = 365


def _load_parquets(pattern: Path) -> pd.DataFrame:
    files = sorted(pattern.parent.glob(pattern.name))
    if not files:
        return pd.DataFrame()
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def _normalize_cik(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.zfill(10)


def _load_registry(raw_dir: Path, processed_dir: Path) -> pd.DataFrame:
    for p in (raw_dir / "xbrl_amendment_registry.parquet",
              processed_dir / "xbrl_amendment_registry.parquet"):
        if p.exists():
            logger.info("Loading XBRL amendment registry from %s", p)
            df = pd.read_parquet(p)
            df["cik"] = _normalize_cik(df["cik"])
            df["period_end"] = pd.to_datetime(df["period_end"]).dt.date
            df["filed_date"] = pd.to_datetime(df["filed_date"]).dt.date
            return df
    raise FileNotFoundError(
        "xbrl_amendment_registry.parquet not found in raw or processed dirs. "
        "Run build_xbrl_dataset with write_amendment_registry=True first."
    )


def _load_labels(processed_dir: Path) -> pd.DataFrame:
    path = processed_dir / "label_database.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found — run build_label_database first.")
    df = pd.read_parquet(path)
    df["cik"] = _normalize_cik(df["cik"])
    df["period_end"] = pd.to_datetime(df["period_end"]).dt.date
    return df[["cik", "period_end", "earnings_restate"]]


def _load_edgar_index(raw_dir: Path) -> pd.DataFrame:
    index_dir = raw_dir / "edgar_index"
    files = sorted(index_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No EDGAR index parquets found at {index_dir}.")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df["cik"] = _normalize_cik(df["cik"])
    df["form_type"] = df["form_type"].astype(str).str.strip()
    df["date_filed"] = pd.to_datetime(df["date_filed"], errors="coerce")
    return df


def _xbrl_amended_pairs(registry: pd.DataFrame) -> pd.DataFrame:
    """Return (cik, period_end) pairs where the registry has ≥2 filings."""
    counts = registry.groupby(["cik", "period_end"]).size().reset_index(name="n")
    return counts.loc[counts["n"] >= 2, ["cik", "period_end"]].reset_index(drop=True)


def classify_divergence(
    registry: pd.DataFrame,
    labels: pd.DataFrame,
    edgar: pd.DataFrame,
) -> pd.DataFrame:
    """Return a DataFrame with a ``bucket`` column for every amendment."""
    amended = _xbrl_amended_pairs(registry)
    logger.info("XBRL-detected amendments: %d (cik, period_end) pairs", len(amended))

    amended = amended.merge(labels, on=["cik", "period_end"], how="left")

    # EDGAR amendments per CIK with both date_filed and form_type
    edgar_amends = edgar.loc[
        edgar["form_type"].isin(EDGAR_AMENDMENT_FORMS_LOOSE),
        ["cik", "form_type", "date_filed"],
    ].copy()

    def _bucket(row: pd.Series) -> str:
        label_val = row["earnings_restate"]
        if not pd.isna(label_val) and int(label_val) == 1:
            return "both"
        cik_edgar = edgar_amends.loc[edgar_amends["cik"] == row["cik"]]
        if cik_edgar.empty:
            return "xbrl_only"
        period_end = pd.Timestamp(row["period_end"])
        # Did any strict 10-K/A exist for this CIK?
        strict = cik_edgar.loc[cik_edgar["form_type"].isin(EDGAR_AMENDMENT_FORMS_STRICT)]
        if not strict.empty:
            # If any 10-K/A filed after the narrow window, that's window-miss.
            after_window = strict.loc[
                strict["date_filed"] > period_end + pd.Timedelta(days=WINDOW_DAYS_NARROW)
            ]
            if not after_window.empty:
                return "edgar_misses_by_window"
            # 10-K/A exists within window → label should have been 1; flag separately.
            return "label_only"
        # Only variant forms → missed by strict form-type filter
        return "edgar_misses_by_form"

    amended["bucket"] = amended.apply(_bucket, axis=1)
    return amended


def summarize(amended: pd.DataFrame) -> dict:
    counts = amended["bucket"].value_counts().to_dict()
    return {
        "n_xbrl_amendments": int(len(amended)),
        "n_labeled_positive": int((amended["earnings_restate"] == 1).sum()),
        "positive_rate": (
            float((amended["earnings_restate"] == 1).mean()) if len(amended) else None
        ),
        "buckets": {k: int(v) for k, v in counts.items()},
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else "")
    p.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    p.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    p.add_argument("--output-dir", type=Path, default=Path("results/study0"))
    p.add_argument("--sample-n", type=int, default=20,
                   help="Rows to sample per bucket into the spot-check CSV.")
    p.add_argument("--log-level", default="INFO")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    )

    registry = _load_registry(args.raw_dir, args.processed_dir)
    labels = _load_labels(args.processed_dir)
    edgar = _load_edgar_index(args.raw_dir)

    amended = classify_divergence(registry, labels, edgar)
    summary = summarize(amended)
    logger.info("Audit summary: %s", json.dumps(summary, indent=2))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "restatement_audit.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    sample_frames: list[pd.DataFrame] = []
    for bucket, grp in amended.groupby("bucket"):
        n = min(args.sample_n, len(grp))
        sample_frames.append(grp.sample(n=n, random_state=0))
    if sample_frames:
        pd.concat(sample_frames, ignore_index=True).to_csv(
            args.output_dir / "restatement_audit_samples.csv", index=False
        )

    logger.info("Wrote %s and samples CSV", args.output_dir / "restatement_audit.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python
"""Prepare and optionally upload the fin-jepa Study 0 dataset to HuggingFace.

Merges the four source parquets (xbrl_features, label_database,
company_universe, market_aligned) into temporal train/val/test splits,
generates a dataset card (README.md), and copies the raw files for
advanced users.

Output layout::

    {output_dir}/
    ├── README.md                    # HF dataset card
    ├── default/
    │   ├── train.parquet
    │   ├── validation.parquet
    │   └── test.parquet
    └── raw/
        ├── xbrl_features.parquet
        ├── label_database.parquet
        ├── company_universe.parquet
        └── market_aligned.parquet

Usage
-----
    python scripts/prepare_hf_dataset.py
    python scripts/prepare_hf_dataset.py --upload --repo-id elroyg/fin-jepa-study0
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
import time
from pathlib import Path

# ── Logging setup ────────────────────────────────────────────────────────────

LOG_FILE = Path(__file__).parent.parent / "logs" / "hf_dataset.log"
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8"),
    ],
)
logger = logging.getLogger("hf_dataset")

# ── Imports (after logging) ──────────────────────────────────────────────────

import pandas as pd  # noqa: E402

# ── Constants ────────────────────────────────────────────────────────────────

TRAIN_END = "2017-12-31"
VAL_END = "2019-12-31"
TEST_END = "2023-12-31"

XBRL_FEATURES = [
    "total_assets",
    "total_liabilities",
    "total_equity",
    "current_assets",
    "current_liabilities",
    "retained_earnings",
    "cash_equivalents",
    "total_debt",
    "total_revenue",
    "cost_of_sales",
    "operating_income",
    "net_income",
    "interest_expense",
    "cash_from_operations",
    "cash_from_investing",
    "cash_from_financing",
]

LABEL_COLS = [
    "stock_decline",
    "earnings_restate",
    "audit_qualification",
    "sec_enforcement",
    "bankruptcy",
]

MARKET_COLS = ["fwd_ret_252d", "mkt_adj_252d", "delisted"]

ID_COLS = ["cik", "ticker", "fiscal_year", "period_end", "filed_date", "sector", "sic_code"]

AUDIT_COLS = ["period_end_xbrl", "period_end_label"]

OUTPUT_COLS = ID_COLS + XBRL_FEATURES + LABEL_COLS + MARKET_COLS + AUDIT_COLS


def _elapsed(t0: float) -> str:
    dt = time.time() - t0
    if dt < 60:
        return f"{dt:.1f}s"
    return f"{dt / 60:.1f}m"


# ── Data loading ─────────────────────────────────────────────────────────────


def load_source_data(
    raw_dir: Path,
    processed_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the four source parquets and validate existence."""
    paths = {
        "xbrl_features": raw_dir / "xbrl_features.parquet",
        "label_database": processed_dir / "label_database.parquet",
        "company_universe": raw_dir / "company_universe.parquet",
        "market_aligned": raw_dir / "market" / "market_aligned.parquet",
    }
    for name, p in paths.items():
        if not p.exists():
            raise FileNotFoundError(f"{name} not found at {p}")

    xbrl_df = pd.read_parquet(paths["xbrl_features"])
    labels_df = pd.read_parquet(paths["label_database"])
    universe_df = pd.read_parquet(paths["company_universe"])
    market_df = pd.read_parquet(paths["market_aligned"])

    logger.info("Loaded source data:")
    logger.info("  xbrl_features:    %d rows", len(xbrl_df))
    logger.info("  label_database:   %d rows", len(labels_df))
    logger.info("  company_universe: %d rows", len(universe_df))
    logger.info("  market_aligned:   %d rows", len(market_df))

    return xbrl_df, labels_df, universe_df, market_df


# ── Merge & split ────────────────────────────────────────────────────────────


def build_default_config(
    xbrl_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    universe_df: pd.DataFrame,
    market_df: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Merge sources and split into train/validation/test DataFrames.

    Joins XBRL features to labels on ``(cik, join_year)`` — where
    ``join_year = period_end.dt.year`` — rather than exact ``period_end``,
    to recover ~3k rows where the two pipelines disagree on the canonical
    fiscal year-end date by a few days (e.g. Dec 29 vs Dec 26).

    Guards:
    1. Dedup XBRL side on (cik, join_year): keep latest filed_date
       (amended filings supersede originals).
    2. Assert uniqueness on both sides after dedup.
    3. After join, drop any row where the two period_ends differ by >14
       days (wrong pairing).
    """
    # Normalise period_end to datetime64 across all frames
    xbrl_df = xbrl_df.copy()
    labels_df = labels_df.copy()
    market_df = market_df.copy()
    xbrl_df["period_end"] = pd.to_datetime(xbrl_df["period_end"])
    labels_df["period_end"] = pd.to_datetime(labels_df["period_end"])
    market_df["period_end"] = pd.to_datetime(market_df["period_end"])

    # Derive join key: calendar year of period_end (NOT xbrl's fiscal_year
    # column, which uses a different convention and is offset by 1-2 years
    # for non-calendar fiscal years)
    xbrl_df["_join_year"] = xbrl_df["period_end"].dt.year
    labels_df["_join_year"] = labels_df["period_end"].dt.year

    # ── Dedup XBRL: keep EARLIEST filed_date per (cik, _join_year) ──────
    #    Duplicates are 10-K/A amendments for the same fiscal period.
    #    We MUST keep the original 10-K (earliest filed_date), NOT the
    #    amendment.  Keeping the amendment would be label leakage:
    #    earnings_restate is defined as "10-K/A filed within 365 days",
    #    so the corrected financials from the 10-K/A ARE the outcome
    #    variable.  The original filing reflects what was known at
    #    decision time, consistent with a forward-looking prediction
    #    scenario.
    xbrl_df["filed_date"] = pd.to_datetime(xbrl_df["filed_date"])
    pre_dedup = len(xbrl_df)

    # Identify the rows that will be dropped (10-K/A amendments)
    xbrl_sorted = xbrl_df.sort_values("filed_date", ascending=True)
    is_dup = xbrl_sorted.duplicated(subset=["cik", "_join_year"], keep="first")
    dropped_amendments = xbrl_sorted[is_dup].copy()

    xbrl_df = (
        xbrl_sorted
        .drop_duplicates(subset=["cik", "_join_year"], keep="first")
        .sort_values(["cik", "_join_year"])
        .reset_index(drop=True)
    )
    n_dropped = pre_dedup - len(xbrl_df)
    if n_dropped > 0:
        logger.info(
            "  Deduped xbrl on (cik, join_year): %d -> %d "
            "(dropped %d amended filings, kept originals)",
            pre_dedup, len(xbrl_df), n_dropped,
        )

    # ── Audit: check amendment-restatement correlation ──────────────────
    #    The dropped 10-K/A rows should correspond to earnings_restate=1
    #    observations.  If they don't, the label pipeline and XBRL pipeline
    #    are detecting amendments differently — a consistency problem.
    if n_dropped > 0 and "earnings_restate" in labels_df.columns:
        amended_cik_years = set(
            zip(dropped_amendments["cik"], dropped_amendments["_join_year"])
        )
        labels_at_amendments = labels_df[
            labels_df.apply(
                lambda r: (r["cik"], r["_join_year"]) in amended_cik_years, axis=1,
            )
        ]
        n_amended_with_label = len(labels_at_amendments)
        if n_amended_with_label > 0:
            n_restate_1 = int((labels_at_amendments["earnings_restate"] == 1).sum())
            n_restate_0 = int((labels_at_amendments["earnings_restate"] == 0).sum())
            n_restate_na = int(labels_at_amendments["earnings_restate"].isna().sum())
            pct_match = 100.0 * n_restate_1 / n_amended_with_label
            logger.info(
                "  Amendment-restatement audit: %d amended company-years matched "
                "to labels: restate=1: %d (%.1f%%), restate=0: %d, restate=NaN: %d",
                n_amended_with_label, n_restate_1, pct_match,
                n_restate_0, n_restate_na,
            )
            if pct_match < 50.0:
                logger.warning(
                    "  LOW CORRELATION: only %.1f%% of dropped amendments have "
                    "earnings_restate=1. The label pipeline and XBRL pipeline may "
                    "detect amendments differently — investigate before publication.",
                    pct_match,
                )
        else:
            logger.info(
                "  Amendment-restatement audit: no amended company-years "
                "found in label database (different coverage).",
            )

    # ── Guard: assert uniqueness on (cik, _join_year) for both sides ─────
    xbrl_dup_max = xbrl_df.groupby(["cik", "_join_year"]).size().max()
    label_dup_max = labels_df.groupby(["cik", "_join_year"]).size().max()
    assert xbrl_dup_max == 1, (
        f"xbrl still has dupes after dedup (max group = {xbrl_dup_max})"
    )
    assert label_dup_max == 1, (
        f"labels has dupes on (cik, _join_year) (max group = {label_dup_max})"
    )

    # ── Inner join on (cik, _join_year) ──────────────────────────────────
    xbrl_join = xbrl_df.rename(columns={"period_end": "period_end_xbrl"})
    labels_join = labels_df.rename(columns={"period_end": "period_end_label"})

    merged = xbrl_join.merge(labels_join, on=["cik", "_join_year"], how="inner")
    logger.info(
        "After xbrl + labels inner join on (cik, join_year): %d rows", len(merged),
    )

    # ── Guard: drop rows with >14 day date mismatch ─────────────────────
    merged["_date_diff_days"] = (
        merged["period_end_xbrl"] - merged["period_end_label"]
    ).dt.days.abs()

    n_misaligned = int((merged["_date_diff_days"] > 14).sum())
    n_near_miss = int(
        ((merged["_date_diff_days"] > 0) & (merged["_date_diff_days"] <= 14)).sum()
    )
    n_exact = int((merged["_date_diff_days"] == 0).sum())

    if n_misaligned > 0:
        logger.warning(
            "  %d rows have period_end mismatch > 14 days — dropping them",
            n_misaligned,
        )
        merged = merged[merged["_date_diff_days"] <= 14].copy()

    logger.info(
        "  Date alignment: %d exact, %d within 1-14 days, %d dropped (>14 days)",
        n_exact, n_near_miss, n_misaligned,
    )

    # Use XBRL's period_end as canonical (from filing metadata)
    merged["period_end"] = merged["period_end_xbrl"]
    merged = merged.drop(columns=["_date_diff_days", "_join_year"])

    # ── Left join: market context columns ────────────────────────────────
    # Market was built from the same pipeline as labels, so join on the
    # label's period_end to match correctly
    market_subset = market_df[["cik", "period_end"] + MARKET_COLS].copy()
    market_subset = market_subset.rename(columns={"period_end": "_mkt_pe"})
    merged = merged.merge(
        market_subset,
        left_on=["cik", "period_end_label"],
        right_on=["cik", "_mkt_pe"],
        how="left",
    )
    merged = merged.drop(columns=["_mkt_pe"], errors="ignore")
    logger.info("After market left join: %d rows", len(merged))

    # Left join: sector + sic_code from universe (deduplicated on cik)
    universe_subset = universe_df[["cik", "sector", "sic_code"]].drop_duplicates(subset=["cik"])
    merged = merged.merge(universe_subset, on="cik", how="left")

    # ── Select and reorder columns ───────────────────────────────────────
    available_cols = [c for c in OUTPUT_COLS if c in merged.columns]
    merged = merged[available_cols]

    # Temporal split (on canonical period_end = XBRL's)
    train_end = pd.Timestamp(TRAIN_END)
    val_end = pd.Timestamp(VAL_END)
    test_end = pd.Timestamp(TEST_END)

    splits = {
        "train": merged[merged["period_end"] <= train_end].copy(),
        "validation": merged[
            (merged["period_end"] > train_end) & (merged["period_end"] <= val_end)
        ].copy(),
        "test": merged[
            (merged["period_end"] > val_end) & (merged["period_end"] <= test_end)
        ].copy(),
    }

    for name, df in splits.items():
        logger.info("  %s: %d rows", name, len(df))

    return splits


# ── Parquet I/O ──────────────────────────────────────────────────────────────


def write_parquets(splits: dict[str, pd.DataFrame], output_dir: Path) -> None:
    """Write train/validation/test parquets to output_dir/default/."""
    config_dir = output_dir / "default"
    config_dir.mkdir(parents=True, exist_ok=True)

    for split_name, df in splits.items():
        path = config_dir / f"{split_name}.parquet"
        df.to_parquet(path, engine="pyarrow", compression="snappy", index=False)
        size_mb = path.stat().st_size / (1024 * 1024)
        logger.info("  Wrote %s (%.2f MB, %d rows)", path, size_mb, len(df))


def copy_raw_files(raw_dir: Path, processed_dir: Path, output_dir: Path) -> None:
    """Copy the four source parquets into output_dir/raw/."""
    raw_out = output_dir / "raw"
    raw_out.mkdir(parents=True, exist_ok=True)

    sources = [
        raw_dir / "xbrl_features.parquet",
        processed_dir / "label_database.parquet",
        raw_dir / "company_universe.parquet",
        raw_dir / "market" / "market_aligned.parquet",
    ]
    for src in sources:
        dst = raw_out / src.name
        shutil.copy2(src, dst)
        size_mb = dst.stat().st_size / (1024 * 1024)
        logger.info("  Copied %s (%.2f MB)", dst, size_mb)


# ── Dataset card generation ──────────────────────────────────────────────────


def _label_stats(splits: dict[str, pd.DataFrame]) -> dict[str, dict]:
    """Compute per-label prevalence stats across all splits combined."""
    combined = pd.concat(splits.values(), ignore_index=True)
    stats = {}
    for col in LABEL_COLS:
        if col not in combined.columns:
            continue
        total = len(combined)
        n_pos = int((combined[col] == 1).sum())
        n_neg = int((combined[col] == 0).sum())
        n_na = int(combined[col].isna().sum())
        coverage = 100.0 * (n_pos + n_neg) / max(total, 1)
        pos_rate = 100.0 * n_pos / max(n_pos + n_neg, 1)
        stats[col] = {
            "n_pos": n_pos,
            "n_neg": n_neg,
            "n_na": n_na,
            "coverage": coverage,
            "pos_rate": pos_rate,
        }
    return stats


def generate_dataset_card(
    splits: dict[str, pd.DataFrame],
    output_dir: Path,
    repo_id: str,
) -> None:
    """Generate a HuggingFace-compatible README.md dataset card."""
    label_stats = _label_stats(splits)
    total_rows = sum(len(df) for df in splits.values())
    n_companies = pd.concat(splits.values())["cik"].nunique()

    # Pre-compute label positive rates for the card (avoids f-string escaping issues)
    def _pos_rate(col: str) -> str:
        return f"{label_stats.get(col, {}).get('pos_rate', 0):.1f}"

    stock_decline_rate = _pos_rate("stock_decline")
    earnings_restate_rate = _pos_rate("earnings_restate")
    audit_qualification_rate = _pos_rate("audit_qualification")
    sec_enforcement_rate = _pos_rate("sec_enforcement")
    bankruptcy_rate = _pos_rate("bankruptcy")

    # ── YAML frontmatter ─────────────────────────────────────────────────
    yaml_features = ""
    for col in OUTPUT_COLS:
        if col in ["cik", "ticker", "sector", "sic_code"]:
            yaml_features += f"      - name: {col}\n        dtype: string\n"
        elif col in ["fiscal_year"]:
            yaml_features += f"      - name: {col}\n        dtype: int64\n"
        elif col in ["period_end", "filed_date"]:
            yaml_features += f"      - name: {col}\n        dtype: date32\n"
        elif col in LABEL_COLS:
            yaml_features += f"      - name: {col}\n        dtype: int8\n"
        elif col == "delisted":
            yaml_features += f"      - name: {col}\n        dtype: bool\n"
        elif col in AUDIT_COLS:
            yaml_features += f"      - name: {col}\n        dtype: date32\n"
        else:
            yaml_features += f"      - name: {col}\n        dtype: float64\n"

    yaml_splits = ""
    for name, df in splits.items():
        yaml_splits += f"      - name: {name}\n        num_examples: {len(df)}\n"

    card = f"""---
license: mit
task_categories:
  - tabular-classification
tags:
  - finance
  - xbrl
  - distress-prediction
  - sec-edgar
  - tabular
  - financial-statements
pretty_name: "Fin-JEPA Study 0: XBRL Financial Distress Dataset"
size_categories:
  - 10K<n<100K
language:
  - en
configs:
  - config_name: default
    data_files:
      - split: train
        path: default/train.parquet
      - split: validation
        path: default/validation.parquet
      - split: test
        path: default/test.parquet
    default: true
dataset_info:
  - config_name: default
    features:
{yaml_features}    splits:
{yaml_splits}---

# Fin-JEPA Study 0: XBRL Financial Distress Dataset

A tabular dataset of **{total_rows:,}** company-year observations for
**{n_companies:,}** unique SEC filers (2012--2023), linking XBRL financial
statement features extracted from 10-K filings to five binary distress
outcomes. Built for the first gate of the
[Financial JEPA](https://github.com/elroy-galbraith/fin-jepa) project.

## Quick Start

```python
from datasets import load_dataset

ds = load_dataset("{repo_id}")
print(ds)
# DatasetDict({{
#     train: Dataset(num_rows={len(splits["train"])}, ...),
#     validation: Dataset(num_rows={len(splits["validation"])}, ...),
#     test: Dataset(num_rows={len(splits["test"])}, ...),
# }})

train = ds["train"].to_pandas()
print(train.columns.tolist())
```

## Dataset Structure

### Configs

| Config | Description | Files |
|--------|-------------|-------|
| `default` | Merged features + labels, temporally split | `default/{{train,validation,test}}.parquet` |
| (raw files) | Individual source parquets for advanced use | `raw/*.parquet` (see below) |

### Columns ({len(OUTPUT_COLS)} total, including 2 audit columns)

**Identifiers / Metadata (7)**

| Column | Type | Description |
|--------|------|-------------|
| `cik` | string | SEC Central Index Key (10-digit, zero-padded) |
| `ticker` | string | Equity ticker symbol (nullable) |
| `fiscal_year` | int64 | Fiscal year of the 10-K filing |
| `period_end` | date | Fiscal year-end date |
| `filed_date` | date | Date the 10-K was filed with the SEC |
| `sector` | string | Fama-French 12-industry sector |
| `sic_code` | string | 4-digit SIC code (nullable) |

**XBRL Financial Features (16)** --- all `float64`, sourced from SEC EDGAR Company Facts API

| Column | Statement | XBRL Concept(s) |
|--------|-----------|-----------------|
| `total_assets` | Balance Sheet | Assets |
| `total_liabilities` | Balance Sheet | Liabilities |
| `total_equity` | Balance Sheet | StockholdersEquity (+fallback) |
| `current_assets` | Balance Sheet | AssetsCurrent |
| `current_liabilities` | Balance Sheet | LiabilitiesCurrent |
| `retained_earnings` | Balance Sheet | RetainedEarningsAccumulatedDeficit |
| `cash_equivalents` | Balance Sheet | CashAndCashEquivalentsAtCarryingValue (+fallbacks) |
| `total_debt` | Balance Sheet | Computed: LongTermDebt + ShortTermBorrowings |
| `total_revenue` | Income | Revenues (+fallbacks) |
| `cost_of_sales` | Income | CostOfGoodsSold (+fallbacks) |
| `operating_income` | Income | OperatingIncomeLoss |
| `net_income` | Income | NetIncomeLoss (+fallbacks) |
| `interest_expense` | Income | InterestExpense (+fallback) |
| `cash_from_operations` | Cash Flow | NetCashProvidedByUsedInOperatingActivities (+fallback) |
| `cash_from_investing` | Cash Flow | NetCashProvidedByUsedInInvestingActivities (+fallback) |
| `cash_from_financing` | Cash Flow | NetCashProvidedByUsedInFinancingActivities (+fallback) |

> Features are **un-normalised** (raw USD values as reported). NaN means the
> concept was not reported in the filing. See the
> [feature engineering module](https://github.com/elroy-galbraith/fin-jepa/blob/main/src/fin_jepa/data/feature_engineering.py)
> for ratio computation, YoY changes, winsorisation, and quantile normalisation.

**Distress Labels (5)** --- nullable `Int8` (0 = no event, 1 = event, NaN = unavailable)

| Column | Definition | Source | Positive Rate |
|--------|-----------|--------|--------------|
| `stock_decline` | Market-adjusted 252-day return < -20% | yfinance (forward returns from filing date) | {stock_decline_rate}% |
| `earnings_restate` | 10-K/A form type found in EDGAR quarterly index within 365 days of period_end | EDGAR quarterly index search (noisy proxy -- see note) | {earnings_restate_rate}% |
| `audit_qualification` | Going-concern or adverse audit opinion | External CSV (Audit Analytics) | {audit_qualification_rate}% |
| `sec_enforcement` | SEC enforcement action disclosure | EDGAR EFTS full-text search (noisy proxy) | {sec_enforcement_rate}% |
| `bankruptcy` | Chapter 7/11 filing (8-K Item 1.03) | EDGAR EFTS full-text search | {bankruptcy_rate}% |

> **NaN semantics:** NaN means "data unavailable", **not** "no event".
> Training code must mask NaN labels in the loss function.
>
> **Delisted companies:** Firms with fewer than 252 trading days of data
> after their filing date are marked `delisted=True` and assigned
> `stock_decline=1`.
>
> **`earnings_restate` caveat:** This label is a **noisy proxy with a high
> false-negative rate.** The label pipeline searches the EDGAR quarterly
> index for filings with form type `10-K/A`. However, the XBRL feature
> pipeline independently detects ~10x more amended filings (same
> `period_end`, different `filed_date`) than the label pipeline flags.
> Only ~9% of XBRL-detected amendments have `earnings_restate=1`.
> The discrepancy likely reflects differences in how each pipeline
> resolves filing metadata. Treat this label as approximate signal,
> not ground truth. For rigorous restatement detection, use Audit
> Analytics or reconcile the two detection methods.
>
> **`audit_qualification` coverage:** This label may be all-NaN if no
> external Audit Analytics data was available at build time.

**Market Context (3)**

| Column | Type | Description |
|--------|------|-------------|
| `fwd_ret_252d` | float64 | Log return over 252 trading days from filing date |
| `mkt_adj_252d` | float64 | Market-adjusted return (stock - S&P 500) |
| `delisted` | bool | True if < 252 trading days remain post-filing |

**Audit Columns (2)** --- for diagnosing date mismatches between pipelines

| Column | Type | Description |
|--------|------|-------------|
| `period_end_xbrl` | date | Fiscal year-end date from XBRL filing metadata (= `period_end`) |
| `period_end_label` | date | Fiscal year-end date from the label/market pipeline |

> The XBRL and label pipelines sometimes disagree on the canonical fiscal
> year-end date by a few days (e.g. Dec 29 vs Dec 26 for companies whose
> fiscal year ends on the last Saturday of December). The dataset joins on
> `(cik, fiscal_year)` and uses the XBRL date as canonical `period_end`.
> Rows where the two dates differ by more than 14 days are dropped as
> mismatches. These audit columns let you verify alignment.

## Temporal Splits

Splits are **strictly temporal** to prevent look-ahead bias:

| Split | Period End Range | Rows | Purpose |
|-------|-----------------|------|---------|
| `train` | &le; 2017-12-31 | {len(splits["train"]):,} | Model training |
| `validation` | 2018-01-01 to 2019-12-31 | {len(splits["validation"]):,} | Hyperparameter tuning |
| `test` | 2020-01-01 to 2023-12-31 | {len(splits["test"]):,} | Final evaluation |

> The test period covers COVID-19, high-inflation, and rising-rate regimes,
> testing out-of-distribution generalisation.

## Data Sources

All data is derived from **public, free sources**:

| Component | Source | URL |
|-----------|--------|-----|
| XBRL Features | SEC EDGAR Company Facts API | `https://data.sec.gov/api/xbrl/companyfacts/` |
| Filing Index | SEC EDGAR Quarterly Index | `https://www.sec.gov/Archives/edgar/full-index/` |
| Market Data | Yahoo Finance (via yfinance) | `https://finance.yahoo.com/` |
| SEC Enforcement | EDGAR EFTS Full-Text Search | `https://efts.sec.gov/LATEST/search-index` |
| Bankruptcy | EDGAR EFTS (8-K Item 1.03) | `https://efts.sec.gov/LATEST/search-index` |

The universe includes **delisted, bankrupt, and acquired companies** ---
there is no survivorship bias.

## Raw Files

For advanced users who want the individual source tables, four parquets
are available in the `raw/` directory:

```python
from huggingface_hub import hf_hub_download

path = hf_hub_download(
    repo_id="{repo_id}",
    repo_type="dataset",
    filename="raw/xbrl_features.parquet",
)
```

| File | Rows | Description |
|------|------|-------------|
| `raw/xbrl_features.parquet` | ~45k | 16 raw XBRL features per company-year |
| `raw/label_database.parquet` | ~45k | 5 binary distress labels |
| `raw/company_universe.parquet` | ~14k | Company metadata (one row per filer) |
| `raw/market_aligned.parquet` | ~45k | Forward returns at 4 horizons + market benchmarks |

## Citation

```bibtex
@dataset{{galbraith2025finjepa,
  author    = {{Galbraith, Elroy}},
  title     = {{Fin-JEPA Study 0: XBRL Financial Distress Dataset}},
  year      = {{2025}},
  publisher = {{Hugging Face}},
  url       = {{https://huggingface.co/datasets/{repo_id}}}
}}
```

## License

MIT
"""

    readme_path = output_dir / "README.md"
    readme_path.write_text(card, encoding="utf-8")
    logger.info("  Wrote dataset card: %s", readme_path)


# ── Upload ───────────────────────────────────────────────────────────────────


def upload_to_hub(output_dir: Path, repo_id: str, private: bool = False) -> None:
    """Upload the dataset directory to HuggingFace Hub."""
    try:
        from huggingface_hub import HfApi
    except ImportError:
        logger.error(
            "huggingface_hub is not installed.  Run: pip install -e '.[hf]'"
        )
        sys.exit(1)

    api = HfApi()
    api.create_repo(repo_id, repo_type="dataset", private=private, exist_ok=True)
    api.upload_folder(
        folder_path=str(output_dir),
        repo_id=repo_id,
        repo_type="dataset",
    )
    logger.info("Uploaded to https://huggingface.co/datasets/%s", repo_id)


# ── CLI ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare fin-jepa Study 0 dataset for HuggingFace.",
    )
    parser.add_argument(
        "--raw-dir", type=Path, default=Path("data/raw"),
        help="Path to raw data directory (default: data/raw).",
    )
    parser.add_argument(
        "--processed-dir", type=Path, default=Path("data/processed"),
        help="Path to processed data directory (default: data/processed).",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/hf_dataset"),
        help="Output directory for HF dataset files (default: data/hf_dataset).",
    )
    parser.add_argument(
        "--upload", action="store_true",
        help="Upload to HuggingFace Hub after building.",
    )
    parser.add_argument(
        "--repo-id", default="elroyg/fin-jepa-study0",
        help="HuggingFace repo ID (default: elroyg/fin-jepa-study0).",
    )
    parser.add_argument(
        "--private", action="store_true",
        help="Create as private HF repo.",
    )
    args = parser.parse_args()

    t0 = time.time()
    logger.info("Preparing HuggingFace dataset")
    logger.info("  raw_dir       = %s", args.raw_dir.resolve())
    logger.info("  processed_dir = %s", args.processed_dir.resolve())
    logger.info("  output_dir    = %s", args.output_dir.resolve())
    logger.info("  repo_id       = %s", args.repo_id)

    # Load
    xbrl_df, labels_df, universe_df, market_df = load_source_data(
        args.raw_dir, args.processed_dir,
    )

    # Build default config (merged + split)
    logger.info("Building default config …")
    splits = build_default_config(xbrl_df, labels_df, universe_df, market_df)

    # Write parquets
    logger.info("Writing parquets …")
    write_parquets(splits, args.output_dir)

    # Copy raw files
    logger.info("Copying raw files …")
    copy_raw_files(args.raw_dir, args.processed_dir, args.output_dir)

    # Generate dataset card
    logger.info("Generating dataset card …")
    generate_dataset_card(splits, args.output_dir, args.repo_id)

    # Upload
    if args.upload:
        logger.info("Uploading to HuggingFace Hub …")
        upload_to_hub(args.output_dir, args.repo_id, private=args.private)

    logger.info("Done.  Total elapsed: %s", _elapsed(t0))


if __name__ == "__main__":
    main()

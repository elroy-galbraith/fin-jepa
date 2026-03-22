#!/usr/bin/env python
"""Market data pipeline runner for ATS-163.

Steps
-----
1. Build company universe from EDGAR (~20 min first run; cached after).
2. Expand universe to company-year observation grid.
3. Fetch adjusted OHLCV via yfinance (batched; ~20-30 min for ~8k tickers).
4. Fetch market and sector index returns (^GSPC + SPDR ETFs; fast).
5. Optionally fetch corporate actions (off by default; adds ~1-2 h).
6. Align market data to each (cik, period_end) filing date.
7. Write market_aligned.parquet and print a quality summary.

The pipeline is fully resumable: every download is cached to disk, so
interrupted runs skip already-fetched tickers on restart.

Usage
-----
    python scripts/run_market_pipeline.py                    # standard run
    python scripts/run_market_pipeline.py --with-actions     # + corporate actions
    python scripts/run_market_pipeline.py --skip-universe    # use existing universe
    python scripts/run_market_pipeline.py --help
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

# ── Logging setup ────────────────────────────────────────────────────────────

LOG_FILE = Path(__file__).parent.parent / "logs" / "market_pipeline.log"
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8"),
    ],
)
logger = logging.getLogger("market_pipeline")

# ── Imports (after logging so import errors are logged) ─────────────────────

from fin_jepa.data.market_data import (  # noqa: E402
    INDEX_TICKERS,
    MarketDataConfig,
    align_to_filing_dates,
    build_company_year_grid,
    fetch_corporate_actions,
    fetch_index_returns,
    fetch_ohlcv,
)
from fin_jepa.data.universe import build_company_universe, load_company_universe  # noqa: E402


# ── Helpers ──────────────────────────────────────────────────────────────────


def _elapsed(t0: float) -> str:
    s = time.time() - t0
    if s < 90:
        return f"{s:.0f}s"
    return f"{s / 60:.1f}m"


def _summary(aligned_df) -> None:
    import pandas as pd

    logger.info("=" * 60)
    logger.info("PIPELINE SUMMARY")
    logger.info("=" * 60)
    logger.info("  Rows (company-years)   : %d", len(aligned_df))
    logger.info("  Unique CIKs            : %d", aligned_df["cik"].nunique())
    logger.info("  Unique tickers         : %d", aligned_df["ticker"].nunique())
    logger.info(
        "  Date range             : %s → %s",
        pd.to_datetime(aligned_df["filing_date"]).min().date(),
        pd.to_datetime(aligned_df["filing_date"]).max().date(),
    )
    logger.info(
        "  Delisted flag          : %.1f%%",
        aligned_df["delisted"].mean() * 100,
    )
    for col in ("fwd_ret_21d", "fwd_ret_63d", "fwd_ret_126d", "fwd_ret_252d"):
        if col in aligned_df.columns:
            pct = aligned_df[col].notna().mean() * 100
            logger.info("  Coverage %-16s: %.1f%%", col, pct)
    logger.info("=" * 60)


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fin-JEPA market data collection pipeline (ATS-163).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--raw-dir", default="data/raw", help="Root raw data directory."
    )
    parser.add_argument(
        "--start-date", default="2010-01-01", help="Earliest price date to fetch."
    )
    parser.add_argument(
        "--end-date", default="2024-12-31", help="Latest price date to fetch."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Tickers per yfinance batch download.",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.5,
        help="Seconds to sleep between OHLCV batches.",
    )
    parser.add_argument(
        "--with-actions",
        action="store_true",
        help="Also fetch corporate actions (adds ~1-2 h; off by default).",
    )
    parser.add_argument(
        "--skip-universe",
        action="store_true",
        help="Load existing company_universe.parquet instead of rebuilding.",
    )
    parser.add_argument(
        "--min-filings",
        type=int,
        default=2,
        help="Minimum 10-K filings for a company to be included.",
    )
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = raw_dir / "market"
    cache_dir.mkdir(parents=True, exist_ok=True)
    universe_path = raw_dir / "company_universe.parquet"

    pipeline_start = time.time()
    logger.info("▶ Market data pipeline starting  (log → %s)", LOG_FILE)

    # ── Step 1: Company universe ─────────────────────────────────────────────
    if universe_path.exists() and args.skip_universe:
        logger.info("Step 1/6  Loading existing universe from %s …", universe_path)
        t0 = time.time()
        universe_df, provenance = load_company_universe(universe_path)
        logger.info(
            "Step 1/6  Done (%s) — %d companies", _elapsed(t0), len(universe_df)
        )
    else:
        logger.info("Step 1/6  Building company universe (EDGAR 2012–2024) …")
        logger.info(
            "          This fetches quarterly EDGAR indexes + per-company"
            " submissions.  Expect ~20 min on first run."
        )
        t0 = time.time()
        universe_df = build_company_universe(
            raw_dir=raw_dir,
            min_filings=args.min_filings,
        )
        logger.info(
            "Step 1/6  Done (%s) — %d companies (min_filings=%d)",
            _elapsed(t0),
            len(universe_df),
            args.min_filings,
        )

    # ── Step 2: Company-year observation grid ────────────────────────────────
    logger.info("Step 2/6  Building company-year observation grid …")
    t0 = time.time()
    filings_df = build_company_year_grid(universe_df)
    logger.info(
        "Step 2/6  Done (%s) — %d company-year observations",
        _elapsed(t0),
        len(filings_df),
    )

    tickers: list[str] = (
        filings_df["ticker"]
        .dropna()
        .astype(str)
        .str.strip()
        .loc[lambda s: s != ""]
        .unique()
        .tolist()
    )
    logger.info("          %d unique tickers with exchange listings", len(tickers))

    cfg = MarketDataConfig(
        start_date=args.start_date,
        end_date=args.end_date,
        batch_size=args.batch_size,
        sleep_between_batches=args.sleep,
    )

    # ── Step 3: Adjusted OHLCV ───────────────────────────────────────────────
    logger.info(
        "Step 3/6  Fetching adjusted OHLCV (%d tickers, batches of %d) …",
        len(tickers),
        cfg.batch_size,
    )
    logger.info(
        "          Estimated time: %d–%d minutes (first run)",
        len(tickers) // cfg.batch_size * 10 // 60,
        len(tickers) // cfg.batch_size * 20 // 60,
    )
    t0 = time.time()
    prices_dict = fetch_ohlcv(
        tickers=tickers,
        start=cfg.start_date,
        end=cfg.end_date,
        cache_dir=cache_dir,
        batch_size=cfg.batch_size,
        sleep_between_batches=cfg.sleep_between_batches,
    )
    n_with_data = sum(1 for df in prices_dict.values() if not df.empty)
    logger.info(
        "Step 3/6  Done (%s) — %d/%d tickers returned data",
        _elapsed(t0),
        n_with_data,
        len(tickers),
    )

    # ── Step 4: Market and sector index returns ──────────────────────────────
    logger.info("Step 4/6  Fetching market/sector index returns (%d series) …", len(INDEX_TICKERS))
    t0 = time.time()
    index_prices = fetch_index_returns(
        index_tickers=INDEX_TICKERS,
        start=cfg.start_date,
        end=cfg.end_date,
        cache_dir=cache_dir,
    )
    n_idx = sum(1 for df in index_prices.values() if not df.empty)
    logger.info("Step 4/6  Done (%s) — %d/%d index series", _elapsed(t0), n_idx, len(INDEX_TICKERS))

    # ── Step 5: Corporate actions (optional) ─────────────────────────────────
    if args.with_actions:
        logger.info(
            "Step 5/6  Fetching corporate actions (%d tickers) …", len(tickers)
        )
        logger.info("          This makes one API call per ticker; expect 1-2 hours.")
        t0 = time.time()
        actions_dict = fetch_corporate_actions(
            tickers=tickers,
            cache_dir=cache_dir,
        )
        n_splits = sum(
            1
            for df in actions_dict.values()
            if not df.empty
            and "Stock Splits" in df.columns
            and (df["Stock Splits"] != 0).any()
        )
        logger.info(
            "Step 5/6  Done (%s) — %d tickers had ≥1 stock split",
            _elapsed(t0),
            n_splits,
        )
    else:
        logger.info(
            "Step 5/6  Skipped (corporate actions) — re-run with --with-actions"
            " to fetch splits/dividends."
        )

    # ── Step 6: Align to filing dates ────────────────────────────────────────
    logger.info("Step 6/6  Aligning market data to filing dates …")
    t0 = time.time()
    aligned_df = align_to_filing_dates(
        filings_df=filings_df,
        prices_dict=prices_dict,
        index_prices=index_prices,
        horizons_days=cfg.horizons_days,
        primary_horizon=cfg.return_window_days,
    )
    logger.info("Step 6/6  Done (%s) — %d rows", _elapsed(t0), len(aligned_df))

    # ── Output ───────────────────────────────────────────────────────────────
    out_path = cache_dir / "market_aligned.parquet"
    aligned_df.to_parquet(out_path, index=False)
    logger.info("Output written → %s", out_path)

    _summary(aligned_df)
    logger.info("▶ Pipeline complete in %s", _elapsed(pipeline_start))


if __name__ == "__main__":
    main()

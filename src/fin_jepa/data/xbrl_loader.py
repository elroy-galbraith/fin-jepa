"""
XBRL data ingestion from SEC EDGAR.

Workstream: Collect and align XBRL financial statement data.

TODO:
  - Download 10-K/10-Q XBRL filings via EDGAR bulk data or sec-edgar-downloader
  - Parse instance documents into a flat feature table (one row per filing)
  - Standardize concept names across reporting periods and taxonomies
  - Align filing dates with fiscal period end dates
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_xbrl_features(raw_dir: Path) -> pd.DataFrame:
    """Load and concatenate raw XBRL feature parquet files from *raw_dir*.

    Returns a DataFrame with columns: [cik, ticker, period_end, *xbrl_features].
    """
    raise NotImplementedError("Implement XBRL ingestion pipeline.")

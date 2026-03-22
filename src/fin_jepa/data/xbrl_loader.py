"""XBRL data ingestion from SEC EDGAR.

Thin wrapper around :mod:`fin_jepa.data.xbrl_pipeline` that provides a
simple ``load_xbrl_features(raw_dir)`` interface for downstream consumers.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from fin_jepa.data.xbrl_pipeline import (
    build_xbrl_dataset,
    load_xbrl_features as _load_parquet,
    XBRLConfig,
)


def load_xbrl_features(raw_dir: Path) -> pd.DataFrame:
    """Load XBRL feature data from *raw_dir*.

    Looks for ``raw_dir/xbrl_features.parquet``.  If the file does not
    exist, raises FileNotFoundError — run the extraction pipeline first
    via :func:`fin_jepa.data.xbrl_pipeline.build_xbrl_dataset`.

    Returns a DataFrame with columns: [cik, ticker, period_end, fiscal_year,
    filed_date, *xbrl_features].
    """
    parquet_path = Path(raw_dir) / "xbrl_features.parquet"
    if not parquet_path.exists():
        raise FileNotFoundError(
            f"XBRL features not found at {parquet_path}. "
            "Run the extraction pipeline first: "
            "fin_jepa.data.xbrl_pipeline.build_xbrl_dataset()"
        )
    df, _provenance = _load_parquet(parquet_path)
    return df

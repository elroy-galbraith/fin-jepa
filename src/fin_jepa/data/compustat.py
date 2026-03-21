"""Compustat cross-reference interface for the company universe.

The core company-universe pipeline works with EDGAR data only.  This module
provides a **clean loading interface** for Compustat data when it is available,
plus a merge helper that links Compustat GVKEYs to the EDGAR CIK-based
universe.

Compustat is not freely available; this module does not download it.  If you
have access through WRDS or a campus license, export the ``company`` table
(COMP/COMPNA) and the ``names`` file as CSV or parquet and point the functions
below at those files.

Expected Compustat columns (all names are lowercase after loading)
------------------------------------------------------------------
gvkey        str   Global Company Key (6-digit, zero-padded)
cik          str   SEC CIK (may be null; not always populated by Compustat)
cusip        str   9-digit CUSIP
tic          str   Ticker symbol
conm         str   Company name
sic          str   4-digit SIC code
naics        str   6-digit NAICS code (optional)
exchg        int   Stock exchange code (11=NYSE, 12=AMEX, 14=NASDAQ, ...)
fyr          int   Fiscal year-end month (1–12)
ipodate      date  IPO date (nullable)
dldte        date  Deletion / delisting date (nullable)
dlrsn        str   Deletion reason code (nullable)
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# Compustat stock exchange codes → exchange name
COMPUSTAT_EXCHANGE_MAP: dict[int, str] = {
    11: "NYSE",
    12: "AMEX",
    14: "NASDAQ",
    15: "NYSE MKT",
    19: "OTC",
    90: "TSX",
    91: "LSE",
}

# Compustat deletion reason codes (dlrsn) → human-readable label
COMPUSTAT_DELIST_REASONS: dict[str, str] = {
    "01": "Dropped — no longer fits S&P's coverage criteria",
    "02": "Fiscal year change",
    "03": "Bankruptcy",
    "04": "Reverse acquisition",
    "06": "Acquisition / merger",
    "07": "Leveraged buyout",
    "08": "Private",
    "09": "Foreign incorporation",
    "10": "Delisted / insufficient SEC filings",
    "11": "Reverse merger",
}


def load_compustat_crossref(path: Path | str) -> pd.DataFrame:
    """Load a Compustat company-header file and normalise column names.

    Supports CSV (``*.csv``, ``*.csv.gz``) and Parquet (``*.parquet``,
    ``*.pq``) inputs.

    Args:
        path: Path to the Compustat company file.

    Returns:
        DataFrame with normalised lowercase column names and CIK
        zero-padded to 10 digits where available.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If required columns are missing.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Compustat file not found: {path}")

    suffix = "".join(path.suffixes).lower()
    if ".parquet" in suffix or ".pq" in suffix:
        df = pd.read_parquet(path)
    elif ".csv" in suffix:
        df = pd.read_csv(path, dtype=str, low_memory=False)
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")

    # Normalise to lowercase
    df.columns = [c.lower().strip() for c in df.columns]

    required = {"gvkey"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Compustat file is missing required columns: {missing}")

    # Zero-pad keys
    if "gvkey" in df.columns:
        df["gvkey"] = df["gvkey"].astype(str).str.zfill(6)
    if "cik" in df.columns:
        df["cik"] = (
            df["cik"]
            .where(df["cik"].notna() & (df["cik"].astype(str).str.strip() != ""))
            .astype("string")
            .str.strip()
            .str.zfill(10)
        )
    if "sic" in df.columns:
        df["sic"] = df["sic"].astype(str).str.strip().str.zfill(4)

    # Parse date columns
    for date_col in ("ipodate", "dldte"):
        if date_col in df.columns:
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

    # Map exchange codes
    if "exchg" in df.columns:
        df["exchg"] = pd.to_numeric(df["exchg"], errors="coerce")
        df["exchange_name"] = df["exchg"].map(COMPUSTAT_EXCHANGE_MAP).fillna("Other")

    # Map delist reasons
    if "dlrsn" in df.columns:
        df["delist_reason"] = df["dlrsn"].map(COMPUSTAT_DELIST_REASONS).fillna("Unknown")

    logger.info("Loaded %d Compustat records from %s", len(df), path)
    return df.reset_index(drop=True)


def merge_compustat(
    universe_df: pd.DataFrame,
    compustat_df: pd.DataFrame,
) -> pd.DataFrame:
    """Merge Compustat metadata into the EDGAR company universe.

    Performs a left join (universe is authoritative) using CIK as the
    primary key.  A fallback ticker-based match is applied for companies
    where Compustat's CIK field is null.

    Adds the following columns to the universe (prefixed with ``cstat_``)::

        cstat_gvkey         str     Compustat GVKEY
        cstat_cusip         str     CUSIP
        cstat_naics         str     NAICS code (if available)
        cstat_ipodate       date    IPO date
        cstat_dldte         date    Compustat deletion date
        cstat_dlrsn         str     Deletion reason code
        cstat_delist_reason str     Human-readable deletion reason

    Args:
        universe_df: Output of :func:`~fin_jepa.data.universe.build_company_universe`.
            Must have a ``cik`` column.
        compustat_df: Output of :func:`load_compustat_crossref`.
            Must have a ``gvkey`` column.

    Returns:
        universe_df with Compustat columns appended.  Rows without a
        Compustat match have NaN in all ``cstat_*`` columns.
    """
    compustat_cols = {
        "gvkey": "cstat_gvkey",
        "cusip": "cstat_cusip",
        "naics": "cstat_naics",
        "ipodate": "cstat_ipodate",
        "dldte": "cstat_dldte",
        "dlrsn": "cstat_dlrsn",
        "delist_reason": "cstat_delist_reason",
    }
    # Only keep columns that exist in compustat_df
    keep = {src: dst for src, dst in compustat_cols.items() if src in compustat_df.columns}

    cstat = compustat_df[["cik"] + list(keep.keys())].rename(columns=keep).copy()
    # Drop rows where CIK is null (can't match on CIK alone)
    cstat_with_cik = cstat.dropna(subset=["cik"]).drop_duplicates(subset="cik", keep="first")

    merged = universe_df.merge(cstat_with_cik, on="cik", how="left")

    # Ticker-based fallback for unmatched rows
    unmatched_mask = merged["cstat_gvkey"].isna()
    if unmatched_mask.any() and "tic" in compustat_df.columns and "ticker" in merged.columns:
        cstat_by_ticker = (
            compustat_df[list(keep.keys()) + ["tic"]]
            .rename(columns={**keep, "tic": "ticker"})
            .dropna(subset=["ticker"])
            .drop_duplicates(subset="ticker", keep="first")
        )
        fallback = universe_df[unmatched_mask][["cik", "ticker"]].merge(
            cstat_by_ticker, on="ticker", how="left"
        ).set_index("cik")
        for col in keep.values():
            if col in fallback.columns:
                merged.loc[unmatched_mask, col] = (
                    merged.loc[unmatched_mask, "cik"].map(fallback[col])
                )
        n_recovered = unmatched_mask.sum() - merged["cstat_gvkey"].isna().sum()
        if n_recovered > 0:
            logger.info("  Ticker fallback recovered %d additional Compustat matches", n_recovered)

    n_matched = merged["cstat_gvkey"].notna().sum()
    logger.info(
        "Compustat merge: %d / %d universe companies matched (%.1f%%)",
        n_matched, len(merged), 100 * n_matched / max(len(merged), 1),
    )
    return merged

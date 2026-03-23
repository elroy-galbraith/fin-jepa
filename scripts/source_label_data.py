#!/usr/bin/env python
"""Source external label datasets for ATS-172.

Downloads SEC enforcement and bankruptcy label data from the EDGAR
Full-Text Search (EFTS) API and writes CSVs that the label builder
(``src/fin_jepa/data/labels.py``) can consume.

Data sources
------------
* **sec_enforcement** — Companies disclosing SEC enforcement actions
  against them, identified via EFTS full-text search of 8-K and 10-K
  filings for specific enforcement-related phrases (Wells notices,
  settlements, AAERs, formal investigations).
* **bankruptcy** — 8-K filings disclosing Item 1.03 (Bankruptcy or
  Receivership), which public companies are legally required to file.

Both sources are free, require no API key, and return CIKs directly.

Output
------
``data/raw/labels/sec_enforcement.csv``  (columns: cik, aaer_date)
``data/raw/labels/bankruptcy.csv``       (columns: cik, filing_date, chapter)

Usage
-----
    python scripts/source_label_data.py
    python scripts/source_label_data.py --validate-only
    python scripts/source_label_data.py --raw-dir data/raw --help
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from urllib.parse import quote, urlencode

# ── Logging setup ────────────────────────────────────────────────────────────

LOG_FILE = Path(__file__).parent.parent / "logs" / "label_sourcing.log"
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8"),
    ],
)
logger = logging.getLogger("label_sourcing")

# ── Imports (after logging so import errors are logged) ───────────────────────

import pandas as pd  # noqa: E402
import requests  # noqa: E402

# ── Constants ─────────────────────────────────────────────────────────────────

EFTS_URL = "https://efts.sec.gov/LATEST/search-index"

_DEFAULT_USER_AGENT = (
    "fin-jepa-research admin@example.com"  # EDGAR requires contact info
)

# Date range matching the project universe (2012-2024)
DEFAULT_START = "2012-01-01"
DEFAULT_END = "2024-12-31"

# EDGAR rate-limit: 10 req/s max — sleep 0.15s between requests
RATE_LIMIT_SLEEP = 0.15


def _elapsed(t0: float) -> str:
    """Human-readable elapsed time since *t0*."""
    dt = time.time() - t0
    if dt < 60:
        return f"{dt:.1f}s"
    return f"{dt / 60:.1f}m"


# ── EFTS helpers ──────────────────────────────────────────────────────────────


def _get_session() -> requests.Session:
    """Create a requests.Session with EDGAR-compliant headers."""
    session = requests.Session()
    ua = os.environ.get("EDGAR_USER_AGENT", _DEFAULT_USER_AGENT)
    session.headers.update({
        "User-Agent": ua,
        "Accept-Encoding": "gzip, deflate",
        "Accept": "application/json",
    })
    return session


def _efts_get(
    session: requests.Session,
    url: str,
    *,
    retries: int = 3,
    backoff: float = 2.0,
) -> dict:
    """GET from EFTS with retry on transient errors."""
    for attempt in range(max(retries, 1)):
        try:
            resp = session.get(url, timeout=30)
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            status = getattr(getattr(exc, "response", None), "status_code", None)
            if isinstance(status, int) and 400 <= status < 500 and status != 429:
                raise
            if attempt == retries - 1:
                raise
            wait = backoff ** attempt
            logger.warning(
                "Retry %d/%d for EFTS GET after %.1fs: %s",
                attempt + 1, retries, wait, exc,
            )
            time.sleep(wait)
    return {}  # unreachable


def _efts_search(
    session: requests.Session,
    query: str,
    forms: list[str] | None = None,
    start_date: str = DEFAULT_START,
    end_date: str = DEFAULT_END,
    max_results: int = 10000,
) -> list[dict]:
    """Paginated EFTS full-text search via GET.

    Returns all ``_source`` dicts up to *max_results*.
    """
    all_hits: list[dict] = []
    offset = 0
    page_size = 100  # EFTS max per request
    total = 0

    while True:
        # Build URL with query params
        params: dict[str, str] = {
            "dateRange": "custom",
            "startdt": start_date,
            "enddt": end_date,
            "from": str(offset),
        }
        if query:
            params["q"] = query
        if forms:
            # EFTS accepts comma-separated form types in a single 'forms' param
            params["forms"] = ",".join(forms)

        url = f"{EFTS_URL}?{urlencode(params, quote_via=quote)}"
        data = _efts_get(session, url)

        hits = data.get("hits", {})
        total = hits.get("total", {}).get("value", 0)
        results = hits.get("hits", [])

        if not results:
            break

        for hit in results:
            source = hit.get("_source", {})
            if source:
                all_hits.append(source)

        offset += page_size
        if offset >= total or offset >= max_results:
            break

        time.sleep(RATE_LIMIT_SLEEP)

    logger.info(
        "  EFTS query (q=%r, forms=%s): %d total, %d fetched",
        query[:60] if query else "<none>",
        forms or "<all>",
        total,
        len(all_hits),
    )
    return all_hits


def _extract_cik_date_pairs(
    hits: list[dict],
    date_field: str = "file_date",
) -> pd.DataFrame:
    """Extract (cik, date) pairs from EFTS hit sources.

    Each EFTS hit has ``ciks`` (list) and ``file_date``.
    We take the first CIK (primary filer) and the filing date.
    """
    records = []
    for src in hits:
        ciks = src.get("ciks", [])
        fdate = src.get(date_field)
        if not ciks or not fdate:
            continue
        cik = str(ciks[0]).strip().zfill(10)
        records.append({"cik": cik, "date": str(fdate).strip()})

    df = pd.DataFrame(records)
    if df.empty:
        return df

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    return df


def _load_universe_ciks(raw_dir: Path) -> set[str]:
    """Load the set of CIKs from the company universe."""
    universe_path = raw_dir / "company_universe.parquet"
    if not universe_path.exists():
        raise FileNotFoundError(
            f"Company universe not found at {universe_path}.  "
            "Run the market pipeline first (scripts/run_market_pipeline.py)."
        )
    universe = pd.read_parquet(universe_path, columns=["cik"])
    ciks = set(universe["cik"].astype(str).str.strip().str.zfill(10))
    logger.info("Loaded universe: %d unique CIKs", len(ciks))
    return ciks


# ── SEC Enforcement sourcing ─────────────────────────────────────────────────

# Queries targeting company self-disclosures of SEC enforcement actions.
# These search the full text of EDGAR filings for specific phrases
# indicating the filer is subject to SEC enforcement.  Ordered from
# highest to lowest precision.
_ENFORCEMENT_QUERIES: list[tuple[str, list[str] | None]] = [
    # Tier 1: high precision — direct AAER / Wells notice references
    ('"AAER"', None),                                           # ~56 hits
    ('"Wells notice"', ["8-K"]),                                # ~227 hits
    ('"Wells notice"', ["10-K"]),                               # ~224 hits
    ('"formal order of investigation"', ["8-K"]),               # ~48 hits
    # Tier 2: moderate precision — SEC settlement/charge disclosures
    ('"SEC settlement" "enforcement"', ["8-K"]),                # ~54 hits
    ('"settled charges" "Securities and Exchange Commission"', ["8-K"]),  # ~36 hits
    ('"Accounting and Auditing Enforcement Release"', None),    # ~25 hits
]


def source_sec_enforcement(
    raw_dir: Path,
    session: requests.Session,
    start_date: str = DEFAULT_START,
    end_date: str = DEFAULT_END,
) -> pd.DataFrame:
    """Source SEC enforcement data from EFTS.

    Combines multiple targeted EFTS queries to identify companies
    subject to SEC enforcement actions, then filters to the project
    universe.

    Returns DataFrame with columns ``[cik, aaer_date]``.
    """
    t0 = time.time()
    logger.info("Sourcing SEC enforcement labels …")

    universe_ciks = _load_universe_ciks(raw_dir)

    all_hits: list[dict] = []
    for i, (query, forms) in enumerate(_ENFORCEMENT_QUERIES, 1):
        logger.info("  Query %d: q=%r, forms=%s", i, query, forms or "<all>")
        hits = _efts_search(
            session, query=query, forms=forms,
            start_date=start_date, end_date=end_date,
        )
        all_hits.extend(hits)

    logger.info("  Total raw hits across all queries: %d", len(all_hits))

    df = _extract_cik_date_pairs(all_hits, date_field="file_date")
    if df.empty:
        logger.warning("  No SEC enforcement events found from EFTS.")
        df = pd.DataFrame(columns=["cik", "aaer_date"])
    else:
        df = df.rename(columns={"date": "aaer_date"})
        df = df.drop_duplicates(subset=["cik", "aaer_date"])
        logger.info("  Unique (cik, aaer_date) pairs: %d", len(df))

        # Filter to universe
        pre_filter = len(df)
        df = df[df["cik"].isin(universe_ciks)].reset_index(drop=True)
        logger.info(
            "  After universe filter: %d / %d (%.1f%%)",
            len(df), pre_filter, 100 * len(df) / max(pre_filter, 1),
        )

    # Write CSV
    labels_dir = raw_dir / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)
    out_path = labels_dir / "sec_enforcement.csv"
    df.to_csv(out_path, index=False)
    logger.info(
        "  Wrote %s (%d rows)  [%s]", out_path, len(df), _elapsed(t0),
    )
    return df


# ── Bankruptcy sourcing ──────────────────────────────────────────────────────


def source_bankruptcy(
    raw_dir: Path,
    session: requests.Session,
    start_date: str = DEFAULT_START,
    end_date: str = DEFAULT_END,
) -> pd.DataFrame:
    """Source bankruptcy data from EFTS via 8-K Item 1.03 filings.

    Returns DataFrame with columns ``[cik, filing_date, chapter]``.
    """
    t0 = time.time()
    logger.info("Sourcing bankruptcy labels …")

    universe_ciks = _load_universe_ciks(raw_dir)

    # Query 1: 8-K with "Item 1.03" — legally required bankruptcy disclosure
    logger.info('  Query 1: 8-K filings with "Item 1.03"')
    hits1 = _efts_search(
        session,
        query='"Item 1.03"',
        forms=["8-K"],
        start_date=start_date,
        end_date=end_date,
    )

    # Query 2: 8-K with exact phrase "bankruptcy or receivership"
    logger.info('  Query 2: 8-K filings with "bankruptcy or receivership"')
    hits2 = _efts_search(
        session,
        query='"bankruptcy or receivership"',
        forms=["8-K"],
        start_date=start_date,
        end_date=end_date,
    )

    all_hits = hits1 + hits2
    logger.info("  Total raw hits: %d", len(all_hits))

    df = _extract_cik_date_pairs(all_hits, date_field="file_date")
    if df.empty:
        logger.warning("  No bankruptcy events found from EFTS.")
        df = pd.DataFrame(columns=["cik", "filing_date", "chapter"])
    else:
        df = df.rename(columns={"date": "filing_date"})
        df = df.drop_duplicates(subset=["cik", "filing_date"])
        # Default to Chapter 11 — far more common for public companies
        df["chapter"] = 11
        logger.info("  Unique (cik, filing_date) pairs: %d", len(df))

        # Filter to universe
        pre_filter = len(df)
        df = df[df["cik"].isin(universe_ciks)].reset_index(drop=True)
        logger.info(
            "  After universe filter: %d / %d (%.1f%%)",
            len(df), pre_filter, 100 * len(df) / max(pre_filter, 1),
        )

    # Write CSV
    labels_dir = raw_dir / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)
    out_path = labels_dir / "bankruptcy.csv"
    df.to_csv(out_path, index=False)
    logger.info(
        "  Wrote %s (%d rows)  [%s]", out_path, len(df), _elapsed(t0),
    )
    return df


# ── Validation ────────────────────────────────────────────────────────────────


def validate_sourced_labels(raw_dir: Path) -> dict:
    """Rebuild the label database and report coverage stats.

    Imports the label builder, runs it with external_csv sources,
    and prints the validation report.
    """
    logger.info("Validating sourced labels …")

    from fin_jepa.data.labels import (
        LabelConfig,
        build_label_database,
        validate_label_database,
    )

    output_path = raw_dir.parent / "processed" / "label_database.parquet"
    config = LabelConfig(
        decline_threshold=-0.20,
        treat_delisted_as_decline=True,
        restatement_source="edgar_amendments",
        audit_qualification_source="external_csv",
        sec_enforcement_source="external_csv",
        bankruptcy_source="external_csv",
        external_label_dir=raw_dir / "labels",
        horizon_days=365,
    )

    labels_df = build_label_database(
        raw_dir=raw_dir,
        output_path=output_path,
        config=config,
    )

    stats = validate_label_database(labels_df)

    # Summary
    logger.info("=" * 60)
    logger.info("Label coverage summary:")
    for outcome, s in stats["per_label"].items():
        status = "OK" if s["n_positive"] >= 20 else "LOW"
        logger.info(
            "  [%s] %-25s  pos=%-5d  neg=%-5d  NA=%-5d  coverage=%.1f%%",
            status, outcome, s["n_positive"], s["n_negative"],
            s["n_missing"], s["coverage_pct"],
        )
    logger.info(
        "Labels with majority coverage: %d / %d",
        stats["n_labels_with_majority_coverage"],
        len(stats["per_label"]),
    )
    logger.info("=" * 60)

    # Check minimum threshold
    for target in ("sec_enforcement", "bankruptcy"):
        n_pos = stats["per_label"].get(target, {}).get("n_positive", 0)
        if n_pos < 20:
            logger.warning(
                "WARNING: %s has only %d positives (need >=20 for train split).",
                target, n_pos,
            )

    return stats


# ── CLI ───────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Source AAER and bankruptcy label datasets (ATS-172).",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data/raw"),
        help="Path to raw data directory (default: data/raw).",
    )
    parser.add_argument(
        "--start-date",
        default=DEFAULT_START,
        help=f"Start date for EFTS queries (default: {DEFAULT_START}).",
    )
    parser.add_argument(
        "--end-date",
        default=DEFAULT_END,
        help=f"End date for EFTS queries (default: {DEFAULT_END}).",
    )
    parser.add_argument(
        "--skip-enforcement",
        action="store_true",
        help="Skip SEC enforcement sourcing.",
    )
    parser.add_argument(
        "--skip-bankruptcy",
        action="store_true",
        help="Skip bankruptcy sourcing.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only run validation on existing CSVs (skip sourcing).",
    )
    args = parser.parse_args()

    raw_dir = args.raw_dir
    t0 = time.time()

    logger.info("ATS-172: Source external label datasets")
    logger.info("  raw_dir    = %s", raw_dir.resolve())
    logger.info("  date range = %s to %s", args.start_date, args.end_date)

    if not args.validate_only:
        session = _get_session()

        if not args.skip_enforcement:
            source_sec_enforcement(
                raw_dir, session,
                start_date=args.start_date,
                end_date=args.end_date,
            )

        if not args.skip_bankruptcy:
            source_bankruptcy(
                raw_dir, session,
                start_date=args.start_date,
                end_date=args.end_date,
            )

    # Validate
    validate_sourced_labels(raw_dir)

    logger.info("Done.  Total elapsed: %s", _elapsed(t0))


if __name__ == "__main__":
    main()

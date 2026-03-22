"""XBRL financial extraction pipeline from SEC EDGAR Company Facts API.

Downloads structured financial statement data for all companies in the
fin-jepa universe using the EDGAR Company Facts endpoint
(``https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json``).

Design decisions
----------------
* **Company Facts API**: returns all XBRL facts for a company in one JSON
  response — avoids raw XBRL/XML parsing entirely.
* **Taxonomy fallbacks**: each canonical feature maps to an ordered list of
  us-gaap concept names; the first available tag wins.
* **Annual 10-K only**: filters to ``form=10-K``, ``fp=FY`` to extract
  fiscal-year data.  Amendments are deduplicated by keeping the latest
  ``filed`` date.
* **Disk caching**: every API response is cached to
  ``cache_dir/companyfacts/{cik}.json`` so subsequent runs are instant.
* **Rate limiting**: reuses the HTTP/threading patterns from ``universe.py``
  to stay within EDGAR's ≤10 req/s policy.

Output schema (one row per company-year)
----------------------------------------
cik                  str       10-digit zero-padded CIK
ticker               str       primary exchange ticker (nullable)
fiscal_year          int       fiscal year (e.g. 2020)
period_end           date      fiscal period end date
filed_date           date      10-K filing date (latest amendment)
total_assets         float64   Balance sheet
total_liabilities    float64   Balance sheet
total_equity         float64   Balance sheet
current_assets       float64   Balance sheet
current_liabilities  float64   Balance sheet
retained_earnings    float64   Balance sheet
cash_equivalents     float64   Balance sheet
total_debt           float64   Balance sheet (computed)
total_revenue        float64   Income statement
cost_of_sales        float64   Income statement
operating_income     float64   Income statement
net_income           float64   Income statement
interest_expense     float64   Income statement
cash_from_operations float64   Cash flow statement
cash_from_investing  float64   Cash flow statement
cash_from_financing  float64   Cash flow statement
"""

from __future__ import annotations

import json
import logging
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fin_jepa.data.universe import _get_session, _fetch, EDGAR_DATA_BASE

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class XBRLConfig:
    """Configures the XBRL extraction pipeline.

    Attributes:
        start_year: First fiscal year to include.
        end_year: Last fiscal year to include.
        max_workers: Concurrent HTTP workers for batch fetching.
        rate_limit_per_sec: Max EDGAR requests per second (policy: ≤10).
        user_agent: EDGAR User-Agent header value.
    """

    start_year: int = 2012
    end_year: int = 2024
    max_workers: int = 4
    rate_limit_per_sec: float = 8.0
    user_agent: str = field(
        default_factory=lambda: __import__("os").environ.get(
            "EDGAR_USER_AGENT", "fin-jepa-research research@example.com"
        )
    )

    def __post_init__(self) -> None:
        if self.rate_limit_per_sec <= 0:
            raise ValueError(
                f"rate_limit_per_sec must be > 0, got {self.rate_limit_per_sec}"
            )


# ---------------------------------------------------------------------------
# Canonical feature schema: maps feature name → ordered XBRL concept tags
# ---------------------------------------------------------------------------

# Each entry is (canonical_name, [primary_tag, fallback_tag1, ...]).
# For "computed" features, the value is a tuple of concept groups to sum.

XBRL_FEATURE_SCHEMA: dict[str, list[str] | dict] = {
    # ── Balance Sheet (instantaneous) ───────────────────────────────────
    "total_assets": ["Assets"],
    "total_liabilities": ["Liabilities"],
    "total_equity": [
        "StockholdersEquity",
        "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
    ],
    "current_assets": ["AssetsCurrent"],
    "current_liabilities": ["LiabilitiesCurrent"],
    "retained_earnings": ["RetainedEarningsAccumulatedDeficit"],
    "cash_equivalents": [
        "CashAndCashEquivalentsAtCarryingValue",
        "Cash",
        "CashCashEquivalentsAndShortTermInvestments",
    ],
    # total_debt is computed as sum of long-term + short-term components
    "total_debt": {
        "type": "sum",
        "components": [
            ["LongTermDebt", "LongTermDebtNoncurrent"],
            ["ShortTermBorrowings", "DebtCurrent"],
        ],
    },

    # ── Income Statement (duration) ─────────────────────────────────────
    "total_revenue": [
        "Revenues",
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "SalesRevenueNet",
        "SalesRevenueGoodsNet",
    ],
    "cost_of_sales": [
        "CostOfGoodsSold",
        "CostOfRevenue",
        "CostOfGoodsAndServicesSold",
    ],
    "operating_income": ["OperatingIncomeLoss"],
    "net_income": [
        "NetIncomeLoss",
        "ProfitLoss",
        "NetIncomeLossAvailableToCommonStockholdersBasic",
    ],
    "interest_expense": [
        "InterestExpense",
        "InterestExpenseDebt",
    ],

    # ── Cash Flow Statement (duration) ──────────────────────────────────
    "cash_from_operations": [
        "NetCashProvidedByUsedInOperatingActivities",
        "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
    ],
    "cash_from_investing": [
        "NetCashProvidedByUsedInInvestingActivities",
        "NetCashProvidedByUsedInInvestingActivitiesContinuingOperations",
    ],
    "cash_from_financing": [
        "NetCashProvidedByUsedInFinancingActivities",
        "NetCashProvidedByUsedInFinancingActivitiesContinuingOperations",
    ],
}

# Ordered list of canonical feature names (excluding computed ones for simple lookup)
FEATURE_NAMES = list(XBRL_FEATURE_SCHEMA.keys())


# ---------------------------------------------------------------------------
# Single-company API fetch
# ---------------------------------------------------------------------------

def fetch_company_facts(
    cik: str,
    cache_dir: Path,
    session=None,
) -> dict[str, Any]:
    """Fetch the EDGAR Company Facts JSON for one company.

    Source: ``https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json``

    The response is cached to ``cache_dir/companyfacts/{cik}.json``.

    Args:
        cik: CIK as a string (will be zero-padded to 10 digits).
        cache_dir: Root cache directory.
        session: Optional requests.Session.

    Returns:
        Parsed JSON dict.  Empty dict on fetch failure (e.g. 404).
    """
    cik_padded = cik.zfill(10)
    cache_path = cache_dir / "companyfacts" / f"{cik_padded}.json"

    if cache_path.exists():
        with open(cache_path, encoding="utf-8") as fh:
            return json.load(fh)

    url = f"{EDGAR_DATA_BASE}/api/xbrl/companyfacts/CIK{cik_padded}.json"
    if session is None:
        session = _get_session()

    try:
        data = _fetch(url, session, as_json=True)
    except Exception as exc:
        logger.debug("Could not fetch company facts for CIK %s: %s", cik_padded, exc)
        return {}

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as fh:
        json.dump(data, fh)
    return data


# ---------------------------------------------------------------------------
# Taxonomy resolution helpers
# ---------------------------------------------------------------------------

def _resolve_simple_feature(
    us_gaap: dict,
    tags: list[str],
    fy: int,
    annual_entries: dict[int, dict],
) -> float | None:
    """Resolve a simple (non-computed) feature for a given fiscal year.

    Tries each tag in order; returns the first value found for the given
    fiscal year from 10-K/FY filings.

    Args:
        us_gaap: The ``facts.us-gaap`` dict from Company Facts JSON.
        tags: Ordered list of XBRL concept names to try.
        fy: Fiscal year to extract.
        annual_entries: Pre-filtered entries keyed by fiscal year (used by caller).

    Returns:
        The value as a float, or None if no tag yields data for this year.
    """
    for tag in tags:
        concept = us_gaap.get(tag)
        if concept is None:
            continue
        # USD units for monetary values
        units = concept.get("units", {})
        entries = units.get("USD", [])
        if not entries:
            continue
        # Filter to 10-K annual filings for this fiscal year
        matches = [
            e for e in entries
            if e.get("form") == "10-K"
            and e.get("fp") == "FY"
            and e.get("fy") == fy
        ]
        if not matches:
            continue
        # Deduplicate: keep the entry with the latest filed date
        best = max(matches, key=lambda e: e.get("filed", ""))
        return float(best["val"])
    return None


def _resolve_computed_feature(
    us_gaap: dict,
    spec: dict,
    fy: int,
) -> float | None:
    """Resolve a computed feature (e.g. total_debt = sum of components).

    Each component is itself a list of fallback tags — the first available
    tag is used per component.  The result is the sum of all resolved
    components; if all components are None, returns None.
    """
    if spec.get("type") != "sum":
        return None

    total = 0.0
    any_found = False

    for component_tags in spec["components"]:
        val = _resolve_simple_feature(us_gaap, component_tags, fy, {})
        if val is not None:
            total += val
            any_found = True

    return total if any_found else None


# ---------------------------------------------------------------------------
# Per-company extraction
# ---------------------------------------------------------------------------

def extract_annual_facts(
    facts_json: dict[str, Any],
    cik: str,
    config: XBRLConfig | None = None,
) -> pd.DataFrame:
    """Extract canonical annual features from a Company Facts JSON response.

    Filters to 10-K / FY filings, resolves taxonomy fallbacks, deduplicates
    by fiscal year (keeping latest filing date), and returns a tidy DataFrame.

    Args:
        facts_json: Parsed JSON from :func:`fetch_company_facts`.
        cik: CIK string for the company.
        config: Optional config to constrain the year range.

    Returns:
        DataFrame with columns: cik, fiscal_year, period_end, filed_date,
        + all feature columns from XBRL_FEATURE_SCHEMA.
        Empty DataFrame if no us-gaap data is found.
    """
    if not facts_json:
        return pd.DataFrame()

    us_gaap = facts_json.get("facts", {}).get("us-gaap", {})
    if not us_gaap:
        return pd.DataFrame()

    start_year = config.start_year if config else 2012
    end_year = config.end_year if config else 2024

    # Discover all fiscal years present in the data by scanning a common concept
    all_fiscal_years: set[int] = set()
    for concept_name, concept_data in us_gaap.items():
        units = concept_data.get("units", {})
        for unit_entries in units.values():
            for entry in unit_entries:
                if entry.get("form") == "10-K" and entry.get("fp") == "FY":
                    fy = entry.get("fy")
                    if isinstance(fy, int) and start_year <= fy <= end_year:
                        all_fiscal_years.add(fy)

    if not all_fiscal_years:
        return pd.DataFrame()

    # Extract features for each fiscal year
    records: list[dict[str, Any]] = []
    cik_padded = cik.zfill(10)

    for fy in sorted(all_fiscal_years):
        row: dict[str, Any] = {
            "cik": cik_padded,
            "fiscal_year": fy,
        }

        # Extract period_end and filed_date from a common concept
        period_end = None
        filed_date = None
        for probe_tag in ["Assets", "Revenues", "NetIncomeLoss", "Liabilities"]:
            concept = us_gaap.get(probe_tag)
            if concept is None:
                continue
            entries = concept.get("units", {}).get("USD", [])
            matches = [
                e for e in entries
                if e.get("form") == "10-K"
                and e.get("fp") == "FY"
                and e.get("fy") == fy
            ]
            if matches:
                best = max(matches, key=lambda e: e.get("filed", ""))
                period_end = best.get("end")
                filed_date = best.get("filed")
                break

        row["period_end"] = period_end
        row["filed_date"] = filed_date

        # Resolve each feature
        for feature_name, spec in XBRL_FEATURE_SCHEMA.items():
            if isinstance(spec, list):
                row[feature_name] = _resolve_simple_feature(us_gaap, spec, fy, {})
            elif isinstance(spec, dict):
                row[feature_name] = _resolve_computed_feature(us_gaap, spec, fy)
            else:
                row[feature_name] = None

        records.append(row)

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    # Convert date columns
    df["period_end"] = pd.to_datetime(df["period_end"], errors="coerce").dt.date
    df["filed_date"] = pd.to_datetime(df["filed_date"], errors="coerce").dt.date
    return df


# ---------------------------------------------------------------------------
# Batch fetcher with rate limiting
# ---------------------------------------------------------------------------

def fetch_all_company_facts(
    ciks: list[str],
    cache_dir: Path,
    config: XBRLConfig,
    ticker_map: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Fetch and extract XBRL features for a list of companies.

    Uses a thread pool with rate limiting to stay within EDGAR's
    fair-use policy (≤10 requests/second).

    Args:
        ciks: List of CIK strings to fetch.
        cache_dir: Root cache directory (responses cached per-CIK).
        config: XBRL pipeline configuration.
        ticker_map: Optional dict mapping CIK → ticker symbol.

    Returns:
        DataFrame with all company-year observations concatenated.
    """
    min_sleep = 1.0 / config.rate_limit_per_sec
    lock = threading.Lock()
    last_request_time: list[float] = [0.0]
    _thread_local = threading.local()

    def _get_thread_session():
        session = getattr(_thread_local, "session", None)
        if session is None:
            session = _get_session(user_agent=config.user_agent)
            _thread_local.session = session
        return session

    def _throttled_fetch(cik: str) -> pd.DataFrame:
        session = _get_thread_session()
        with lock:
            now = time.monotonic()
            elapsed = now - last_request_time[0]
            sleep_time = max(0.0, min_sleep - elapsed)
            last_request_time[0] = time.monotonic() + sleep_time
        if sleep_time > 0:
            time.sleep(sleep_time)

        facts = fetch_company_facts(cik, cache_dir, session)
        return extract_annual_facts(facts, cik, config)

    frames: list[pd.DataFrame] = []
    total = len(ciks)
    logger.info("Fetching XBRL company facts for %d companies …", total)

    with ThreadPoolExecutor(max_workers=config.max_workers) as pool:
        futures = {pool.submit(_throttled_fetch, cik): cik for cik in ciks}
        for i, future in enumerate(as_completed(futures), 1):
            cik = futures[future]
            try:
                df = future.result()
                if not df.empty:
                    frames.append(df)
            except Exception as exc:
                logger.debug("XBRL extraction failed for CIK %s: %s", cik, exc)
            if i % 500 == 0 or i == total:
                logger.info("  %d / %d companies processed", i, total)

    if not frames:
        raise RuntimeError("No XBRL data could be extracted.")

    # Filter out empty DataFrames before concat
    frames = [f for f in frames if not f.empty]
    if not frames:
        raise RuntimeError("No XBRL data could be extracted.")

    # Ensure all frames have the same columns to avoid FutureWarning
    # about all-NA columns during concat
    all_cols = ["cik", "fiscal_year", "period_end", "filed_date"] + FEATURE_NAMES
    aligned: list[pd.DataFrame] = []
    for f in frames:
        missing = {col: np.nan for col in all_cols if col not in f.columns}
        aligned.append(f.assign(**missing) if missing else f)

    combined = pd.concat(aligned, ignore_index=True)

    # Add ticker column from universe mapping
    if ticker_map:
        combined["ticker"] = combined["cik"].map(ticker_map)
    else:
        combined["ticker"] = None

    return combined.sort_values(["cik", "fiscal_year"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Top-level pipeline orchestrator
# ---------------------------------------------------------------------------

def build_xbrl_dataset(
    raw_dir: Path,
    universe_path: Path,
    output_path: Path | None = None,
    config: XBRLConfig | None = None,
    cik_subset: list[str] | None = None,
) -> pd.DataFrame:
    """Build the XBRL features dataset for the full company universe.

    Steps:
    1. Load company universe to get CIK list and ticker mapping.
    2. Fetch Company Facts API for each CIK (cached, rate-limited).
    3. Extract annual 10-K features with taxonomy fallback resolution.
    4. Validate and write to Parquet with provenance metadata.

    Args:
        raw_dir: Directory for raw data and caching.
        universe_path: Path to the company_universe.parquet file.
        output_path: Where to write the output Parquet.  Defaults to
            ``raw_dir / "xbrl_features.parquet"``.
        config: Pipeline configuration.  Uses defaults if not provided.
        cik_subset: Optional list of CIKs to process (for testing/debugging).
            If None, processes all CIKs from the universe.

    Returns:
        The assembled DataFrame.
    """
    if config is None:
        config = XBRLConfig()
    if output_path is None:
        output_path = raw_dir / "xbrl_features.parquet"

    # Load universe
    from fin_jepa.data.universe import load_company_universe

    universe_df, _ = load_company_universe(universe_path)
    logger.info("Loaded universe with %d companies", len(universe_df))

    # Build CIK list and ticker mapping
    if cik_subset is not None:
        ciks = [c.zfill(10) for c in cik_subset]
    else:
        ciks = universe_df["cik"].tolist()

    ticker_map = dict(zip(universe_df["cik"], universe_df["ticker"]))

    # Set up cache directory
    cache_dir = raw_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Fetch and extract
    df = fetch_all_company_facts(ciks, cache_dir, config, ticker_map)
    logger.info(
        "Extracted %d company-year observations for %d companies",
        len(df), df["cik"].nunique(),
    )

    # Validate
    report = validate_xbrl_dataset(df)
    logger.info("Validation report: %s", json.dumps(report, indent=2))

    # Reorder columns: identifiers first, then features
    id_cols = ["cik", "ticker", "fiscal_year", "period_end", "filed_date"]
    feature_cols = [c for c in FEATURE_NAMES if c in df.columns]
    df = df[id_cols + feature_cols]

    # Write with provenance metadata
    build_date = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    provenance = {
        "build_date": build_date,
        "start_year": config.start_year,
        "end_year": config.end_year,
        "n_companies": int(df["cik"].nunique()),
        "n_observations": len(df),
        "feature_coverage": report.get("feature_coverage", {}),
        "data_source": "SEC EDGAR Company Facts API",
    }

    try:
        import pyarrow as pa
        import pyarrow.parquet as pq

        table = pa.Table.from_pandas(df)
        meta = table.schema.metadata or {}
        meta[b"fin_jepa_provenance"] = json.dumps(provenance).encode()
        table = table.replace_schema_metadata(meta)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, output_path)
    except ImportError:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(output_path, index=False)
        prov_path = output_path.with_suffix(".provenance.json")
        with open(prov_path, "w", encoding="utf-8") as fh:
            json.dump(provenance, fh, indent=2)

    logger.info(
        "XBRL dataset written to %s  (%d observations, %d companies)",
        output_path, len(df), df["cik"].nunique(),
    )
    return df


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def load_xbrl_features(path: Path) -> tuple[pd.DataFrame, dict]:
    """Load the XBRL features Parquet file with provenance metadata.

    Args:
        path: Path to ``xbrl_features.parquet``.

    Returns:
        Tuple of ``(features_df, provenance_dict)``.
    """
    path = Path(path)
    provenance: dict = {}

    try:
        import pyarrow.parquet as pq

        pf = pq.ParquetFile(path)
        raw_meta = pf.schema_arrow.metadata or {}
        if b"fin_jepa_provenance" in raw_meta:
            provenance = json.loads(raw_meta[b"fin_jepa_provenance"].decode())
        df = pf.read().to_pandas()
    except Exception:
        df = pd.read_parquet(path)
        prov_path = path.with_suffix(".provenance.json")
        if prov_path.exists():
            with open(prov_path, encoding="utf-8") as fh:
                provenance = json.load(fh)

    return df, provenance


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_xbrl_dataset(df: pd.DataFrame) -> dict[str, Any]:
    """Compute coverage and quality statistics for the XBRL dataset.

    Args:
        df: DataFrame from :func:`build_xbrl_dataset` or
            :func:`fetch_all_company_facts`.

    Returns:
        Dict with validation statistics:
        - n_observations: total rows
        - n_companies: unique CIKs
        - year_range: [min_year, max_year]
        - feature_coverage: dict of feature_name → pct non-null
        - duplicate_count: number of duplicate (cik, fiscal_year) pairs
    """
    feature_cols = [c for c in FEATURE_NAMES if c in df.columns]

    coverage = {}
    for col in feature_cols:
        n_non_null = df[col].notna().sum()
        coverage[col] = round(n_non_null / len(df) * 100, 1) if len(df) > 0 else 0.0

    fy_col = "fiscal_year" if "fiscal_year" in df.columns else None
    year_range = (
        [int(df[fy_col].min()), int(df[fy_col].max())]
        if fy_col and len(df) > 0
        else []
    )

    dup_count = 0
    if "cik" in df.columns and fy_col:
        dup_count = int(df.duplicated(subset=["cik", fy_col]).sum())

    return {
        "n_observations": len(df),
        "n_companies": int(df["cik"].nunique()) if "cik" in df.columns else 0,
        "year_range": year_range,
        "feature_coverage": coverage,
        "duplicate_count": dup_count,
    }

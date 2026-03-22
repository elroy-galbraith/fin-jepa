"""Company universe construction for fin-jepa Study 0.

Builds a survivorship-bias-free universe of all SEC 10-K filers for
2012–2024 using EDGAR bulk index files and the submissions API.

Key design decisions
--------------------
* **No survivorship bias**: includes companies that delisted, filed for
  bankruptcy, were acquired, or otherwise stopped filing before 2024.
* **EDGAR-only core**: works without any paid data source.
* **Incremental / cached**: every HTTP response is written to disk so
  subsequent runs are fast and the pipeline can be resumed after failure.
* **Compustat-ready**: schema includes columns that align with Compustat
  GVKEY cross-references when that data is available (see compustat.py).

Data sources
------------
1. ``https://www.sec.gov/Archives/edgar/full-index/{year}/QTR{q}/form.idx``
   — quarterly EDGAR index of all filings; filtered to 10-K / 10-K/A.
2. ``https://www.sec.gov/files/company_tickers_exchange.json``
   — CIK → ticker / exchange mapping maintained by EDGAR.
3. ``https://data.sec.gov/submissions/CIK{cik:010d}.json``
   — per-company metadata: SIC, state, fiscal year end, full filing history.

EDGAR fair-use policy requires a ``User-Agent`` header and ≤10 req/s.
Set the env var ``EDGAR_USER_AGENT`` to ``"FirstName LastName email@example.com"``
or accept the default placeholder (fine for research use).

Output schema (one row per company)
-------------------------------------
cik                  str     10-digit zero-padded CIK
entity_name          str     EDGAR registrant name (latest known)
ticker               str     primary exchange ticker (nullable)
exchange             str     primary listing exchange (nullable)
sic_code             str     4-digit SIC code (nullable if not in EDGAR)
sic_description      str     EDGAR SIC description (nullable)
sector               str     Fama-French 12-industry sector
state_of_inc         str     2-letter state of incorporation (nullable)
fiscal_year_end      str     MMDD fiscal year end (nullable)
first_10k_date       date    earliest 10-K filing date in the sample window
last_10k_date        date    most recent 10-K filing date in the sample window
n_10k_filings        int     number of 10-K filings (excl. 10-K/A amendments)
n_xbrl_filings       int     10-K filings estimated to include XBRL (≥2012)
xbrl_coverage_pct    float   n_xbrl_filings / n_10k_filings
filing_years         list    sorted list of calendar years with a 10-K filing
gap_years            list    years in [first_10k_year, last_10k_year] with no 10-K filing
is_current_filer     bool    filed a 10-K in the final two years of the window
"""

from __future__ import annotations

import io
import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from fin_jepa.data.sector_map import sic_to_sector

# ---------------------------------------------------------------------------
# Optional dependency: requests (W8)
# ---------------------------------------------------------------------------
try:
    import requests as _requests_module
except ImportError:
    _requests_module = None

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EDGAR_BASE = "https://www.sec.gov"
EDGAR_DATA_BASE = "https://data.sec.gov"
XBRL_MANDATORY_DATE = date(2012, 1, 1)  # XBRL required for all filers by 2011

_DEFAULT_USER_AGENT = (
    "fin-jepa-research research@example.com"
)


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class UniverseConfig:
    """Configures the company-universe build pipeline.

    Attributes:
        start_year: First calendar year of 10-K filings to include.
        end_year: Last calendar year of 10-K filings to include.
        form_types: EDGAR form types to include. 10-K/A (amendments) are
            counted separately for the XBRL audit but not in n_10k_filings.
        min_filings: Minimum number of distinct 10-K filings required for
            a company to be included in the universe.
        fetch_submissions: Whether to call the EDGAR submissions API per-CIK
            to enrich company metadata (SIC, state, fiscal year end). Adds
            ~20 minutes on first run for a full universe but is cached.
        max_workers: Concurrent HTTP workers for submissions fetching.
        rate_limit_per_sec: Max EDGAR requests per second (policy: ≤10).
        user_agent: Value for the HTTP ``User-Agent`` header as required by
            EDGAR fair-use policy.  Falls back to ``EDGAR_USER_AGENT`` env
            var, then the built-in placeholder.
    """
    start_year: int = 2012
    end_year: int = 2024
    form_types: list[str] = field(default_factory=lambda: ["10-K", "10-K/A"])
    min_filings: int = 1
    fetch_submissions: bool = True
    max_workers: int = 4
    rate_limit_per_sec: float = 8.0
    user_agent: str = field(
        default_factory=lambda: os.environ.get("EDGAR_USER_AGENT", _DEFAULT_USER_AGENT)
    )

    def __post_init__(self) -> None:
        if self.rate_limit_per_sec <= 0:
            raise ValueError(
                f"rate_limit_per_sec must be > 0, got {self.rate_limit_per_sec}"
            )


# ---------------------------------------------------------------------------
# Internal HTTP helpers
# ---------------------------------------------------------------------------

def _get_session(user_agent: str | None = None):
    """Create a requests.Session with EDGAR-compliant headers.

    Args:
        user_agent: Explicit ``User-Agent`` string.  Falls back to the
            ``EDGAR_USER_AGENT`` env var, then the built-in placeholder.
            Pass ``config.user_agent`` so programmatic overrides via
            :class:`UniverseConfig` take effect (addresses the gap where
            the config value was silently ignored).

    .. note::
        Do NOT hard-code a ``Host`` header here.  The HTTP stack derives
        the correct ``Host`` value from each request URL, allowing the same
        session to target both ``www.sec.gov`` and ``data.sec.gov`` without
        silently sending the wrong value (C1).
    """
    if _requests_module is None:
        raise ImportError(
            "The 'requests' package is required. "
            "Install with: pip install requests"
        )
    session = _requests_module.Session()
    ua = user_agent or os.environ.get("EDGAR_USER_AGENT", _DEFAULT_USER_AGENT)
    session.headers.update({
        "User-Agent": ua,
        "Accept-Encoding": "gzip, deflate",
    })
    return session


def _fetch(url: str, session, *, as_json: bool = False, retries: int = 3, backoff: float = 2.0):
    """Fetch a URL with exponential-backoff retry on transient errors (W1).

    Args:
        url: URL to fetch.
        session: requests.Session to use.
        as_json: If True, parse and return the response as JSON; otherwise
            return the response text.
        retries: Number of attempts (clamped to ≥ 1; passing 0 is safe).
        backoff: Multiplier for exponential back-off between retries.

    Returns:
        Parsed JSON dict (``as_json=True``) or raw text string.
    """
    retries = max(retries, 1)
    for attempt in range(retries):
        try:
            resp = session.get(url, timeout=30)
            resp.raise_for_status()
            return resp.json() if as_json else resp.text
        except Exception as exc:
            # Fail fast on deterministic client errors (4xx) except 429
            # (Too Many Requests), which is transient by nature.
            status = getattr(getattr(exc, "response", None), "status_code", None)
            if isinstance(status, int) and 400 <= status < 500 and status != 429:
                raise
            if attempt == retries - 1:
                raise
            wait = backoff ** attempt
            logger.warning("Retry %d/%d for %s after %.1fs: %s", attempt + 1, retries, url, wait, exc)
            time.sleep(wait)
    raise RuntimeError(f"Failed to fetch {url}")  # unreachable


# ---------------------------------------------------------------------------
# EDGAR form.idx parsing
# ---------------------------------------------------------------------------

def _parse_form_idx(content: str) -> pd.DataFrame:
    """Parse a raw EDGAR ``form.idx`` fixed-width text file.

    The file has a two-line header followed by a separator line and then
    data.  Column boundaries are inferred from the header so the parser
    is robust to minor formatting variation across quarters.

    Args:
        content: Raw text content of a ``form.idx`` file.

    Returns:
        DataFrame with columns: form_type, company_name, cik, date_filed,
        filename.
    """
    lines = content.splitlines()
    if len(lines) < 3:
        return pd.DataFrame(columns=["form_type", "company_name", "cik", "date_filed", "filename"])

    # Find the header line (contains "Form Type")
    header_idx = 0
    for i, line in enumerate(lines[:5]):
        if "Form Type" in line and "CIK" in line:
            header_idx = i
            break

    header = lines[header_idx]
    # Determine column start positions from header text
    col_starts = {
        "form_type": 0,
        "company_name": header.index("Company Name"),
        "cik": header.index("CIK"),
        "date_filed": header.index("Date Filed"),
        "filename": header.index("Filename"),
    }
    # Build colspecs for pd.read_fwf
    keys = list(col_starts.keys())
    positions = list(col_starts.values())
    colspecs = [
        (positions[i], positions[i + 1])
        for i in range(len(positions) - 1)
    ] + [(positions[-1], None)]

    data_lines = [
        line for line in lines[header_idx + 2:]  # skip header + separator
        if line.strip() and not line.startswith("---")
    ]
    if not data_lines:
        return pd.DataFrame(columns=keys)

    raw = "\n".join(data_lines)
    df = pd.read_fwf(
        io.StringIO(raw),
        colspecs=colspecs,
        names=keys,
        dtype=str,
    )
    df = df.dropna(subset=["cik", "date_filed"])
    df["cik"] = df["cik"].str.strip().str.zfill(10)
    df["form_type"] = df["form_type"].str.strip()
    df["company_name"] = df["company_name"].str.strip()
    df["date_filed"] = pd.to_datetime(df["date_filed"].str.strip(), format="%Y-%m-%d", errors="coerce")
    df = df.dropna(subset=["date_filed"])
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Quarterly index fetching
# ---------------------------------------------------------------------------

def fetch_quarterly_index(
    year: int,
    quarter: int,
    cache_dir: Path,
    config: UniverseConfig,
    session=None,
) -> pd.DataFrame:
    """Download and parse one quarterly EDGAR full-index form.idx file.

    Results are cached to ``cache_dir/edgar_index/{year}_Q{quarter}.parquet``
    so repeated calls are free.

    Args:
        year: Calendar year (e.g. 2012).
        quarter: Quarter number 1–4.
        cache_dir: Directory for caching HTTP responses and parsed data.
        config: Universe configuration (user agent, etc.).
        session: Optional requests.Session; one is created if not provided.

    Returns:
        DataFrame with columns: form_type, company_name, cik, date_filed,
        filename (filtered to config.form_types).
    """
    cache_path = cache_dir / "edgar_index" / f"{year}_Q{quarter}.parquet"
    if cache_path.exists():
        return pd.read_parquet(cache_path)

    url = f"{EDGAR_BASE}/Archives/edgar/full-index/{year}/QTR{quarter}/form.idx"
    logger.debug("Fetching %s", url)

    if session is None:
        session = _get_session()

    content = _fetch(url, session)
    df = _parse_form_idx(content)

    # Filter to requested form types
    df = df[df["form_type"].isin(config.form_types)].copy()

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_path, index=False)
    return df


# ---------------------------------------------------------------------------
# Full filing index builder
# ---------------------------------------------------------------------------

def build_filing_index(
    cache_dir: Path,
    config: UniverseConfig,
    session=None,
) -> pd.DataFrame:
    """Aggregate quarterly EDGAR indexes across the full sample window.

    Downloads one ``form.idx`` per quarter for ``config.start_year``–
    ``config.end_year`` (inclusive) and concatenates them into a single
    DataFrame representing every 10-K (and 10-K/A) filing in the period.

    Quarters that fail to download are skipped with a warning.

    Args:
        cache_dir: Root cache directory.
        config: Universe configuration.
        session: Optional requests.Session.

    Returns:
        DataFrame with columns: form_type, company_name, cik, date_filed,
        filename, plus derived ``filing_year`` (int).
    """
    if session is None:
        session = _get_session()

    quarters = [
        (y, q)
        for y in range(config.start_year, config.end_year + 1)
        for q in range(1, 5)
    ]
    frames: list[pd.DataFrame] = []
    min_sleep = 1.0 / config.rate_limit_per_sec

    for year, quarter in quarters:
        try:
            df = fetch_quarterly_index(year, quarter, cache_dir, config, session)
            frames.append(df)
            time.sleep(min_sleep)
        except Exception as exc:
            logger.warning("Skipping %d Q%d: %s", year, quarter, exc)

    if not frames:
        raise RuntimeError("No quarterly indexes could be downloaded.")

    combined = pd.concat(frames, ignore_index=True)
    combined["filing_year"] = combined["date_filed"].dt.year
    return combined.sort_values("date_filed").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Company tickers from EDGAR
# ---------------------------------------------------------------------------

def fetch_company_tickers(cache_dir: Path, session=None) -> pd.DataFrame:
    """Fetch the EDGAR company-tickers-exchange mapping.

    Source: ``https://www.sec.gov/files/company_tickers_exchange.json``

    Args:
        cache_dir: Root cache directory.
        session: Optional requests.Session.

    Returns:
        DataFrame with columns: cik (zero-padded 10-digit str), ticker,
        exchange, name.
    """
    cache_path = cache_dir / "company_tickers_exchange.parquet"
    if cache_path.exists():
        return pd.read_parquet(cache_path)

    url = f"{EDGAR_BASE}/files/company_tickers_exchange.json"
    logger.debug("Fetching company tickers from %s", url)

    if session is None:
        session = _get_session()

    data = _fetch(url, session, as_json=True)
    fields = data.get("fields", ["cik", "name", "ticker", "exchange"])
    rows = data.get("data", [])

    df = pd.DataFrame(rows, columns=fields)
    df["cik"] = df["cik"].astype(str).str.zfill(10)
    # Prefer primary-exchange listings over OTC / foreign venues (W5).
    # For dual-listed companies the first Nasdaq/NYSE/NYSE MKT row wins;
    # companies absent from those exchanges fall back to first occurrence.
    _PRIMARY_EXCHANGES = {"Nasdaq", "NYSE", "NYSE MKT"}
    primary_mask = df["exchange"].isin(_PRIMARY_EXCHANGES)
    df_primary = df[primary_mask].drop_duplicates(subset="cik", keep="first")
    df_rest = df[~df["cik"].isin(df_primary["cik"])].drop_duplicates(subset="cik", keep="first")
    df = pd.concat([df_primary, df_rest], ignore_index=True)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_path, index=False)
    return df


# ---------------------------------------------------------------------------
# Per-company EDGAR submissions metadata
# ---------------------------------------------------------------------------

def fetch_company_submissions(
    cik: str,
    cache_dir: Path,
    session=None,
) -> dict[str, Any]:
    """Fetch the EDGAR submissions JSON for one company.

    Source: ``https://data.sec.gov/submissions/CIK{cik:010d}.json``

    The response is cached to ``cache_dir/submissions/{cik}.json``.

    Args:
        cik: CIK as a string (will be zero-padded to 10 digits).
        cache_dir: Root cache directory.
        session: Optional requests.Session.

    Returns:
        Parsed JSON dict.  Empty dict on fetch failure.
    """
    cik_padded = cik.zfill(10)
    cache_path = cache_dir / "submissions" / f"{cik_padded}.json"

    if cache_path.exists():
        with open(cache_path, encoding="utf-8") as fh:
            return json.load(fh)

    url = f"{EDGAR_DATA_BASE}/submissions/CIK{cik_padded}.json"
    if session is None:
        session = _get_session()

    try:
        data = _fetch(url, session, as_json=True)
    except Exception as exc:
        logger.debug("Could not fetch submissions for CIK %s: %s", cik_padded, exc)
        return {}

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as fh:
        json.dump(data, fh)
    return data


def _extract_submissions_metadata(sub: dict) -> dict[str, str | None]:
    """Extract flat metadata fields from a submissions JSON dict."""
    return {
        "sic_code": sub.get("sic") or None,
        "sic_description": sub.get("sicDescription") or None,
        "state_of_inc": sub.get("stateOfIncorporation") or None,
        "fiscal_year_end": sub.get("fiscalYearEnd") or None,
        "entity_type": sub.get("entityType") or None,
    }


# ---------------------------------------------------------------------------
# Batch submissions fetch with rate-limiting
# ---------------------------------------------------------------------------

def fetch_all_submissions(
    ciks: list[str],
    cache_dir: Path,
    config: UniverseConfig,
) -> dict[str, dict[str, str | None]]:
    """Fetch EDGAR submissions metadata for a list of CIKs.

    Uses a thread pool with a rate limiter to stay within EDGAR's
    fair-use policy (≤10 requests/second).

    Args:
        ciks: List of CIK strings to fetch.
        cache_dir: Root cache directory (responses cached per-CIK).
        config: Universe configuration (max_workers, rate_limit_per_sec).

    Returns:
        Dict mapping zero-padded CIK → metadata dict.
    """
    import threading
    min_sleep = 1.0 / config.rate_limit_per_sec
    lock = threading.Lock()
    last_request_time: list[float] = [0.0]
    # Thread-local storage so each worker thread reuses one Session,
    # keeping TCP connection pools alive across CIK fetches (avoids
    # creating a new Session — and new connections — per CIK).
    _thread_local = threading.local()

    def _get_thread_session():
        """Lazily create one requests.Session per worker thread."""
        session = getattr(_thread_local, "session", None)
        if session is None:
            session = _get_session(user_agent=config.user_agent)
            _thread_local.session = session
        return session

    def _throttled_fetch(cik: str) -> tuple[str, dict]:
        session = _get_thread_session()
        # Enforce rate limit: compute the required sleep duration while
        # holding the lock, then release before sleeping so other threads
        # are not serialized waiting on the sleep itself (W2).
        with lock:
            now = time.monotonic()
            elapsed = now - last_request_time[0]
            sleep_time = max(0.0, min_sleep - elapsed)
            last_request_time[0] = time.monotonic() + sleep_time
        if sleep_time > 0:
            time.sleep(sleep_time)
        sub = fetch_company_submissions(cik, cache_dir, session)
        return cik.zfill(10), _extract_submissions_metadata(sub)

    results: dict[str, dict] = {}
    total = len(ciks)
    logger.info("Fetching EDGAR submissions for %d companies …", total)

    with ThreadPoolExecutor(max_workers=config.max_workers) as pool:
        futures = {pool.submit(_throttled_fetch, cik): cik for cik in ciks}
        for i, future in enumerate(as_completed(futures), 1):
            try:
                cik_padded, meta = future.result()
                results[cik_padded] = meta
            except Exception as exc:
                cik = futures[future]
                logger.debug("Submissions fetch failed for CIK %s: %s", cik, exc)
                results[cik.zfill(10)] = _extract_submissions_metadata({})
            if i % 500 == 0 or i == total:
                logger.info("  %d / %d submissions fetched", i, total)

    return results


# ---------------------------------------------------------------------------
# XBRL coverage audit
# ---------------------------------------------------------------------------

def audit_xbrl_coverage(
    filings_df: pd.DataFrame,
    end_year: int | None = None,
) -> pd.DataFrame:
    """Estimate XBRL coverage for each company in the filing index.

    **Estimation approach**: all 10-K filings dated on or after
    2012-01-01 are assumed to include XBRL, since EDGAR mandated inline
    XBRL for all filers by fiscal years ending on or after 28 June 2011.
    Earlier filings (if any appear in the index) are counted as
    non-XBRL.

    A per-company ``xbrl_gap_flag`` is set True when the ratio of XBRL
    filings to total 10-K filings (excl. 10-K/A) falls below 95 %, which
    would indicate data quality issues worth investigating.

    Args:
        filings_df: Output of :func:`build_filing_index` with at least
            columns ``cik``, ``form_type``, ``date_filed``.
        end_year: Last year of the universe window (e.g. 2024).  Used to
            determine ``is_current_filer``: True iff the company filed a
            10-K in ``{end_year, end_year - 1}``.  When omitted, defaults
            to the maximum ``filing_year`` present in *filings_df*; note
            that in that case ``is_current_filer`` will be ``True`` for
            every company that filed at all (since ``last_10k_year`` is
            always in ``filing_years``), so callers should always supply
            this value.

    Returns:
        DataFrame indexed by cik with columns:
        ``n_10k_filings``, ``n_xbrl_filings``, ``xbrl_coverage_pct``,
        ``xbrl_gap_flag``, ``first_10k_date``, ``last_10k_date``,
        ``filing_years``, ``gap_years``, ``is_current_filer``.
    """
    # Separate clean 10-Ks from amendments
    filings_10k = filings_df[filings_df["form_type"] == "10-K"].copy()
    xbrl_cutoff = pd.Timestamp(XBRL_MANDATORY_DATE)

    if filings_10k.empty:
        empty_cols = [
            "cik", "n_10k_filings", "n_xbrl_filings", "xbrl_coverage_pct",
            "xbrl_gap_flag", "first_10k_date", "last_10k_date",
            "filing_years", "gap_years", "is_current_filer",
        ]
        return pd.DataFrame(columns=empty_cols).set_index("cik")

    filings_10k = filings_10k.copy()
    filings_10k["_is_xbrl"] = filings_10k["date_filed"] >= xbrl_cutoff

    # Vectorized groupby aggregation replaces the pure-Python loop (W3)
    agg = filings_10k.groupby("cik").agg(
        n_10k_filings=("date_filed", "count"),
        n_xbrl_filings=("_is_xbrl", "sum"),
        first_10k_date=("date_filed", "min"),
        last_10k_date=("date_filed", "max"),
        filing_years=(
            "filing_year",
            lambda s: sorted(s.dropna().astype(int).unique().tolist()),
        ),
    )
    agg["n_xbrl_filings"] = agg["n_xbrl_filings"].astype(int)
    agg["xbrl_coverage_pct"] = (
        agg["n_xbrl_filings"] / agg["n_10k_filings"].clip(lower=1)
    ).round(4)
    agg["xbrl_gap_flag"] = agg["xbrl_coverage_pct"] < 0.95
    agg["first_10k_date"] = agg["first_10k_date"].dt.date
    agg["last_10k_date"] = agg["last_10k_date"].dt.date

    agg["gap_years"] = agg["filing_years"].apply(
        lambda years: (
            sorted(set(range(years[0], years[-1] + 1)) - set(years))
            if years else []
        )
    )
    # is_current_filer: True iff the company filed in the final two years
    # of the universe window.  Use the supplied end_year; if not given,
    # fall back to the maximum year in the dataset.  Note: using
    # last_10k_date.year here would make is_current_filer always True
    # (every company's last year is by definition in filing_years).
    _end_year: int = (
        end_year
        if end_year is not None
        else int(filings_10k["filing_year"].max())
    )
    final_two = {_end_year, _end_year - 1}
    agg["is_current_filer"] = agg["filing_years"].apply(
        lambda years: bool(set(years) & final_two)
    )
    return agg


# ---------------------------------------------------------------------------
# Main universe builder
# ---------------------------------------------------------------------------

def build_company_universe(
    raw_dir: Path | str,
    start_year: int = 2012,
    end_year: int = 2024,
    output_path: Path | str | None = None,
    fetch_submissions: bool = True,
    max_workers: int = 4,
    min_filings: int = 1,
    rate_limit_per_sec: float = 8.0,
) -> pd.DataFrame:
    """Build the full SEC filer universe and return it as a DataFrame.

    Orchestrates:
    1. Download & parse quarterly EDGAR form.idx files (2012–2024).
    2. Deduplicate to one row per company (CIK).
    3. Fetch per-company metadata from EDGAR submissions API.
    4. Join ticker/exchange from EDGAR company-tickers JSON.
    5. Apply Fama-French 12-industry sector classification.
    6. Run XBRL coverage audit.
    7. Write output to ``output_path`` as a Parquet file with JSON
       provenance metadata embedded in the file-level schema.

    The pipeline is idempotent: every HTTP response is cached, so
    interrupted runs can be resumed without re-downloading data.

    Args:
        raw_dir: Root data directory.  A ``cache/universe/`` subdirectory
            is created here for HTTP caches.
        start_year: First calendar year (default 2012).
        end_year: Last calendar year (default 2024).
        output_path: Where to write the output parquet.  Defaults to
            ``{raw_dir}/company_universe.parquet``.
        fetch_submissions: Enrich with SIC / state / fiscal-year-end from
            the EDGAR submissions API.  Takes ~20 min on first run for a
            full universe.  Disable for quick testing.
        max_workers: Parallel HTTP workers for submissions fetch.
        min_filings: Minimum number of 10-K filings for inclusion.
        rate_limit_per_sec: EDGAR request rate limit (policy: ≤10).

    Returns:
        DataFrame with one row per company and the columns described in
        the module docstring.
    """
    raw_dir = Path(raw_dir)
    cache_dir = raw_dir / "cache" / "universe"
    cache_dir.mkdir(parents=True, exist_ok=True)

    if output_path is None:
        output_path = raw_dir / "company_universe.parquet"
    output_path = Path(output_path)

    config = UniverseConfig(
        start_year=start_year,
        end_year=end_year,
        fetch_submissions=fetch_submissions,
        max_workers=max_workers,
        min_filings=min_filings,
        rate_limit_per_sec=rate_limit_per_sec,
    )

    # ── Step 1: Filing index ─────────────────────────────────────────────────
    logger.info("Building filing index for %d–%d …", start_year, end_year)
    session = _get_session(user_agent=config.user_agent)
    filings_df = build_filing_index(cache_dir, config, session)
    n_total_filings = len(filings_df)
    logger.info("  %d total 10-K/10-K/A filings found", n_total_filings)

    # ── Step 2: XBRL coverage audit ──────────────────────────────────────────
    logger.info("Running XBRL coverage audit …")
    coverage_df = audit_xbrl_coverage(filings_df, end_year=config.end_year)

    # Apply min_filings filter
    coverage_df = coverage_df[coverage_df["n_10k_filings"] >= config.min_filings]
    logger.info("  %d companies pass min_filings=%d filter", len(coverage_df), config.min_filings)

    # ── Step 3: Company name (most-recent filing) ────────────────────────────
    latest_name = (
        filings_df[filings_df["form_type"] == "10-K"]
        .sort_values("date_filed")
        .groupby("cik")["company_name"]
        .last()
    )
    universe = coverage_df.join(latest_name.rename("entity_name"), how="left")

    # ── Step 4: Ticker / exchange ────────────────────────────────────────────
    logger.info("Fetching company ticker/exchange mapping …")
    time.sleep(1.0 / config.rate_limit_per_sec)
    tickers_df = fetch_company_tickers(cache_dir, session)
    tickers_df = tickers_df.set_index("cik")[["ticker", "exchange"]]
    universe = universe.join(tickers_df, how="left")

    # ── Step 5: EDGAR submissions metadata ──────────────────────────────────
    if config.fetch_submissions:
        ciks = universe.index.tolist()
        submissions_meta = fetch_all_submissions(ciks, cache_dir, config)
        meta_df = pd.DataFrame.from_dict(submissions_meta, orient="index")
        universe = universe.join(meta_df, how="left")
    else:
        for col in ["sic_code", "sic_description", "state_of_inc", "fiscal_year_end", "entity_type"]:
            universe[col] = None

    # ── Step 6: Sector classification ────────────────────────────────────────
    universe["sector"] = universe["sic_code"].apply(sic_to_sector)

    # ── Step 7: Final column ordering & types ────────────────────────────────
    ordered_cols = [
        "entity_name",
        "ticker",
        "exchange",
        "sic_code",
        "sic_description",
        "sector",
        "state_of_inc",
        "fiscal_year_end",
        "entity_type",
        "first_10k_date",
        "last_10k_date",
        "n_10k_filings",
        "n_xbrl_filings",
        "xbrl_coverage_pct",
        "xbrl_gap_flag",
        "filing_years",
        "gap_years",
        "is_current_filer",
    ]
    # Add any columns we have but aren't in the ordered list
    extra = [c for c in universe.columns if c not in ordered_cols]
    universe = universe[ordered_cols + extra]
    universe.index.name = "cik"
    universe = universe.reset_index()

    # ── Step 8: Provenance metadata ──────────────────────────────────────────
    build_date = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")  # W6
    provenance = {
        "build_date": build_date,
        "start_year": start_year,
        "end_year": end_year,
        "n_companies": len(universe),
        "n_filings_in_index": n_total_filings,
        "form_types": list(config.form_types),
        "min_filings": config.min_filings,
        "xbrl_mandatory_date": XBRL_MANDATORY_DATE.isoformat(),
        "data_sources": [
            "EDGAR full-index quarterly form.idx files",
            "EDGAR company_tickers_exchange.json",
            "EDGAR submissions API (per-company)",
        ],
    }
    # Embed provenance as Parquet schema metadata
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq

        table = pa.Table.from_pandas(universe)
        meta = table.schema.metadata or {}
        meta[b"fin_jepa_provenance"] = json.dumps(provenance).encode()
        table = table.replace_schema_metadata(meta)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, output_path)
    except ImportError:
        # pyarrow not available — fall back to plain parquet without
        # embedded metadata (C2: only ImportError warrants this fallback;
        # disk/permission errors should propagate so callers can react).
        output_path.parent.mkdir(parents=True, exist_ok=True)
        universe.to_parquet(output_path, index=False)
        # Write provenance as sidecar JSON
        prov_path = output_path.with_suffix(".provenance.json")
        with open(prov_path, "w", encoding="utf-8") as fh:
            json.dump(provenance, fh, indent=2)

    logger.info(
        "Universe written to %s  (%d companies, %d filings)",
        output_path, len(universe), n_total_filings,
    )
    return universe


# ---------------------------------------------------------------------------
# Universe loader
# ---------------------------------------------------------------------------

def load_company_universe(
    path: Path | str,
) -> tuple[pd.DataFrame, dict]:
    """Load a pre-built company universe Parquet file.

    Args:
        path: Path to the ``company_universe.parquet`` file produced by
            :func:`build_company_universe`.

    Returns:
        Tuple of ``(universe_df, provenance_dict)``.  The provenance dict
        is extracted from the Parquet schema metadata; an empty dict is
        returned if metadata is absent.
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
        # Try sidecar provenance JSON
        prov_path = path.with_suffix(".provenance.json")
        if prov_path.exists():
            with open(prov_path, encoding="utf-8") as fh:
                provenance = json.load(fh)

    return df, provenance

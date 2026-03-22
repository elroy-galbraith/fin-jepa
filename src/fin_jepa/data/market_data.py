"""Market data collection and alignment.

Collects and aligns daily OHLCV data, corporate actions, and index returns
for the full company universe.  Required for:

  * ``stock_decline`` distress labels — 12-month market-adjusted forward
    return below ``decline_threshold`` (default −30 %).
  * Study 1 market signal redundancy test.

Data sources
------------
* **yfinance** — adjusted daily OHLCV; corporate actions (splits / dividends).
  Splits are baked into the ``auto_adjust=True`` price series, so no manual
  split adjustment is needed.
* **SPDR sector ETFs** — sector-level benchmark returns for risk adjustment.
  XLC (Communications) was only created 2018-06-18; Telecom companies use
  ``^GSPC`` as a fallback before that date.
* **^GSPC (S&P 500)** — market-wide benchmark.

Short-interest note
-------------------
FINRA publishes bi-monthly equity short interest files free of charge at
https://www.finra.org/filing-reporting/regulatory-filing-systems/short-interest.
These are *not* auto-downloaded here because the files require per-security
aggregation and the FINRA bulk format changes over time.  To incorporate
short interest, download the desired FINRA files, aggregate to daily
(cik, period_end) granularity, and pass the result to
``build_market_dataset(short_interest_df=...)``.

Caching
-------
All downloads are cached to ``{raw_dir}/market/``:

  ``prices/{TICKER}.parquet``   — daily adjusted OHLCV per equity ticker
  ``indices/{TICKER}.parquet``  — daily adjusted OHLCV per index / ETF
  ``actions/{TICKER}.parquet``  — splits and dividends per equity ticker
  ``market_aligned.parquet``    — final aligned dataset (pipeline output)

Output schema
-------------
``build_market_dataset()`` / ``align_to_filing_dates()`` returns a DataFrame
keyed by ``(cik, period_end)`` with columns:

  cik            str    10-digit zero-padded CIK
  period_end     date   fiscal year end date
  filing_date    date   10-K filing date (forward-window start)
  ticker         str    equity ticker
  sector         str    Fama-French 12-industry sector
  fwd_ret_21d    float  log return over next 21 trading days (~1 month)
  fwd_ret_63d    float  log return over next 63 trading days (~3 months)
  fwd_ret_126d   float  log return over next 126 trading days (~6 months)
  fwd_ret_252d   float  log return over next 252 trading days (~12 months)
  mkt_ret_252d   float  S&P 500 log return over the same 252-day window
  sec_ret_252d   float  sector ETF log return over the same 252-day window
  mkt_adj_252d   float  fwd_ret_252d − mkt_ret_252d
  sec_adj_252d   float  fwd_ret_252d − sec_ret_252d
  volume_avg_63d float  mean daily volume over the 63 trading days before filing
  delisted       bool   True if fewer than ``primary_horizon`` days of data
                        remain after the filing date
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional dependency: yfinance
# ---------------------------------------------------------------------------
try:
    import yfinance as _yf
except ImportError:  # pragma: no cover
    _yf = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Return horizons (trading days) used throughout the module.
RETURN_HORIZONS: list[int] = [21, 63, 126, 252]

#: S&P 500 — market-wide benchmark ticker.
MARKET_INDEX_TICKER: str = "^GSPC"

#: Fama-French 12-sector → SPDR sector ETF mapping.
#:
#: XLC (Communications Services) began trading 2018-06-18.  For filings
#: before that date the Telecom bucket falls back to ^GSPC.
FF12_TO_ETF: dict[str, str] = {
    "Consumer NonDurables": "XLP",
    "Consumer Durables": "XLY",
    "Manufacturing": "XLI",
    "Energy": "XLE",
    "Chemicals": "XLB",
    "Business Equipment": "XLK",
    "Telecom": "XLC",
    "Utilities": "XLU",
    "Shops": "XLY",
    "Health Care": "XLV",
    "Finance": "XLF",
    "Other": "^GSPC",
}

#: All unique index / ETF tickers that the pipeline needs.
INDEX_TICKERS: list[str] = sorted({MARKET_INDEX_TICKER} | set(FF12_TO_ETF.values()))

#: Date from which XLC data is available (SPDR Communications Services ETF).
_XLC_START = pd.Timestamp("2018-06-18")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class MarketDataConfig:
    """Configuration for the market data collection pipeline.

    Attributes:
        start_date: Earliest date to fetch prices for ("YYYY-MM-DD").
            Should precede the earliest filing date by enough to compute
            pre-filing volume statistics (default: 2010-01-01).
        end_date: Latest date to fetch prices for ("YYYY-MM-DD").
        horizons_days: Return horizons in trading days.
        return_window_days: Primary forward-return horizon used for the
            ``stock_decline`` label (must be in ``horizons_days``).
        batch_size: Tickers per yfinance batch download call.
        sleep_between_batches: Seconds to sleep between yfinance calls
            to avoid rate-limiting.
        overwrite_cache: Re-download even when a cached parquet file
            already exists on disk.
    """

    start_date: str = "2010-01-01"
    end_date: str = "2024-12-31"
    horizons_days: list[int] = field(default_factory=lambda: list(RETURN_HORIZONS))
    return_window_days: int = 252
    batch_size: int = 100
    sleep_between_batches: float = 0.5
    overwrite_cache: bool = False

    def __post_init__(self) -> None:
        if self.return_window_days not in self.horizons_days:
            raise ValueError(
                f"return_window_days={self.return_window_days} must be in "
                f"horizons_days={self.horizons_days}"
            )


# ---------------------------------------------------------------------------
# Caching helpers
# ---------------------------------------------------------------------------


def _cache_path(cache_dir: Path, subdir: str, ticker: str) -> Path:
    """Return the parquet cache path for *ticker* under *cache_dir/subdir*."""
    safe = ticker.replace("^", "_idx_").replace("/", "_")
    return cache_dir / subdir / f"{safe}.parquet"


def _load_cached(path: Path) -> pd.DataFrame | None:
    """Return the cached DataFrame, or ``None`` if absent / unreadable."""
    if path.exists():
        try:
            return pd.read_parquet(path)
        except Exception:
            logger.warning("Cache read failed for %s — will re-download", path)
    return None


def _save_cache(df: pd.DataFrame, path: Path) -> None:
    """Persist *df* to *path* as parquet, creating parent dirs as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=True)


# ---------------------------------------------------------------------------
# yfinance download helpers
# ---------------------------------------------------------------------------


def _require_yfinance() -> None:
    if _yf is None:  # pragma: no cover
        raise ImportError(
            "yfinance is required for market data collection. "
            "Install with: pip install 'yfinance>=0.2.40'"
        )


def _yf_download(
    tickers: list[str],
    start: str,
    end: str,
) -> dict[str, pd.DataFrame]:
    """Download adjusted OHLCV for *tickers* via yfinance.

    Returns a dict mapping ticker → DataFrame (empty for delistings /
    unknown symbols).  Uses ``auto_adjust=True`` so all prices are already
    split- and dividend-adjusted.
    """
    _require_yfinance()
    if not tickers:
        return {}

    if len(tickers) == 1:
        raw = _yf.download(
            tickers[0],
            start=start,
            end=end,
            auto_adjust=True,
            progress=False,
            threads=False,
        )
        # yfinance ≥0.2 returns a MultiIndex even for single tickers when
        # group_by="ticker"; handle both flat and multi-index DataFrames.
        if isinstance(raw.columns, pd.MultiIndex):
            raw = raw[tickers[0]] if tickers[0] in raw.columns.get_level_values(0) else raw
        return {tickers[0]: raw}

    raw = _yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        threads=False,
        group_by="ticker",
    )
    result: dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        try:
            df = raw[ticker].copy()
            # Drop all-NaN rows that yfinance inserts for missing days
            df = df.dropna(how="all")
            result[ticker] = df
        except (KeyError, TypeError):
            result[ticker] = pd.DataFrame()
    return result


# ---------------------------------------------------------------------------
# Public fetch functions
# ---------------------------------------------------------------------------


def fetch_ohlcv(
    tickers: list[str],
    start: str,
    end: str,
    cache_dir: Path,
    batch_size: int = 100,
    sleep_between_batches: float = 0.5,
    overwrite: bool = False,
) -> dict[str, pd.DataFrame]:
    """Fetch adjusted daily OHLCV for *tickers* with per-ticker disk caching.

    Each ticker is cached to ``{cache_dir}/prices/{ticker}.parquet``.
    Subsequent calls hit the cache (unless *overwrite* is ``True``).

    Args:
        tickers: Equity tickers (e.g. ``["AAPL", "MSFT"]``).
        start: Start date "YYYY-MM-DD".
        end: End date "YYYY-MM-DD".
        cache_dir: Root cache directory (``data/raw/market``).
        batch_size: Tickers per yfinance download call.
        sleep_between_batches: Seconds between batches.
        overwrite: Re-download even when a cache file exists.

    Returns:
        Dict mapping ticker → OHLCV DataFrame with columns
        ``[Open, High, Low, Close, Volume]`` indexed by date.
        Empty DataFrame for delisted / unknown tickers.
    """
    result: dict[str, pd.DataFrame] = {}
    to_download: list[str] = []

    for ticker in tickers:
        path = _cache_path(cache_dir, "prices", ticker)
        cached = None if overwrite else _load_cached(path)
        if cached is not None:
            result[ticker] = cached
        else:
            to_download.append(ticker)

    if not to_download:
        return result

    logger.info(
        "Downloading OHLCV for %d tickers (batches of %d)…",
        len(to_download),
        batch_size,
    )
    for i in range(0, len(to_download), batch_size):
        batch = to_download[i : i + batch_size]
        try:
            batch_data = _yf_download(batch, start, end)
        except Exception as exc:
            logger.warning("Batch download failed (%s) — retrying one-by-one", exc)
            batch_data = {}
            for ticker in batch:
                try:
                    batch_data.update(_yf_download([ticker], start, end))
                except Exception as exc2:
                    logger.warning("Single download failed for %s: %s", ticker, exc2)
                    batch_data[ticker] = pd.DataFrame()
                time.sleep(0.2)

        for ticker, df in batch_data.items():
            result[ticker] = df
            _save_cache(df, _cache_path(cache_dir, "prices", ticker))

        if i + batch_size < len(to_download):
            time.sleep(sleep_between_batches)

    return result


def fetch_index_returns(
    index_tickers: list[str],
    start: str,
    end: str,
    cache_dir: Path,
    overwrite: bool = False,
) -> dict[str, pd.DataFrame]:
    """Fetch daily OHLCV for market / sector index tickers.

    Caches to ``{cache_dir}/indices/{ticker}.parquet``.

    Args:
        index_tickers: Index / ETF tickers (e.g. ``["^GSPC", "XLK"]``).
        start: Start date "YYYY-MM-DD".
        end: End date "YYYY-MM-DD".
        cache_dir: Root cache directory.
        overwrite: Re-download even when a cache file exists.

    Returns:
        Dict mapping ticker → daily OHLCV DataFrame.
    """
    result: dict[str, pd.DataFrame] = {}
    to_download: list[str] = []

    for ticker in index_tickers:
        path = _cache_path(cache_dir, "indices", ticker)
        cached = None if overwrite else _load_cached(path)
        if cached is not None:
            result[ticker] = cached
        else:
            to_download.append(ticker)

    if not to_download:
        return result

    logger.info("Downloading %d index/ETF series…", len(to_download))
    _require_yfinance()
    for ticker in to_download:
        try:
            raw = _yf_download([ticker], start, end)
            df = raw.get(ticker, pd.DataFrame())
        except Exception as exc:
            logger.warning("Index download failed for %s: %s", ticker, exc)
            df = pd.DataFrame()
        result[ticker] = df
        _save_cache(df, _cache_path(cache_dir, "indices", ticker))
        time.sleep(0.2)

    return result


def fetch_corporate_actions(
    tickers: list[str],
    cache_dir: Path,
    overwrite: bool = False,
) -> dict[str, pd.DataFrame]:
    """Fetch corporate actions (splits, dividends) via yfinance.

    Caches to ``{cache_dir}/actions/{ticker}.parquet``.

    yfinance returns *adjusted* prices (``auto_adjust=True``), so splits and
    dividends are already baked into the OHLCV series.  This function stores
    the raw corporate-actions table for downstream analysis (e.g. detecting
    delistings via large reverse splits) and for audit purposes.

    Args:
        tickers: Equity tickers.
        cache_dir: Root cache directory.
        overwrite: Re-download even when a cache file exists.

    Returns:
        Dict mapping ticker → DataFrame with columns
        ``[Dividends, Stock Splits]`` indexed by date.
    """
    _require_yfinance()
    _EMPTY_ACTIONS = pd.DataFrame(columns=["Dividends", "Stock Splits"])
    result: dict[str, pd.DataFrame] = {}

    for ticker in tickers:
        path = _cache_path(cache_dir, "actions", ticker)
        cached = None if overwrite else _load_cached(path)
        if cached is not None:
            result[ticker] = cached
            continue

        try:
            obj = _yf.Ticker(ticker)
            actions = obj.actions
            if actions is None or actions.empty:
                actions = _EMPTY_ACTIONS.copy()
        except Exception as exc:
            logger.warning("Corporate actions fetch failed for %s: %s", ticker, exc)
            actions = _EMPTY_ACTIONS.copy()

        result[ticker] = actions
        _save_cache(actions, path)
        time.sleep(0.1)

    return result


# ---------------------------------------------------------------------------
# Forward return computation
# ---------------------------------------------------------------------------


def _close_series(prices: pd.DataFrame) -> pd.Series:
    """Extract the adjusted close price series from an OHLCV DataFrame."""
    for col in ("Close", "close", "Adj Close", "adj_close"):
        if col in prices.columns:
            return prices[col].dropna()
    raise KeyError(
        f"No Close column found in prices DataFrame. Columns: {list(prices.columns)}"
    )


def _fwd_returns_wide(close: pd.Series, horizons_days: list[int]) -> pd.DataFrame:
    """Compute wide-format log forward returns for a single price series.

    Returns a DataFrame indexed by date with columns ``fwd_ret_{h}d``.
    Trailing rows where the full window is not yet available are NaN.
    """
    log_p = np.log(close)
    df = pd.DataFrame(index=close.index)
    for h in horizons_days:
        df[f"fwd_ret_{h}d"] = log_p.shift(-h) - log_p
    return df


def compute_forward_returns(
    prices: pd.DataFrame,
    horizons_days: list[int] = RETURN_HORIZONS,
) -> pd.DataFrame:
    """Compute log forward returns at each horizon for all tickers in *prices*.

    Args:
        prices: Wide-format adjusted close price DataFrame returned by
            :func:`fetch_prices` (date index, ticker columns).
        horizons_days: Forward-looking windows in trading days.

    Returns:
        Long-format DataFrame with columns:
        ``[ticker, date, horizon, fwd_return]``.
    """
    records: list[pd.DataFrame] = []
    for ticker in prices.columns:
        close = prices[ticker].dropna()
        if close.empty:
            continue
        wide = _fwd_returns_wide(close, horizons_days)
        wide.index.name = "date"
        melted = wide.reset_index().melt(
            id_vars="date",
            var_name="horizon_col",
            value_name="fwd_return",
        )
        melted["ticker"] = ticker
        melted["horizon"] = melted["horizon_col"].str.extract(r"(\d+)").astype(int)
        melted = melted.drop(columns="horizon_col")
        records.append(melted[["ticker", "date", "horizon", "fwd_return"]])

    if not records:
        return pd.DataFrame(columns=["ticker", "date", "horizon", "fwd_return"])
    return pd.concat(records, ignore_index=True)


def compute_market_adjusted_returns(
    stock_returns: pd.Series,
    benchmark_returns: pd.Series,
) -> pd.Series:
    """Compute market-adjusted (excess) returns.

    Args:
        stock_returns: Stock log returns indexed by date.
        benchmark_returns: Benchmark log returns indexed by date.

    Returns:
        Series of excess log returns (stock − benchmark), aligned on the
        intersection of both indexes.
    """
    aligned_stock, aligned_bench = stock_returns.align(benchmark_returns, join="inner")
    return aligned_stock - aligned_bench


# ---------------------------------------------------------------------------
# Company-year grid construction
# ---------------------------------------------------------------------------


def build_company_year_grid(universe_df: pd.DataFrame) -> pd.DataFrame:
    """Expand the per-company universe into per-company-year observations.

    For each company in *universe_df*, emits one row per calendar year in
    ``filing_years``.  The ``period_end`` date is constructed from the
    company's ``fiscal_year_end`` (MMDD) and the filing year.

    Args:
        universe_df: Company universe from ``build_company_universe()``
            (one row per company).  Must contain columns ``cik``,
            ``ticker``, ``sector``, ``filing_years``, and
            ``fiscal_year_end`` (MMDD string, nullable).

    Returns:
        DataFrame with one row per (cik, fiscal_year):
        ``[cik, ticker, sector, fiscal_year, period_end, filing_date]``

        *filing_date* is estimated as ``period_end + 90 days`` because the
        actual per-filing dates come from the XBRL loader (ATS-162).  When
        that module is available, pass a ``filings_df`` directly to
        :func:`align_to_filing_dates` instead.
    """
    required = {"cik", "ticker", "sector", "filing_years"}
    missing = required - set(universe_df.columns)
    if missing:
        raise ValueError(f"universe_df missing required columns: {missing}")

    rows: list[dict] = []
    for _, row in universe_df.iterrows():
        filing_years = row.get("fiscal_year", None)  # unused default
        fy_list = row["filing_years"]
        if not fy_list:
            continue

        fy_end_mmdd = str(row.get("fiscal_year_end") or "1231")
        # Normalise: strip hyphens so "12-31" → "1231"
        fy_end_mmdd = fy_end_mmdd.replace("-", "").zfill(4)
        mm = fy_end_mmdd[:2]
        dd = fy_end_mmdd[2:]

        for year in fy_list:
            try:
                period_end = pd.Timestamp(f"{year}-{mm}-{dd}")
            except Exception:
                # Fallback: December 31 of the fiscal year
                period_end = pd.Timestamp(f"{year}-12-31")

            filing_date = period_end + pd.Timedelta(days=90)

            rows.append(
                {
                    "cik": row["cik"],
                    "ticker": row.get("ticker"),
                    "sector": row.get("sector", "Other"),
                    "fiscal_year": int(year),
                    "period_end": period_end,
                    "filing_date": filing_date,
                }
            )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Alignment to filing dates
# ---------------------------------------------------------------------------


def align_to_filing_dates(
    filings_df: pd.DataFrame,
    prices_dict: dict[str, pd.DataFrame],
    index_prices: dict[str, pd.DataFrame],
    horizons_days: list[int] = RETURN_HORIZONS,
    primary_horizon: int = 252,
) -> pd.DataFrame:
    """Align market data to each (cik, period_end, filing_date) observation.

    The *filing date* (not the fiscal year end) is used as the start of the
    forward return window: market participants can only react after the 10-K
    is publicly available.

    Args:
        filings_df: Per-company-year observations.  Must contain columns
            ``cik``, ``ticker``, ``sector``, ``period_end``, and
            ``filing_date``.  Rows without a ticker are dropped.
        prices_dict: Dict mapping ticker → adjusted OHLCV DataFrame
            (output of :func:`fetch_ohlcv`).
        index_prices: Dict mapping index / ETF ticker → OHLCV DataFrame
            (output of :func:`fetch_index_returns`).
        horizons_days: Return horizons to compute (trading days).
        primary_horizon: Horizon used for the ``delisted`` flag.  A row is
            flagged delisted when fewer than *primary_horizon* trading days
            of price data remain after the filing date.

    Returns:
        DataFrame with the output schema described in the module docstring.
    """
    required = {"cik", "ticker", "sector", "period_end", "filing_date"}
    missing = required - set(filings_df.columns)
    if missing:
        raise ValueError(f"filings_df missing required columns: {missing}")

    # Pre-compute index forward returns once (expensive to repeat per row)
    idx_fwd: dict[str, pd.DataFrame] = {}
    for idx_ticker, idx_df in index_prices.items():
        if not idx_df.empty:
            try:
                close = _close_series(idx_df)
                idx_fwd[idx_ticker] = _fwd_returns_wide(close, horizons_days)
            except Exception:
                idx_fwd[idx_ticker] = pd.DataFrame()
        else:
            idx_fwd[idx_ticker] = pd.DataFrame()

    market_fwd = idx_fwd.get(MARKET_INDEX_TICKER, pd.DataFrame())

    rows: list[dict] = []

    for _, obs in filings_df.iterrows():
        ticker = obs.get("ticker")
        if not ticker or pd.isna(ticker):
            continue

        cik = obs["cik"]
        sector = obs.get("sector", "Other")
        period_end = pd.Timestamp(obs["period_end"])
        filing_date = pd.Timestamp(obs["filing_date"])

        prices = prices_dict.get(str(ticker), pd.DataFrame())

        # Determine sector ETF; fall back to ^GSPC for XLC before 2018-06-18
        etf_ticker = FF12_TO_ETF.get(str(sector), MARKET_INDEX_TICKER)
        if etf_ticker == "XLC" and filing_date < _XLC_START:
            etf_ticker = MARKET_INDEX_TICKER
        sector_fwd = idx_fwd.get(etf_ticker, market_fwd)

        record: dict = {
            "cik": cik,
            "period_end": period_end,
            "filing_date": filing_date,
            "ticker": ticker,
            "sector": sector,
        }

        if prices.empty:
            _fill_nan(record, horizons_days, primary_horizon)
            record["delisted"] = True
            rows.append(record)
            continue

        close = _close_series(prices)
        trading_dates = close.index

        # First trading day on or after the filing date
        on_or_after = trading_dates[trading_dates >= filing_date]
        if on_or_after.empty:
            _fill_nan(record, horizons_days, primary_horizon)
            record["delisted"] = True
            rows.append(record)
            continue

        ref_date = on_or_after[0]
        days_remaining = int((trading_dates > ref_date).sum())
        record["delisted"] = days_remaining < primary_horizon

        fwd_wide = _fwd_returns_wide(close, horizons_days)

        for h in horizons_days:
            col = f"fwd_ret_{h}d"
            record[col] = _lookup_fwd(fwd_wide, ref_date, col)

        # Market- and sector-adjusted returns for each horizon
        for h in horizons_days:
            fwd_col = f"fwd_ret_{h}d"
            s_ret = record[fwd_col]
            m_ret = _lookup_fwd(market_fwd, ref_date, fwd_col)
            sec_ret = _lookup_fwd(sector_fwd, ref_date, fwd_col)
            record[f"mkt_adj_{h}d"] = (
                s_ret - m_ret
                if not (_isnan(s_ret) or _isnan(m_ret))
                else np.nan
            )
            record[f"sec_adj_{h}d"] = (
                s_ret - sec_ret
                if not (_isnan(s_ret) or _isnan(sec_ret))
                else np.nan
            )

        # Convenience aliases for the primary (12-month) horizon
        ph = primary_horizon
        record[f"mkt_ret_{ph}d"] = _lookup_fwd(market_fwd, ref_date, f"fwd_ret_{ph}d")
        record[f"sec_ret_{ph}d"] = _lookup_fwd(sector_fwd, ref_date, f"fwd_ret_{ph}d")

        # Average daily volume over the 63 trading days before filing
        if "Volume" in prices.columns:
            pre_filing = prices.loc[prices.index < ref_date, "Volume"].iloc[-63:]
            record["volume_avg_63d"] = (
                float(pre_filing.mean()) if len(pre_filing) > 0 else np.nan
            )
        else:
            record["volume_avg_63d"] = np.nan

        rows.append(record)

    return pd.DataFrame(rows)


def _lookup_fwd(fwd_df: pd.DataFrame, ref_date: pd.Timestamp, col: str) -> float:
    """Look up the forward return value at or nearest to *ref_date*."""
    if fwd_df.empty or col not in fwd_df.columns:
        return np.nan
    on_or_after = fwd_df.index[fwd_df.index >= ref_date]
    if on_or_after.empty:
        return np.nan
    try:
        return float(fwd_df.loc[on_or_after[0], col])
    except (KeyError, TypeError):
        return np.nan


def _fill_nan(record: dict, horizons_days: list[int], primary_horizon: int) -> None:
    """Fill all return columns with NaN for delisted / missing-data rows."""
    for h in horizons_days:
        record[f"fwd_ret_{h}d"] = np.nan
        record[f"mkt_adj_{h}d"] = np.nan
        record[f"sec_adj_{h}d"] = np.nan
    record[f"mkt_ret_{primary_horizon}d"] = np.nan
    record[f"sec_ret_{primary_horizon}d"] = np.nan
    record["volume_avg_63d"] = np.nan


def _isnan(v: object) -> bool:
    try:
        return bool(np.isnan(v))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return True


# ---------------------------------------------------------------------------
# Full pipeline orchestrator
# ---------------------------------------------------------------------------


def build_market_dataset(
    universe_df: pd.DataFrame,
    raw_dir: Path,
    config: MarketDataConfig | None = None,
    filings_df: pd.DataFrame | None = None,
    short_interest_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Full market data collection and alignment pipeline.

    Downloads prices, indices, and corporate actions for every company in
    *universe_df*, then aligns them to each company-year's filing date.

    Args:
        universe_df: Company universe from ``build_company_universe()``.
        raw_dir: Root raw data directory (e.g. ``data/raw``).
        config: Pipeline configuration.  Defaults to ``MarketDataConfig()``.
        filings_df: Optional pre-built per-company-year observations with
            columns ``[cik, ticker, sector, period_end, filing_date]``.
            If ``None``, the grid is built automatically from *universe_df*
            via :func:`build_company_year_grid`.
        short_interest_df: Optional short-interest table keyed by
            ``(ticker, date)`` with a ``short_interest_ratio`` column.
            When provided, it is left-joined onto the aligned dataset.

    Returns:
        Aligned DataFrame (see module docstring for full schema).
        Persisted to ``{raw_dir}/market/market_aligned.parquet``.
    """
    if config is None:
        config = MarketDataConfig()

    cache_dir = Path(raw_dir) / "market"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # 1. Build or validate the company-year observation grid
    if filings_df is None:
        logger.info("Building company-year grid from universe (%d companies)…", len(universe_df))
        filings_df = build_company_year_grid(universe_df)
    else:
        required = {"cik", "ticker", "sector", "period_end", "filing_date"}
        missing = required - set(filings_df.columns)
        if missing:
            raise ValueError(f"filings_df missing required columns: {missing}")

    logger.info("Company-year grid: %d observations", len(filings_df))

    # 2. Collect unique equity tickers
    tickers: list[str] = (
        filings_df["ticker"]
        .dropna()
        .astype(str)
        .str.strip()
        .loc[lambda s: s != ""]
        .unique()
        .tolist()
    )
    logger.info("Unique equity tickers: %d", len(tickers))

    # 3. Fetch adjusted OHLCV for equity tickers
    prices_dict = fetch_ohlcv(
        tickers=tickers,
        start=config.start_date,
        end=config.end_date,
        cache_dir=cache_dir,
        batch_size=config.batch_size,
        sleep_between_batches=config.sleep_between_batches,
        overwrite=config.overwrite_cache,
    )

    # 4. Fetch market and sector index prices
    index_prices = fetch_index_returns(
        index_tickers=INDEX_TICKERS,
        start=config.start_date,
        end=config.end_date,
        cache_dir=cache_dir,
        overwrite=config.overwrite_cache,
    )

    # 5. Fetch corporate actions (cached; summary logged)
    actions_dict = fetch_corporate_actions(
        tickers=tickers,
        cache_dir=cache_dir,
        overwrite=config.overwrite_cache,
    )
    n_splits = sum(
        1
        for df in actions_dict.values()
        if not df.empty
        and "Stock Splits" in df.columns
        and (df["Stock Splits"] != 0).any()
    )
    logger.info(
        "Corporate actions fetched: %d/%d tickers had ≥1 stock split",
        n_splits,
        len(tickers),
    )

    # 6. Align market data to filing dates
    aligned_df = align_to_filing_dates(
        filings_df=filings_df,
        prices_dict=prices_dict,
        index_prices=index_prices,
        horizons_days=config.horizons_days,
        primary_horizon=config.return_window_days,
    )

    # 7. Optionally join short-interest data
    if short_interest_df is not None:
        required_si = {"ticker", "date", "short_interest_ratio"}
        missing_si = required_si - set(short_interest_df.columns)
        if missing_si:
            logger.warning(
                "short_interest_df missing columns %s — skipping join", missing_si
            )
        else:
            si_at_filing = short_interest_df.rename(columns={"date": "filing_date"})
            aligned_df = aligned_df.merge(
                si_at_filing[["ticker", "filing_date", "short_interest_ratio"]],
                on=["ticker", "filing_date"],
                how="left",
            )
            logger.info(
                "Short-interest join: %.1f%% of rows matched",
                100 * aligned_df["short_interest_ratio"].notna().mean(),
            )

    # 8. Persist
    out_path = cache_dir / "market_aligned.parquet"
    aligned_df.to_parquet(out_path, index=False)
    logger.info(
        "Saved aligned market dataset → %s  (%d rows, %d columns)",
        out_path,
        len(aligned_df),
        len(aligned_df.columns),
    )

    return aligned_df


# ---------------------------------------------------------------------------
# Original public API stubs — now implemented
# ---------------------------------------------------------------------------


def fetch_prices(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """Fetch adjusted close prices for *tickers* between *start* and *end*.

    Returns a DataFrame indexed by date with ticker columns (wide format).

    This is a convenience wrapper around :func:`fetch_ohlcv` that returns
    only the ``Close`` column for each ticker.  For full OHLCV data (Open,
    High, Low, Volume) use :func:`fetch_ohlcv` with a persistent cache dir.

    .. note::
        This function does **not** cache results to disk.  For production
        pipelines use :func:`fetch_ohlcv` with a ``cache_dir``.
    """
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        prices_dict = fetch_ohlcv(
            tickers=tickers,
            start=start,
            end=end,
            cache_dir=Path(tmpdir),
        )

    frames: dict[str, pd.Series] = {}
    for ticker, df in prices_dict.items():
        if not df.empty:
            try:
                frames[ticker] = _close_series(df)
            except KeyError:
                pass

    if not frames:
        return pd.DataFrame()
    return pd.DataFrame(frames)

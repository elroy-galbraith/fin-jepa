"""
Market data collection and alignment.

Workstream: Collect and align prices, volumes, and corporate actions.

TODO:
  - Pull adjusted daily OHLCV via yfinance (or a vendor API)
  - Handle splits, dividends, delistings
  - Compute forward returns over multiple horizons (1m, 3m, 6m, 12m)
  - Align market data with XBRL filing dates
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def fetch_prices(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """Fetch adjusted close prices for *tickers* between *start* and *end*.

    Returns a DataFrame indexed by date with ticker columns.
    """
    raise NotImplementedError("Implement market data fetch.")


def compute_forward_returns(prices: pd.DataFrame, horizons_days: list[int]) -> pd.DataFrame:
    """Compute forward returns at each horizon for all tickers.

    Returns a long DataFrame with columns: [ticker, date, horizon, fwd_return].
    """
    raise NotImplementedError("Implement forward return computation.")

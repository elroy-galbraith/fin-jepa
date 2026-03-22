"""Tests for fin_jepa.data.market_data.

All tests use mocked yfinance calls so the suite runs fully offline.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from fin_jepa.data.market_data import (
    FF12_TO_ETF,
    INDEX_TICKERS,
    MARKET_INDEX_TICKER,
    RETURN_HORIZONS,
    MarketDataConfig,
    _close_series,
    _fwd_returns_wide,
    align_to_filing_dates,
    build_company_year_grid,
    build_market_dataset,
    compute_forward_returns,
    compute_market_adjusted_returns,
    fetch_corporate_actions,
    fetch_index_returns,
    fetch_ohlcv,
    fetch_prices,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _make_prices(
    start: str = "2020-01-02",
    periods: int = 300,
    ticker: str = "AAPL",
    volume: float = 1_000_000,
) -> pd.DataFrame:
    """Create a synthetic daily OHLCV DataFrame."""
    dates = pd.bdate_range(start=start, periods=periods)
    close = 100 * np.exp(np.cumsum(np.random.default_rng(42).normal(0, 0.01, periods)))
    return pd.DataFrame(
        {
            "Open": close * 0.99,
            "High": close * 1.01,
            "Low": close * 0.98,
            "Close": close,
            "Volume": volume,
        },
        index=dates,
    )


def _make_universe(n: int = 3) -> pd.DataFrame:
    """Create a minimal company universe DataFrame."""
    tickers = ["AAA", "BBB", "CCC"][:n]
    return pd.DataFrame(
        {
            "cik": [f"{i:010d}" for i in range(1, n + 1)],
            "ticker": tickers,
            "sector": ["Finance", "Health Care", "Energy"][:n],
            "fiscal_year_end": ["1231", "0630", "0930"][:n],
            "filing_years": [[2020, 2021], [2020], [2021]][:n],
        }
    )


# ---------------------------------------------------------------------------
# MarketDataConfig
# ---------------------------------------------------------------------------


class TestMarketDataConfig:
    def test_defaults(self):
        cfg = MarketDataConfig()
        assert cfg.start_date == "2010-01-01"
        assert cfg.return_window_days == 252
        assert 252 in cfg.horizons_days

    def test_invalid_primary_horizon_raises(self):
        with pytest.raises(ValueError, match="return_window_days"):
            MarketDataConfig(horizons_days=[21, 63], return_window_days=252)


# ---------------------------------------------------------------------------
# Caching: fetch_ohlcv
# ---------------------------------------------------------------------------


class TestFetchOhlcv:
    def test_cache_miss_triggers_download(self, tmp_path):
        prices = _make_prices()
        with patch("fin_jepa.data.market_data._yf_download") as mock_dl:
            mock_dl.return_value = {"AAPL": prices}
            result = fetch_ohlcv(["AAPL"], "2020-01-01", "2021-01-01", tmp_path)

        assert "AAPL" in result
        assert not result["AAPL"].empty
        mock_dl.assert_called_once()

    def test_cache_hit_skips_download(self, tmp_path):
        prices = _make_prices()
        # Pre-populate cache
        cache_file = tmp_path / "prices" / "AAPL.parquet"
        cache_file.parent.mkdir(parents=True)
        prices.to_parquet(cache_file)

        with patch("fin_jepa.data.market_data._yf_download") as mock_dl:
            result = fetch_ohlcv(["AAPL"], "2020-01-01", "2021-01-01", tmp_path)

        mock_dl.assert_not_called()
        assert "AAPL" in result

    def test_overwrite_ignores_cache(self, tmp_path):
        prices = _make_prices()
        cache_file = tmp_path / "prices" / "AAPL.parquet"
        cache_file.parent.mkdir(parents=True)
        prices.to_parquet(cache_file)

        fresh = _make_prices(periods=50)
        with patch("fin_jepa.data.market_data._yf_download") as mock_dl:
            mock_dl.return_value = {"AAPL": fresh}
            result = fetch_ohlcv(
                ["AAPL"], "2020-01-01", "2021-01-01", tmp_path, overwrite=True
            )

        mock_dl.assert_called_once()
        assert len(result["AAPL"]) == 50

    def test_empty_df_for_unknown_ticker(self, tmp_path):
        with patch("fin_jepa.data.market_data._yf_download") as mock_dl:
            mock_dl.return_value = {"ZZZZZ": pd.DataFrame()}
            result = fetch_ohlcv(["ZZZZZ"], "2020-01-01", "2021-01-01", tmp_path)

        assert result["ZZZZZ"].empty

    def test_batch_split(self, tmp_path):
        """Tickers exceeding batch_size are downloaded in multiple calls."""
        prices_a = _make_prices()
        prices_b = _make_prices(ticker="BBBB")
        call_count = []

        def mock_dl(tickers, start, end):
            call_count.append(len(tickers))
            return {t: prices_a if t == "AAPL" else prices_b for t in tickers}

        with patch("fin_jepa.data.market_data._yf_download", side_effect=mock_dl):
            fetch_ohlcv(
                ["AAPL", "BBBB"],
                "2020-01-01",
                "2021-01-01",
                tmp_path,
                batch_size=1,
                sleep_between_batches=0.0,
            )

        assert len(call_count) == 2  # one call per ticker

    def test_cache_written_to_disk(self, tmp_path):
        prices = _make_prices()
        with patch("fin_jepa.data.market_data._yf_download") as mock_dl:
            mock_dl.return_value = {"AAPL": prices}
            fetch_ohlcv(["AAPL"], "2020-01-01", "2021-01-01", tmp_path)

        cache_file = tmp_path / "prices" / "AAPL.parquet"
        assert cache_file.exists()


# ---------------------------------------------------------------------------
# fetch_index_returns
# ---------------------------------------------------------------------------


class TestFetchIndexReturns:
    def test_downloads_and_caches(self, tmp_path):
        spy_prices = _make_prices(ticker="^GSPC")
        with patch("fin_jepa.data.market_data._yf_download") as mock_dl:
            mock_dl.return_value = {"^GSPC": spy_prices}
            result = fetch_index_returns(["^GSPC"], "2020-01-01", "2021-01-01", tmp_path)

        assert "^GSPC" in result
        assert not result["^GSPC"].empty
        # Cached under indices/
        idx_file = tmp_path / "indices" / "_idx_GSPC.parquet"
        assert idx_file.exists()

    def test_cache_hit_skips_download(self, tmp_path):
        spy_prices = _make_prices()
        cache_file = tmp_path / "indices" / "_idx_GSPC.parquet"
        cache_file.parent.mkdir(parents=True)
        spy_prices.to_parquet(cache_file)

        with patch("fin_jepa.data.market_data._yf_download") as mock_dl:
            fetch_index_returns(["^GSPC"], "2020-01-01", "2021-01-01", tmp_path)

        mock_dl.assert_not_called()


# ---------------------------------------------------------------------------
# fetch_corporate_actions
# ---------------------------------------------------------------------------


class TestFetchCorporateActions:
    def _mock_ticker(self, dividends=None, splits=None):
        actions = pd.DataFrame(
            {
                "Dividends": dividends or [0.0],
                "Stock Splits": splits or [0.0],
            },
            index=pd.DatetimeIndex(["2020-06-01"]),
        )
        mock_ticker = MagicMock()
        mock_ticker.actions = actions
        return mock_ticker

    def test_fetches_and_caches(self, tmp_path):
        mock_ticker = self._mock_ticker(splits=[2.0])
        with patch("fin_jepa.data.market_data._yf") as mock_yf:
            mock_yf.Ticker.return_value = mock_ticker
            result = fetch_corporate_actions(["AAPL"], tmp_path)

        assert "AAPL" in result
        assert "Stock Splits" in result["AAPL"].columns
        assert (tmp_path / "actions" / "AAPL.parquet").exists()

    def test_cache_hit_skips_api(self, tmp_path):
        actions = pd.DataFrame(
            {"Dividends": [0.25], "Stock Splits": [0.0]},
            index=pd.DatetimeIndex(["2021-03-01"]),
        )
        cache_file = tmp_path / "actions" / "AAPL.parquet"
        cache_file.parent.mkdir(parents=True)
        actions.to_parquet(cache_file)

        with patch("fin_jepa.data.market_data._yf") as mock_yf:
            fetch_corporate_actions(["AAPL"], tmp_path)

        mock_yf.Ticker.assert_not_called()


# ---------------------------------------------------------------------------
# Forward return computation
# ---------------------------------------------------------------------------


class TestFwdReturnsWide:
    def test_shape(self):
        prices = _make_prices(periods=300)
        close = prices["Close"]
        result = _fwd_returns_wide(close, [21, 63])
        assert list(result.columns) == ["fwd_ret_21d", "fwd_ret_63d"]
        assert len(result) == len(close)

    def test_log_return_correct(self):
        """fwd_ret_1d[t] = log(P[t+1] / P[t])."""
        close = pd.Series(
            [100.0, 110.0, 99.0, 121.0],
            index=pd.bdate_range("2020-01-02", periods=4),
        )
        result = _fwd_returns_wide(close, [1])
        expected = np.log(110.0 / 100.0)
        assert abs(result["fwd_ret_1d"].iloc[0] - expected) < 1e-10

    def test_trailing_nan(self):
        """Last *h* rows must be NaN (window not yet available)."""
        close = pd.Series(
            np.ones(10),
            index=pd.bdate_range("2020-01-02", periods=10),
        )
        result = _fwd_returns_wide(close, [3])
        assert result["fwd_ret_3d"].iloc[-3:].isna().all()
        assert not result["fwd_ret_3d"].iloc[:-3].isna().any()


class TestComputeForwardReturns:
    def test_long_format_columns(self):
        prices = _make_prices(periods=100)
        wide = pd.DataFrame({"AAPL": prices["Close"]})
        result = compute_forward_returns(wide, horizons_days=[21])
        assert set(result.columns) == {"ticker", "date", "horizon", "fwd_return"}

    def test_ticker_values(self):
        prices = _make_prices(periods=60)
        wide = pd.DataFrame({"MSFT": prices["Close"]})
        result = compute_forward_returns(wide, horizons_days=[21])
        assert (result["ticker"] == "MSFT").all()

    def test_horizon_values(self):
        prices = _make_prices(periods=100)
        wide = pd.DataFrame({"X": prices["Close"]})
        result = compute_forward_returns(wide, horizons_days=[21, 63])
        assert set(result["horizon"].unique()) == {21, 63}

    def test_empty_prices_returns_empty(self):
        result = compute_forward_returns(pd.DataFrame(), horizons_days=[21])
        assert result.empty
        assert set(result.columns) == {"ticker", "date", "horizon", "fwd_return"}


# ---------------------------------------------------------------------------
# Market-adjusted returns
# ---------------------------------------------------------------------------


class TestComputeMarketAdjustedReturns:
    def test_zero_for_identical_returns(self):
        idx = pd.bdate_range("2020-01-02", periods=50)
        stock = pd.Series(np.random.default_rng(1).normal(0, 0.01, 50), index=idx)
        result = compute_market_adjusted_returns(stock, stock)
        assert (result.abs() < 1e-12).all()

    def test_correct_excess_return(self):
        idx = pd.bdate_range("2020-01-02", periods=5)
        stock = pd.Series([0.10, 0.05, -0.02, 0.03, 0.08], index=idx)
        market = pd.Series([0.05, 0.05, 0.05, 0.05, 0.05], index=idx)
        result = compute_market_adjusted_returns(stock, market)
        expected = stock - market
        pd.testing.assert_series_equal(result, expected)

    def test_alignment_on_intersection(self):
        idx1 = pd.bdate_range("2020-01-02", periods=10)
        idx2 = pd.bdate_range("2020-01-06", periods=10)  # overlapping
        stock = pd.Series(np.ones(10), index=idx1)
        market = pd.Series(np.zeros(10), index=idx2)
        result = compute_market_adjusted_returns(stock, market)
        # Only intersection dates should appear
        assert len(result) == len(idx1.intersection(idx2))


# ---------------------------------------------------------------------------
# build_company_year_grid
# ---------------------------------------------------------------------------


class TestBuildCompanyYearGrid:
    def test_output_columns(self):
        univ = _make_universe(2)
        grid = build_company_year_grid(univ)
        for col in ("cik", "ticker", "sector", "fiscal_year", "period_end", "filing_date"):
            assert col in grid.columns

    def test_row_count(self):
        univ = _make_universe()  # filing_years: [2020,2021], [2020], [2021]
        grid = build_company_year_grid(univ)
        assert len(grid) == 4  # 2 + 1 + 1

    def test_period_end_uses_fiscal_year_end(self):
        univ = pd.DataFrame(
            {
                "cik": ["0000000001"],
                "ticker": ["AAA"],
                "sector": ["Finance"],
                "fiscal_year_end": ["0630"],
                "filing_years": [[2021]],
            }
        )
        grid = build_company_year_grid(univ)
        assert grid.iloc[0]["period_end"] == pd.Timestamp("2021-06-30")

    def test_filing_date_is_90_days_after_period_end(self):
        univ = pd.DataFrame(
            {
                "cik": ["0000000001"],
                "ticker": ["AAA"],
                "sector": ["Finance"],
                "fiscal_year_end": ["1231"],
                "filing_years": [[2021]],
            }
        )
        grid = build_company_year_grid(univ)
        period_end = grid.iloc[0]["period_end"]
        filing_date = grid.iloc[0]["filing_date"]
        assert filing_date == period_end + pd.Timedelta(days=90)

    def test_missing_fiscal_year_end_defaults_to_dec31(self):
        univ = pd.DataFrame(
            {
                "cik": ["0000000001"],
                "ticker": ["AAA"],
                "sector": ["Finance"],
                "fiscal_year_end": [None],
                "filing_years": [[2020]],
            }
        )
        grid = build_company_year_grid(univ)
        assert grid.iloc[0]["period_end"].month == 12
        assert grid.iloc[0]["period_end"].day == 31

    def test_missing_required_column_raises(self):
        bad = pd.DataFrame({"cik": ["001"], "ticker": ["A"]})
        with pytest.raises(ValueError, match="missing required columns"):
            build_company_year_grid(bad)


# ---------------------------------------------------------------------------
# align_to_filing_dates
# ---------------------------------------------------------------------------


class TestAlignToFilingDates:
    def _make_filings(
        self,
        cik: str = "0000000001",
        ticker: str = "AAPL",
        sector: str = "Business Equipment",
        period_end: str = "2021-12-31",
        filing_date: str = "2022-03-01",
    ) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "cik": [cik],
                "ticker": [ticker],
                "sector": [sector],
                "period_end": [pd.Timestamp(period_end)],
                "filing_date": [pd.Timestamp(filing_date)],
            }
        )

    def _make_index_prices(self, start="2020-01-02", periods=800) -> dict[str, pd.DataFrame]:
        """Return a minimal index_prices dict for all required tickers."""
        p = _make_prices(start=start, periods=periods)
        return {t: p.copy() for t in INDEX_TICKERS}

    def test_output_columns_present(self):
        filings = self._make_filings()
        prices = {"AAPL": _make_prices(periods=600)}
        index_prices = self._make_index_prices()
        result = align_to_filing_dates(filings, prices, index_prices)
        for col in ("cik", "period_end", "filing_date", "ticker", "sector",
                    "fwd_ret_252d", "mkt_adj_252d", "sec_adj_252d",
                    "volume_avg_63d", "delisted"):
            assert col in result.columns, f"Missing column: {col}"

    def test_missing_ticker_skipped(self):
        filings = pd.DataFrame(
            {
                "cik": ["001"],
                "ticker": [None],
                "sector": ["Finance"],
                "period_end": [pd.Timestamp("2021-12-31")],
                "filing_date": [pd.Timestamp("2022-03-01")],
            }
        )
        result = align_to_filing_dates(filings, {}, {})
        assert len(result) == 0

    def test_delisted_flag_for_unknown_ticker(self):
        filings = self._make_filings(ticker="ZZZZZ")
        result = align_to_filing_dates(filings, prices_dict={}, index_prices={})
        assert bool(result.iloc[0]["delisted"]) is True
        assert pd.isna(result.iloc[0]["fwd_ret_252d"])

    def test_delisted_when_insufficient_data_after_filing(self):
        """Only 10 days of data after filing_date → delisted=True."""
        filings = self._make_filings(filing_date="2021-12-15")
        # 20 days of prices total, filing date near the end → < 252 remaining
        prices = {"AAPL": _make_prices(start="2021-12-01", periods=20)}
        index_prices = self._make_index_prices()
        result = align_to_filing_dates(
            filings, prices, index_prices, primary_horizon=252
        )
        assert bool(result.iloc[0]["delisted"]) is True

    def test_not_delisted_with_sufficient_data(self):
        filings = self._make_filings(filing_date="2020-03-01")
        prices = {"AAPL": _make_prices(start="2019-01-02", periods=1000)}
        index_prices = self._make_index_prices(periods=1500)
        result = align_to_filing_dates(
            filings, prices, index_prices, primary_horizon=252
        )
        assert bool(result.iloc[0]["delisted"]) is False

    def test_fwd_ret_is_log_return(self):
        """fwd_ret_21d should equal log(P[ref+21] / P[ref])."""
        filing_date = pd.Timestamp("2021-01-04")
        # Simple price series: 1, 2, 3, ... (linear)
        dates = pd.bdate_range("2021-01-04", periods=400)
        close = np.arange(1, 401, dtype=float)
        prices_df = pd.DataFrame({"Close": close, "Volume": 1e6}, index=dates)

        filings = self._make_filings(
            ticker="FAKE",
            filing_date=filing_date.strftime("%Y-%m-%d"),
            period_end="2020-12-31",
        )
        result = align_to_filing_dates(
            filings,
            {"FAKE": prices_df},
            self._make_index_prices(),
            horizons_days=[21],
        )
        expected = np.log(22.0 / 1.0)  # log(P[21] / P[0]) = log(22/1)
        assert abs(result.iloc[0]["fwd_ret_21d"] - expected) < 1e-8

    def test_mkt_adj_equals_stock_minus_market(self):
        """mkt_adj_21d should be exactly fwd_ret_21d − market fwd_ret_21d."""
        filing_date = pd.Timestamp("2021-01-04")
        dates = pd.bdate_range("2021-01-04", periods=400)
        close = np.arange(1, 401, dtype=float)
        prices_df = pd.DataFrame({"Close": close, "Volume": 1e6}, index=dates)

        filings = self._make_filings(
            ticker="FAKE",
            sector="Finance",  # maps to XLF
            filing_date=filing_date.strftime("%Y-%m-%d"),
        )
        # Use same prices for market/sector index so stock and market move identically
        idx_prices = {t: prices_df.copy() for t in INDEX_TICKERS}

        result = align_to_filing_dates(
            filings,
            {"FAKE": prices_df},
            idx_prices,
            horizons_days=[21],
        )
        # Stock and market are identical → excess return ≈ 0
        assert abs(result.iloc[0]["mkt_adj_21d"]) < 1e-8

    def test_volume_avg_63d(self):
        filing_date = pd.Timestamp("2021-06-01")
        dates = pd.bdate_range("2020-01-02", periods=400)
        close = np.ones(400)
        volume = np.full(400, 500_000.0)
        prices_df = pd.DataFrame({"Close": close, "Volume": volume}, index=dates)

        filings = self._make_filings(
            ticker="FAKE", filing_date="2021-06-01"
        )
        result = align_to_filing_dates(
            filings,
            {"FAKE": prices_df},
            self._make_index_prices(),
            horizons_days=[21],
        )
        assert abs(result.iloc[0]["volume_avg_63d"] - 500_000.0) < 1.0

    def test_missing_required_column_raises(self):
        bad = pd.DataFrame({"cik": ["001"]})
        with pytest.raises(ValueError, match="missing required columns"):
            align_to_filing_dates(bad, {}, {})

    def test_xlc_fallback_before_2018(self):
        """Telecom sector before XLC's inception should not crash."""
        filings = self._make_filings(
            ticker="T",
            sector="Telecom",
            period_end="2015-12-31",
            filing_date="2016-03-01",
        )
        prices = {"T": _make_prices(start="2013-01-02", periods=1000)}
        index_prices = self._make_index_prices(start="2013-01-02", periods=2000)
        result = align_to_filing_dates(filings, prices, index_prices)
        # Should complete without error; mkt_adj_252d should be finite or NaN (no crash)
        assert len(result) == 1


# ---------------------------------------------------------------------------
# build_market_dataset (integration / smoke test)
# ---------------------------------------------------------------------------


class TestBuildMarketDataset:
    def test_smoke_test_runs_end_to_end(self, tmp_path):
        """Verify the full pipeline runs with mocked yfinance."""
        universe = _make_universe(2)
        prices = _make_prices(start="2019-01-02", periods=1200)

        def mock_download(tickers, start, end):
            return {t: prices.copy() for t in tickers}

        mock_ticker = MagicMock()
        mock_ticker.actions = pd.DataFrame(
            {"Dividends": [0.0], "Stock Splits": [0.0]},
            index=pd.DatetimeIndex(["2020-06-01"]),
        )

        with (
            patch("fin_jepa.data.market_data._yf_download", side_effect=mock_download),
            patch("fin_jepa.data.market_data._yf") as mock_yf,
        ):
            mock_yf.Ticker.return_value = mock_ticker
            cfg = MarketDataConfig(
                start_date="2019-01-01",
                end_date="2022-12-31",
                sleep_between_batches=0.0,
            )
            result = build_market_dataset(universe, tmp_path / "raw", config=cfg)

        assert len(result) > 0
        assert "fwd_ret_252d" in result.columns
        assert "mkt_adj_252d" in result.columns
        assert "delisted" in result.columns
        # Parquet output should be written
        assert (tmp_path / "raw" / "market" / "market_aligned.parquet").exists()

    def test_custom_filings_df_respected(self, tmp_path):
        """Providing a custom filings_df bypasses build_company_year_grid."""
        filings = pd.DataFrame(
            {
                "cik": ["0000000001"],
                "ticker": ["AAPL"],
                "sector": ["Business Equipment"],
                "period_end": [pd.Timestamp("2021-12-31")],
                "filing_date": [pd.Timestamp("2022-03-01")],
            }
        )
        prices = _make_prices(start="2020-01-02", periods=1200)

        with (
            patch("fin_jepa.data.market_data._yf_download") as mock_dl,
            patch("fin_jepa.data.market_data._yf") as mock_yf,
        ):
            mock_dl.return_value = {"AAPL": prices}
            mock_yf.Ticker.return_value = MagicMock(
                actions=pd.DataFrame(
                    {"Dividends": [0.0], "Stock Splits": [0.0]},
                    index=pd.DatetimeIndex(["2021-01-01"]),
                )
            )
            universe = _make_universe(1)
            result = build_market_dataset(
                universe,
                tmp_path / "raw",
                filings_df=filings,
                config=MarketDataConfig(sleep_between_batches=0.0),
            )

        assert len(result) == 1
        assert result.iloc[0]["ticker"] == "AAPL"

    def test_short_interest_join(self, tmp_path):
        filings = pd.DataFrame(
            {
                "cik": ["0000000001"],
                "ticker": ["AAPL"],
                "sector": ["Business Equipment"],
                "period_end": [pd.Timestamp("2021-12-31")],
                "filing_date": [pd.Timestamp("2022-03-01")],
            }
        )
        prices = _make_prices(start="2020-01-02", periods=1200)
        si = pd.DataFrame(
            {
                "ticker": ["AAPL"],
                "date": [pd.Timestamp("2022-03-01")],
                "short_interest_ratio": [0.05],
            }
        )

        with (
            patch("fin_jepa.data.market_data._yf_download") as mock_dl,
            patch("fin_jepa.data.market_data._yf") as mock_yf,
        ):
            mock_dl.return_value = {"AAPL": prices}
            mock_yf.Ticker.return_value = MagicMock(
                actions=pd.DataFrame(
                    {"Dividends": [0.0], "Stock Splits": [0.0]},
                    index=pd.DatetimeIndex(["2021-01-01"]),
                )
            )
            universe = _make_universe(1)
            result = build_market_dataset(
                universe,
                tmp_path / "raw",
                filings_df=filings,
                short_interest_df=si,
                config=MarketDataConfig(sleep_between_batches=0.0),
            )

        assert "short_interest_ratio" in result.columns
        assert abs(result.iloc[0]["short_interest_ratio"] - 0.05) < 1e-9


# ---------------------------------------------------------------------------
# fetch_prices (legacy API wrapper)
# ---------------------------------------------------------------------------


class TestFetchPrices:
    def test_returns_wide_close_dataframe(self):
        prices = _make_prices(periods=100)
        with patch("fin_jepa.data.market_data._yf_download") as mock_dl:
            mock_dl.return_value = {"AAPL": prices}
            result = fetch_prices(["AAPL"], "2020-01-01", "2021-01-01")

        assert "AAPL" in result.columns
        assert not result.empty

    def test_empty_result_for_no_data(self):
        with patch("fin_jepa.data.market_data._yf_download") as mock_dl:
            mock_dl.return_value = {"ZZZ": pd.DataFrame()}
            result = fetch_prices(["ZZZ"], "2020-01-01", "2021-01-01")

        assert result.empty


# ---------------------------------------------------------------------------
# Constants / mapping integrity
# ---------------------------------------------------------------------------


class TestConstants:
    def test_market_index_in_index_tickers(self):
        assert MARKET_INDEX_TICKER in INDEX_TICKERS

    def test_all_ff12_etfs_in_index_tickers(self):
        for etf in FF12_TO_ETF.values():
            assert etf in INDEX_TICKERS

    def test_return_horizons_sorted(self):
        assert RETURN_HORIZONS == sorted(RETURN_HORIZONS)

    def test_252_in_return_horizons(self):
        """Primary horizon for stock_decline must be present."""
        assert 252 in RETURN_HORIZONS

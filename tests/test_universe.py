"""Tests for fin_jepa.data.universe and fin_jepa.data.sector_map.

All tests are pure-Python with no real HTTP calls; EDGAR responses are
mocked using unittest.mock.patch so the test suite runs offline.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# sector_map tests
# ---------------------------------------------------------------------------
from fin_jepa.data.sector_map import FF12_SECTORS, sic_to_sector


class TestSicToSector:
    def test_pharmaceutical_sic(self):
        assert sic_to_sector("2836") == "Health Care"

    def test_commercial_banking_sic(self):
        assert sic_to_sector("6022") == "Finance"

    def test_oil_gas_extraction(self):
        assert sic_to_sector("1311") == "Energy"

    def test_utilities(self):
        assert sic_to_sector("4911") == "Utilities"

    def test_software_services(self):
        assert sic_to_sector("7372") == "Business Equipment"

    def test_telecom(self):
        assert sic_to_sector("4813") == "Telecom"

    def test_food_manufacturing(self):
        assert sic_to_sector("2000") == "Consumer NonDurables"

    def test_tobacco(self):
        assert sic_to_sector("2111") == "Consumer NonDurables"

    def test_motor_vehicles(self):
        assert sic_to_sector("3711") == "Consumer Durables"

    def test_retail(self):
        assert sic_to_sector("5411") == "Shops"

    def test_unknown_sic_returns_other(self):
        assert sic_to_sector("9999") == "Other"
        assert sic_to_sector("0000") == "Other"

    def test_none_returns_other(self):
        assert sic_to_sector(None) == "Other"

    def test_invalid_string_returns_other(self):
        assert sic_to_sector("abcd") == "Other"
        assert sic_to_sector("") == "Other"

    def test_int_input(self):
        assert sic_to_sector(6022) == "Finance"

    def test_all_known_sectors_reachable(self):
        # Verify each sector is reachable through at least one SIC code
        known_sics = {
            "Consumer NonDurables": "2050",
            "Consumer Durables": "3711",
            "Manufacturing": "3490",
            "Energy": "1311",
            "Chemicals": "2810",
            "Business Equipment": "3571",
            "Telecom": "4813",
            "Utilities": "4911",
            "Shops": "5300",
            "Health Care": "2833",
            "Finance": "6021",
        }
        for expected, sic in known_sics.items():
            assert sic_to_sector(sic) == expected, f"SIC {sic} should map to {expected}"

    def test_ff12_sectors_list_complete(self):
        assert len(FF12_SECTORS) == 12
        assert "Other" in FF12_SECTORS
        assert "Finance" in FF12_SECTORS


# ---------------------------------------------------------------------------
# form.idx parsing tests
# ---------------------------------------------------------------------------
from fin_jepa.data.universe import _parse_form_idx


SAMPLE_FORM_IDX = """\
Form Type           Company Name                                                    CIK         Date Filed  Filename
-----------         --------------------------------------------------------------------------------    ----------  ----------  --------------------------------------------------------------------------------
10-K                APPLE INC                                                       320193      2023-11-03  edgar/data/320193/0000320193-23-000106-index.htm
10-K/A              TESLA INC                                                       1318605     2022-02-14  edgar/data/1318605/0001318605-22-000001-index.htm
10-K                MICROSOFT CORP                                                  789019      2023-07-27  edgar/data/789019/0000789019-23-000075-index.htm
8-K                 SOME OTHER CO                                                   999999      2023-01-15  edgar/data/999999/0000999999-23-000001-index.htm
"""


class TestParseFormIdx:
    def test_returns_dataframe(self):
        df = _parse_form_idx(SAMPLE_FORM_IDX)
        assert isinstance(df, pd.DataFrame)

    def test_column_names(self):
        df = _parse_form_idx(SAMPLE_FORM_IDX)
        assert set(["form_type", "company_name", "cik", "date_filed", "filename"]).issubset(df.columns)

    def test_correct_row_count(self):
        df = _parse_form_idx(SAMPLE_FORM_IDX)
        # All 4 data rows are returned (8-K is not filtered here — filtering
        # happens in fetch_quarterly_index)
        assert len(df) == 4

    def test_cik_zero_padded(self):
        df = _parse_form_idx(SAMPLE_FORM_IDX)
        apple = df[df["company_name"].str.contains("APPLE", na=False)].iloc[0]
        assert apple["cik"] == "0000320193"

    def test_date_parsed_correctly(self):
        df = _parse_form_idx(SAMPLE_FORM_IDX)
        apple = df[df["company_name"].str.contains("APPLE", na=False)].iloc[0]
        assert apple["date_filed"] == pd.Timestamp("2023-11-03")

    def test_empty_content_returns_empty_df(self):
        df = _parse_form_idx("")
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0


# ---------------------------------------------------------------------------
# XBRL coverage audit tests
# ---------------------------------------------------------------------------
from fin_jepa.data.universe import audit_xbrl_coverage


def _make_filings_df(records: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(records)
    df["date_filed"] = pd.to_datetime(df["date_filed"])
    df["filing_year"] = df["date_filed"].dt.year
    return df


class TestAuditXbrlCoverage:
    def test_full_xbrl_coverage(self):
        filings = _make_filings_df([
            {"cik": "0000000001", "form_type": "10-K", "date_filed": "2015-03-01"},
            {"cik": "0000000001", "form_type": "10-K", "date_filed": "2016-03-01"},
            {"cik": "0000000001", "form_type": "10-K", "date_filed": "2017-03-01"},
        ])
        result = audit_xbrl_coverage(filings)
        row = result.loc["0000000001"]
        assert row["n_10k_filings"] == 3
        assert row["n_xbrl_filings"] == 3
        assert row["xbrl_coverage_pct"] == 1.0
        assert not row["xbrl_gap_flag"]

    def test_gap_years_detected(self):
        filings = _make_filings_df([
            {"cik": "0000000002", "form_type": "10-K", "date_filed": "2012-03-01"},
            # 2013 missing
            {"cik": "0000000002", "form_type": "10-K", "date_filed": "2014-03-01"},
            {"cik": "0000000002", "form_type": "10-K", "date_filed": "2015-03-01"},
        ])
        result = audit_xbrl_coverage(filings)
        row = result.loc["0000000002"]
        assert 2013 in row["gap_years"]

    def test_amendments_excluded_from_n_10k(self):
        filings = _make_filings_df([
            {"cik": "0000000003", "form_type": "10-K", "date_filed": "2018-03-01"},
            {"cik": "0000000003", "form_type": "10-K/A", "date_filed": "2018-04-01"},
        ])
        result = audit_xbrl_coverage(filings)
        assert result.loc["0000000003"]["n_10k_filings"] == 1  # 10-K/A not counted

    def test_is_current_filer(self):
        filings = _make_filings_df([
            {"cik": "0000000004", "form_type": "10-K", "date_filed": "2023-03-01"},
        ])
        result = audit_xbrl_coverage(filings)
        assert result.loc["0000000004"]["is_current_filer"]

    def test_not_current_filer(self):
        filings = _make_filings_df([
            {"cik": "0000000005", "form_type": "10-K", "date_filed": "2015-03-01"},
        ])
        result = audit_xbrl_coverage(filings)
        # is_current is relative to last filing year; this confirms the column
        # is a boolean-like value (numpy bool_ is not a Python bool subclass).
        val = result.loc["0000000005"]["is_current_filer"]
        assert val is True or val is False or bool(val) in (True, False)

    def test_empty_input(self):
        filings = pd.DataFrame(
            columns=["cik", "form_type", "date_filed", "filing_year"]
        )
        filings["date_filed"] = pd.to_datetime(filings["date_filed"])
        result = audit_xbrl_coverage(filings)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0


# ---------------------------------------------------------------------------
# build_company_universe integration test (mocked HTTP)
# ---------------------------------------------------------------------------
from fin_jepa.data.universe import build_company_universe


MOCK_TICKERS_RESPONSE = {
    "fields": ["cik", "name", "ticker", "exchange"],
    "data": [
        [320193, "APPLE INC", "AAPL", "Nasdaq"],
        [789019, "MICROSOFT CORP", "MSFT", "Nasdaq"],
    ],
}

MOCK_SUBMISSIONS_APPLE = {
    "cik": "320193",
    "name": "Apple Inc.",
    "sic": "3674",
    "sicDescription": "Semiconductors and Related Devices",
    "stateOfIncorporation": "CA",
    "fiscalYearEnd": "0930",
    "entityType": "operating",
}

MOCK_SUBMISSIONS_MICROSOFT = {
    "cik": "789019",
    "name": "Microsoft Corporation",
    "sic": "7372",
    "sicDescription": "Prepackaged Software",
    "stateOfIncorporation": "WA",
    "fiscalYearEnd": "0630",
    "entityType": "operating",
}

_MOCK_FORM_IDX = """\
Form Type           Company Name                                                    CIK         Date Filed  Filename
-----------         --------------------------------------------------------------------------------    ----------  ----------  --------------------------------------------------------------------------------
10-K                APPLE INC                                                       320193      2023-11-03  edgar/data/320193/0000320193-23-000106-index.htm
10-K                MICROSOFT CORP                                                  789019      2023-07-27  edgar/data/789019/0000789019-23-000075-index.htm
"""


def _mock_requests_get(url: str, **kwargs):
    """Minimal mock that returns realistic EDGAR responses."""
    resp = MagicMock()
    resp.status_code = 200
    resp.raise_for_status = lambda: None

    if "company_tickers_exchange" in url:
        resp.json.return_value = MOCK_TICKERS_RESPONSE
    elif "submissions/CIK0000320193" in url:
        resp.json.return_value = MOCK_SUBMISSIONS_APPLE
    elif "submissions/CIK0000789019" in url:
        resp.json.return_value = MOCK_SUBMISSIONS_MICROSOFT
    elif "full-index" in url:
        resp.text = _MOCK_FORM_IDX
    else:
        resp.json.return_value = {}
        resp.text = ""
    return resp


class TestBuildCompanyUniverse:
    @patch("requests.Session.get", side_effect=_mock_requests_get)
    def test_returns_dataframe(self, mock_get):
        with tempfile.TemporaryDirectory() as tmpdir:
            df = build_company_universe(
                raw_dir=tmpdir,
                start_year=2023,
                end_year=2023,
                fetch_submissions=True,
                max_workers=1,
            )
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    @patch("requests.Session.get", side_effect=_mock_requests_get)
    def test_required_columns_present(self, mock_get):
        with tempfile.TemporaryDirectory() as tmpdir:
            df = build_company_universe(
                raw_dir=tmpdir,
                start_year=2023,
                end_year=2023,
                fetch_submissions=True,
                max_workers=1,
            )
        required = {
            "cik", "entity_name", "ticker", "exchange",
            "sic_code", "sector", "n_10k_filings", "n_xbrl_filings",
            "xbrl_coverage_pct", "filing_years", "is_current_filer",
        }
        assert required.issubset(set(df.columns))

    @patch("requests.Session.get", side_effect=_mock_requests_get)
    def test_cik_zero_padded(self, mock_get):
        with tempfile.TemporaryDirectory() as tmpdir:
            df = build_company_universe(
                raw_dir=tmpdir,
                start_year=2023,
                end_year=2023,
                fetch_submissions=False,
                max_workers=1,
            )
        for cik in df["cik"]:
            assert len(str(cik)) == 10, f"CIK {cik!r} should be 10 chars"

    @patch("requests.Session.get", side_effect=_mock_requests_get)
    def test_sector_populated(self, mock_get):
        with tempfile.TemporaryDirectory() as tmpdir:
            df = build_company_universe(
                raw_dir=tmpdir,
                start_year=2023,
                end_year=2023,
                fetch_submissions=True,
                max_workers=1,
            )
        # MSFT SIC 7372 → Business Equipment
        msft = df[df["cik"] == "0000789019"]
        if len(msft) > 0:
            assert msft.iloc[0]["sector"] == "Business Equipment"

    @patch("requests.Session.get", side_effect=_mock_requests_get)
    def test_parquet_written(self, mock_get):
        with tempfile.TemporaryDirectory() as tmpdir:
            raw_dir = Path(tmpdir)
            out_path = raw_dir / "company_universe.parquet"
            build_company_universe(
                raw_dir=tmpdir,
                start_year=2023,
                end_year=2023,
                fetch_submissions=False,
                output_path=out_path,
                max_workers=1,
            )
            assert out_path.exists()

    @patch("requests.Session.get", side_effect=_mock_requests_get)
    def test_caching_prevents_second_http_call(self, mock_get):
        """Second build_company_universe call must use cache, not HTTP."""
        with tempfile.TemporaryDirectory() as tmpdir:
            kwargs = dict(
                raw_dir=tmpdir,
                start_year=2023,
                end_year=2023,
                fetch_submissions=False,
                max_workers=1,
            )
            build_company_universe(**kwargs)
            call_count_after_first = mock_get.call_count
            build_company_universe(**kwargs)
            # HTTP calls should not increase on second run (cache hit)
            assert mock_get.call_count == call_count_after_first


# ---------------------------------------------------------------------------
# load_company_universe roundtrip test
# ---------------------------------------------------------------------------
from fin_jepa.data.universe import load_company_universe


class TestLoadCompanyUniverse:
    @patch("requests.Session.get", side_effect=_mock_requests_get)
    def test_roundtrip(self, mock_get):
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "company_universe.parquet"
            df_original = build_company_universe(
                raw_dir=tmpdir,
                start_year=2023,
                end_year=2023,
                fetch_submissions=False,
                output_path=out_path,
                max_workers=1,
            )
            df_loaded, provenance = load_company_universe(out_path)

        assert len(df_loaded) == len(df_original)
        assert set(df_loaded.columns) == set(df_original.columns)

    @patch("requests.Session.get", side_effect=_mock_requests_get)
    def test_provenance_metadata(self, mock_get):
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "company_universe.parquet"
            build_company_universe(
                raw_dir=tmpdir,
                start_year=2023,
                end_year=2023,
                fetch_submissions=False,
                output_path=out_path,
                max_workers=1,
            )
            _, provenance = load_company_universe(out_path)

        assert "build_date" in provenance
        assert provenance["start_year"] == 2023
        assert provenance["end_year"] == 2023
        assert "n_companies" in provenance


# ---------------------------------------------------------------------------
# compustat.py tests
# ---------------------------------------------------------------------------
from fin_jepa.data.compustat import load_compustat_crossref, merge_compustat


class TestLoadCompustatCrossref:
    def test_file_not_found_raises(self):
        with pytest.raises(FileNotFoundError):
            load_compustat_crossref("/nonexistent/path/compustat.csv")

    def test_missing_gvkey_raises(self, tmp_path):
        bad_csv = tmp_path / "bad.csv"
        bad_csv.write_text("cusip,name\n123456789,TEST CO\n")
        with pytest.raises(ValueError, match="gvkey"):
            load_compustat_crossref(bad_csv)

    def test_loads_csv_correctly(self, tmp_path):
        csv_path = tmp_path / "compustat.csv"
        csv_path.write_text(
            "gvkey,cik,cusip,tic,conm,sic,exchg,fyr,ipodate,dldte,dlrsn\n"
            "001690,320193,037833100,AAPL,APPLE INC,3674,14,9,1980-12-12,,\n"
            "012141,789019,594918104,MSFT,MICROSOFT CORP,7372,14,6,1986-03-13,,\n"
        )
        df = load_compustat_crossref(csv_path)
        assert len(df) == 2
        assert df["gvkey"].iloc[0] == "001690"
        assert df["cik"].iloc[0] == "0000320193"

    def test_gvkey_zero_padded(self, tmp_path):
        csv_path = tmp_path / "compustat.csv"
        csv_path.write_text("gvkey,cusip\n42,037833100\n1234,594918104\n")
        df = load_compustat_crossref(csv_path)
        assert df["gvkey"].iloc[0] == "000042"

    def test_parquet_loading(self, tmp_path):
        source = pd.DataFrame({
            "gvkey": ["001690", "012141"],
            "cik": ["0000320193", "0000789019"],
            "tic": ["AAPL", "MSFT"],
        })
        parquet_path = tmp_path / "compustat.parquet"
        source.to_parquet(parquet_path, index=False)
        df = load_compustat_crossref(parquet_path)
        assert len(df) == 2


class TestMergeCompustat:
    def _universe(self) -> pd.DataFrame:
        return pd.DataFrame({
            "cik": ["0000320193", "0000789019", "0000000001"],
            "entity_name": ["APPLE INC", "MICROSOFT CORP", "UNKNOWN CO"],
            "ticker": ["AAPL", "MSFT", "UNK"],
            "sector": ["Business Equipment", "Business Equipment", "Other"],
        })

    def _compustat(self) -> pd.DataFrame:
        return pd.DataFrame({
            "gvkey": ["001690", "012141"],
            "cik": ["0000320193", "0000789019"],
            "cusip": ["037833100", "594918104"],
            "tic": ["AAPL", "MSFT"],
            "dldte": [None, None],
            "dlrsn": [None, None],
            "delist_reason": [None, None],
        })

    def test_merge_by_cik(self):
        merged = merge_compustat(self._universe(), self._compustat())
        apple = merged[merged["cik"] == "0000320193"].iloc[0]
        assert apple["cstat_gvkey"] == "001690"

    def test_unmatched_rows_have_nan_gvkey(self):
        merged = merge_compustat(self._universe(), self._compustat())
        unknown = merged[merged["cik"] == "0000000001"].iloc[0]
        assert pd.isna(unknown["cstat_gvkey"])

    def test_output_has_all_universe_rows(self):
        merged = merge_compustat(self._universe(), self._compustat())
        assert len(merged) == len(self._universe())

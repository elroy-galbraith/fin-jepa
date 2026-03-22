"""Tests for fin_jepa.data.xbrl_pipeline.

All tests are pure-Python with no real HTTP calls; EDGAR responses are
mocked using unittest.mock.patch so the test suite runs offline.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from fin_jepa.data.xbrl_pipeline import (
    FEATURE_NAMES,
    XBRL_FEATURE_SCHEMA,
    XBRLConfig,
    extract_annual_facts,
    fetch_company_facts,
    load_xbrl_features,
    validate_xbrl_dataset,
)


# ---------------------------------------------------------------------------
# Test fixtures: synthetic Company Facts API responses
# ---------------------------------------------------------------------------

def _make_entry(fy: int, val: float, form: str = "10-K", fp: str = "FY",
                end: str | None = None, filed: str | None = None) -> dict:
    """Build a single XBRL fact entry."""
    if end is None:
        end = f"{fy}-12-31"
    if filed is None:
        filed = f"{fy + 1}-02-15"
    return {
        "end": end, "val": val, "accn": f"0000000000-{fy}-000001",
        "fy": fy, "fp": fp, "form": form, "filed": filed,
    }


def _make_facts_json(
    cik: int = 320193,
    entity_name: str = "Test Corp",
    concepts: dict[str, list[dict]] | None = None,
) -> dict:
    """Build a synthetic Company Facts JSON response."""
    if concepts is None:
        concepts = {
            "Assets": [
                _make_entry(2020, 100_000_000),
                _make_entry(2021, 120_000_000),
            ],
            "Liabilities": [
                _make_entry(2020, 60_000_000),
                _make_entry(2021, 70_000_000),
            ],
            "StockholdersEquity": [
                _make_entry(2020, 40_000_000),
                _make_entry(2021, 50_000_000),
            ],
            "Revenues": [
                _make_entry(2020, 200_000_000),
                _make_entry(2021, 250_000_000),
            ],
            "NetIncomeLoss": [
                _make_entry(2020, 30_000_000),
                _make_entry(2021, 35_000_000),
            ],
            "NetCashProvidedByUsedInOperatingActivities": [
                _make_entry(2020, 45_000_000),
                _make_entry(2021, 50_000_000),
            ],
        }
    us_gaap = {}
    for concept_name, entries in concepts.items():
        us_gaap[concept_name] = {
            "label": concept_name,
            "description": f"Test {concept_name}",
            "units": {"USD": entries},
        }
    return {
        "cik": cik,
        "entityName": entity_name,
        "facts": {"us-gaap": us_gaap},
    }


# ---------------------------------------------------------------------------
# Schema tests
# ---------------------------------------------------------------------------

class TestXBRLFeatureSchema:
    def test_all_features_have_tags(self):
        for name, spec in XBRL_FEATURE_SCHEMA.items():
            if isinstance(spec, list):
                assert len(spec) >= 1, f"{name} has no tags"
            elif isinstance(spec, dict):
                assert spec.get("type") == "sum", f"{name} has invalid computed spec"
                assert len(spec.get("components", [])) >= 1

    def test_feature_names_match_schema(self):
        assert FEATURE_NAMES == list(XBRL_FEATURE_SCHEMA.keys())

    def test_expected_feature_count(self):
        assert len(FEATURE_NAMES) == 16


class TestXBRLConfig:
    def test_defaults(self):
        cfg = XBRLConfig()
        assert cfg.start_year == 2012
        assert cfg.end_year == 2024
        assert cfg.rate_limit_per_sec > 0

    def test_invalid_rate_limit(self):
        with pytest.raises(ValueError):
            XBRLConfig(rate_limit_per_sec=0)
        with pytest.raises(ValueError):
            XBRLConfig(rate_limit_per_sec=-1)


# ---------------------------------------------------------------------------
# extract_annual_facts tests
# ---------------------------------------------------------------------------

class TestExtractAnnualFacts:
    def test_basic_extraction(self):
        facts = _make_facts_json()
        df = extract_annual_facts(facts, "0000320193")

        assert len(df) == 2
        assert list(df["fiscal_year"]) == [2020, 2021]
        assert df["cik"].iloc[0] == "0000320193"

    def test_feature_values(self):
        facts = _make_facts_json()
        df = extract_annual_facts(facts, "0000320193")

        row_2020 = df[df["fiscal_year"] == 2020].iloc[0]
        assert row_2020["total_assets"] == 100_000_000
        assert row_2020["total_liabilities"] == 60_000_000
        assert row_2020["total_equity"] == 40_000_000
        assert row_2020["total_revenue"] == 200_000_000
        assert row_2020["net_income"] == 30_000_000
        assert row_2020["cash_from_operations"] == 45_000_000

    def test_missing_features_are_nan(self):
        facts = _make_facts_json()
        df = extract_annual_facts(facts, "0000320193")

        # interest_expense not in our test data → should be NaN
        row = df.iloc[0]
        assert pd.isna(row["interest_expense"])
        assert pd.isna(row["current_assets"])

    def test_fallback_tag_resolution(self):
        """When primary tag is missing, fallback tag should be used."""
        concepts = {
            "Assets": [_make_entry(2020, 100_000_000)],
            # Use fallback tag for equity
            "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest": [
                _make_entry(2020, 55_000_000),
            ],
            # Use fallback tag for revenue
            "RevenueFromContractWithCustomerExcludingAssessedTax": [
                _make_entry(2020, 180_000_000),
            ],
            "NetIncomeLoss": [_make_entry(2020, 20_000_000)],
        }
        facts = _make_facts_json(concepts=concepts)
        df = extract_annual_facts(facts, "0000320193")

        row = df.iloc[0]
        assert row["total_equity"] == 55_000_000
        assert row["total_revenue"] == 180_000_000

    def test_primary_tag_takes_precedence_over_fallback(self):
        """When both primary and fallback tags exist, primary wins."""
        concepts = {
            "Assets": [_make_entry(2020, 100_000_000)],
            "StockholdersEquity": [_make_entry(2020, 40_000_000)],
            "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest": [
                _make_entry(2020, 45_000_000),
            ],
            "NetIncomeLoss": [_make_entry(2020, 20_000_000)],
        }
        facts = _make_facts_json(concepts=concepts)
        df = extract_annual_facts(facts, "0000320193")

        # Primary tag (StockholdersEquity=40M) should win over fallback (45M)
        assert df.iloc[0]["total_equity"] == 40_000_000

    def test_10q_data_excluded(self):
        """Only 10-K filings should be included, not 10-Q."""
        concepts = {
            "Assets": [
                _make_entry(2020, 100_000_000, form="10-K"),
                _make_entry(2020, 90_000_000, form="10-Q", fp="Q1"),
                _make_entry(2020, 95_000_000, form="10-Q", fp="Q2"),
            ],
            "NetIncomeLoss": [_make_entry(2020, 20_000_000)],
        }
        facts = _make_facts_json(concepts=concepts)
        df = extract_annual_facts(facts, "0000320193")

        assert len(df) == 1
        assert df.iloc[0]["total_assets"] == 100_000_000

    def test_deduplication_keeps_latest_filing(self):
        """When multiple 10-K entries exist for same FY, latest filed wins."""
        concepts = {
            "Assets": [
                _make_entry(2020, 100_000_000, filed="2021-02-15"),
                _make_entry(2020, 105_000_000, filed="2021-06-01"),  # amended
            ],
            "NetIncomeLoss": [_make_entry(2020, 20_000_000)],
        }
        facts = _make_facts_json(concepts=concepts)
        df = extract_annual_facts(facts, "0000320193")

        # Amended value (105M, filed later) should win
        assert df.iloc[0]["total_assets"] == 105_000_000

    def test_computed_feature_total_debt(self):
        """total_debt should be the sum of long-term + short-term debt."""
        concepts = {
            "Assets": [_make_entry(2020, 100_000_000)],
            "LongTermDebt": [_make_entry(2020, 30_000_000)],
            "ShortTermBorrowings": [_make_entry(2020, 5_000_000)],
            "NetIncomeLoss": [_make_entry(2020, 20_000_000)],
        }
        facts = _make_facts_json(concepts=concepts)
        df = extract_annual_facts(facts, "0000320193")

        assert df.iloc[0]["total_debt"] == 35_000_000

    def test_computed_feature_partial_components(self):
        """total_debt with only one component available."""
        concepts = {
            "Assets": [_make_entry(2020, 100_000_000)],
            "LongTermDebt": [_make_entry(2020, 30_000_000)],
            # No short-term debt
            "NetIncomeLoss": [_make_entry(2020, 20_000_000)],
        }
        facts = _make_facts_json(concepts=concepts)
        df = extract_annual_facts(facts, "0000320193")

        # Should still compute with available component
        assert df.iloc[0]["total_debt"] == 30_000_000

    def test_computed_feature_fallback_tags(self):
        """total_debt using fallback tags (LongTermDebtNoncurrent, DebtCurrent)."""
        concepts = {
            "Assets": [_make_entry(2020, 100_000_000)],
            "LongTermDebtNoncurrent": [_make_entry(2020, 25_000_000)],
            "DebtCurrent": [_make_entry(2020, 8_000_000)],
            "NetIncomeLoss": [_make_entry(2020, 20_000_000)],
        }
        facts = _make_facts_json(concepts=concepts)
        df = extract_annual_facts(facts, "0000320193")

        assert df.iloc[0]["total_debt"] == 33_000_000

    def test_empty_facts_returns_empty_df(self):
        df = extract_annual_facts({}, "0000000001")
        assert df.empty

    def test_no_usgaap_returns_empty_df(self):
        facts = {"facts": {"dei": {}}}
        df = extract_annual_facts(facts, "0000000001")
        assert df.empty

    def test_year_range_filtering(self):
        concepts = {
            "Assets": [
                _make_entry(2010, 50_000_000),
                _make_entry(2020, 100_000_000),
                _make_entry(2025, 150_000_000),
            ],
            "NetIncomeLoss": [
                _make_entry(2010, 10_000_000),
                _make_entry(2020, 20_000_000),
                _make_entry(2025, 30_000_000),
            ],
        }
        facts = _make_facts_json(concepts=concepts)
        config = XBRLConfig(start_year=2012, end_year=2024)
        df = extract_annual_facts(facts, "0000320193", config)

        # Only 2020 should be in range (2010 < start, 2025 > end)
        assert len(df) == 1
        assert df.iloc[0]["fiscal_year"] == 2020

    def test_period_end_and_filed_date(self):
        concepts = {
            "Assets": [
                _make_entry(2020, 100_000_000, end="2020-09-30", filed="2020-11-15"),
            ],
            "NetIncomeLoss": [_make_entry(2020, 20_000_000)],
        }
        facts = _make_facts_json(concepts=concepts)
        df = extract_annual_facts(facts, "0000320193")

        from datetime import date
        assert df.iloc[0]["period_end"] == date(2020, 9, 30)
        assert df.iloc[0]["filed_date"] == date(2020, 11, 15)

    def test_cik_zero_padding(self):
        facts = _make_facts_json()
        df = extract_annual_facts(facts, "320193")
        assert df.iloc[0]["cik"] == "0000320193"


# ---------------------------------------------------------------------------
# fetch_company_facts tests
# ---------------------------------------------------------------------------

class TestFetchCompanyFacts:
    def test_caching_prevents_second_fetch(self):
        """Second call should read from cache, not HTTP."""
        facts_data = _make_facts_json()

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)

            mock_session = MagicMock()
            mock_response = MagicMock()
            mock_response.json.return_value = facts_data
            mock_response.raise_for_status.return_value = None
            mock_session.get.return_value = mock_response

            # First call: should hit HTTP
            result1 = fetch_company_facts("320193", cache_dir, mock_session)
            assert mock_session.get.call_count == 1
            assert result1["entityName"] == "Test Corp"

            # Second call: should read from cache
            result2 = fetch_company_facts("320193", cache_dir, mock_session)
            assert mock_session.get.call_count == 1  # no additional call
            assert result2["entityName"] == "Test Corp"

    def test_cache_file_path(self):
        """Cache file should be at companyfacts/{cik}.json."""
        facts_data = _make_facts_json()

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)

            mock_session = MagicMock()
            mock_response = MagicMock()
            mock_response.json.return_value = facts_data
            mock_response.raise_for_status.return_value = None
            mock_session.get.return_value = mock_response

            fetch_company_facts("320193", cache_dir, mock_session)

            cache_path = cache_dir / "companyfacts" / "0000320193.json"
            assert cache_path.exists()

    def test_failed_fetch_returns_empty_dict(self):
        """HTTP error should return empty dict, not raise."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)

            mock_session = MagicMock()
            mock_session.get.side_effect = Exception("404 Not Found")

            result = fetch_company_facts("999999", cache_dir, mock_session)
            assert result == {}


# ---------------------------------------------------------------------------
# validate_xbrl_dataset tests
# ---------------------------------------------------------------------------

class TestValidateXBRLDataset:
    def test_basic_validation(self):
        facts = _make_facts_json()
        df = extract_annual_facts(facts, "0000320193")
        report = validate_xbrl_dataset(df)

        assert report["n_observations"] == 2
        assert report["n_companies"] == 1
        assert report["year_range"] == [2020, 2021]
        assert report["duplicate_count"] == 0

    def test_feature_coverage(self):
        facts = _make_facts_json()
        df = extract_annual_facts(facts, "0000320193")
        report = validate_xbrl_dataset(df)

        coverage = report["feature_coverage"]
        # total_assets should be 100% covered
        assert coverage["total_assets"] == 100.0
        # interest_expense should be 0% (not in test data)
        assert coverage["interest_expense"] == 0.0

    def test_detects_duplicates(self):
        facts = _make_facts_json()
        df = extract_annual_facts(facts, "0000320193")
        # Manually create a duplicate
        df = pd.concat([df, df.iloc[[0]]], ignore_index=True)
        report = validate_xbrl_dataset(df)
        assert report["duplicate_count"] == 1


# ---------------------------------------------------------------------------
# load_xbrl_features tests
# ---------------------------------------------------------------------------

class TestLoadXBRLFeatures:
    def test_parquet_roundtrip(self):
        """Write and read back a Parquet file with provenance."""
        facts = _make_facts_json()
        df = extract_annual_facts(facts, "0000320193")

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "xbrl_features.parquet"

            # Write with provenance
            import pyarrow as pa
            import pyarrow.parquet as pq
            import json

            provenance = {"build_date": "2026-01-01", "n_companies": 1}
            table = pa.Table.from_pandas(df)
            meta = table.schema.metadata or {}
            meta[b"fin_jepa_provenance"] = json.dumps(provenance).encode()
            table = table.replace_schema_metadata(meta)
            pq.write_table(table, path)

            # Read back
            loaded_df, loaded_prov = load_xbrl_features(path)
            assert len(loaded_df) == len(df)
            assert loaded_prov["n_companies"] == 1
            assert loaded_prov["build_date"] == "2026-01-01"


# ---------------------------------------------------------------------------
# Integration test with mocked HTTP
# ---------------------------------------------------------------------------

class TestBuildXBRLDatasetIntegration:
    """Integration test that mocks HTTP to verify the full pipeline."""

    def _make_mock_universe(self, tmpdir: Path) -> Path:
        """Create a minimal universe Parquet for testing."""
        universe_df = pd.DataFrame({
            "cik": ["0000320193", "0000789019"],
            "entity_name": ["Apple Inc", "Microsoft Corp"],
            "ticker": ["AAPL", "MSFT"],
            "exchange": ["Nasdaq", "Nasdaq"],
            "sic_code": ["3571", "7372"],
            "sector": ["Business Equipment", "Business Equipment"],
        })
        path = tmpdir / "company_universe.parquet"
        universe_df.to_parquet(path, index=False)
        return path

    @patch("fin_jepa.data.xbrl_pipeline._fetch")
    @patch("fin_jepa.data.xbrl_pipeline._get_session")
    def test_full_pipeline(self, mock_get_session, mock_fetch):
        """End-to-end test: universe → fetch facts → extract → validate → Parquet."""
        # Mock session
        mock_session = MagicMock()
        mock_get_session.return_value = mock_session

        # Build two different Company Facts responses
        aapl_facts = _make_facts_json(
            cik=320193,
            entity_name="Apple Inc",
            concepts={
                "Assets": [_make_entry(2020, 323_888_000_000, end="2020-09-26")],
                "Liabilities": [_make_entry(2020, 258_549_000_000, end="2020-09-26")],
                "StockholdersEquity": [_make_entry(2020, 65_339_000_000, end="2020-09-26")],
                "Revenues": [_make_entry(2020, 274_515_000_000)],
                "NetIncomeLoss": [_make_entry(2020, 57_411_000_000)],
                "NetCashProvidedByUsedInOperatingActivities": [
                    _make_entry(2020, 80_674_000_000),
                ],
            },
        )
        msft_facts = _make_facts_json(
            cik=789019,
            entity_name="Microsoft Corp",
            concepts={
                "Assets": [_make_entry(2020, 301_311_000_000, end="2020-06-30")],
                "Revenues": [_make_entry(2020, 143_015_000_000)],
                "NetIncomeLoss": [_make_entry(2020, 44_281_000_000)],
            },
        )

        # Mock _fetch to return company facts based on URL
        def _mock_fetch(url, session, **kwargs):
            if "0000320193" in url:
                return aapl_facts
            elif "0000789019" in url:
                return msft_facts
            raise Exception(f"Unexpected URL: {url}")

        mock_fetch.side_effect = _mock_fetch

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            universe_path = self._make_mock_universe(tmpdir)
            raw_dir = tmpdir / "raw"
            raw_dir.mkdir()
            output_path = raw_dir / "xbrl_features.parquet"

            config = XBRLConfig(start_year=2020, end_year=2020)

            from fin_jepa.data.xbrl_pipeline import build_xbrl_dataset
            df = build_xbrl_dataset(
                raw_dir=raw_dir,
                universe_path=universe_path,
                output_path=output_path,
                config=config,
            )

            # Verify output shape
            assert len(df) == 2  # 2 companies, 1 year each
            assert set(df["cik"]) == {"0000320193", "0000789019"}

            # Verify AAPL values
            aapl = df[df["cik"] == "0000320193"].iloc[0]
            assert aapl["total_assets"] == 323_888_000_000
            assert aapl["total_revenue"] == 274_515_000_000
            assert aapl["ticker"] == "AAPL"

            # Verify MSFT has some NaN features (no liabilities/equity in test data)
            msft = df[df["cik"] == "0000789019"].iloc[0]
            assert msft["total_assets"] == 301_311_000_000
            assert pd.isna(msft["total_liabilities"])

            # Verify Parquet file was written
            assert output_path.exists()

            # Verify provenance
            loaded_df, prov = load_xbrl_features(output_path)
            assert prov["n_companies"] == 2
            assert prov["n_observations"] == 2

            # Verify column order: IDs first, then features
            expected_id_cols = ["cik", "ticker", "fiscal_year", "period_end", "filed_date"]
            assert list(df.columns[:5]) == expected_id_cols

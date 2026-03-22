"""Smoke test for XBRL extraction pipeline — hits real EDGAR API.

Run with: pytest tests/smoke_test_xbrl.py -m smoke -v

This test is NOT run by default in CI (requires network access and takes
~30 seconds due to EDGAR rate limiting).  It validates the full pipeline
end-to-end against a small subset of well-known companies.
"""

from __future__ import annotations

import tempfile
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from fin_jepa.data.xbrl_pipeline import (
    FEATURE_NAMES,
    XBRLConfig,
    build_xbrl_dataset,
    extract_annual_facts,
    fetch_company_facts,
    load_xbrl_features,
    validate_xbrl_dataset,
)


# Well-known companies for spot-checking
SMOKE_COMPANIES = {
    "0000320193": {"name": "Apple Inc", "ticker": "AAPL"},
    "0000789019": {"name": "Microsoft Corp", "ticker": "MSFT"},
    "0000019617": {"name": "JPMorgan Chase", "ticker": "JPM"},
    "0000034088": {"name": "Exxon Mobil", "ticker": "XOM"},
    "0000200406": {"name": "Johnson & Johnson", "ticker": "JNJ"},
}

# Known financial values for spot-checking (approximate, within 5% tolerance)
# Source: public 10-K filings
KNOWN_VALUES = {
    # Apple FY2023 (period ending 2023-09-30)
    ("0000320193", 2023): {
        "total_assets": 352_583_000_000,
        "total_revenue": 383_285_000_000,
        "net_income": 96_995_000_000,
    },
    # Microsoft FY2023 (period ending 2023-06-30)
    # Note: EDGAR fy numbering for non-Dec FY-end companies can be offset;
    # use wider tolerance for MSFT
    ("0000789019", 2023): {
        "total_assets": 411_976_000_000,
    },
    # JPMorgan FY2023 (period ending 2023-12-31)
    ("0000019617", 2023): {
        "total_assets": 3_875_393_000_000,
    },
}


@pytest.mark.smoke
class TestSmokeXBRLPipeline:
    """End-to-end smoke test against real EDGAR API."""

    @pytest.fixture(autouse=True)
    def setup_temp_dir(self):
        """Create a temp directory for cache and output."""
        with tempfile.TemporaryDirectory() as tmpdir:
            self.tmpdir = Path(tmpdir)
            self.cache_dir = self.tmpdir / "cache"
            self.cache_dir.mkdir()
            yield

    def test_fetch_single_company(self):
        """Fetch Apple's Company Facts and verify structure."""
        facts = fetch_company_facts("320193", self.cache_dir)

        assert facts, "Empty response for Apple (CIK 320193)"
        assert "facts" in facts
        assert "us-gaap" in facts["facts"]

        us_gaap = facts["facts"]["us-gaap"]
        assert "Assets" in us_gaap, "Missing 'Assets' concept for Apple"
        assert "USD" in us_gaap["Assets"]["units"]

    def test_extract_apple_annual_facts(self):
        """Extract Apple's annual facts and verify known values."""
        facts = fetch_company_facts("320193", self.cache_dir)
        config = XBRLConfig(start_year=2015, end_year=2024)
        df = extract_annual_facts(facts, "0000320193", config)

        assert len(df) >= 5, f"Expected ≥5 years for Apple, got {len(df)}"
        assert "total_assets" in df.columns
        assert "total_revenue" in df.columns
        assert "net_income" in df.columns

        # All fiscal years should be in range
        assert df["fiscal_year"].min() >= 2015
        assert df["fiscal_year"].max() <= 2024

        # No duplicate fiscal years
        assert df["fiscal_year"].is_unique, "Duplicate fiscal years for Apple"

        # Spot-check FY2023 total_assets
        fy2023 = df[df["fiscal_year"] == 2023]
        if len(fy2023) > 0:
            actual = fy2023.iloc[0]["total_assets"]
            expected = KNOWN_VALUES[("0000320193", 2023)]["total_assets"]
            pct_diff = abs(actual - expected) / expected
            assert pct_diff < 0.05, (
                f"Apple FY2023 total_assets: expected ~{expected:,.0f}, "
                f"got {actual:,.0f} ({pct_diff:.1%} diff)"
            )

    def test_full_pipeline_small_subset(self):
        """Run build_xbrl_dataset on 5 companies end-to-end."""
        # Create a minimal universe Parquet
        universe_df = pd.DataFrame({
            "cik": list(SMOKE_COMPANIES.keys()),
            "entity_name": [v["name"] for v in SMOKE_COMPANIES.values()],
            "ticker": [v["ticker"] for v in SMOKE_COMPANIES.values()],
            "exchange": ["Nasdaq", "Nasdaq", "NYSE", "NYSE", "NYSE"],
            "sic_code": ["3571", "7372", "6022", "1311", "2836"],
            "sector": [
                "Business Equipment", "Business Equipment", "Finance",
                "Energy", "Health Care",
            ],
        })
        universe_path = self.tmpdir / "company_universe.parquet"
        universe_df.to_parquet(universe_path, index=False)

        raw_dir = self.tmpdir / "raw"
        raw_dir.mkdir()
        output_path = raw_dir / "xbrl_features.parquet"

        config = XBRLConfig(start_year=2018, end_year=2023)
        df = build_xbrl_dataset(
            raw_dir=raw_dir,
            universe_path=universe_path,
            output_path=output_path,
            config=config,
        )

        # --- Shape assertions ---
        n_companies = df["cik"].nunique()
        assert n_companies == 5, f"Expected 5 companies, got {n_companies}"

        # Each company should have ~6 years (2018-2023), some may have fewer
        assert len(df) >= 20, f"Expected ≥20 rows (5 companies × ~4+ years), got {len(df)}"
        assert len(df) <= 40, f"Unexpectedly many rows: {len(df)}"

        # --- Column assertions ---
        expected_id_cols = ["cik", "ticker", "fiscal_year", "period_end", "filed_date"]
        for col in expected_id_cols:
            assert col in df.columns, f"Missing column: {col}"

        # --- Data type assertions ---
        assert df["period_end"].dtype == object  # date objects stored as object in pandas
        assert all(
            isinstance(d, date)
            for d in df["period_end"].dropna()
        ), "period_end should contain date objects"

        # All period_end dates should be within reasonable range
        # Note: EDGAR fy field can be offset from period_end year by 1-2 years
        # for some companies, so allow a wider window than the config range
        period_ends = pd.to_datetime(df["period_end"].dropna())
        assert period_ends.min().year >= 2015, "period_end too early"
        assert period_ends.max().year <= 2024, "period_end too late"

        # --- No duplicate (cik, fiscal_year) pairs ---
        dups = df.duplicated(subset=["cik", "fiscal_year"])
        assert not dups.any(), f"Found {dups.sum()} duplicate (cik, fiscal_year) pairs"

        # --- Core features should be mostly non-null ---
        core_features = ["total_assets", "total_revenue", "net_income"]
        for feat in core_features:
            coverage = df[feat].notna().mean()
            assert coverage >= 0.8, (
                f"Feature '{feat}' coverage too low: {coverage:.1%} "
                f"(expected ≥80% for core features)"
            )

        # --- Spot-check known values (Apple only — Dec FY-end, reliable fy mapping) ---
        aapl_key = ("0000320193", 2023)
        if aapl_key in KNOWN_VALUES:
            aapl_row = df[(df["cik"] == "0000320193") & (df["fiscal_year"] == 2023)]
            if len(aapl_row) > 0:
                for feat, expected in KNOWN_VALUES[aapl_key].items():
                    actual = aapl_row.iloc[0][feat]
                    if pd.isna(actual):
                        continue
                    pct_diff = abs(actual - expected) / expected
                    assert pct_diff < 0.05, (
                        f"Apple FY2023 {feat}: "
                        f"expected ~{expected:,.0f}, got {actual:,.0f} ({pct_diff:.1%} diff)"
                    )

        # --- Parquet file assertions ---
        assert output_path.exists(), "Output Parquet file not written"
        loaded_df, prov = load_xbrl_features(output_path)
        assert len(loaded_df) == len(df)
        assert prov.get("n_companies") == 5
        assert prov.get("data_source") == "SEC EDGAR Company Facts API"

        # --- Validation report ---
        report = validate_xbrl_dataset(df)
        assert report["duplicate_count"] == 0

        # Print coverage summary for manual review
        print("\n=== XBRL Smoke Test Coverage Summary ===")
        print(f"Companies: {report['n_companies']}")
        print(f"Observations: {report['n_observations']}")
        print(f"Year range: {report['year_range']}")
        print("\nFeature coverage:")
        for feat, pct in sorted(report["feature_coverage"].items()):
            bar = "#" * int(pct / 5) + "." * (20 - int(pct / 5))
            print(f"  {feat:30s} [{bar}] {pct:5.1f}%")

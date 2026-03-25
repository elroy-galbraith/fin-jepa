"""Tests for fin_jepa.data.labels.

All tests use synthetic data and tmp_path fixtures — no network calls.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from fin_jepa.data.labels import (
    ALL_OUTCOMES,
    LabelConfig,
    OutcomeType,
    _build_audit_qualification,
    _build_bankruptcy,
    _build_earnings_restate,
    _build_sec_enforcement,
    _compute_stock_decline,
    _load_external_label,
    build_label_database,
    load_label_database,
    validate_label_database,
)

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _make_market_aligned(n: int = 6) -> pd.DataFrame:
    """Create a synthetic market_aligned DataFrame."""
    return pd.DataFrame(
        {
            "cik": [f"{i:010d}" for i in range(1, n + 1)],
            "period_end": pd.to_datetime(
                ["2019-12-31", "2019-12-31", "2020-06-30",
                 "2020-12-31", "2021-06-30", "2021-12-31"][:n]
            ).date,
            "ticker": ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"][:n],
            "sector": ["Finance"] * n,
            "fwd_ret_252d": [-0.30, -0.10, 0.05, -0.50, np.nan, -0.15][:n],
            "mkt_adj_252d": [-0.25, -0.05, 0.10, -0.45, np.nan, -0.10][:n],
            "sec_adj_252d": [-0.20, -0.03, 0.08, -0.40, np.nan, -0.08][:n],
            "volume_avg_63d": [1e6] * n,
            "delisted": [False, False, False, False, True, False][:n],
        }
    )


def _write_market_aligned(tmp_path: Path, df: pd.DataFrame | None = None) -> Path:
    """Write market_aligned.parquet to the expected location."""
    if df is None:
        df = _make_market_aligned()
    market_dir = tmp_path / "market"
    market_dir.mkdir(parents=True, exist_ok=True)
    path = market_dir / "market_aligned.parquet"
    df.to_parquet(path, index=False)
    return tmp_path  # raw_dir


def _write_edgar_index(
    tmp_path: Path, amendments: list[tuple[str, str]] | None = None
) -> None:
    """Write synthetic EDGAR quarterly index parquets.

    *amendments* is a list of (cik, date_filed) tuples for 10-K/A filings.
    """
    index_dir = tmp_path / "edgar_index"
    index_dir.mkdir(parents=True, exist_ok=True)

    # Base 10-K filings (always present)
    base_rows = [
        {"form_type": "10-K", "cik": f"{i:010d}", "date_filed": f"2020-03-{15 + i:02d}",
         "company_name": f"Company {i}", "filename": f"edgar/{i}/10k.htm"}
        for i in range(1, 7)
    ]

    amendment_rows = []
    if amendments:
        for cik, date_filed in amendments:
            amendment_rows.append(
                {
                    "form_type": "10-K/A",
                    "cik": cik.zfill(10),
                    "date_filed": date_filed,
                    "company_name": f"Company {cik}",
                    "filename": f"edgar/{cik}/10ka.htm",
                }
            )

    df = pd.DataFrame(base_rows + amendment_rows)
    df["date_filed"] = pd.to_datetime(df["date_filed"])
    # Write as a single quarterly index file
    df.to_parquet(index_dir / "2020_Q1.parquet", index=False)


def _write_external_csv(
    tmp_path: Path,
    label_name: str,
    data: list[dict],
) -> Path:
    """Write an external label CSV and return the directory path."""
    label_dir = tmp_path / "labels"
    label_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(data)
    df.to_csv(label_dir / f"{label_name}.csv", index=False)
    return label_dir


# ---------------------------------------------------------------------------
# TestLabelConfig
# ---------------------------------------------------------------------------


class TestLabelConfig:
    def test_defaults(self):
        cfg = LabelConfig()
        assert cfg.decline_threshold == -0.20
        assert cfg.treat_delisted_as_decline is True
        assert cfg.restatement_source == "edgar_amendments"
        assert cfg.horizon_days == 365

    def test_custom_threshold(self):
        cfg = LabelConfig(decline_threshold=-0.30)
        assert cfg.decline_threshold == -0.30


# ---------------------------------------------------------------------------
# TestStockDecline
# ---------------------------------------------------------------------------


class TestStockDecline:
    def test_threshold_labels_correctly(self):
        """Values below threshold → 1, above → 0, NaN → NaN."""
        df = _make_market_aligned()
        cfg = LabelConfig(decline_threshold=-0.20, treat_delisted_as_decline=False)
        labels = _compute_stock_decline(df, cfg)

        # mkt_adj_252d: [-0.25, -0.05, 0.10, -0.45, NaN, -0.10]
        assert labels.iloc[0] == 1   # -0.25 < -0.20
        assert labels.iloc[1] == 0   # -0.05 > -0.20
        assert labels.iloc[2] == 0   # 0.10 > -0.20
        assert labels.iloc[3] == 1   # -0.45 < -0.20
        assert pd.isna(labels.iloc[4])  # NaN return, not delisted → NaN
        assert labels.iloc[5] == 0   # -0.10 > -0.20

    def test_delisted_as_decline(self):
        """When treat_delisted_as_decline=True, delisted rows get label=1."""
        df = _make_market_aligned()
        cfg = LabelConfig(treat_delisted_as_decline=True)
        labels = _compute_stock_decline(df, cfg)

        # Row 4 is delisted=True, mkt_adj_252d=NaN → should be 1
        assert labels.iloc[4] == 1

    def test_delisted_not_decline(self):
        """When treat_delisted_as_decline=False, delisted+NaN rows stay NaN."""
        df = _make_market_aligned()
        cfg = LabelConfig(treat_delisted_as_decline=False)
        labels = _compute_stock_decline(df, cfg)

        assert pd.isna(labels.iloc[4])

    def test_all_nan_returns(self):
        """All NaN returns → all NaN labels (no delisted)."""
        df = pd.DataFrame({
            "cik": ["0000000001", "0000000002"],
            "mkt_adj_252d": [np.nan, np.nan],
            "delisted": [False, False],
        })
        cfg = LabelConfig(treat_delisted_as_decline=False)
        labels = _compute_stock_decline(df, cfg)
        assert labels.isna().all()


# ---------------------------------------------------------------------------
# TestEarningsRestate
# ---------------------------------------------------------------------------


class TestEarningsRestate:
    def test_10ka_within_window(self, tmp_path):
        """A 10-K/A filed within horizon_days → label=1."""
        raw_dir = _write_market_aligned(tmp_path)
        # Company 1 (CIK=0000000001) has period_end=2019-12-31.
        # 10-K/A filed 2020-06-15 → within 365 days.
        _write_edgar_index(tmp_path, amendments=[("1", "2020-06-15")])

        grid = _make_market_aligned()[["cik", "period_end"]].copy()
        cfg = LabelConfig(restatement_source="edgar_amendments", horizon_days=365)
        labels = _build_earnings_restate(grid, raw_dir, cfg)

        assert labels.iloc[0] == 1  # CIK 1 had a 10-K/A in window

    def test_10ka_outside_window(self, tmp_path):
        """A 10-K/A filed outside horizon_days → label=0."""
        raw_dir = _write_market_aligned(tmp_path)
        # Company 1 period_end=2019-12-31, 10-K/A filed 2021-06-15 → >365 days
        _write_edgar_index(tmp_path, amendments=[("1", "2021-06-15")])

        grid = _make_market_aligned()[["cik", "period_end"]].copy()
        cfg = LabelConfig(restatement_source="edgar_amendments", horizon_days=365)
        labels = _build_earnings_restate(grid, raw_dir, cfg)

        assert labels.iloc[0] == 0

    def test_no_amendments_all_zero(self, tmp_path):
        """No 10-K/A filings at all → all labels = 0."""
        raw_dir = _write_market_aligned(tmp_path)
        _write_edgar_index(tmp_path, amendments=None)

        grid = _make_market_aligned()[["cik", "period_end"]].copy()
        cfg = LabelConfig(restatement_source="edgar_amendments")
        labels = _build_earnings_restate(grid, raw_dir, cfg)

        assert (labels == 0).all()

    def test_external_csv_fallback(self, tmp_path):
        """Load earnings_restate from external CSV."""
        raw_dir = _write_market_aligned(tmp_path)
        label_dir = _write_external_csv(
            tmp_path,
            "earnings_restate",
            [
                {"cik": "0000000001", "period_end": "2019-12-31", "earnings_restate": 1},
                {"cik": "0000000002", "period_end": "2019-12-31", "earnings_restate": 0},
            ],
        )
        grid = _make_market_aligned()[["cik", "period_end"]].copy()
        cfg = LabelConfig(
            restatement_source="external_csv",
            external_label_dir=label_dir,
        )
        labels = _build_earnings_restate(grid, raw_dir, cfg)

        # Index 0 = CIK 1, period_end 2019-12-31 → 1
        # Index 1 = CIK 2, period_end 2019-12-31 → 0
        assert labels[0] == 1
        assert labels[1] == 0

    def test_missing_index_returns_nan(self, tmp_path):
        """No EDGAR index cache → all NaN."""
        raw_dir = _write_market_aligned(tmp_path)
        # Don't create edgar_index dir
        grid = _make_market_aligned()[["cik", "period_end"]].copy()
        cfg = LabelConfig(restatement_source="edgar_amendments")
        labels = _build_earnings_restate(grid, raw_dir, cfg)

        assert labels.isna().all()


# ---------------------------------------------------------------------------
# TestAuditQualification
# ---------------------------------------------------------------------------


class TestAuditQualification:
    def test_all_nan_when_no_file(self):
        """No external file → all NaN."""
        grid = _make_market_aligned()[["cik", "period_end"]].copy()
        cfg = LabelConfig(external_label_dir=None)
        labels = _build_audit_qualification(grid, cfg)
        assert labels.isna().all()

    def test_loads_from_csv(self, tmp_path):
        """Correctly merges audit_qualification from external CSV."""
        label_dir = _write_external_csv(
            tmp_path,
            "audit_qualification",
            [
                {"cik": "0000000003", "period_end": "2020-06-30", "audit_qualification": 1},
            ],
        )
        grid = _make_market_aligned()[["cik", "period_end"]].copy()
        cfg = LabelConfig(external_label_dir=label_dir)
        labels = _build_audit_qualification(grid, cfg)

        # Row 2 = CIK 3, period_end 2020-06-30 → 1
        assert labels[2] == 1
        # Other rows should be NaN (not in external file)
        assert pd.isna(labels[0])


# ---------------------------------------------------------------------------
# TestSecEnforcement
# ---------------------------------------------------------------------------


class TestSecEnforcement:
    def test_all_nan_when_no_file(self):
        """No external file → all NaN."""
        grid = _make_market_aligned()[["cik", "period_end"]].copy()
        cfg = LabelConfig(external_label_dir=None)
        labels = _build_sec_enforcement(grid, cfg)
        assert labels.isna().all()

    def test_aaer_match_with_date_range(self, tmp_path):
        """AAER with start/end date overlapping period → label=1."""
        label_dir = _write_external_csv(
            tmp_path,
            "sec_enforcement",
            [
                {
                    "cik": "0000000001",
                    "start_date": "2019-01-01",
                    "end_date": "2020-06-30",
                },
            ],
        )
        grid = _make_market_aligned()[["cik", "period_end"]].copy()
        cfg = LabelConfig(external_label_dir=label_dir, horizon_days=365)
        labels = _build_sec_enforcement(grid, cfg)

        # CIK 1, period_end 2019-12-31 → violation window overlaps
        assert labels.iloc[0] == 1

    def test_aaer_no_match(self, tmp_path):
        """AAER for a different CIK → label=0."""
        label_dir = _write_external_csv(
            tmp_path,
            "sec_enforcement",
            [
                {
                    "cik": "0000099999",
                    "start_date": "2019-01-01",
                    "end_date": "2020-06-30",
                },
            ],
        )
        grid = _make_market_aligned()[["cik", "period_end"]].copy()
        cfg = LabelConfig(external_label_dir=label_dir)
        labels = _build_sec_enforcement(grid, cfg)

        assert (labels == 0).all()

    def test_binary_column_merge(self, tmp_path):
        """CSV with pre-computed binary sec_enforcement column."""
        label_dir = _write_external_csv(
            tmp_path,
            "sec_enforcement",
            [
                {"cik": "0000000002", "period_end": "2019-12-31", "sec_enforcement": 1},
            ],
        )
        grid = _make_market_aligned()[["cik", "period_end"]].copy()
        cfg = LabelConfig(external_label_dir=label_dir)
        labels = _build_sec_enforcement(grid, cfg)

        assert labels[1] == 1


# ---------------------------------------------------------------------------
# TestBankruptcy
# ---------------------------------------------------------------------------


class TestBankruptcy:
    def test_compustat_dlrsn_03(self, tmp_path):
        """Compustat delist reason '03' within horizon → label=1."""
        raw_dir = _write_market_aligned(tmp_path)

        # Write a universe with Compustat delist data
        universe = pd.DataFrame({
            "cik": ["0000000004"],
            "cstat_dlrsn": ["03"],
            "cstat_dldte": ["2021-03-15"],
        })
        universe.to_parquet(raw_dir / "company_universe.parquet", index=False)

        grid = _make_market_aligned()[["cik", "period_end"]].copy()
        cfg = LabelConfig(bankruptcy_source="compustat", horizon_days=365)
        labels = _build_bankruptcy(grid, raw_dir, cfg)

        # CIK 4, period_end=2020-12-31, dldte=2021-03-15 → within 365 days
        assert labels.iloc[3] == 1
        # Other CIKs not bankrupt
        assert labels.iloc[0] == 0

    def test_compustat_no_bankruptcy(self, tmp_path):
        """Compustat delist reason != '03' → label=0."""
        raw_dir = _write_market_aligned(tmp_path)

        universe = pd.DataFrame({
            "cik": ["0000000001"],
            "cstat_dlrsn": ["06"],  # Acquisition, not bankruptcy
            "cstat_dldte": ["2020-06-15"],
        })
        universe.to_parquet(raw_dir / "company_universe.parquet", index=False)

        grid = _make_market_aligned()[["cik", "period_end"]].copy()
        cfg = LabelConfig(bankruptcy_source="compustat")
        labels = _build_bankruptcy(grid, raw_dir, cfg)

        assert (labels == 0).all()

    def test_external_csv_with_filing_date(self, tmp_path):
        """Load bankruptcy from external CSV with filing_date column."""
        raw_dir = _write_market_aligned(tmp_path)
        label_dir = _write_external_csv(
            tmp_path,
            "bankruptcy",
            [
                {"cik": "0000000001", "filing_date": "2020-06-01", "chapter": "11"},
            ],
        )
        grid = _make_market_aligned()[["cik", "period_end"]].copy()
        cfg = LabelConfig(
            bankruptcy_source="external_csv",
            external_label_dir=label_dir,
        )
        labels = _build_bankruptcy(grid, raw_dir, cfg)

        # CIK 1, period_end=2019-12-31, bankruptcy filed 2020-06-01 → in window
        assert labels.iloc[0] == 1

    def test_all_nan_when_no_source(self, tmp_path):
        """No Compustat, no external CSV → all NaN."""
        raw_dir = _write_market_aligned(tmp_path)
        # No universe file, no external CSV
        grid = _make_market_aligned()[["cik", "period_end"]].copy()
        cfg = LabelConfig(bankruptcy_source="compustat", external_label_dir=None)
        labels = _build_bankruptcy(grid, raw_dir, cfg)

        assert labels.isna().all()


# ---------------------------------------------------------------------------
# TestLoadExternalLabel
# ---------------------------------------------------------------------------


class TestLoadExternalLabel:
    def test_returns_none_when_no_dir(self):
        result = _load_external_label("test_label", None)
        assert result is None

    def test_returns_none_when_no_file(self, tmp_path):
        result = _load_external_label("nonexistent", tmp_path)
        assert result is None

    def test_loads_csv(self, tmp_path):
        data = [{"cik": "1", "period_end": "2020-01-01", "val": "1"}]
        pd.DataFrame(data).to_csv(tmp_path / "my_label.csv", index=False)
        result = _load_external_label("my_label", tmp_path)
        assert result is not None
        assert len(result) == 1
        assert result["cik"].iloc[0] == "0000000001"

    def test_loads_parquet(self, tmp_path):
        df = pd.DataFrame({"cik": ["0000000001"], "period_end": ["2020-01-01"]})
        df.to_parquet(tmp_path / "my_label.parquet", index=False)
        result = _load_external_label("my_label", tmp_path)
        assert result is not None
        assert len(result) == 1


# ---------------------------------------------------------------------------
# TestValidateLabelDatabase
# ---------------------------------------------------------------------------


class TestValidateLabelDatabase:
    def test_coverage_report(self):
        df = pd.DataFrame({
            "cik": ["0000000001", "0000000002", "0000000003", "0000000004"],
            "period_end": pd.to_datetime(
                ["2019-12-31", "2019-12-31", "2020-06-30", "2020-12-31"]
            ).date,
            "stock_decline": pd.array([1, 0, 0, 1], dtype="Int8"),
            "earnings_restate": pd.array([0, 0, 1, 0], dtype="Int8"),
            "audit_qualification": pd.array([pd.NA, pd.NA, pd.NA, pd.NA], dtype="Int8"),
            "sec_enforcement": pd.array([pd.NA, pd.NA, pd.NA, pd.NA], dtype="Int8"),
            "bankruptcy": pd.array([0, 0, 0, 1], dtype="Int8"),
        })
        stats = validate_label_database(df)

        assert stats["n_observations"] == 4
        assert stats["per_label"]["stock_decline"]["n_positive"] == 2
        assert stats["per_label"]["stock_decline"]["coverage_pct"] == 100.0
        assert stats["per_label"]["audit_qualification"]["n_missing"] == 4
        assert stats["per_label"]["audit_qualification"]["coverage_pct"] == 0.0
        assert stats["n_labels_with_majority_coverage"] == 3  # stock, restate, bankruptcy


# ---------------------------------------------------------------------------
# TestBuildLabelDatabase (integration)
# ---------------------------------------------------------------------------


class TestBuildLabelDatabase:
    def test_output_schema(self, tmp_path):
        """Output has exactly the expected columns."""
        raw_dir = _write_market_aligned(tmp_path)
        _write_edgar_index(tmp_path)

        output = tmp_path / "output" / "label_database.parquet"
        df = build_label_database(raw_dir, output_path=output)

        expected_cols = {"cik", "period_end"} | set(ALL_OUTCOMES)
        assert set(df.columns) == expected_cols

    def test_output_dtypes(self, tmp_path):
        """Label columns are nullable Int8."""
        raw_dir = _write_market_aligned(tmp_path)
        _write_edgar_index(tmp_path)

        df = build_label_database(raw_dir, output_path=tmp_path / "out.parquet")
        for col in ALL_OUTCOMES:
            assert df[col].dtype == pd.Int8Dtype(), f"{col} dtype is {df[col].dtype}"

    def test_roundtrip_parquet(self, tmp_path):
        """Write to parquet and reload; data matches."""
        raw_dir = _write_market_aligned(tmp_path)
        _write_edgar_index(tmp_path)

        output = tmp_path / "output" / "label_database.parquet"
        original = build_label_database(raw_dir, output_path=output)

        loaded, provenance = load_label_database(output)
        assert len(loaded) == len(original)
        assert set(loaded.columns) == set(original.columns)
        assert provenance.get("n_observations") == len(original)

    def test_provenance_metadata(self, tmp_path):
        """Provenance dict is embedded in parquet metadata."""
        raw_dir = _write_market_aligned(tmp_path)
        _write_edgar_index(tmp_path)

        output = tmp_path / "output" / "labels.parquet"
        build_label_database(raw_dir, output_path=output)

        _, provenance = load_label_database(output)
        assert "build_date" in provenance
        assert "decline_threshold" in provenance
        assert "label_coverage" in provenance
        assert provenance["decline_threshold"] == -0.20

    def test_graceful_with_only_stock_decline(self, tmp_path):
        """When no external data exists, only stock_decline is populated."""
        raw_dir = _write_market_aligned(tmp_path)
        # No EDGAR index, no external CSVs, no Compustat

        output = tmp_path / "output" / "labels.parquet"
        df = build_label_database(raw_dir, output_path=output)

        # stock_decline should have values
        assert df["stock_decline"].notna().any()
        # audit_qualification and sec_enforcement should be all NaN
        assert df["audit_qualification"].isna().all()
        assert df["sec_enforcement"].isna().all()

    def test_stock_decline_values(self, tmp_path):
        """Verify stock_decline labels match expected threshold logic."""
        raw_dir = _write_market_aligned(tmp_path)
        _write_edgar_index(tmp_path)

        cfg = LabelConfig(decline_threshold=-0.20, treat_delisted_as_decline=True)
        df = build_label_database(raw_dir, output_path=tmp_path / "out.parquet", config=cfg)

        # mkt_adj_252d: [-0.25, -0.05, 0.10, -0.45, NaN(delisted), -0.10]
        assert df["stock_decline"].iloc[0] == 1   # -0.25 < -0.20
        assert df["stock_decline"].iloc[1] == 0   # -0.05 > -0.20
        assert df["stock_decline"].iloc[2] == 0   # 0.10 > -0.20
        assert df["stock_decline"].iloc[3] == 1   # -0.45 < -0.20
        assert df["stock_decline"].iloc[4] == 1   # delisted → 1
        assert df["stock_decline"].iloc[5] == 0   # -0.10 > -0.20


# ---------------------------------------------------------------------------
# Enum / constants
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# TestDelistingEdgeCases
# ---------------------------------------------------------------------------


class TestDelistingEdgeCases:
    """Edge-case tests for delisting handling in stock_decline labels."""

    def test_delisted_marked_as_decline(self):
        """When treat_delisted_as_decline=True, delisted + NaN return → label=1."""
        df = pd.DataFrame({
            "cik": ["0000000001"],
            "mkt_adj_252d": [np.nan],
            "delisted": [True],
        })
        cfg = LabelConfig(treat_delisted_as_decline=True)
        labels = _compute_stock_decline(df, cfg)
        assert labels.iloc[0] == 1

    def test_delisted_not_marked_when_flag_off(self):
        """When treat_delisted_as_decline=False, delisted + NaN → NaN (not 0)."""
        df = pd.DataFrame({
            "cik": ["0000000001"],
            "mkt_adj_252d": [np.nan],
            "delisted": [True],
        })
        cfg = LabelConfig(treat_delisted_as_decline=False)
        labels = _compute_stock_decline(df, cfg)
        assert pd.isna(labels.iloc[0])

    def test_delisted_with_valid_negative_return(self):
        """A delisted company with mkt_adj_252d below threshold gets
        stock_decline=1 from the return logic (not just from the delist flag)."""
        df = pd.DataFrame({
            "cik": ["0000000001"],
            "mkt_adj_252d": [-0.50],
            "delisted": [True],
        })
        cfg = LabelConfig(treat_delisted_as_decline=False, decline_threshold=-0.20)
        labels = _compute_stock_decline(df, cfg)
        assert labels.iloc[0] == 1

    def test_delisted_with_positive_return_still_gets_1(self):
        """A delisted company with a positive return but treat_delisted_as_decline=True
        should get stock_decline=1 (the delist flag overrides the return)."""
        df = pd.DataFrame({
            "cik": ["0000000001"],
            "mkt_adj_252d": [0.15],
            "delisted": [True],
        })
        cfg = LabelConfig(treat_delisted_as_decline=True, decline_threshold=-0.20)
        labels = _compute_stock_decline(df, cfg)
        assert labels.iloc[0] == 1


class TestOutcomeType:
    def test_all_outcomes_list(self):
        assert len(ALL_OUTCOMES) == 5
        assert ALL_OUTCOMES == [
            "stock_decline",
            "earnings_restate",
            "audit_qualification",
            "sec_enforcement",
            "bankruptcy",
        ]

    def test_enum_values(self):
        assert OutcomeType.STOCK_DECLINE == "stock_decline"
        assert OutcomeType.BANKRUPTCY == "bankruptcy"

"""End-to-end tests for the data specification generator."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml

from fin_jepa.data.data_spec import generate_data_spec
from fin_jepa.data.splits import (
    RollingSplitConfig,
    SplitConfig,
    make_rolling_splits,
    make_splits,
)


def _make_sample_df(n: int = 120) -> pd.DataFrame:
    """Create a synthetic feature DataFrame spanning 2012–2023."""
    dates = pd.date_range("2012-01-01", periods=n, freq="QE")
    return pd.DataFrame({
        "period_end": dates,
        "cik": [f"{(i % 10) + 1:010d}" for i in range(n)],
        "sector_idx": [i % 12 for i in range(n)],
        "value": range(n),
    })


class TestDataSpec:
    @pytest.fixture()
    def df(self):
        return _make_sample_df()

    @pytest.fixture()
    def split_config(self):
        return SplitConfig(
            train_end="2017-12-31", val_end="2019-12-31", test_end="2023-12-31",
        )

    def test_spec_matches_config(self, df, split_config):
        """Verify the spec document's split dates match the config."""
        splits = make_splits(df, split_config)
        spec = generate_data_spec(splits, split_config=split_config)
        assert spec["splits"]["static"]["train_end"] == split_config.train_end
        assert spec["splits"]["static"]["val_end"] == split_config.val_end
        assert spec["splits"]["static"]["test_end"] == split_config.test_end

    def test_per_split_stats_row_counts(self, df, split_config):
        """Per-split row counts match actual split sizes."""
        splits = make_splits(df, split_config)
        spec = generate_data_spec(splits, split_config=split_config)
        for name in ("train", "val", "test"):
            assert spec["per_split_stats"][name]["n_rows"] == len(splits[name])

    def test_yaml_roundtrip(self, df, split_config, tmp_path):
        """Write data_spec.yaml, re-read it, verify all top-level keys present."""
        splits = make_splits(df, split_config)
        spec = generate_data_spec(splits, split_config=split_config, output_path=tmp_path)

        yaml_path = tmp_path / "data_spec.yaml"
        assert yaml_path.exists()
        with open(yaml_path, encoding="utf-8") as fh:
            loaded = yaml.safe_load(fh)

        expected_keys = {"generated_at", "universe", "features", "labels", "splits",
                         "per_split_stats", "reproducibility"}
        assert expected_keys.issubset(set(loaded.keys()))

    def test_markdown_generated(self, df, split_config, tmp_path):
        """Markdown document is written alongside YAML."""
        splits = make_splits(df, split_config)
        generate_data_spec(splits, split_config=split_config, output_path=tmp_path)
        md_path = tmp_path / "data_spec.md"
        assert md_path.exists()
        content = md_path.read_text(encoding="utf-8")
        assert "Study 0" in content
        assert "Feature Pipeline" in content

    def test_feature_counts(self, df, split_config):
        """Feature counts reflect the actual pipeline constants."""
        splits = make_splits(df, split_config)
        spec = generate_data_spec(splits, split_config=split_config)
        f = spec["features"]
        assert f["raw_xbrl"]["count"] == 16
        assert f["ratios"]["count"] == 12
        assert f["yoy_changes"]["count"] == 16
        assert f["total_numeric"] == 45  # 16 + 12 + 16 + 1

    def test_rolling_config_included(self, df):
        """Rolling split config appears in spec when provided."""
        split_config = SplitConfig(
            train_end="2017-12-31", val_end="2019-12-31", test_end="2023-12-31",
        )
        rolling_config = RollingSplitConfig(
            first_train_end="2014-12-31",
            val_window_years=1,
            test_window_years=2,
            step_years=1,
            last_test_end="2023-12-31",
        )
        splits = make_splits(df, split_config)
        spec = generate_data_spec(
            splits, split_config=split_config, rolling_config=rolling_config,
        )
        assert "rolling" in spec["splits"]
        assert spec["splits"]["rolling"]["first_train_end"] == "2014-12-31"

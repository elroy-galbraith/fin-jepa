"""Tests for the SSL experiment (ATS-167): _pretrain_encoder + run_ssl_experiment."""

import copy

import numpy as np
import pytest
import torch

from fin_jepa.models.ft_transformer import FTTransformer
from fin_jepa.training.dataset import TabularDataset
from fin_jepa.training.pretrain_ssl import _pretrain_encoder


# ── Tiny model / data fixtures ──────────────────────────────────────────

_N_FEATURES = 5
_MODEL_KWARGS = {
    "n_features": _N_FEATURES,
    "d_token": 8,
    "n_heads": 1,
    "n_layers": 1,
    "n_outputs": 1,
}


def _make_loader(n_rows: int = 60, batch_size: int = 16):
    """Structured synthetic data (repeated pattern) for SSL."""
    base = np.random.randn(10, _N_FEATURES).astype(np.float32)
    X = np.tile(base, (max(n_rows // 10, 1), 1))[:n_rows]
    ds = TabularDataset(X)
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True)


# ── _pretrain_encoder tests ─────────────────────────────────────────────


class TestPretrainEncoder:
    def test_returns_state_dict_and_losses(self):
        torch.manual_seed(0)
        np.random.seed(0)
        encoder = FTTransformer(**_MODEL_KWARGS)
        loader = _make_loader()

        state_dict, loss_history = _pretrain_encoder(
            encoder, loader, mask_ratio=0.15, device=torch.device("cpu"),
            epochs=3, warmup_epochs=1,
        )

        assert isinstance(state_dict, dict)
        assert isinstance(loss_history, list)
        assert len(loss_history) == 3
        # Keys should match a fresh FTTransformer's state dict
        expected_keys = set(FTTransformer(**_MODEL_KWARGS).state_dict().keys())
        assert set(state_dict.keys()) == expected_keys

    def test_weights_change(self):
        torch.manual_seed(1)
        np.random.seed(1)
        encoder = FTTransformer(**_MODEL_KWARGS)
        before = copy.deepcopy(encoder.state_dict())
        loader = _make_loader()

        _pretrain_encoder(
            encoder, loader, mask_ratio=0.15, device=torch.device("cpu"),
            epochs=5, warmup_epochs=1,
        )

        after = encoder.state_dict()
        changed = sum(
            1 for k in before if not torch.equal(before[k], after[k])
        )
        assert changed > 0, "Pretraining should modify at least some weights"

    def test_loss_decreases(self):
        torch.manual_seed(42)
        np.random.seed(42)
        encoder = FTTransformer(**_MODEL_KWARGS)
        loader = _make_loader(n_rows=200, batch_size=64)

        _, loss_history = _pretrain_encoder(
            encoder, loader, mask_ratio=0.15, device=torch.device("cpu"),
            epochs=20, lr=5e-4, warmup_epochs=2,
        )

        early_avg = np.mean(loss_history[:3])
        late_avg = np.mean(loss_history[-3:])
        assert late_avg < early_avg, (
            f"Loss should decrease: early={early_avg:.4f} → late={late_avg:.4f}"
        )

    def test_all_losses_finite(self):
        torch.manual_seed(2)
        np.random.seed(2)
        encoder = FTTransformer(**_MODEL_KWARGS)
        loader = _make_loader()

        _, loss_history = _pretrain_encoder(
            encoder, loader, mask_ratio=0.30, device=torch.device("cpu"),
            epochs=5, warmup_epochs=1,
        )

        for i, loss in enumerate(loss_history):
            assert np.isfinite(loss), f"Non-finite loss at epoch {i}: {loss}"


# ── run_ssl_experiment smoke test ────────────────────────────────────────


class TestRunSSLExperiment:
    def test_smoke(self, monkeypatch, tmp_path):
        """End-to-end smoke test with synthetic data and tiny model."""
        import pandas as pd

        from fin_jepa.training import pretrain_ssl
        from fin_jepa.training.ablations import _BENCHMARK_DEFAULTS

        # Build synthetic DataFrame with features + one outcome column
        rng = np.random.RandomState(42)
        n = 180
        feature_cols = [f"feat_{i}" for i in range(_N_FEATURES)]
        df = pd.DataFrame(rng.randn(n, _N_FEATURES).astype(np.float32), columns=feature_cols)
        df["cik"] = range(n)
        # Span 2010–2024 so all three splits (train/val/test) have data
        df["period_end"] = pd.date_range("2010-01-01", periods=n, freq="ME")
        df["fiscal_year"] = df["period_end"].dt.year
        df["filed_date"] = df["period_end"] + pd.Timedelta(days=30)
        # Single outcome with linearly separable signal
        df["stock_decline"] = (df["feat_0"] > 0).astype(float)

        # Monkeypatch data loading to return synthetic data
        def mock_load_xbrl(raw_dir):
            return df[["cik", "period_end", "fiscal_year", "filed_date"] + feature_cols].copy()

        def mock_load_labels(path):
            return df[["cik", "period_end", "stock_decline"]].copy(), {"stock_decline": "Stock decline > 20%"}

        monkeypatch.setattr(pretrain_ssl, "load_xbrl_features", mock_load_xbrl)
        monkeypatch.setattr(
            "fin_jepa.data.labels.load_label_database", mock_load_labels,
        )

        # Monkeypatch build_feature_matrix to skip real feature engineering
        def mock_build_feature_matrix(merged_df, split_cfg, feat_cfg=None, universe_df=None):
            train = merged_df[merged_df["period_end"] <= "2017-12-31"].copy()
            val = merged_df[
                (merged_df["period_end"] > "2017-12-31")
                & (merged_df["period_end"] <= "2019-12-31")
            ].copy()
            test = merged_df[merged_df["period_end"] > "2019-12-31"].copy()
            return {"train": train, "val": val, "test": test}, None, feature_cols, []

        monkeypatch.setattr(pretrain_ssl, "build_feature_matrix", mock_build_feature_matrix)

        # Override benchmark defaults to tiny values for speed
        monkeypatch.setattr(
            "fin_jepa.training.ablations._BENCHMARK_DEFAULTS",
            {**_BENCHMARK_DEFAULTS, "epochs": 2, "patience": 1, "batch_size": 32},
        )

        config = {
            "data": {
                "raw_dir": str(tmp_path),
                "processed_dir": str(tmp_path),
                "split": {
                    "train_end": "2017-12-31",
                    "val_end": "2019-12-31",
                    "test_end": "2023-12-31",
                },
            },
            "features": {
                "use_raw": True,
                "use_ratios": False,
                "use_yoy": False,
                "use_missingness_flags": False,
                "coverage_threshold": 0.0,
                "normalization_method": "quantile",
                "median_impute": True,
            },
            "outcomes": ["stock_decline"],
            "ssl_experiment": {"mask_ratios": [0.15]},
            "pretrain": {
                "batch_size": 32,
                "epochs": 3,
                "learning_rate": 1e-3,
                "weight_decay": 1e-5,
                "warmup_epochs": 1,
            },
            "ft_transformer": {
                "d_token": 8,
                "n_heads": 1,
                "n_layers": 1,
                "d_ffn_factor": 2,
                "dropout": 0.0,
            },
            "training": {"seed": 42},
            "checkpoint_dir": str(tmp_path / "ckpts"),
            "results_dir": str(tmp_path / "results"),
        }

        from fin_jepa.training.pretrain_ssl import run_ssl_experiment

        results = run_ssl_experiment(config)

        # Verify structure
        assert "baseline" in results
        assert "pretrained" in results
        assert "loss_curves" in results
        assert "comparison" in results
        assert "recommendation" in results

        assert "stock_decline" in results["baseline"]
        assert "0.15" in results["pretrained"]
        assert "stock_decline" in results["pretrained"]["0.15"]
        assert "0.15" in results["loss_curves"]
        assert len(results["loss_curves"]["0.15"]) == 3

        assert results["recommendation"] in ("proceed", "marginal", "no_gain")

        # Results file should exist
        results_path = tmp_path / "results" / "ssl_experiment_results.json"
        assert results_path.exists()

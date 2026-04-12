"""
Ablation studies and scaling curves.

Workstream: Run ablation studies and produce scaling curves.

Usage:
    python -m fin_jepa.training.ablations experiment=study0/ablations
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from fin_jepa.data.feature_engineering import (
    N_SECTORS,
    FeatureConfig,
    build_feature_matrix,
)
from fin_jepa.data.splits import SplitConfig
from fin_jepa.data.xbrl_loader import load_xbrl_features
from fin_jepa.data.labels import load_label_database
from fin_jepa.models.ft_transformer import FTTransformer
from fin_jepa.models.ssl_head import MaskedFeatureSSL
from fin_jepa.training.dataset import make_dataloader
from fin_jepa.training.metrics import compute_all_metrics
from fin_jepa.training.train_study0 import (
    MIN_POSITIVES,
    _cfg,
    _predict_scores,
    train_ft_transformer,
)
from fin_jepa.utils.reproducibility import seed_everything

log = logging.getLogger(__name__)

# Defaults matching configs/study0/benchmark.yaml — keep in sync if those change.
_BENCHMARK_DEFAULTS = {
    "d_token": 192,
    "n_heads": 8,
    "n_layers": 3,
    "d_ffn_factor": 4,
    "dropout": 0.0,
    "batch_size": 256,
    "epochs": 100,
    "learning_rate": 1e-4,
    "weight_decay": 1e-5,
    "patience": 10,
}


def run_ablations(config) -> dict:
    """Entry point for ablation study sweeps.

    Runs one-dimensional sweeps over architecture and data hyperparameters,
    holding all other parameters at their default values.

    Parameters
    ----------
    config : dict or DictConfig
        Hydra configuration (from ``configs/study0/ablations.yaml``).

    Returns
    -------
    dict mapping sweep name to list of per-point result dicts.
    """
    seed = int(_cfg(config, "seed", 42))
    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Load and prepare data ────────────────────────────────────────
    raw_dir = Path(_cfg(config, "data.raw_dir", "data/raw"))
    processed_dir = Path(_cfg(config, "data.processed_dir", "data/processed"))

    xbrl_df = load_xbrl_features(raw_dir)
    labels_df, _ = load_label_database(processed_dir / "label_database.parquet")
    xbrl_df["period_end"] = pd.to_datetime(xbrl_df["period_end"])
    labels_df["period_end"] = pd.to_datetime(labels_df["period_end"])
    merged = xbrl_df.merge(labels_df, on=["cik", "period_end"], how="inner", suffixes=("", "_label"))

    split_cfg = SplitConfig(
        train_end=_cfg(config, "data.split.train_end", "2017-12-31"),
        val_end=_cfg(config, "data.split.val_end", "2019-12-31"),
        test_end=_cfg(config, "data.split.test_end", "2023-12-31"),
    )
    # Load universe for SIC join
    universe_df = None
    universe_path = raw_dir / "company_universe.parquet"
    if universe_path.exists():
        universe_df = pd.read_parquet(universe_path)

    splits, scaler, feature_cols, categorical_cols = build_feature_matrix(
        merged, split_cfg, universe_df=universe_df,
    )
    n_cat = len(categorical_cols)
    cat_cards = [N_SECTORS] * n_cat if n_cat > 0 else None

    outcomes = _cfg(config, "outcomes", ["stock_decline"])
    # Use first outcome for ablations unless specified
    if isinstance(outcomes, list):
        outcome = outcomes[0]
    else:
        outcome = outcomes

    results_dir = Path(_cfg(config, "results_dir", "results/study0/ablations"))
    results_dir.mkdir(parents=True, exist_ok=True)

    all_sweep_results: dict[str, list[dict]] = {}

    # ── n_layers sweep ───────────────────────────────────────────────
    n_layers_grid = _cfg(config, "sweep.n_layers", [1, 2, 3, 6])
    if n_layers_grid:
        log.info("Running n_layers sweep: %s", n_layers_grid)
        results = _run_architecture_sweep(
            splits, feature_cols, outcome, device,
            param_name="n_layers",
            param_values=n_layers_grid,
            seed=seed,
            cat_feature_cols=categorical_cols,
            n_cat=n_cat,
            cat_cards=cat_cards,
        )
        all_sweep_results["n_layers"] = results
        _save_sweep(results_dir / "n_layers_sweep.json", results)

    # ── d_token sweep ────────────────────────────────────────────────
    d_token_grid = _cfg(config, "sweep.d_token", [64, 128, 192, 256])
    if d_token_grid:
        log.info("Running d_token sweep: %s", d_token_grid)
        results = _run_architecture_sweep(
            splits, feature_cols, outcome, device,
            param_name="d_token",
            param_values=d_token_grid,
            seed=seed,
            cat_feature_cols=categorical_cols,
            n_cat=n_cat,
            cat_cards=cat_cards,
        )
        all_sweep_results["d_token"] = results
        _save_sweep(results_dir / "d_token_sweep.json", results)

    # ── train_fraction sweep (scaling curve) ─────────────────────────
    fractions = _cfg(config, "sweep.train_fractions", [0.10, 0.25, 0.50, 0.75, 1.0])
    if fractions:
        log.info("Running train_fraction sweep: %s", fractions)
        results = _run_fraction_sweep(
            splits, feature_cols, outcome, device,
            fractions=fractions,
            seed=seed,
            cat_feature_cols=categorical_cols,
            n_cat=n_cat,
            cat_cards=cat_cards,
        )
        all_sweep_results["train_fraction"] = results
        _save_sweep(results_dir / "train_fraction_sweep.json", results)

    # ── mask_ratio sweep (SSL pretrain + fine-tune vs from-scratch) ──
    mask_ratios = _cfg(config, "sweep.mask_ratio", [0.10, 0.15, 0.20, 0.30])
    if mask_ratios:
        log.info("Running mask_ratio sweep: %s", mask_ratios)
        results = _run_mask_ratio_sweep(
            splits, feature_cols, outcome, device,
            mask_ratios=mask_ratios,
            seed=seed,
            cat_feature_cols=categorical_cols,
            n_cat=n_cat,
            cat_cards=cat_cards,
        )
        all_sweep_results["mask_ratio"] = results
        _save_sweep(results_dir / "mask_ratio_sweep.json", results)

    log.info("All ablation sweeps complete. Results in %s", results_dir)
    return all_sweep_results


# ── Sweep helpers ────────────────────────────────────────────────────────


def _run_architecture_sweep(
    splits: dict,
    feature_cols: list[str],
    outcome: str,
    device: torch.device,
    param_name: str,
    param_values: list,
    seed: int = 42,
    cat_feature_cols: list[str] | None = None,
    n_cat: int = 0,
    cat_cards: list[int] | None = None,
) -> list[dict]:
    """Sweep a single FT-Transformer hyperparameter, holding others at default."""
    results = []

    for value in param_values:
        seed_everything(seed)
        log.info("  %s=%s", param_name, value)

        model_kwargs = {
            "n_features": len(feature_cols),
            "d_token": _BENCHMARK_DEFAULTS["d_token"],
            "n_heads": _BENCHMARK_DEFAULTS["n_heads"],
            "n_layers": _BENCHMARK_DEFAULTS["n_layers"],
            "d_ffn_factor": _BENCHMARK_DEFAULTS["d_ffn_factor"],
            "dropout": _BENCHMARK_DEFAULTS["dropout"],
            "n_outputs": 1,
            "n_cat_features": n_cat,
            "cat_cardinalities": cat_cards,
        }
        model_kwargs[param_name] = int(value)

        # Adjust n_heads if d_token changes to stay divisible
        if param_name == "d_token":
            d = int(value)
            # Use largest power-of-2 head count ≤ 8 that divides d_token
            for nh in [8, 4, 2, 1]:
                if d % nh == 0:
                    model_kwargs["n_heads"] = nh
                    break

        metrics = _train_and_evaluate(
            splits, feature_cols, outcome, device, model_kwargs,
            cat_feature_cols=cat_feature_cols,
        )
        results.append({param_name: value, **metrics})

    return results


def _run_fraction_sweep(
    splits: dict,
    feature_cols: list[str],
    outcome: str,
    device: torch.device,
    fractions: list[float],
    seed: int = 42,
    cat_feature_cols: list[str] | None = None,
    n_cat: int = 0,
    cat_cards: list[int] | None = None,
) -> list[dict]:
    """Sweep training set size (scaling curve)."""
    results = []
    full_train = splits["train"]

    for frac in fractions:
        seed_everything(seed)
        n_sample = max(1, int(len(full_train) * frac))
        subsample = full_train.sample(n=n_sample, random_state=seed)
        log.info("  train_fraction=%.2f (%d rows)", frac, n_sample)

        sub_splits = {**splits, "train": subsample}
        model_kwargs = {
            "n_features": len(feature_cols),
            "d_token": _BENCHMARK_DEFAULTS["d_token"],
            "n_heads": _BENCHMARK_DEFAULTS["n_heads"],
            "n_layers": _BENCHMARK_DEFAULTS["n_layers"],
            "d_ffn_factor": _BENCHMARK_DEFAULTS["d_ffn_factor"],
            "dropout": _BENCHMARK_DEFAULTS["dropout"],
            "n_outputs": 1,
            "n_cat_features": n_cat,
            "cat_cardinalities": cat_cards,
        }
        metrics = _train_and_evaluate(
            sub_splits, feature_cols, outcome, device, model_kwargs,
            cat_feature_cols=cat_feature_cols,
        )
        results.append({"train_fraction": frac, "n_train": n_sample, **metrics})

    return results


def _train_and_evaluate(
    splits: dict,
    feature_cols: list[str],
    outcome: str,
    device: torch.device,
    model_kwargs: dict,
    init_state_dict: dict | None = None,
    cat_feature_cols: list[str] | None = None,
    lr: float | None = None,
) -> dict:
    """Train an FT-Transformer and return test metrics.

    Parameters
    ----------
    init_state_dict:
        If provided, load these weights into the model before fine-tuning
        (e.g. from SSL pretraining). Uses ``strict=False`` so the
        classification head can differ.
    cat_feature_cols:
        Categorical feature column names to pass to the dataloader.
    lr:
        Learning rate override.  Falls back to
        ``_BENCHMARK_DEFAULTS["learning_rate"]`` when *None*.
    """
    train_df = splits["train"][splits["train"][outcome].notna()]
    val_df = splits["val"][splits["val"][outcome].notna()]
    test_df = splits["test"][splits["test"][outcome].notna()]

    n_pos = int(train_df[outcome].sum())
    if n_pos < MIN_POSITIVES:
        log.warning("Fewer than %d positives — returning null metrics.", MIN_POSITIVES)
        return {"auroc": None, "auprc": None, "skipped": True}

    n_neg = len(train_df) - n_pos
    pos_weight = n_neg / max(n_pos, 1)

    batch_size = _BENCHMARK_DEFAULTS["batch_size"]
    _cat_cols = cat_feature_cols or None
    train_loader = make_dataloader(train_df, feature_cols, outcome, batch_size, shuffle=True, cat_feature_cols=_cat_cols)
    val_loader = make_dataloader(val_df, feature_cols, outcome, batch_size, shuffle=False, cat_feature_cols=_cat_cols)
    test_loader = make_dataloader(test_df, feature_cols, outcome, batch_size, shuffle=False, cat_feature_cols=_cat_cols)

    model = FTTransformer(**model_kwargs).to(device)
    if init_state_dict is not None:
        model.load_state_dict(init_state_dict, strict=False)
    _lr = lr if lr is not None else _BENCHMARK_DEFAULTS["learning_rate"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=_lr,
        weight_decay=_BENCHMARK_DEFAULTS["weight_decay"],
    )
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight], device=device),
    )

    train_ft_transformer(
        model, train_loader, val_loader, criterion, optimizer,
        device,
        epochs=_BENCHMARK_DEFAULTS["epochs"],
        patience=_BENCHMARK_DEFAULTS["patience"],
    )

    y_true, y_score = _predict_scores(model, test_loader, device)
    metrics = compute_all_metrics(y_true, y_score)
    return metrics


def _run_mask_ratio_sweep(
    splits: dict,
    feature_cols: list[str],
    outcome: str,
    device: torch.device,
    mask_ratios: list[float],
    seed: int = 42,
    pretrain_epochs: int = 50,
    cat_feature_cols: list[str] | None = None,
    n_cat: int = 0,
    cat_cards: list[int] | None = None,
) -> list[dict]:
    """Sweep SSL mask ratio: pretrain with each ratio, fine-tune, compare."""
    results = []
    n_features = len(feature_cols)
    model_kwargs = {
        "n_features": n_features,
        "d_token": _BENCHMARK_DEFAULTS["d_token"],
        "n_heads": _BENCHMARK_DEFAULTS["n_heads"],
        "n_layers": _BENCHMARK_DEFAULTS["n_layers"],
        "d_ffn_factor": _BENCHMARK_DEFAULTS["d_ffn_factor"],
        "dropout": _BENCHMARK_DEFAULTS["dropout"],
        "n_outputs": 1,
        "n_cat_features": n_cat,
        "cat_cardinalities": cat_cards,
    }

    has_cat = bool(cat_feature_cols)
    _cat_cols = cat_feature_cols or None

    for ratio in mask_ratios:
        seed_everything(seed)
        log.info("  mask_ratio=%.2f", ratio)

        # ── SSL pretraining ──────────────────────────────────────────
        encoder = FTTransformer(**model_kwargs)
        ssl_model = MaskedFeatureSSL(encoder, mask_ratio=ratio).to(device)
        ssl_optimizer = torch.optim.AdamW(
            ssl_model.parameters(), lr=1e-4, weight_decay=1e-5,
        )

        ssl_loader = make_dataloader(
            splits["train"], feature_cols, label_col=None,
            batch_size=_BENCHMARK_DEFAULTS["batch_size"], shuffle=True,
            cat_feature_cols=_cat_cols,
        )

        for _epoch in range(pretrain_epochs):
            ssl_model.train()
            for batch in ssl_loader:
                if has_cat and len(batch) >= 2:
                    x_batch, x_cat_batch = batch[0].to(device), batch[1].to(device)
                else:
                    x_batch = batch[0].to(device)
                    x_cat_batch = None
                ssl_optimizer.zero_grad()
                loss, _, _ = ssl_model(x_batch, x_cat=x_cat_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(ssl_model.parameters(), max_norm=1.0)
                ssl_optimizer.step()

        # ── Fine-tune pretrained encoder on downstream task ──────────
        pretrained_state = encoder.state_dict()

        metrics = _train_and_evaluate(
            splits, feature_cols, outcome, device, model_kwargs,
            init_state_dict=pretrained_state,
            cat_feature_cols=cat_feature_cols,
        )
        metrics["mask_ratio"] = ratio
        results.append(metrics)

    return results


def _save_sweep(path: Path, results: list[dict]) -> None:
    """Write sweep results to JSON."""
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    log.info("Sweep results saved to %s", path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    try:
        import hydra
        from omegaconf import DictConfig

        @hydra.main(config_path="../../../configs/study0", config_name="ablations", version_base=None)
        def main(cfg: DictConfig) -> None:
            run_ablations(cfg)

        main()
    except ImportError:
        log.warning("Hydra not available — running with empty config.")
        run_ablations({})

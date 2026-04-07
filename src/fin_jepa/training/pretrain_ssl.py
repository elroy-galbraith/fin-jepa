"""
Self-supervised pretraining script (masked feature reconstruction).

Workstream: Self-supervised pretraining experiment.

Usage:
    python -m fin_jepa.training.pretrain_ssl experiment=study0/pretrain
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader

from fin_jepa.data.feature_engineering import (
    CATEGORICAL_FEATURES,
    N_SECTORS,
    FeatureConfig,
    build_feature_matrix,
)
from fin_jepa.data.splits import SplitConfig
from fin_jepa.data.xbrl_loader import load_xbrl_features
from fin_jepa.models.ft_transformer import FTTransformer
from fin_jepa.models.ssl_head import MaskedFeatureSSL
from fin_jepa.training.dataset import make_dataloader
from fin_jepa.utils.reproducibility import seed_everything

log = logging.getLogger(__name__)


# ── Reusable pretraining helper ──────────────────────────────────────────


def _pretrain_encoder(
    encoder: FTTransformer,
    train_loader: DataLoader,
    mask_ratio: float,
    device: torch.device,
    epochs: int = 200,
    lr: float = 1e-4,
    wd: float = 1e-5,
    warmup_epochs: int = 10,
) -> tuple[dict, list[float]]:
    """Pretrain an encoder via masked feature reconstruction.

    Parameters
    ----------
    encoder : FTTransformer
        The encoder to pretrain (modified in-place).
    train_loader : DataLoader
        Unlabelled feature batches (each item is a tuple ``(x_batch,)``).
    mask_ratio : float
        Fraction of features to mask per sample.
    device : torch.device
        Training device.
    epochs, lr, wd, warmup_epochs
        Training hyperparameters.

    Returns
    -------
    (state_dict, loss_history)
        ``state_dict`` — encoder weights after pretraining.
        ``loss_history`` — per-epoch average reconstruction loss.
    """
    ssl_model = MaskedFeatureSSL(encoder, mask_ratio=mask_ratio).to(device)
    optimizer = torch.optim.AdamW(ssl_model.parameters(), lr=lr, weight_decay=wd)

    warmup_scheduler = LinearLR(optimizer, start_factor=1e-3, total_iters=warmup_epochs)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=max(epochs - warmup_epochs, 1))
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_epochs],
    )

    loss_history: list[float] = []

    has_cat = hasattr(ssl_model.encoder.tokenizer, "n_cat_features") and ssl_model.encoder.tokenizer.n_cat_features > 0

    for epoch in range(1, epochs + 1):
        ssl_model.train()
        total_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            if has_cat and len(batch) >= 2:
                x_batch, x_cat_batch = batch[0].to(device), batch[1].to(device)
            else:
                x_batch = batch[0].to(device)
                x_cat_batch = None
            optimizer.zero_grad()
            loss, _x_hat, _mask = ssl_model(x_batch, x_cat=x_cat_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ssl_model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = total_loss / max(n_batches, 1)
        loss_history.append(avg_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        if epoch <= 5 or epoch % 10 == 0 or epoch == epochs:
            log.info(
                "Epoch %3d/%d | loss=%.6f | lr=%.2e",
                epoch, epochs, avg_loss, current_lr,
            )

    return encoder.state_dict(), loss_history


# ── Original entry point (single-ratio pretraining) ─────────────────────


def run_pretraining(config) -> Path:
    """Entry point for SSL pretraining experiment.

    Parameters
    ----------
    config : dict or DictConfig
        Hydra configuration (from ``configs/study0/pretrain.yaml``).

    Returns
    -------
    Path to the saved encoder checkpoint.
    """
    from fin_jepa.training.train_study0 import _cfg

    seed = int(_cfg(config, "training.seed", 42))
    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    # ── Load data (train split only) ─────────────────────────────────
    raw_dir = Path(_cfg(config, "data.raw_dir", "data/raw"))

    xbrl_df = load_xbrl_features(raw_dir)

    split_cfg = SplitConfig(
        train_end=_cfg(config, "data.split.train_end", "2017-12-31"),
        val_end=_cfg(config, "data.split.val_end", "2019-12-31"),
        test_end=_cfg(config, "data.split.test_end", "2023-12-31"),
    )
    feat_cfg = FeatureConfig(
        use_raw=_cfg(config, "features.use_raw", True),
        use_ratios=_cfg(config, "features.use_ratios", True),
        use_yoy=_cfg(config, "features.use_yoy", True),
        use_sic=_cfg(config, "features.use_sic", True),
        use_missingness_flags=_cfg(config, "features.use_missingness_flags", True),
        coverage_threshold=_cfg(config, "features.coverage_threshold", 0.50),
        normalization_method=_cfg(config, "features.normalization_method", "quantile"),
        median_impute=_cfg(config, "features.median_impute", True),
    )

    # Load universe for SIC join
    universe_df = None
    universe_path = raw_dir / "company_universe.parquet"
    if universe_path.exists() and feat_cfg.use_sic:
        universe_df = pd.read_parquet(universe_path)

    splits, scaler, feature_cols, categorical_cols = build_feature_matrix(
        xbrl_df, split_cfg, feat_cfg, universe_df=universe_df,
    )

    train_loader = make_dataloader(
        splits["train"],
        feature_cols,
        label_col=None,
        batch_size=int(_cfg(config, "training.batch_size", 512)),
        shuffle=True,
        cat_feature_cols=categorical_cols or None,
    )

    n_features = len(feature_cols)
    n_cat = len(categorical_cols)
    cat_cards = [N_SECTORS] * n_cat if n_cat > 0 else None
    log.info(
        "Training SSL on %d continuous + %d categorical features, %d train rows.",
        n_features, n_cat, len(splits["train"]),
    )

    # ── Model ────────────────────────────────────────────────────────
    encoder = FTTransformer(
        n_features=n_features,
        d_token=int(_cfg(config, "ft_transformer.d_token", 192)),
        n_heads=int(_cfg(config, "ft_transformer.n_heads", 8)),
        n_layers=int(_cfg(config, "ft_transformer.n_layers", 3)),
        d_ffn_factor=int(_cfg(config, "ft_transformer.d_ffn_factor", 4)),
        dropout=float(_cfg(config, "ft_transformer.dropout", 0.0)),
        n_outputs=1,
        n_cat_features=n_cat,
        cat_cardinalities=cat_cards,
    )
    mask_ratio = float(_cfg(config, "ssl.mask_ratio", 0.15))

    state_dict, _loss_history = _pretrain_encoder(
        encoder,
        train_loader,
        mask_ratio=mask_ratio,
        device=device,
        epochs=int(_cfg(config, "training.epochs", 200)),
        lr=float(_cfg(config, "training.learning_rate", 1e-4)),
        wd=float(_cfg(config, "training.weight_decay", 1e-5)),
        warmup_epochs=int(_cfg(config, "training.warmup_epochs", 10)),
    )

    # ── Save checkpoint ──────────────────────────────────────────────
    ckpt_dir = Path(_cfg(config, "checkpoint_dir", "models/checkpoints"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "encoder_pretrained.pt"
    torch.save(state_dict, ckpt_path)
    log.info("Pretrained encoder saved to %s", ckpt_path)

    return ckpt_path


# ── SSL Experiment: pretrained vs from-scratch (ATS-167) ─────────────────


_DEFAULT_OUTCOMES = [
    "stock_decline",
    "earnings_restate",
    "audit_qualification",
    "sec_enforcement",
    "bankruptcy",
]


def run_ssl_experiment(
    config,
    *,
    prebuilt_splits: dict | None = None,
    prebuilt_feature_cols: list[str] | None = None,
    prebuilt_cat_cols: list[str] | None = None,
) -> dict:
    """Run the full SSL pretraining experiment (ATS-167).

    For each mask ratio, pretrain an encoder (200 epochs, warmup + cosine),
    then fine-tune on all 5 distress outcomes.  Also runs a from-scratch
    baseline for comparison.

    Parameters
    ----------
    config : dict or DictConfig
        Hydra configuration (from ``configs/study0/ssl_experiment.yaml``).
    prebuilt_splits : dict, optional
        Pre-built ``{train, val, test}`` DataFrames from
        ``build_feature_matrix``.  When provided, the function skips
        internal data loading and reuses the caller's preprocessing
        pipeline — ensuring scratch baselines are identical across
        experiments (ATS-217).
    prebuilt_feature_cols : list[str], optional
        Continuous feature columns matching *prebuilt_splits*.
    prebuilt_cat_cols : list[str], optional
        Categorical feature columns matching *prebuilt_splits*.

    Returns
    -------
    dict with keys ``baseline``, ``pretrained``, ``loss_curves``,
    ``comparison``, ``recommendation``.
    """
    from fin_jepa.data.labels import load_label_database
    from fin_jepa.training.ablations import _BENCHMARK_DEFAULTS, _train_and_evaluate
    from fin_jepa.training.train_study0 import _cfg

    seed = int(_cfg(config, "training.seed", 42))
    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("SSL experiment — device: %s", device)

    # ── Load data ────────────────────────────────────────────────────
    if prebuilt_splits is not None:
        # Reuse caller's preprocessing pipeline (ATS-217: eliminates
        # QuantileTransformer divergence between benchmark and SSL tables).
        splits = prebuilt_splits
        feature_cols = prebuilt_feature_cols or []
        categorical_cols = prebuilt_cat_cols or []
        log.info("SSL experiment: using prebuilt splits (%d train rows).", len(splits["train"]))
    else:
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
        feat_cfg = FeatureConfig(
            use_raw=_cfg(config, "features.use_raw", True),
            use_ratios=_cfg(config, "features.use_ratios", True),
            use_yoy=_cfg(config, "features.use_yoy", True),
            use_sic=_cfg(config, "features.use_sic", True),
            use_missingness_flags=_cfg(config, "features.use_missingness_flags", True),
            coverage_threshold=_cfg(config, "features.coverage_threshold", 0.50),
            normalization_method=_cfg(config, "features.normalization_method", "quantile"),
            median_impute=_cfg(config, "features.median_impute", True),
        )

        # Load universe for SIC join
        universe_df = None
        universe_path = raw_dir / "company_universe.parquet"
        if universe_path.exists() and feat_cfg.use_sic:
            universe_df = pd.read_parquet(universe_path)

        splits, scaler, feature_cols, categorical_cols = build_feature_matrix(
            merged, split_cfg, feat_cfg, universe_df=universe_df,
        )

    n_features = len(feature_cols)
    n_cat = len(categorical_cols)
    cat_cards = [N_SECTORS] * n_cat if n_cat > 0 else None
    log.info(
        "SSL experiment: %d continuous + %d categorical features, %d train rows.",
        n_features, n_cat, len(splits["train"]),
    )

    outcomes = _cfg(config, "outcomes", _DEFAULT_OUTCOMES)
    mask_ratios = _cfg(config, "ssl_experiment.mask_ratios", [0.15, 0.30, 0.50])

    model_kwargs = {
        "n_features": n_features,
        "d_token": int(_cfg(config, "ft_transformer.d_token", _BENCHMARK_DEFAULTS["d_token"])),
        "n_heads": int(_cfg(config, "ft_transformer.n_heads", _BENCHMARK_DEFAULTS["n_heads"])),
        "n_layers": int(_cfg(config, "ft_transformer.n_layers", _BENCHMARK_DEFAULTS["n_layers"])),
        "d_ffn_factor": int(_cfg(config, "ft_transformer.d_ffn_factor", _BENCHMARK_DEFAULTS["d_ffn_factor"])),
        "dropout": float(_cfg(config, "ft_transformer.dropout", _BENCHMARK_DEFAULTS["dropout"])),
        "n_outputs": 1,
        "n_cat_features": n_cat,
        "cat_cardinalities": cat_cards,
    }

    pretrain_epochs = int(_cfg(config, "pretrain.epochs", 200))
    pretrain_lr = float(_cfg(config, "pretrain.learning_rate", 1e-4))
    pretrain_wd = float(_cfg(config, "pretrain.weight_decay", 1e-5))
    pretrain_warmup = int(_cfg(config, "pretrain.warmup_epochs", 10))
    pretrain_batch_size = int(_cfg(config, "pretrain.batch_size", 512))

    # ATS-217: fine-tuning lr can be overridden via config so the SSL
    # baseline matches the final-benchmark ft_scratch exactly.
    finetune_lr = float(
        _cfg(config, "training.learning_rate",
             _BENCHMARK_DEFAULTS["learning_rate"])
    )

    # ── FROM-SCRATCH BASELINE ────────────────────────────────────────
    log.info("Running from-scratch baseline on %d outcomes...", len(outcomes))
    baseline_results: dict[str, dict] = {}
    for outcome in outcomes:
        seed_everything(seed)
        log.info("  baseline — %s", outcome)
        metrics = _train_and_evaluate(
            splits, feature_cols, outcome, device, model_kwargs,
            cat_feature_cols=categorical_cols,
            lr=finetune_lr,
        )
        baseline_results[outcome] = metrics

    # ── PRETRAINED VARIANTS ──────────────────────────────────────────
    pretrained_results: dict[str, dict[str, dict]] = {}
    loss_curves: dict[str, list[float]] = {}

    force_pretrain = bool(_cfg(config, "ssl_experiment.force_pretrain", False))
    ckpt_dir = Path(_cfg(config, "checkpoint_dir", "models/checkpoints/study0_ssl_experiment"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    for ratio in mask_ratios:
        ratio_key = f"{ratio:.2f}"
        ckpt_path = ckpt_dir / f"encoder_mr{ratio_key}.pt"

        if not force_pretrain and ckpt_path.exists():
            log.info(
                "Skipping pretraining for mask_ratio=%.2f — loading checkpoint from %s",
                ratio, ckpt_path,
            )
            state_dict = torch.load(ckpt_path, map_location=device)
            loss_curves[ratio_key] = []
        else:
            log.info("Pretraining with mask_ratio=%.2f ...", ratio)
            seed_everything(seed)

            # Re-create DataLoader after re-seeding so shuffle order is
            # identical across mask ratios (reproducibility).
            ssl_loader = make_dataloader(
                splits["train"], feature_cols, label_col=None,
                batch_size=pretrain_batch_size, shuffle=True,
                cat_feature_cols=categorical_cols or None,
            )

            encoder = FTTransformer(**model_kwargs)
            state_dict, losses = _pretrain_encoder(
                encoder, ssl_loader,
                mask_ratio=ratio,
                device=device,
                epochs=pretrain_epochs,
                lr=pretrain_lr,
                wd=pretrain_wd,
                warmup_epochs=pretrain_warmup,
            )
            loss_curves[ratio_key] = losses
            torch.save(state_dict, ckpt_path)
            log.info("  checkpoint saved to %s", ckpt_path)

        pretrained_results[ratio_key] = {}
        for outcome in outcomes:
            seed_everything(seed)
            log.info("  fine-tune mr=%.2f — %s", ratio, outcome)
            metrics = _train_and_evaluate(
                splits, feature_cols, outcome, device, model_kwargs,
                init_state_dict=state_dict,
                cat_feature_cols=categorical_cols,
                lr=finetune_lr,
            )
            pretrained_results[ratio_key][outcome] = metrics

    # ── COMPARISON ───────────────────────────────────────────────────
    comparison: dict[str, dict] = {}
    n_improved = 0

    for outcome in outcomes:
        baseline_auroc = baseline_results[outcome].get("auroc")
        if baseline_auroc is None:
            comparison[outcome] = {"skipped": True}
            continue

        best_auroc = baseline_auroc
        best_ratio = "none"
        for ratio_key, outcome_metrics in pretrained_results.items():
            pt_auroc = outcome_metrics[outcome].get("auroc")
            if pt_auroc is not None and pt_auroc > best_auroc:
                best_auroc = pt_auroc
                best_ratio = ratio_key

        delta = best_auroc - baseline_auroc
        if delta >= 0.005:
            n_improved += 1

        comparison[outcome] = {
            "baseline_auroc": baseline_auroc,
            "best_pretrained_auroc": best_auroc,
            "best_mask_ratio": best_ratio,
            "delta": round(delta, 6),
        }

    if n_improved >= 3:
        recommendation = "proceed"
    elif n_improved >= 1:
        recommendation = "marginal"
    else:
        recommendation = "no_gain"

    log.info("SSL experiment recommendation: %s (%d/5 outcomes improved)", recommendation, n_improved)

    # ── Save results ─────────────────────────────────────────────────
    output = {
        "baseline": baseline_results,
        "pretrained": pretrained_results,
        "loss_curves": loss_curves,
        "comparison": comparison,
        "recommendation": recommendation,
    }

    results_dir = Path(_cfg(config, "results_dir", "results/study0"))
    results_dir.mkdir(parents=True, exist_ok=True)
    results_path = results_dir / "ssl_experiment_results.json"
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    log.info("Results saved to %s", results_path)

    # ── Optional MLflow logging ──────────────────────────────────────
    try:
        import mlflow

        with mlflow.start_run(run_name="study0_ssl_experiment"):
            mlflow.log_param("mask_ratios", mask_ratios)
            mlflow.log_param("pretrain_epochs", pretrain_epochs)
            mlflow.log_metric("recommendation_n_improved", n_improved)
            for outcome, comp in comparison.items():
                if not comp.get("skipped"):
                    mlflow.log_metric(f"{outcome}_baseline_auroc", comp["baseline_auroc"])
                    mlflow.log_metric(f"{outcome}_best_pretrained_auroc", comp["best_pretrained_auroc"])
                    mlflow.log_metric(f"{outcome}_delta", comp["delta"])
            mlflow.log_artifact(str(results_path))
    except Exception as exc:
        log.debug("MLflow logging skipped: %s", exc)

    return output


# ── Multi-seed SSL evaluation (ATS-252) ───────────────────────────────────


def run_multiseed_ssl(
    config,
    seeds: list[int] | None = None,
    *,
    prebuilt_splits: dict | None = None,
    prebuilt_feature_cols: list[str] | None = None,
    prebuilt_cat_cols: list[str] | None = None,
) -> dict:
    """Run SSL pretraining + fine-tuning across multiple seeds.

    For each seed, pretrain with each mask ratio (200 epochs), fine-tune
    on all outcomes, and evaluate on test.  Reports mean +/- std AUROC
    per (outcome, mask_ratio) to confirm which SSL gains survive
    initialisation variance.

    Parameters
    ----------
    config : dict or DictConfig
        Hydra configuration (from ``configs/study0/ssl_experiment.yaml``).
    seeds : list[int] or None
        Override seeds list; falls back to ``ssl_experiment.seeds``,
        then ``training.seeds``, then ``[42, 123, 456]``.
    prebuilt_splits : dict or None
        If provided, reuse caller's preprocessed train/val/test splits
        (ATS-217 consistency pattern).
    prebuilt_feature_cols : list[str] or None
        Feature column names corresponding to *prebuilt_splits*.
    prebuilt_cat_cols : list[str] or None
        Categorical column names corresponding to *prebuilt_splits*.

    Returns
    -------
    dict with keys ``multiseed_ssl`` (per-outcome per-ratio stats),
    ``seeds``, and ``mask_ratios``.
    """
    from fin_jepa.data.labels import load_label_database
    from fin_jepa.training.ablations import _BENCHMARK_DEFAULTS, _train_and_evaluate
    from fin_jepa.training.train_study0 import _cfg

    if seeds is None:
        seeds = list(
            _cfg(config, "ssl_experiment.seeds",
                 _cfg(config, "training.seeds", [42, 123, 456]))
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Multi-seed SSL | seeds=%s | device=%s", seeds, device)

    # ── Load data ────────────────────────────────────────────────────
    if prebuilt_splits is not None:
        splits = prebuilt_splits
        feature_cols = prebuilt_feature_cols or []
        categorical_cols = prebuilt_cat_cols or []
        log.info(
            "Multi-seed SSL: using prebuilt splits (%d train rows).",
            len(splits["train"]),
        )
    else:
        raw_dir = Path(_cfg(config, "data.raw_dir", "data/raw"))
        processed_dir = Path(_cfg(config, "data.processed_dir", "data/processed"))

        xbrl_df = load_xbrl_features(raw_dir)
        labels_df, _ = load_label_database(processed_dir / "label_database.parquet")
        xbrl_df["period_end"] = pd.to_datetime(xbrl_df["period_end"])
        labels_df["period_end"] = pd.to_datetime(labels_df["period_end"])
        merged = xbrl_df.merge(
            labels_df, on=["cik", "period_end"], how="inner", suffixes=("", "_label"),
        )

        split_cfg = SplitConfig(
            train_end=_cfg(config, "data.split.train_end", "2017-12-31"),
            val_end=_cfg(config, "data.split.val_end", "2019-12-31"),
            test_end=_cfg(config, "data.split.test_end", "2023-12-31"),
        )
        feat_cfg = FeatureConfig(
            use_raw=_cfg(config, "features.use_raw", True),
            use_ratios=_cfg(config, "features.use_ratios", True),
            use_yoy=_cfg(config, "features.use_yoy", True),
            use_sic=_cfg(config, "features.use_sic", True),
            use_missingness_flags=_cfg(config, "features.use_missingness_flags", True),
            coverage_threshold=_cfg(config, "features.coverage_threshold", 0.50),
            normalization_method=_cfg(config, "features.normalization_method", "quantile"),
            median_impute=_cfg(config, "features.median_impute", True),
        )

        universe_df = None
        universe_path = raw_dir / "company_universe.parquet"
        if universe_path.exists() and feat_cfg.use_sic:
            universe_df = pd.read_parquet(universe_path)

        splits, _scaler, feature_cols, categorical_cols = build_feature_matrix(
            merged, split_cfg, feat_cfg, universe_df=universe_df,
        )

    n_features = len(feature_cols)
    n_cat = len(categorical_cols)
    cat_cards = [N_SECTORS] * n_cat if n_cat > 0 else None

    outcomes = _cfg(config, "outcomes", _DEFAULT_OUTCOMES)
    mask_ratios = _cfg(config, "ssl_experiment.mask_ratios", [0.15, 0.30, 0.50])

    model_kwargs = {
        "n_features": n_features,
        "d_token": int(_cfg(config, "ft_transformer.d_token", _BENCHMARK_DEFAULTS["d_token"])),
        "n_heads": int(_cfg(config, "ft_transformer.n_heads", _BENCHMARK_DEFAULTS["n_heads"])),
        "n_layers": int(_cfg(config, "ft_transformer.n_layers", _BENCHMARK_DEFAULTS["n_layers"])),
        "d_ffn_factor": int(_cfg(config, "ft_transformer.d_ffn_factor", _BENCHMARK_DEFAULTS["d_ffn_factor"])),
        "dropout": float(_cfg(config, "ft_transformer.dropout", _BENCHMARK_DEFAULTS["dropout"])),
        "n_outputs": 1,
        "n_cat_features": n_cat,
        "cat_cardinalities": cat_cards,
    }

    pretrain_epochs = int(_cfg(config, "pretrain.epochs", 200))
    pretrain_lr = float(_cfg(config, "pretrain.learning_rate", 1e-4))
    pretrain_wd = float(_cfg(config, "pretrain.weight_decay", 1e-5))
    pretrain_warmup = int(_cfg(config, "pretrain.warmup_epochs", 10))
    pretrain_batch_size = int(_cfg(config, "pretrain.batch_size", 512))
    finetune_lr = float(
        _cfg(config, "training.learning_rate", _BENCHMARK_DEFAULTS["learning_rate"]),
    )

    # per_seed[outcome]["scratch"|ratio_key] = list of AUROC per seed
    per_seed: dict[str, dict[str, list[float]]] = {}
    for outcome in outcomes:
        per_seed[outcome] = {"scratch": []}
        for ratio in mask_ratios:
            per_seed[outcome][f"{ratio:.2f}"] = []

    for seed in seeds:
        log.info("═══ Seed %d ═══════════════════════════════════════════", seed)

        # ── From-scratch baseline ───────────────────────────────────
        for outcome in outcomes:
            seed_everything(seed)
            metrics = _train_and_evaluate(
                splits, feature_cols, outcome, device, model_kwargs,
                cat_feature_cols=categorical_cols,
                lr=finetune_lr,
            )
            auroc = metrics.get("auroc")
            if auroc is not None:
                per_seed[outcome]["scratch"].append(float(auroc))
                log.info("  scratch | seed=%d | %s | AUROC=%.4f", seed, outcome, auroc)

        # ── Pretrained variants ─────────────────────────────────────
        for ratio in mask_ratios:
            ratio_key = f"{ratio:.2f}"
            log.info("  Pretraining mask_ratio=%.2f seed=%d ...", ratio, seed)
            seed_everything(seed)

            ssl_loader = make_dataloader(
                splits["train"], feature_cols, label_col=None,
                batch_size=pretrain_batch_size, shuffle=True,
                cat_feature_cols=categorical_cols or None,
            )

            encoder = FTTransformer(**model_kwargs)
            state_dict, _losses = _pretrain_encoder(
                encoder, ssl_loader,
                mask_ratio=ratio,
                device=device,
                epochs=pretrain_epochs,
                lr=pretrain_lr,
                wd=pretrain_wd,
                warmup_epochs=pretrain_warmup,
            )

            for outcome in outcomes:
                seed_everything(seed)
                metrics = _train_and_evaluate(
                    splits, feature_cols, outcome, device, model_kwargs,
                    init_state_dict=state_dict,
                    cat_feature_cols=categorical_cols,
                    lr=finetune_lr,
                )
                auroc = metrics.get("auroc")
                if auroc is not None:
                    per_seed[outcome][ratio_key].append(float(auroc))
                    log.info(
                        "  mr=%.2f | seed=%d | %s | AUROC=%.4f",
                        ratio, seed, outcome, auroc,
                    )

            # Free GPU memory after each (seed, ratio) pretraining
            del encoder, state_dict
            torch.cuda.empty_cache()

    # ── Aggregate ────────────────────────────────────────────────────
    aggregated: dict[str, dict] = {}
    for outcome in outcomes:
        aggregated[outcome] = {}
        for key, aurocs in per_seed[outcome].items():
            if aurocs:
                aggregated[outcome][key] = {
                    "mean_auroc": float(np.mean(aurocs)),
                    "std_auroc": float(np.std(aurocs)),
                    "per_seed": {str(s): a for s, a in zip(seeds, aurocs)},
                }
                log.info(
                    "  %s | %s | mean=%.4f ± %.4f (n=%d)",
                    outcome, key,
                    aggregated[outcome][key]["mean_auroc"],
                    aggregated[outcome][key]["std_auroc"],
                    len(aurocs),
                )

    output: dict = {
        "multiseed_ssl": aggregated,
        "seeds": list(seeds),
        "mask_ratios": mask_ratios,
    }

    results_dir = Path(_cfg(config, "results_dir", "results/study0"))
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / "multiseed_ssl.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    log.info("Multi-seed SSL results saved to %s", out_path)

    return output


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    # Dispatch: use ssl_experiment config → run_ssl_experiment,
    #           otherwise default pretrain config → run_pretraining.
    _use_multiseed_ssl = any("multiseed_ssl" in arg for arg in sys.argv)
    _use_ssl_experiment = any("ssl_experiment" in arg for arg in sys.argv)
    if _use_multiseed_ssl:
        _config_name = "ssl_experiment"
        _entry_fn = run_multiseed_ssl
    elif _use_ssl_experiment:
        _config_name = "ssl_experiment"
        _entry_fn = run_ssl_experiment
    else:
        _config_name = "pretrain"
        _entry_fn = run_pretraining

    try:
        import hydra
        from omegaconf import DictConfig

        @hydra.main(config_path="../../../configs/study0", config_name=_config_name, version_base=None)
        def main(cfg: DictConfig) -> None:
            _entry_fn(cfg)

        main()
    except ImportError:
        log.warning("Hydra not available — running with empty config.")
        _entry_fn({})

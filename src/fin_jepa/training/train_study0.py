"""
Study 0 benchmark training script.

Workstream: Run FT-Transformer vs. baselines benchmark (go/no-go gate).

Usage:
    python -m fin_jepa.training.train_study0 experiment=study0/benchmark
"""

from __future__ import annotations

import copy
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from sklearn.metrics import roc_auc_score

from fin_jepa.data.feature_engineering import (
    N_SECTORS,
    TRADITIONAL_RATIO_FEATURES,
    FeatureConfig,
    RAW_FEATURES,
    build_feature_matrix,
)
from fin_jepa.data.labels import load_label_database
from fin_jepa.data.splits import RollingSplitConfig, SplitConfig, make_rolling_splits
from fin_jepa.data.xbrl_loader import load_xbrl_features
from fin_jepa.models.baselines import (
    build_gbt,
    build_logistic_regression,
    build_xgboost,
)
from fin_jepa.models.ft_transformer import FTTransformer
from fin_jepa.training.dataset import make_dataloader
from fin_jepa.training.metrics import (
    compute_all_metrics,
    compute_sector_stratified_metrics,
    go_no_go_gate,
)
from fin_jepa.training.temporal_cv import TemporalCV
from fin_jepa.utils.reproducibility import seed_everything

log = logging.getLogger(__name__)

MIN_POSITIVES = 20  # skip outcome if fewer positives in train


# ── Reusable FT-Transformer training helper ──────────────────────────────


def train_ft_transformer(
    model: FTTransformer,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epochs: int = 100,
    patience: int = 10,
) -> dict:
    """Train an FT-Transformer with early stopping on validation AUROC.

    Returns
    -------
    dict with keys:
        best_val_auroc : float
        best_epoch     : int
        state_dict     : OrderedDict (best model weights)
    """
    best_val_auroc = -1.0
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        # ---- train ----
        model.train()
        train_loss = 0.0
        n_batches = 0
        for batch in train_loader:
            if len(batch) == 3:
                x_batch, x_cat_batch, y_batch = batch
                x_cat_batch = x_cat_batch.to(device)
            else:
                x_batch, y_batch = batch
                x_cat_batch = None
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            logits = model(x_batch, x_cat_batch).squeeze(-1)
            loss = criterion(logits, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
            n_batches += 1

        avg_train_loss = train_loss / max(n_batches, 1)

        # ---- validate ----
        val_auroc = _evaluate_auroc(model, val_loader, device)

        log.info(
            "Epoch %3d | train_loss=%.4f | val_auroc=%.4f",
            epoch,
            avg_train_loss,
            val_auroc,
        )

        if val_auroc > best_val_auroc:
            best_val_auroc = val_auroc
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                log.info(
                    "Early stopping at epoch %d (best epoch %d, val_auroc=%.4f)",
                    epoch,
                    best_epoch,
                    best_val_auroc,
                )
                break

    model.load_state_dict(best_state)
    return {
        "best_val_auroc": best_val_auroc,
        "best_epoch": best_epoch,
        "state_dict": best_state,
    }


def _evaluate_auroc(
    model: FTTransformer,
    loader: DataLoader,
    device: torch.device,
) -> float:
    """Compute AUROC on a DataLoader (returns 0.0 on degenerate input)."""
    y_true, y_score = _predict_scores(model, loader, device)

    # Need both classes to compute AUROC
    if len(np.unique(y_true)) < 2:
        return 0.0

    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(y_true, y_score))


def _predict_scores(
    model: FTTransformer,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (y_true, y_score) arrays from a DataLoader."""
    model.eval()
    all_scores: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    with torch.no_grad():
        for batch in loader:
            if len(batch) == 3:
                x_batch, x_cat_batch, y_batch = batch
                x_cat_batch = x_cat_batch.to(device)
            else:
                x_batch, y_batch = batch
                x_cat_batch = None
            x_batch = x_batch.to(device)
            logits = model(x_batch, x_cat_batch).squeeze(-1)
            scores = torch.sigmoid(logits).cpu().numpy()
            all_scores.append(scores)
            all_labels.append(y_batch.numpy())

    return np.concatenate(all_labels), np.concatenate(all_scores)


# ── Optuna-based baseline tuning ─────────────────────────────────────────


def tune_baseline(
    model_name: str,
    build_fn: callable,
    search_space: dict,
    X_train_df: pd.DataFrame,
    y_train: np.ndarray,
    feature_cols: list[str],
    n_splits: int = 3,
    n_trials: int = 30,
    seed: int = 42,
) -> dict:
    """Tune a baseline model using temporal CV and Optuna.

    Parameters
    ----------
    model_name : str
        Label for logging (e.g. ``"xgboost"``).
    build_fn : callable
        Factory that accepts ``**params`` and returns an sklearn-compatible
        classifier.
    search_space : dict
        Maps parameter names to dicts with keys ``type``, ``low``, ``high``,
        and optionally ``log``.
    X_train_df : DataFrame
        Training data with ``fiscal_year`` column for temporal splitting
        and the feature columns.
    y_train : ndarray
        Binary labels aligned with *X_train_df*.
    feature_cols : list[str]
        Columns in *X_train_df* to use as features.
    n_splits : int
        Number of temporal CV folds.
    n_trials : int
        Number of Optuna trials.
    seed : int
        Random seed for the Optuna sampler.

    Returns
    -------
    dict with ``best_params`` and ``mean_val_auroc``.
    """
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    cv = TemporalCV(n_splits=n_splits)

    def objective(trial: optuna.Trial) -> float:
        params: dict = {}
        for name, spec in search_space.items():
            if spec["type"] == "float":
                params[name] = trial.suggest_float(
                    name, spec["low"], spec["high"], log=spec.get("log", False),
                )
            elif spec["type"] == "int":
                params[name] = trial.suggest_int(name, spec["low"], spec["high"])

        aurocs: list[float] = []
        for train_idx, val_idx in cv.split(X_train_df):
            X_tr = X_train_df.iloc[train_idx][feature_cols].to_numpy(dtype=np.float32)
            y_tr = y_train[train_idx]
            X_va = X_train_df.iloc[val_idx][feature_cols].to_numpy(dtype=np.float32)
            y_va = y_train[val_idx]

            X_tr = np.nan_to_num(X_tr, nan=0.0)
            X_va = np.nan_to_num(X_va, nan=0.0)

            model = build_fn(**params)
            model.fit(X_tr, y_tr)
            scores = model.predict_proba(X_va)[:, 1]

            if len(np.unique(y_va)) < 2:
                continue
            aurocs.append(float(roc_auc_score(y_va, scores)))

        return float(np.mean(aurocs)) if aurocs else 0.0

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed),
    )
    study.optimize(objective, n_trials=n_trials)

    log.info(
        "  %s tuning — best val AUROC: %.4f  params: %s",
        model_name, study.best_value, study.best_params,
    )

    return {
        "best_params": study.best_params,
        "mean_val_auroc": study.best_value,
    }


# ── Main benchmark entry point ───────────────────────────────────────────


def run_benchmark(config) -> dict:
    """Entry point for the Study 0 benchmark experiment.

    Parameters
    ----------
    config : dict or DictConfig
        Hydra configuration (from ``configs/study0/benchmark.yaml``).

    Returns
    -------
    dict with all results including per-outcome metrics and gate decision.
    """
    # ── Setup ────────────────────────────────────────────────────────
    seed = config.get("training", {}).get("seed", 42) if isinstance(config, dict) else config.training.seed
    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    # ── Load data ────────────────────────────────────────────────────
    raw_dir = Path(_cfg(config, "data.raw_dir", "data/raw"))
    processed_dir = Path(_cfg(config, "data.processed_dir", "data/processed"))

    xbrl_df = load_xbrl_features(raw_dir)
    labels_df, _ = load_label_database(processed_dir / "label_database.parquet")

    xbrl_df["period_end"] = pd.to_datetime(xbrl_df["period_end"])
    labels_df["period_end"] = pd.to_datetime(labels_df["period_end"])

    # Merge features + labels on (cik, period_end)
    merged = xbrl_df.merge(labels_df, on=["cik", "period_end"], how="inner", suffixes=("", "_label"))

    # ── Build feature matrix with splits ─────────────────────────────
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
    n_cat = len(categorical_cols)
    cat_cards = [N_SECTORS] * n_cat if n_cat > 0 else None

    outcomes = _cfg(config, "outcomes", [
        "stock_decline", "earnings_restate", "audit_qualification",
        "sec_enforcement", "bankruptcy",
    ])

    # ── Feature subsets for distinct baselines ──────────────────────
    # Traditional ratios for LR "interpretable floor"
    trad_feature_cols = [c for c in TRADITIONAL_RATIO_FEATURES if c in feature_cols]
    log.info("Traditional ratio features: %d/%d available", len(trad_feature_cols), len(TRADITIONAL_RATIO_FEATURES))

    # Raw XBRL features only (no ratios, no YoY) for GBT baseline
    raw_feature_cols = [c for c in feature_cols if c in RAW_FEATURES]
    log.info("Raw XBRL features: %d available", len(raw_feature_cols))

    # Baseline tuning config
    do_tune = bool(_cfg(config, "baselines.tune", False))
    n_trials = int(_cfg(config, "baselines.n_trials", 30))
    n_cv_splits = int(_cfg(config, "baselines.n_cv_splits", 3))

    # ── Per-outcome training loop ────────────────────────────────────
    ft_results: dict[str, dict] = {}
    xgb_results: dict[str, dict] = {}
    lr_results: dict[str, dict] = {}
    lr_trad_results: dict[str, dict] = {}
    gbt_results: dict[str, dict] = {}
    all_results: dict[str, dict] = {}

    batch_size = int(_cfg(config, "training.batch_size", 256))
    epochs = int(_cfg(config, "training.epochs", 100))
    lr = float(_cfg(config, "training.learning_rate", 1e-4))
    wd = float(_cfg(config, "training.weight_decay", 1e-5))
    patience_val = int(_cfg(config, "training.patience", 10))

    for outcome in outcomes:
        log.info("=" * 60)
        log.info("Outcome: %s", outcome)

        if outcome not in splits["train"].columns:
            log.warning("Outcome column '%s' not found in data — skipping.", outcome)
            continue

        # Filter to rows with non-null labels per split
        train_valid = splits["train"][splits["train"][outcome].notna()]
        val_valid = splits["val"][splits["val"][outcome].notna()]
        test_valid = splits["test"][splits["test"][outcome].notna()]

        n_pos_train = int(train_valid[outcome].sum())
        n_neg_train = len(train_valid) - n_pos_train

        log.info(
            "  train: %d rows (%d pos, %d neg) | val: %d | test: %d",
            len(train_valid), n_pos_train, n_neg_train,
            len(val_valid), len(test_valid),
        )

        if n_pos_train < MIN_POSITIVES:
            log.warning(
                "  Fewer than %d positives in train — skipping %s.",
                MIN_POSITIVES, outcome,
            )
            continue

        pos_weight = n_neg_train / max(n_pos_train, 1)

        # ── Feature / label extraction ───────────────────────────────
        X_train = train_valid[feature_cols].to_numpy(dtype=np.float32)
        y_train = train_valid[outcome].to_numpy(dtype=np.float32)
        X_test = test_valid[feature_cols].to_numpy(dtype=np.float32)
        y_test = test_valid[outcome].to_numpy(dtype=np.float32)

        # Replace remaining NaN in features
        X_train = np.nan_to_num(X_train, nan=0.0)
        X_test = np.nan_to_num(X_test, nan=0.0)

        # ── Baselines ───────────────────────────────────────────────

        # --- Optional Optuna tuning ---
        xgb_params: dict = {}
        gbt_params: dict = {}
        lr_full_params: dict = {}
        lr_trad_params: dict = {}
        if do_tune:
            lr_search = _cfg(config, "baselines.lr_search", {})

            xgb_search = _cfg(config, "baselines.xgboost_search", {})
            if xgb_search:
                tune_result = tune_baseline(
                    "xgboost", build_xgboost, dict(xgb_search),
                    train_valid, y_train, feature_cols,
                    n_splits=n_cv_splits, n_trials=n_trials, seed=seed,
                )
                xgb_params = tune_result["best_params"]

            gbt_search = _cfg(config, "baselines.gbt_search", {})
            if gbt_search and raw_feature_cols:
                tune_result = tune_baseline(
                    "gbt", build_gbt, dict(gbt_search),
                    train_valid, y_train, raw_feature_cols,
                    n_splits=n_cv_splits, n_trials=n_trials, seed=seed,
                )
                gbt_params = tune_result["best_params"]

            # Tune LR separately for full features and traditional ratios
            if lr_search:
                tune_result = tune_baseline(
                    "lr_full", build_logistic_regression, dict(lr_search),
                    train_valid, y_train, feature_cols,
                    n_splits=n_cv_splits, n_trials=n_trials, seed=seed,
                )
                lr_full_params = tune_result["best_params"]

                if trad_feature_cols:
                    tune_result = tune_baseline(
                        "lr_trad", build_logistic_regression, dict(lr_search),
                        train_valid, y_train, trad_feature_cols,
                        n_splits=n_cv_splits, n_trials=n_trials, seed=seed,
                    )
                    lr_trad_params = tune_result["best_params"]

        # --- XGBoost on full XBRL features (primary benchmark) ---
        xgb_model = build_xgboost(
            n_estimators=int(xgb_params.get("n_estimators", _cfg(config, "xgboost.n_estimators", 500))),
            learning_rate=float(xgb_params.get("learning_rate", _cfg(config, "xgboost.learning_rate", 0.05))),
            max_depth=int(xgb_params.get("max_depth", _cfg(config, "xgboost.max_depth", 6))),
            subsample=float(xgb_params.get("subsample", _cfg(config, "xgboost.subsample", 0.8))),
            colsample_bytree=float(xgb_params.get("colsample_bytree", _cfg(config, "xgboost.colsample_bytree", 0.8))),
            scale_pos_weight=pos_weight,
        )
        xgb_model.fit(X_train, y_train)
        xgb_scores = xgb_model.predict_proba(X_test)[:, 1]
        xgb_metrics = compute_all_metrics(y_test, xgb_scores)
        log.info("  XGBoost   — AUROC: %.4f", xgb_metrics["auroc"])

        # --- Logistic Regression on full features ---
        lr_model = build_logistic_regression(
            C=float(lr_full_params.get("C", _cfg(config, "logistic_regression.C", 1.0))),
            max_iter=int(_cfg(config, "logistic_regression.max_iter", 1000)),
        )
        lr_model.fit(X_train, y_train)
        lr_scores = lr_model.predict_proba(X_test)[:, 1]
        lr_metrics = compute_all_metrics(y_test, lr_scores)
        log.info("  LR (full) — AUROC: %.4f", lr_metrics["auroc"])

        # --- Logistic Regression on traditional ratios (interpretable floor) ---
        if trad_feature_cols:
            X_train_trad = train_valid[trad_feature_cols].to_numpy(dtype=np.float32)
            X_test_trad = test_valid[trad_feature_cols].to_numpy(dtype=np.float32)
            X_train_trad = np.nan_to_num(X_train_trad, nan=0.0)
            X_test_trad = np.nan_to_num(X_test_trad, nan=0.0)

            lr_trad_model = build_logistic_regression(
                C=float(lr_trad_params.get("C", _cfg(config, "logistic_regression.C", 1.0))),
                max_iter=int(_cfg(config, "logistic_regression.max_iter", 1000)),
            )
            lr_trad_model.fit(X_train_trad, y_train)
            lr_trad_scores = lr_trad_model.predict_proba(X_test_trad)[:, 1]
            lr_trad_metrics = compute_all_metrics(y_test, lr_trad_scores)
            log.info("  LR (trad) — AUROC: %.4f", lr_trad_metrics["auroc"])
        else:
            lr_trad_metrics = {}
            log.warning("  No traditional ratio features available — skipping LR (trad).")

        # --- GBT on raw XBRL features (minimal feature engineering) ---
        if raw_feature_cols:
            X_train_raw = train_valid[raw_feature_cols].to_numpy(dtype=np.float32)
            X_test_raw = test_valid[raw_feature_cols].to_numpy(dtype=np.float32)
            # HistGradientBoostingClassifier handles NaN natively — no imputation

            gbt_model = build_gbt(
                max_iter=int(gbt_params.get("max_iter", _cfg(config, "gbt.max_iter", 500))),
                learning_rate=float(gbt_params.get("learning_rate", _cfg(config, "gbt.learning_rate", 0.05))),
                max_depth=int(gbt_params.get("max_depth", _cfg(config, "gbt.max_depth", 6))),
                min_samples_leaf=int(gbt_params.get("min_samples_leaf", _cfg(config, "gbt.min_samples_leaf", 20))),
            )
            gbt_model.fit(X_train_raw, y_train)
            gbt_scores = gbt_model.predict_proba(X_test_raw)[:, 1]
            gbt_metrics = compute_all_metrics(y_test, gbt_scores)
            log.info("  GBT (raw) — AUROC: %.4f", gbt_metrics["auroc"])
        else:
            gbt_metrics = {}
            log.warning("  No raw XBRL features available — skipping GBT.")

        # ── FT-Transformer ──────────────────────────────────────────
        train_loader = make_dataloader(
            train_valid, feature_cols, outcome, batch_size=batch_size, shuffle=True,
            cat_feature_cols=categorical_cols or None,
        )
        val_loader = make_dataloader(
            val_valid, feature_cols, outcome, batch_size=batch_size, shuffle=False,
            cat_feature_cols=categorical_cols or None,
        )
        test_loader = make_dataloader(
            test_valid, feature_cols, outcome, batch_size=batch_size, shuffle=False,
            cat_feature_cols=categorical_cols or None,
        )

        ft_model = FTTransformer(
            n_features=len(feature_cols),
            d_token=int(_cfg(config, "ft_transformer.d_token", 192)),
            n_heads=int(_cfg(config, "ft_transformer.n_heads", 8)),
            n_layers=int(_cfg(config, "ft_transformer.n_layers", 3)),
            d_ffn_factor=int(_cfg(config, "ft_transformer.d_ffn_factor", 4)),
            dropout=float(_cfg(config, "ft_transformer.dropout", 0.0)),
            n_outputs=1,
            n_cat_features=n_cat,
            cat_cardinalities=cat_cards,
        ).to(device)

        optimizer = torch.optim.AdamW(ft_model.parameters(), lr=lr, weight_decay=wd)
        criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_weight], device=device),
        )

        train_result = train_ft_transformer(
            ft_model, train_loader, val_loader, criterion, optimizer,
            device, epochs=epochs, patience=patience_val,
        )
        log.info(
            "  FT-Trans  — best_val_auroc: %.4f (epoch %d)",
            train_result["best_val_auroc"], train_result["best_epoch"],
        )

        # Evaluate on test
        ft_y_true, ft_y_score = _predict_scores(ft_model, test_loader, device)
        ft_metrics = compute_all_metrics(ft_y_true, ft_y_score)
        log.info("  FT-Trans  — test AUROC: %.4f", ft_metrics["auroc"])

        # ── Store results ────────────────────────────────────────────
        ft_results[outcome] = ft_metrics
        xgb_results[outcome] = xgb_metrics
        lr_results[outcome] = lr_metrics
        lr_trad_results[outcome] = lr_trad_metrics
        gbt_results[outcome] = gbt_metrics
        all_results[outcome] = {
            "ft_transformer": ft_metrics,
            "xgboost": xgb_metrics,
            "lr_full": lr_metrics,
            "lr_traditional": lr_trad_metrics,
            "gbt_raw": gbt_metrics,
        }

        # ── Sector-stratified evaluation ──────────────────────────────
        if "sector_idx" in test_valid.columns:
            sector_mask = test_valid["sector_idx"].notna()
            sector_ids = test_valid.loc[sector_mask, "sector_idx"].to_numpy(dtype=int)
            if len(sector_ids) > 0:
                all_results[outcome]["sector_stratified"] = {
                    "ft_transformer": compute_sector_stratified_metrics(
                        ft_y_true[sector_mask.values], ft_y_score[sector_mask.values], sector_ids,
                    ),
                    "xgboost": compute_sector_stratified_metrics(
                        y_test[sector_mask.values], xgb_scores[sector_mask.values], sector_ids,
                    ),
                }

    # ── Go / No-Go gate ──────────────────────────────────────────────
    evaluated_outcomes = list(ft_results.keys())
    if evaluated_outcomes:
        margin = float(_cfg(config, "gate.auroc_margin", 0.01))
        passed, n_wins, gate_detail = go_no_go_gate(
            ft_results, xgb_results, evaluated_outcomes, margin=margin,
        )
        log.info("=" * 60)
        log.info(
            "GO/NO-GO GATE: %s (%d/%d wins, margin=%.3f)",
            "PASSED" if passed else "FAILED", n_wins, len(evaluated_outcomes), margin,
        )
        for oc, detail in gate_detail.items():
            log.info(
                "  %s: FT=%.4f XGB=%.4f %s",
                oc, detail["ft_auroc"], detail["xgb_auroc"],
                "WIN" if detail["win"] else "",
            )
    else:
        passed, n_wins, gate_detail = False, 0, {}
        log.warning("No outcomes evaluated — gate cannot pass.")

    # ── Save results ─────────────────────────────────────────────────
    results_dir = Path(_cfg(config, "results_dir", "results/study0"))
    results_dir.mkdir(parents=True, exist_ok=True)

    output = {
        "gate": {"passed": passed, "n_wins": n_wins, "detail": gate_detail},
        "outcomes": all_results,
    }
    results_path = results_dir / "benchmark_results.json"
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    log.info("Results saved to %s", results_path)

    # ── Save sector-stratified results separately ─────────────────────
    sector_results = {
        oc: res["sector_stratified"]
        for oc, res in all_results.items()
        if "sector_stratified" in res
    }
    if sector_results:
        sector_path = results_dir / "sector_stratified_results.json"
        with open(sector_path, "w") as f:
            json.dump(sector_results, f, indent=2, default=str)
        log.info("Sector-stratified results saved to %s", sector_path)

    # ── Optional MLflow logging ──────────────────────────────────────
    try:
        import mlflow

        with mlflow.start_run(run_name="study0_benchmark"):
            mlflow.log_metric("gate_passed", int(passed))
            mlflow.log_metric("gate_n_wins", n_wins)
            for oc, metrics in ft_results.items():
                mlflow.log_metric(f"{oc}_ft_auroc", metrics["auroc"])
            for oc, metrics in xgb_results.items():
                mlflow.log_metric(f"{oc}_xgb_auroc", metrics["auroc"])
            for oc, metrics in lr_trad_results.items():
                if metrics:
                    mlflow.log_metric(f"{oc}_lr_trad_auroc", metrics["auroc"])
            for oc, metrics in gbt_results.items():
                if metrics:
                    mlflow.log_metric(f"{oc}_gbt_raw_auroc", metrics["auroc"])
            mlflow.log_artifact(str(results_path))
    except Exception as exc:
        log.debug("MLflow logging skipped: %s", exc)

    return output


# ── Multi-seed FT-Transformer variance estimation ────────────────────────


def run_multiseed_benchmark(
    config,
    seeds: list | None = None,
    *,
    prebuilt_splits: dict | None = None,
    prebuilt_feature_cols: list[str] | None = None,
    prebuilt_cat_cols: list[str] | None = None,
) -> dict:
    """Run FT-Transformer across multiple random seeds; report mean ± std AUROC.

    Baselines are deterministic and not re-run.  Seeds control model
    initialisation and dataloader shuffle order.

    Parameters
    ----------
    config : dict or DictConfig
    seeds : list[int] or None
        Override seeds list; falls back to ``training.seeds`` in config,
        then ``[42, 123, 456]``.
    prebuilt_splits : dict or None
        If provided, reuse caller's preprocessed train/val/test splits
        instead of building a fresh feature matrix.  This ensures the
        same QuantileTransformer is used across all experiments (ATS-217).
    prebuilt_feature_cols : list[str] or None
        Feature column names corresponding to *prebuilt_splits*.
    prebuilt_cat_cols : list[str] or None
        Categorical column names corresponding to *prebuilt_splits*.

    Returns
    -------
    dict with keys ``multiseed`` (per-outcome stats) and ``seeds``.
    """
    if seeds is None:
        seeds = list(_cfg(config, "training.seeds", [42, 123, 456]))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Multi-seed benchmark | seeds=%s | device=%s", seeds, device)

    # ── Load data ────────────────────────────────────────────────────
    if prebuilt_splits is not None:
        # ATS-217: reuse caller's preprocessing pipeline so seed-42
        # numbers match the main benchmark exactly.
        splits = prebuilt_splits
        feature_cols = prebuilt_feature_cols or [
            c for c in splits["train"].columns
            if c not in ("cik", "period_end", "stock_decline",
                         "earnings_restate", "audit_qualification",
                         "sec_enforcement", "bankruptcy")
        ]
        categorical_cols = prebuilt_cat_cols or []
        log.info(
            "Multi-seed benchmark: using prebuilt splits (%d train rows).",
            len(splits["train"]),
        )
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
        universe_df = None
        universe_path = raw_dir / "company_universe.parquet"
        if universe_path.exists() and feat_cfg.use_sic:
            universe_df = pd.read_parquet(universe_path)

        splits, _scaler, feature_cols, categorical_cols = build_feature_matrix(
            merged, split_cfg, feat_cfg, universe_df=universe_df,
        )
    n_cat = len(categorical_cols)
    cat_cards = [N_SECTORS] * n_cat if n_cat > 0 else None
    outcomes = _cfg(config, "outcomes", [
        "stock_decline", "earnings_restate", "audit_qualification",
        "sec_enforcement", "bankruptcy",
    ])
    batch_size = int(_cfg(config, "training.batch_size", 256))
    epochs = int(_cfg(config, "training.epochs", 100))
    lr = float(_cfg(config, "training.learning_rate", 1e-4))
    wd = float(_cfg(config, "training.weight_decay", 1e-5))
    patience_val = int(_cfg(config, "training.patience", 10))

    # per_seed_auroc[outcome] = list of test AUROC values, one per seed
    per_seed_auroc: dict[str, list[float]] = {oc: [] for oc in outcomes}

    for seed in seeds:
        log.info("─── Seed %d ───────────────────────────────────────────", seed)
        seed_everything(seed)

        for outcome in outcomes:
            if outcome not in splits["train"].columns:
                continue

            train_valid = splits["train"][splits["train"][outcome].notna()]
            val_valid = splits["val"][splits["val"][outcome].notna()]
            test_valid = splits["test"][splits["test"][outcome].notna()]

            n_pos_train = int(train_valid[outcome].sum())
            n_neg_train = len(train_valid) - n_pos_train
            if n_pos_train < MIN_POSITIVES:
                continue

            pos_weight = n_neg_train / max(n_pos_train, 1)

            train_loader = make_dataloader(
                train_valid, feature_cols, outcome, batch_size=batch_size, shuffle=True,
                cat_feature_cols=categorical_cols or None,
            )
            val_loader = make_dataloader(
                val_valid, feature_cols, outcome, batch_size=batch_size, shuffle=False,
                cat_feature_cols=categorical_cols or None,
            )
            test_loader = make_dataloader(
                test_valid, feature_cols, outcome, batch_size=batch_size, shuffle=False,
                cat_feature_cols=categorical_cols or None,
            )

            ft_model = FTTransformer(
                n_features=len(feature_cols),
                d_token=int(_cfg(config, "ft_transformer.d_token", 192)),
                n_heads=int(_cfg(config, "ft_transformer.n_heads", 8)),
                n_layers=int(_cfg(config, "ft_transformer.n_layers", 3)),
                d_ffn_factor=int(_cfg(config, "ft_transformer.d_ffn_factor", 4)),
                dropout=float(_cfg(config, "ft_transformer.dropout", 0.0)),
                n_outputs=1,
                n_cat_features=n_cat,
                cat_cardinalities=cat_cards,
            ).to(device)

            # Load SSL pretrained weights if checkpoint path is provided
            ssl_ckpt = _cfg(config, "ssl_checkpoint", None)
            if ssl_ckpt is not None:
                ckpt_path = Path(ssl_ckpt)
                if ckpt_path.exists():
                    ssl_state = torch.load(ckpt_path, map_location=device)
                    ft_model.load_state_dict(ssl_state, strict=False)
                    log.info("  Loaded SSL checkpoint: %s", ckpt_path)

            optimizer = torch.optim.AdamW(ft_model.parameters(), lr=lr, weight_decay=wd)
            criterion = nn.BCEWithLogitsLoss(
                pos_weight=torch.tensor([pos_weight], device=device),
            )
            train_ft_transformer(
                ft_model, train_loader, val_loader, criterion, optimizer,
                device, epochs=epochs, patience=patience_val,
            )
            ft_y_true, ft_y_score = _predict_scores(ft_model, test_loader, device)
            if len(np.unique(ft_y_true)) < 2:
                continue
            test_auroc = float(roc_auc_score(ft_y_true, ft_y_score))
            per_seed_auroc[outcome].append(test_auroc)
            log.info("  Seed %d | %s | test AUROC: %.4f", seed, outcome, test_auroc)

    # ── Aggregate ────────────────────────────────────────────────────
    aggregated: dict[str, dict] = {}
    for outcome in outcomes:
        aurocs = per_seed_auroc[outcome]
        if aurocs:
            aggregated[outcome] = {
                "mean_auroc": float(np.mean(aurocs)),
                "std_auroc": float(np.std(aurocs)),
                "seeds": list(seeds[: len(aurocs)]),
                "per_seed_auroc": {str(s): a for s, a in zip(seeds, aurocs)},
            }
            log.info(
                "  %s | mean=%.4f ± %.4f (n=%d seeds)",
                outcome,
                aggregated[outcome]["mean_auroc"],
                aggregated[outcome]["std_auroc"],
                len(aurocs),
            )

    output: dict = {"multiseed": aggregated, "seeds": list(seeds)}

    results_dir = Path(_cfg(config, "results_dir", "results/study0"))
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / "multiseed_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    log.info("Multi-seed results saved to %s", out_path)

    return output


# ── Walk-forward (expanding-window) validation ───────────────────────────


def run_walk_forward(config) -> dict:
    """Evaluate XGBoost and FT-Transformer across expanding-window folds.

    Uses ``rolling_split`` section from benchmark.yaml to generate folds from
    ``first_train_end`` to ``last_test_end``.  Each fold expands the training
    window by ``step_years`` and evaluates on a ``test_window_years``-wide
    held-out window.

    Feature normalisation is fit once on data up to ``first_train_end`` so
    that future data never contaminates the scaler (conservative choice).

    Returns
    -------
    dict with ``walk_forward_folds`` (list of per-fold per-outcome results)
    and ``n_folds``.
    """
    seed = int(_cfg(config, "training.seed", 42))
    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Walk-forward validation | device=%s", device)

    # ── Load data ────────────────────────────────────────────────────
    raw_dir = Path(_cfg(config, "data.raw_dir", "data/raw"))
    processed_dir = Path(_cfg(config, "data.processed_dir", "data/processed"))
    xbrl_df = load_xbrl_features(raw_dir)
    labels_df, _ = load_label_database(processed_dir / "label_database.parquet")
    xbrl_df["period_end"] = pd.to_datetime(xbrl_df["period_end"])
    labels_df["period_end"] = pd.to_datetime(labels_df["period_end"])
    merged = xbrl_df.merge(labels_df, on=["cik", "period_end"], how="inner", suffixes=("", "_label"))

    # ── Rolling split config ─────────────────────────────────────────
    rolling_cfg = RollingSplitConfig(
        first_train_end=_cfg(config, "rolling_split.first_train_end", "2014-12-31"),
        val_window_years=int(_cfg(config, "rolling_split.val_window_years", 1)),
        test_window_years=int(_cfg(config, "rolling_split.test_window_years", 2)),
        step_years=int(_cfg(config, "rolling_split.step_years", 1)),
        last_test_end=_cfg(config, "rolling_split.last_test_end", "2023-12-31"),
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

    # Fit normalisation on data up to the main benchmark's train_end.
    # Using first_train_end alone can produce an empty training split if
    # the dataset starts after that date; the standard train_end (2017)
    # gives the scaler enough data while still avoiding future leakage
    # (all walk-forward test folds start after 2017).
    scaler_train_end = _cfg(config, "data.split.train_end", "2017-12-31")
    norm_split_cfg = SplitConfig(
        train_end=scaler_train_end,
        val_end=scaler_train_end,   # empty val — just to anchor scaler
        test_end=rolling_cfg.last_test_end,
    )
    _splits_norm, _scaler, feature_cols, categorical_cols = build_feature_matrix(
        merged, norm_split_cfg, feat_cfg, universe_df=universe_df,
    )
    n_cat = len(categorical_cols)
    cat_cards = [N_SECTORS] * n_cat if n_cat > 0 else None

    # Reconstruct a single normalised dataframe covering all dates
    full_normalized = (
        pd.concat([_splits_norm["train"], _splits_norm["val"], _splits_norm["test"]])
        .sort_values("period_end")
        .reset_index(drop=True)
    )

    rolling_folds = make_rolling_splits(full_normalized, rolling_cfg)
    log.info("Walk-forward: %d folds generated", len(rolling_folds))

    outcomes = _cfg(config, "outcomes", [
        "stock_decline", "earnings_restate", "audit_qualification",
        "sec_enforcement", "bankruptcy",
    ])
    batch_size = int(_cfg(config, "training.batch_size", 256))
    epochs = int(_cfg(config, "training.epochs", 100))
    lr_rate = float(_cfg(config, "training.learning_rate", 1e-4))
    wd = float(_cfg(config, "training.weight_decay", 1e-5))
    patience_val = int(_cfg(config, "training.patience", 10))

    fold_results: list[dict] = []

    for fold_idx, fold in enumerate(rolling_folds):
        train_fold = fold["train"]
        val_fold = fold["val"]
        test_fold = fold["test"]

        if len(train_fold) == 0 or len(test_fold) == 0:
            continue

        train_dates = pd.to_datetime(train_fold["period_end"])
        test_dates = pd.to_datetime(test_fold["period_end"])
        fold_label = (
            f"train≤{train_dates.max().year}"
            f" → test {test_dates.min().year}–{test_dates.max().year}"
        )
        log.info("Fold %d: %s", fold_idx, fold_label)

        fold_outcome_results: dict[str, dict] = {}
        for outcome in outcomes:
            if outcome not in train_fold.columns:
                continue

            train_valid = train_fold[train_fold[outcome].notna()]
            val_valid = val_fold[val_fold[outcome].notna()]
            test_valid = test_fold[test_fold[outcome].notna()]

            n_pos_train = int(train_valid[outcome].sum())
            n_neg_train = len(train_valid) - n_pos_train
            if n_pos_train < MIN_POSITIVES or len(test_valid) == 0:
                continue
            if len(np.unique(test_valid[outcome].to_numpy())) < 2:
                continue

            pos_weight = n_neg_train / max(n_pos_train, 1)

            # --- XGBoost ---
            X_train = np.nan_to_num(train_valid[feature_cols].to_numpy(dtype=np.float32), nan=0.0)
            y_train = train_valid[outcome].to_numpy(dtype=np.float32)
            X_test = np.nan_to_num(test_valid[feature_cols].to_numpy(dtype=np.float32), nan=0.0)
            y_test = test_valid[outcome].to_numpy(dtype=np.float32)

            xgb_model = build_xgboost(
                n_estimators=int(_cfg(config, "xgboost.n_estimators", 500)),
                learning_rate=float(_cfg(config, "xgboost.learning_rate", 0.05)),
                max_depth=int(_cfg(config, "xgboost.max_depth", 6)),
                subsample=float(_cfg(config, "xgboost.subsample", 0.8)),
                colsample_bytree=float(_cfg(config, "xgboost.colsample_bytree", 0.8)),
                scale_pos_weight=pos_weight,
            )
            xgb_model.fit(X_train, y_train)
            xgb_scores = xgb_model.predict_proba(X_test)[:, 1]
            xgb_auroc = float(roc_auc_score(y_test, xgb_scores))

            # --- FT-Transformer ---
            eff_val = val_valid if len(val_valid) > 0 else train_valid
            train_loader = make_dataloader(
                train_valid, feature_cols, outcome, batch_size=batch_size, shuffle=True,
                cat_feature_cols=categorical_cols or None,
            )
            val_loader = make_dataloader(
                eff_val, feature_cols, outcome, batch_size=batch_size, shuffle=False,
                cat_feature_cols=categorical_cols or None,
            )
            test_loader = make_dataloader(
                test_valid, feature_cols, outcome, batch_size=batch_size, shuffle=False,
                cat_feature_cols=categorical_cols or None,
            )

            ft_model = FTTransformer(
                n_features=len(feature_cols),
                d_token=int(_cfg(config, "ft_transformer.d_token", 192)),
                n_heads=int(_cfg(config, "ft_transformer.n_heads", 8)),
                n_layers=int(_cfg(config, "ft_transformer.n_layers", 3)),
                d_ffn_factor=int(_cfg(config, "ft_transformer.d_ffn_factor", 4)),
                dropout=float(_cfg(config, "ft_transformer.dropout", 0.0)),
                n_outputs=1,
                n_cat_features=n_cat,
                cat_cardinalities=cat_cards,
            ).to(device)

            optimizer = torch.optim.AdamW(ft_model.parameters(), lr=lr_rate, weight_decay=wd)
            criterion = nn.BCEWithLogitsLoss(
                pos_weight=torch.tensor([pos_weight], device=device),
            )
            train_ft_transformer(
                ft_model, train_loader, val_loader, criterion, optimizer,
                device, epochs=epochs, patience=patience_val,
            )
            ft_y_true, ft_y_score = _predict_scores(ft_model, test_loader, device)
            ft_auroc = (
                float(roc_auc_score(ft_y_true, ft_y_score))
                if len(np.unique(ft_y_true)) >= 2
                else float("nan")
            )

            fold_outcome_results[outcome] = {
                "xgb_auroc": xgb_auroc,
                "ft_auroc": ft_auroc,
            }
            log.info("  %s | XGB=%.4f FT=%.4f", outcome, xgb_auroc, ft_auroc)

        fold_results.append({
            "fold": fold_idx,
            "label": fold_label,
            "outcomes": fold_outcome_results,
        })

    output: dict = {"walk_forward_folds": fold_results, "n_folds": len(fold_results)}

    results_dir = Path(_cfg(config, "results_dir", "results/study0"))
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / "walk_forward_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    log.info("Walk-forward results saved to %s", out_path)

    return output


# ── Config access helper ─────────────────────────────────────────────────


def _cfg(config, dotpath: str, default=None):
    """Access a nested config value via dot-separated path.

    Works with both plain dicts and OmegaConf DictConfig objects.
    """
    parts = dotpath.split(".")
    node = config
    for part in parts:
        try:
            node = node[part]
        except (KeyError, TypeError):
            try:
                node = getattr(node, part)
            except AttributeError:
                return default
    return node


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    try:
        import hydra
        from omegaconf import DictConfig

        @hydra.main(config_path="../../../configs/study0", config_name="benchmark", version_base=None)
        def main(cfg: DictConfig) -> None:
            run_benchmark(cfg)

        main()
    except ImportError:
        log.warning("Hydra not available — running with empty config.")
        run_benchmark({})

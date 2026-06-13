 #!/usr/bin/env python
"""Script-driven runner for the two outstanding Study 0 robustness experiments.

Replaces the notebook cells in ``experiments/study0/05_robustness.ipynb`` for the
two experiments that still need to be run to close out Study 0:

  * ``ft-tune``     — Optuna-tune the FT-Transformer (decision-critical gate input)
  * ``walk-forward`` — expanding-window temporal validation across the COVID boundary

Both call the existing, tested functions in
:mod:`fin_jepa.data` / :mod:`fin_jepa.training.train_study0`; this script is a thin
launcher so the whole pipeline can be debugged locally on CPU (``--smoke``) before a
full run on a GPU box, with no notebook state to fight.

Usage
-----
    # Fast local sanity + timing extrapolation (CPU):
    python scripts/run_robustness.py ft-tune      --smoke
    python scripts/run_robustness.py walk-forward --smoke

    # Full runs (point at a GPU if available — auto-detected):
    python scripts/run_robustness.py ft-tune
    python scripts/run_robustness.py walk-forward   # consumes ft-tune output if present
    python scripts/run_robustness.py all

Outputs (under results/study0/):
    ft_transformer_tuning.json   — best_params, mean_val_auroc, all_trials
    walk_forward_results.json    — per-fold per-outcome XGB/FT AUROC
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from fin_jepa.data.feature_engineering import (  # noqa: E402
    N_SECTORS,
    FeatureConfig,
    build_feature_matrix,
)
from fin_jepa.data.labels import load_label_database  # noqa: E402
from fin_jepa.data.splits import SplitConfig  # noqa: E402
from fin_jepa.data.xbrl_loader import load_xbrl_features  # noqa: E402
from fin_jepa.training.train_study0 import (  # noqa: E402
    run_benchmark,
    run_walk_forward,
    tune_ft_transformer,
)

log = logging.getLogger("run_robustness")

RESULTS_DIR = ROOT / "results" / "study0"
FT_TUNE_PATH = RESULTS_DIR / "ft_transformer_tuning.json"


# ── Shared data prep (mirrors run_benchmark's inline build) ──────────────


def _load_config():
    cfg_path = ROOT / "configs" / "study0" / "benchmark.yaml"
    try:
        from omegaconf import OmegaConf

        return OmegaConf.to_container(OmegaConf.load(cfg_path), resolve=True)
    except ImportError:
        import yaml

        with open(cfg_path) as f:
            return yaml.safe_load(f)


def _build_feature_matrix(cfg: dict):
    """Load + merge data and build the train/val/test feature matrix.

    Returns (splits, feature_cols, categorical_cols, n_cat, cat_cards).
    Identical construction to ``run_benchmark`` so numbers are comparable.
    """
    raw_dir = ROOT / cfg["data"]["raw_dir"]
    processed_dir = ROOT / cfg["data"]["processed_dir"]

    xbrl_df = load_xbrl_features(raw_dir)
    labels_df, _ = load_label_database(processed_dir / "label_database.parquet")
    xbrl_df["period_end"] = pd.to_datetime(xbrl_df["period_end"])
    labels_df["period_end"] = pd.to_datetime(labels_df["period_end"])
    merged = xbrl_df.merge(
        labels_df, on=["cik", "period_end"], how="inner", suffixes=("", "_label")
    )

    split_cfg = SplitConfig(
        train_end=cfg["data"]["split"]["train_end"],
        val_end=cfg["data"]["split"]["val_end"],
        test_end=cfg["data"]["split"]["test_end"],
    )
    feat = cfg["features"]
    feat_cfg = FeatureConfig(
        use_raw=feat["use_raw"],
        use_ratios=feat["use_ratios"],
        use_yoy=feat["use_yoy"],
        use_sic=feat["use_sic"],
        use_missingness_flags=feat["use_missingness_flags"],
        coverage_threshold=feat["coverage_threshold"],
        normalization_method=feat["normalization_method"],
        median_impute=feat["median_impute"],
    )

    universe_df = None
    universe_path = raw_dir / "company_universe.parquet"
    if universe_path.exists() and feat_cfg.use_sic:
        universe_df = pd.read_parquet(universe_path)

    splits, _scaler, feature_cols, categorical_cols = build_feature_matrix(
        merged, split_cfg, feat_cfg, universe_df=universe_df
    )
    n_cat = len(categorical_cols)
    cat_cards = [N_SECTORS] * n_cat if n_cat > 0 else None
    return splits, feature_cols, categorical_cols, n_cat, cat_cards


# ── ft-tune ──────────────────────────────────────────────────────────────


def cmd_ft_tune(args) -> None:
    cfg = _load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = int(cfg.get("training", {}).get("seed", 42))

    splits, feature_cols, categorical_cols, n_cat, cat_cards = _build_feature_matrix(cfg)

    tune_outcome = "stock_decline"  # most positives; arch choice is outcome-agnostic
    train_df = splits["train"]
    train_df = train_df[train_df[tune_outcome].notna()]  # drop NaN-label rows
    y_tune = train_df[tune_outcome].to_numpy(dtype=float)

    n_trials = args.n_trials if args.n_trials is not None else (2 if args.smoke else 30)
    n_splits = args.n_splits if args.n_splits is not None else (2 if args.smoke else 3)
    ft_search = cfg.get("ft_transformer_search")

    log.info(
        "FT-tune | device=%s | rows=%d | n_trials=%d | n_splits=%d | smoke=%s",
        device, len(train_df), n_trials, n_splits, args.smoke,
    )

    # Persisted Optuna study makes a long run resumable (skip for smoke).
    storage = None
    if not args.smoke:
        db = RESULTS_DIR / "ft_tuning_optuna.db"
        db.parent.mkdir(parents=True, exist_ok=True)
        storage = f"sqlite:///{db.as_posix()}"

    t0 = time.time()
    result = tune_ft_transformer(
        X_train_df=train_df,
        y_train=y_tune,
        feature_cols=feature_cols,
        device=device,
        search_space=ft_search,
        n_splits=n_splits,
        n_trials=n_trials,
        seed=seed,
        n_cat=n_cat,
        cat_cards=cat_cards,
        categorical_cols=categorical_cols,
        storage=storage,
        study_name="ft_study0" if storage else None,
    )
    elapsed = time.time() - t0

    log.info("Best params:        %s", result["best_params"])
    log.info("Best mean val AUROC: %.4f", result["mean_val_auroc"])
    log.info("Elapsed: %.1f s for %d trials (%.1f s/trial)",
             elapsed, n_trials, elapsed / max(n_trials, 1))

    if args.smoke:
        per_trial = elapsed / max(n_trials, 1)
        log.info(
            "EXTRAPOLATION → full 30-trial run on %s ≈ %.1f min",
            device.type, per_trial * 30 / 60,
        )
    else:
        out = dict(result)
        out["_meta"] = {"elapsed_s": elapsed, "n_trials": n_trials,
                        "n_splits": n_splits, "device": device.type}
        FT_TUNE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(FT_TUNE_PATH, "w") as f:
            json.dump(out, f, indent=2, default=str)
        log.info("Saved → %s", FT_TUNE_PATH)


# ── walk-forward ───────────────────────────────────────────────────────────


def cmd_walk_forward(args) -> None:
    cfg = _load_config()
    cfg["results_dir"] = str(RESULTS_DIR)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Consume tuned FT params if a prior ft-tune run saved them.
    tuned_ft = None
    if FT_TUNE_PATH.exists():
        with open(FT_TUNE_PATH) as f:
            tuned_ft = json.load(f).get("best_params")
        log.info("Using tuned FT params: %s", tuned_ft)
    else:
        log.info("No ft_transformer_tuning.json — FT uses config defaults")

    if args.smoke:
        # One narrow outcome + a 1-fold window to exercise the full code path fast.
        cfg["outcomes"] = ["stock_decline"]
        cfg["rolling_split"] = {
            "first_train_end": "2017-12-31",
            "val_window_years": 1,
            "test_window_years": 2,
            "step_years": 1,
            "last_test_end": "2020-12-31",
        }
        cfg["training"] = {**cfg.get("training", {}), "epochs": 5, "patience": 3}

    log.info("Walk-forward | device=%s | smoke=%s | outcomes=%s",
             device, args.smoke, cfg["outcomes"])

    t0 = time.time()
    result = run_walk_forward(cfg, tuned_ft_params=tuned_ft)
    elapsed = time.time() - t0

    log.info("Walk-forward: %d folds in %.1f s", result["n_folds"], elapsed)
    for fold in result["walk_forward_folds"]:
        log.info("  %s", fold.get("label", fold.get("fold")))
        for oc, m in fold.get("outcomes", {}).items():
            log.info("    %-18s XGB=%.4f FT=%.4f", oc, m["xgb_auroc"], m["ft_auroc"])

    if args.smoke and result["n_folds"] > 0:
        # crude extrapolation: full run is ~6 folds × 4 outcomes vs smoke 1 fold × 1 outcome
        per_fold_outcome = elapsed / max(
            sum(len(f["outcomes"]) for f in result["walk_forward_folds"]), 1
        )
        log.info(
            "EXTRAPOLATION → full run (~6 folds × 4 outcomes) on %s ≈ %.1f min",
            device, per_fold_outcome * 24 / 60,
        )


# ── close-out: full corrected Study 0 sequence ──────────────────────────────


def cmd_close_out(args) -> None:
    """Run the complete corrected close-out, decision-critical steps first.

    1. FT-Transformer Optuna tuning (resumable via SQLite).
    2. Corrected benchmark: re-tuned baselines (fixed CV) + tuned-FT gate.
    3. Walk-forward validation using the tuned FT params.

    All artifacts land in results/study0/corrected/ so existing results are
    never clobbered.
    """
    import copy

    cfg = _load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = int(cfg.get("training", {}).get("seed", 42))
    corrected = RESULTS_DIR / "corrected"
    corrected.mkdir(parents=True, exist_ok=True)

    smoke = args.smoke
    n_trials = args.n_trials if args.n_trials is not None else (2 if smoke else 30)
    n_splits = 2 if smoke else 3

    # ── Step 1: FT tuning (resumable) ───────────────────────────────────
    log.info("=" * 64)
    log.info("STEP 1/3  FT-Transformer Optuna tuning (n_trials=%d)", n_trials)
    splits, feature_cols, categorical_cols, n_cat, cat_cards = _build_feature_matrix(cfg)
    tune_outcome = "stock_decline"
    tr = splits["train"]
    tr = tr[tr[tune_outcome].notna()]
    y_tune = tr[tune_outcome].to_numpy(dtype=float)
    storage = None if smoke else f"sqlite:///{(corrected / 'ft_tuning_optuna.db').as_posix()}"

    t0 = time.time()
    ft = tune_ft_transformer(
        X_train_df=tr, y_train=y_tune, feature_cols=feature_cols, device=device,
        search_space=cfg.get("ft_transformer_search"), n_splits=n_splits,
        n_trials=n_trials, seed=seed, n_cat=n_cat, cat_cards=cat_cards,
        categorical_cols=categorical_cols, storage=storage,
        study_name="ft_study0" if storage else None,
    )
    best = ft["best_params"]
    ft["_meta"] = {"elapsed_s": time.time() - t0, "n_trials": n_trials, "device": device.type}
    for path in (corrected / "ft_transformer_tuning.json", FT_TUNE_PATH):
        with open(path, "w") as f:
            json.dump(ft, f, indent=2, default=str)
    log.info("STEP 1 done in %.1f min | best=%s | val AUROC=%.4f",
             ft["_meta"]["elapsed_s"] / 60, best, ft["mean_val_auroc"])

    # ── Step 2: corrected benchmark (re-tuned baselines + tuned FT) ──────
    log.info("=" * 64)
    log.info("STEP 2/3  Corrected benchmark (re-tuned baselines + tuned-FT gate)")
    cfg2 = copy.deepcopy(cfg)
    cfg2["ft_transformer"] = dict(cfg["ft_transformer"])
    cfg2["ft_transformer"]["d_token"] = int(best.get("d_token", cfg["ft_transformer"]["d_token"]))
    cfg2["ft_transformer"]["n_layers"] = int(best.get("n_layers", cfg["ft_transformer"]["n_layers"]))
    cfg2["training"] = dict(cfg["training"])
    cfg2["training"]["learning_rate"] = float(best.get("learning_rate", cfg["training"]["learning_rate"]))
    cfg2["baselines"] = dict(cfg.get("baselines", {}))
    cfg2["baselines"]["tune"] = True
    cfg2["results_dir"] = str(corrected)
    if smoke:
        cfg2["outcomes"] = ["stock_decline", "bankruptcy"]
        cfg2["training"]["epochs"] = 5
        cfg2["training"]["patience"] = 3
        cfg2["baselines"]["n_trials"] = 3
    t0 = time.time()
    bench = run_benchmark(cfg2)
    log.info("STEP 2 done in %.1f min | gate passed=%s (%d wins)",
             (time.time() - t0) / 60, bench["gate"]["passed"], bench["gate"]["n_wins"])

    # ── Step 3: walk-forward with tuned FT ──────────────────────────────
    log.info("=" * 64)
    log.info("STEP 3/3  Walk-forward validation (tuned FT params)")
    cfg3 = copy.deepcopy(cfg)
    cfg3["results_dir"] = str(corrected)
    if smoke:
        cfg3["outcomes"] = ["stock_decline"]
        cfg3["rolling_split"] = {
            "first_train_end": "2017-12-31", "val_window_years": 1,
            "test_window_years": 2, "step_years": 1, "last_test_end": "2020-12-31",
        }
        cfg3["training"] = dict(cfg3["training"])
        cfg3["training"]["epochs"] = 5
        cfg3["training"]["patience"] = 3
    t0 = time.time()
    wf = run_walk_forward(cfg3, tuned_ft_params=best)
    log.info("STEP 3 done in %.1f min | %d folds", (time.time() - t0) / 60, wf["n_folds"])

    log.info("=" * 64)
    log.info("CLOSE-OUT COMPLETE → %s", corrected)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="task", required=True)

    pf = sub.add_parser("ft-tune", help="Optuna-tune the FT-Transformer")
    pf.add_argument("--smoke", action="store_true", help="fast CPU sanity + timing")
    pf.add_argument("--n-trials", type=int, default=None)
    pf.add_argument("--n-splits", type=int, default=None)
    pf.set_defaults(func=cmd_ft_tune)

    pw = sub.add_parser("walk-forward", help="expanding-window validation")
    pw.add_argument("--smoke", action="store_true", help="fast CPU sanity + timing")
    pw.set_defaults(func=cmd_walk_forward)

    pa = sub.add_parser("all", help="ft-tune then walk-forward")
    pa.add_argument("--smoke", action="store_true")
    pa.add_argument("--n-trials", type=int, default=None)
    pa.add_argument("--n-splits", type=int, default=None)

    pc = sub.add_parser(
        "close-out",
        help="full corrected sequence: FT tune -> benchmark -> walk-forward",
    )
    pc.add_argument("--smoke", action="store_true", help="fast end-to-end CPU validation")
    pc.add_argument("--n-trials", type=int, default=None)
    pc.set_defaults(func=cmd_close_out)

    args = p.parse_args()
    if args.task == "all":
        cmd_ft_tune(args)
        cmd_walk_forward(args)
    else:
        args.func(args)


if __name__ == "__main__":
    main()

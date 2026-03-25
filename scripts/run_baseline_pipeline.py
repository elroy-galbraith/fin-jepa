#!/usr/bin/env python
"""Baseline pipeline runner for ATS-168.

Steps
-----
1. Build EDGAR quarterly filing index  → data/raw/edgar_index/  (needed for
   earnings_restate labels; ~2 min, 52 quarterly files)
2. Build XBRL features from EDGAR Company Facts API  → data/raw/xbrl_features.parquet
   (fetches only the CIKs present in market_aligned.parquet; ~10–20 min)
3. Build distress label database  → data/processed/label_database.parquet
4. Run the three baseline models (LR, XGBoost, GBT) on all five outcomes
   and write results  → results/study0/baseline_results.json

All downloads are cached, so the script is fully resumable.

Usage
-----
    python scripts/run_baseline_pipeline.py
    python scripts/run_baseline_pipeline.py --skip-xbrl   # if xbrl_features.parquet exists
    python scripts/run_baseline_pipeline.py --skip-index  # if edgar_index/ already populated
    python scripts/run_baseline_pipeline.py --help
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

# ── Logging setup ─────────────────────────────────────────────────────────────

LOG_DIR = Path(__file__).parent.parent / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "baseline_pipeline.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8"),
    ],
)
logger = logging.getLogger("baseline_pipeline")


def _elapsed(t0: float) -> str:
    s = time.time() - t0
    return f"{s:.0f}s" if s < 90 else f"{s / 60:.1f}m"


# ── Step 1: EDGAR quarterly index ────────────────────────────────────────────

def build_edgar_index(raw_dir: Path, skip: bool = False) -> None:
    """Download quarterly form.idx files → raw_dir/edgar_index/*.parquet."""
    index_dir = raw_dir / "edgar_index"
    existing = list(index_dir.glob("*.parquet")) if index_dir.exists() else []

    if skip and existing:
        logger.info("Step 1/4  EDGAR index: %d files already present — skipping.", len(existing))
        return

    from fin_jepa.data.universe import UniverseConfig, fetch_quarterly_index, _get_session

    cfg = UniverseConfig(start_year=2012, end_year=2024)
    session = _get_session(cfg.user_agent)

    quarters = [(y, q) for y in range(2012, 2025) for q in range(1, 5)]
    # Skip quarters we already have
    todo = [
        (y, q) for y, q in quarters
        if not (index_dir / f"{y}_Q{q}.parquet").exists()
    ]

    if not todo:
        logger.info("Step 1/4  EDGAR index already complete (%d files).", len(existing))
        return

    logger.info(
        "Step 1/4  Building EDGAR filing index: %d/%d quarters to fetch …",
        len(todo), len(quarters),
    )
    t0 = time.time()
    min_sleep = 1.0 / cfg.rate_limit_per_sec

    for i, (year, quarter) in enumerate(todo, 1):
        try:
            # cache_dir=raw_dir so files land at raw_dir/edgar_index/{year}_Q{quarter}.parquet
            fetch_quarterly_index(year, quarter, raw_dir, cfg, session)
            time.sleep(min_sleep)
        except Exception as exc:
            logger.warning("  Skipping %d Q%d: %s", year, quarter, exc)
        if i % 10 == 0 or i == len(todo):
            logger.info("  %d / %d quarters fetched", i, len(todo))

    n_files = len(list(index_dir.glob("*.parquet")))
    logger.info("Step 1/4  Done (%s) — %d parquet files in edgar_index/", _elapsed(t0), n_files)


# ── Step 2: XBRL features ─────────────────────────────────────────────────────

def build_xbrl(raw_dir: Path, skip: bool = False) -> None:
    """Fetch XBRL Company Facts for the market_aligned CIK set."""
    output_path = raw_dir / "xbrl_features.parquet"

    if skip and output_path.exists():
        import pandas as pd
        df = pd.read_parquet(output_path)
        logger.info(
            "Step 2/4  XBRL features already present (%d rows, %d companies) — skipping.",
            len(df), df["cik"].nunique(),
        )
        return

    import pandas as pd
    from fin_jepa.data.xbrl_pipeline import XBRLConfig, build_xbrl_dataset

    # Use only CIKs present in market_aligned (avoids fetching all 13k companies)
    market_path = raw_dir / "market" / "market_aligned.parquet"
    if not market_path.exists():
        raise FileNotFoundError(
            f"market_aligned.parquet not found at {market_path}. "
            "Run scripts/run_market_pipeline.py first."
        )

    market_df = pd.read_parquet(market_path)
    market_ciks = market_df["cik"].astype(str).str.zfill(10).unique().tolist()
    logger.info(
        "Step 2/4  Building XBRL features for %d CIKs from market_aligned …",
        len(market_ciks),
    )

    t0 = time.time()
    cfg = XBRLConfig(start_year=2012, end_year=2024, max_workers=4, rate_limit_per_sec=8.0)

    build_xbrl_dataset(
        raw_dir=raw_dir,
        universe_path=raw_dir / "company_universe.parquet",
        output_path=output_path,
        config=cfg,
        cik_subset=market_ciks,
    )
    logger.info("Step 2/4  Done (%s) — written to %s", _elapsed(t0), output_path)


# ── Step 3: Label database ────────────────────────────────────────────────────

def build_labels(raw_dir: Path, processed_dir: Path, skip: bool = False) -> None:
    """Construct the five-outcome distress label database."""
    output_path = processed_dir / "label_database.parquet"

    if skip and output_path.exists():
        import pandas as pd
        df = pd.read_parquet(output_path)
        logger.info(
            "Step 3/4  Label database already present (%d rows) — skipping.",
            len(df),
        )
        return

    from fin_jepa.data.labels import LabelConfig, build_label_database

    logger.info("Step 3/4  Building distress label database …")
    t0 = time.time()

    cfg = LabelConfig(
        decline_threshold=-0.20,
        treat_delisted_as_decline=True,
        restatement_source="edgar_amendments",
        audit_qualification_source="external_csv",
        sec_enforcement_source="external_csv",
        bankruptcy_source="compustat",
        external_label_dir=raw_dir / "labels",
        horizon_days=365,
    )

    build_label_database(raw_dir=raw_dir, output_path=output_path, config=cfg)
    logger.info("Step 3/4  Done (%s) — written to %s", _elapsed(t0), output_path)


# ── Step 4: Baseline models ────────────────────────────────────────────────────

def run_baselines(raw_dir: Path, processed_dir: Path, results_dir: Path) -> dict:
    """Train and evaluate LR, XGBoost, and GBT on all five outcomes."""
    import numpy as np
    import pandas as pd

    from fin_jepa.data.feature_engineering import (
        TRADITIONAL_RATIO_FEATURES,
        FeatureConfig,
        RAW_FEATURES,
        build_feature_matrix,
    )
    from fin_jepa.data.labels import load_label_database
    from fin_jepa.data.splits import SplitConfig
    from fin_jepa.data.xbrl_loader import load_xbrl_features
    from fin_jepa.models.baselines import build_gbt, build_logistic_regression, build_xgboost
    from fin_jepa.training.metrics import compute_all_metrics
    from fin_jepa.utils.reproducibility import seed_everything

    seed_everything(42)

    logger.info("Step 4/4  Running baseline models …")
    t0 = time.time()

    # ── Load data ──────────────────────────────────────────────────────────────
    xbrl_df = load_xbrl_features(raw_dir)
    labels_df, _ = load_label_database(processed_dir / "label_database.parquet")

    # Load company universe for SIC sector join
    universe_path = raw_dir / "company_universe.parquet"
    universe_df = None
    if universe_path.exists():
        universe_df = pd.read_parquet(universe_path)
        logger.info("  Universe: %d companies (for SIC join)", len(universe_df))
    else:
        logger.warning("  company_universe.parquet not found — SIC join will be skipped.")

    logger.info(
        "  XBRL: %d rows, %d companies", len(xbrl_df), xbrl_df["cik"].nunique()
    )
    logger.info(
        "  Labels: %d rows, %d companies", len(labels_df), labels_df["cik"].nunique()
    )

    # Normalise period_end to datetime64 in both frames before merging
    xbrl_df = xbrl_df.copy()
    labels_df = labels_df.copy()
    xbrl_df["period_end"] = pd.to_datetime(xbrl_df["period_end"])
    labels_df["period_end"] = pd.to_datetime(labels_df["period_end"])

    # Merge on (cik, period_end)
    merged = xbrl_df.merge(
        labels_df, on=["cik", "period_end"], how="inner", suffixes=("", "_label")
    )
    logger.info("  Merged: %d rows", len(merged))

    if merged.empty:
        raise RuntimeError(
            "Merge of XBRL features and labels produced an empty DataFrame. "
            "Check that period_end dates align between the two datasets."
        )

    # ── Feature matrix ────────────────────────────────────────────────────────
    split_cfg = SplitConfig(
        train_end="2017-12-31",
        val_end="2019-12-31",
        test_end="2023-12-31",
    )
    feat_cfg = FeatureConfig(
        use_raw=True,
        use_ratios=True,
        use_yoy=True,
        use_sic=True,
        use_missingness_flags=True,
        coverage_threshold=0.50,
        normalization_method="quantile",
        median_impute=True,
    )
    splits, _scaler, feature_cols, categorical_cols = build_feature_matrix(
        merged, split_cfg, feat_cfg, universe_df=universe_df,
    )

    # For baseline models: include categorical features as integer columns
    # alongside continuous features (XGBoost/GBT handle this natively; LR
    # treats them as ordinal — imperfect but sufficient for baselines).
    all_baseline_cols = feature_cols + categorical_cols

    logger.info(
        "  Feature matrix: %d continuous + %d categorical | train=%d, val=%d, test=%d",
        len(feature_cols),
        len(categorical_cols),
        len(splits["train"]),
        len(splits.get("val", [])),
        len(splits.get("test", [])),
    )

    # Feature subsets
    trad_cols = [c for c in TRADITIONAL_RATIO_FEATURES if c in feature_cols]
    raw_cols = [c for c in feature_cols if c in RAW_FEATURES]
    logger.info(
        "  Feature subsets: %d traditional, %d raw XBRL",
        len(trad_cols), len(raw_cols),
    )

    outcomes = [
        "stock_decline", "earnings_restate", "audit_qualification",
        "sec_enforcement", "bankruptcy",
    ]
    MIN_POSITIVES = 20
    all_results: dict = {}

    for outcome in outcomes:
        logger.info("─" * 55)
        logger.info("Outcome: %s", outcome)

        if outcome not in splits["train"].columns:
            logger.warning("  Column '%s' not found — skipping.", outcome)
            continue

        train_valid = splits["train"][splits["train"][outcome].notna()]
        test_valid = splits["test"][splits["test"][outcome].notna()]

        n_pos = int(train_valid[outcome].sum())
        n_neg = len(train_valid) - n_pos
        n_pos_test = int(test_valid[outcome].sum()) if len(test_valid) > 0 else 0

        logger.info(
            "  train: %d rows (%d pos, %d neg) | test: %d rows (%d pos)",
            len(train_valid), n_pos, n_neg, len(test_valid), n_pos_test,
        )

        if n_pos < MIN_POSITIVES:
            logger.warning("  Fewer than %d positives in train — skipping.", MIN_POSITIVES)
            continue

        if len(np.unique(test_valid[outcome].to_numpy())) < 2:
            logger.warning("  Test set has only one class — metrics undefined, skipping.")
            continue

        pos_weight = n_neg / max(n_pos, 1)

        X_train = np.nan_to_num(train_valid[all_baseline_cols].to_numpy(dtype=np.float32), nan=0.0)
        y_train = train_valid[outcome].to_numpy(dtype=np.float32)
        X_test  = np.nan_to_num(test_valid[all_baseline_cols].to_numpy(dtype=np.float32), nan=0.0)
        y_test  = test_valid[outcome].to_numpy(dtype=np.float32)

        outcome_results: dict = {}

        # ── XGBoost (full XBRL features — primary benchmark) ──────────────────
        xgb = build_xgboost(
            n_estimators=500, learning_rate=0.05, max_depth=6,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=pos_weight,
        )
        xgb.fit(X_train, y_train)
        xgb_scores = xgb.predict_proba(X_test)[:, 1]
        xgb_m = compute_all_metrics(y_test, xgb_scores)
        logger.info("  XGBoost (full)   — AUROC=%.4f  AUPRC=%.4f  Brier=%.4f  ECE=%.4f",
                    xgb_m["auroc"], xgb_m["auprc"], xgb_m["brier"], xgb_m["ece"])
        outcome_results["xgboost"] = xgb_m

        # ── Logistic Regression (full features) ───────────────────────────────
        lr = build_logistic_regression(C=1.0, max_iter=1000)
        lr.fit(X_train, y_train)
        lr_scores = lr.predict_proba(X_test)[:, 1]
        lr_m = compute_all_metrics(y_test, lr_scores)
        logger.info("  LR (full)        — AUROC=%.4f  AUPRC=%.4f  Brier=%.4f  ECE=%.4f",
                    lr_m["auroc"], lr_m["auprc"], lr_m["brier"], lr_m["ece"])
        outcome_results["lr_full"] = lr_m

        # ── Logistic Regression (traditional ratios — interpretable floor) ───
        if trad_cols:
            X_tr_trad = np.nan_to_num(
                train_valid[trad_cols].to_numpy(dtype=np.float32), nan=0.0
            )
            X_te_trad = np.nan_to_num(
                test_valid[trad_cols].to_numpy(dtype=np.float32), nan=0.0
            )
            lr_trad = build_logistic_regression(C=1.0, max_iter=1000)
            lr_trad.fit(X_tr_trad, y_train)
            lr_trad_scores = lr_trad.predict_proba(X_te_trad)[:, 1]
            lr_trad_m = compute_all_metrics(y_test, lr_trad_scores)
            logger.info(
                "  LR (trad ratios) — AUROC=%.4f  AUPRC=%.4f  Brier=%.4f  ECE=%.4f",
                lr_trad_m["auroc"], lr_trad_m["auprc"], lr_trad_m["brier"], lr_trad_m["ece"],
            )
            outcome_results["lr_traditional"] = lr_trad_m
        else:
            logger.warning("  No traditional ratio features — skipping LR (trad).")

        # ── GBT on raw XBRL features (minimal engineering) ───────────────────
        if raw_cols:
            # HistGBT handles NaN natively — use original values (no imputation)
            X_tr_raw = train_valid[raw_cols].to_numpy(dtype=np.float32)
            X_te_raw = test_valid[raw_cols].to_numpy(dtype=np.float32)
            gbt = build_gbt(max_iter=500, learning_rate=0.05, max_depth=6, min_samples_leaf=20)
            gbt.fit(X_tr_raw, y_train)
            gbt_scores = gbt.predict_proba(X_te_raw)[:, 1]
            gbt_m = compute_all_metrics(y_test, gbt_scores)
            logger.info(
                "  GBT (raw XBRL)   — AUROC=%.4f  AUPRC=%.4f  Brier=%.4f  ECE=%.4f",
                gbt_m["auroc"], gbt_m["auprc"], gbt_m["brier"], gbt_m["ece"],
            )
            outcome_results["gbt_raw"] = gbt_m
        else:
            logger.warning("  No raw XBRL features — skipping GBT.")

        all_results[outcome] = outcome_results

    logger.info("=" * 55)
    logger.info("BASELINE RESULTS SUMMARY")
    logger.info("=" * 55)
    header = f"  {'Outcome':<22}  {'Model':<18}  {'AUROC':>6}  {'AUPRC':>6}  {'Brier':>6}  {'ECE':>6}"
    logger.info(header)
    logger.info("  " + "-" * 70)
    for outcome, models in all_results.items():
        for model_name, m in models.items():
            logger.info(
                "  %-22s  %-18s  %6.4f  %6.4f  %6.4f  %6.4f",
                outcome, model_name,
                m.get("auroc", float("nan")),
                m.get("auprc", float("nan")),
                m.get("brier", float("nan")),
                m.get("ece", float("nan")),
            )

    # ── Save results ──────────────────────────────────────────────────────────
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / "baseline_results.json"
    with open(out_path, "w") as fh:
        json.dump(all_results, fh, indent=2, default=str)
    logger.info("Results written → %s", out_path)
    logger.info("Step 4/4  Done (%s)", _elapsed(t0))

    return all_results


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fin-JEPA baseline pipeline: EDGAR index → XBRL → labels → baselines.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--raw-dir",       default="data/raw",       help="Raw data directory.")
    parser.add_argument("--processed-dir", default="data/processed", help="Processed data directory.")
    parser.add_argument("--results-dir",   default="results/study0", help="Results directory.")
    parser.add_argument("--skip-index",    action="store_true", help="Skip EDGAR index if present.")
    parser.add_argument("--skip-xbrl",     action="store_true", help="Skip XBRL download if present.")
    parser.add_argument("--skip-labels",   action="store_true", help="Skip label build if present.")
    args = parser.parse_args()

    raw_dir       = Path(args.raw_dir)
    processed_dir = Path(args.processed_dir)
    results_dir   = Path(args.results_dir)

    pipeline_start = time.time()
    logger.info("=" * 60)
    logger.info(">> Baseline pipeline starting  (log → %s)", LOG_FILE)
    logger.info("=" * 60)

    build_edgar_index(raw_dir, skip=args.skip_index)
    build_xbrl(raw_dir, skip=args.skip_xbrl)
    build_labels(raw_dir, processed_dir, skip=args.skip_labels)
    run_baselines(raw_dir, processed_dir, results_dir)

    logger.info("=" * 60)
    logger.info(">> Pipeline complete in %s", _elapsed(pipeline_start))
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

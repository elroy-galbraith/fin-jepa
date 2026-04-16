# Study 0 — Financial Encoder Validation

Notebooks for running the full Study 0 pipeline on Google Colab.

## Prerequisites

- A Colab runtime with GPU (A100 recommended, T4 works but slower)
- A GitHub PAT stored in Colab Secrets as `GITHUB_PAT`
- Data files in Google Drive at `My Drive/Fin-JEPA/data/` (`raw/` and `processed/`)

## Notebook series

| # | Notebook | Runtime | Description |
|---|----------|---------|-------------|
| 00 | `00_setup_and_preprocessing` | ~10 min | Mount Drive, clone repo, build feature matrix, save preprocessing artifact to Drive |
| 01 | `01_baselines` | ~1 hr | Train XGBoost, Logistic Regression, and GBT baselines |
| 02 | `02_hp_sweep` | ~6–12 hrs | Grid search over 36 FT-Transformer configs |
| 03 | `03_ssl_pretraining` | ~2–3 hrs | SSL pretraining at 3 mask ratios with best sweep architecture |
| 04 | `04_final_benchmark` | ~4–6 hrs | Re-train all 6 models with bootstrap CIs; go/no-go gate |
| 05 | `05_robustness` | ~13 hrs | Optuna tuning, multi-seed variance, walk-forward validation |
| 06 | `06_verification` | ~1 min | Seed-42 consistency check across result files; Drive backup |

## Execution order

```
00  (must run first — produces the shared preprocessing artifact)
 |
 +---> 01  (baselines)
 +---> 02  (HP sweep)          <-- 01 and 02 can run in parallel
         |
         +---> 03  (SSL)
                |
                +---> 04  (final benchmark + gate)
                       |
                       +---> 05  (robustness)
                              |
                              +---> 06  (verification)
```

**NB 01 and NB 02 are independent** — they can run on separate Colab instances at
the same time since both only need the preprocessing artifact from NB 00.

## How data flows between notebooks

NB 00 saves a preprocessing artifact (pickle) to Google Drive:

```
My Drive/Fin-JEPA/artifacts/study0/preprocessing_v1.pkl
```

This contains the train/val/test splits, fitted scaler, feature column lists, and a
config fingerprint. All downstream notebooks load this artifact instead of rebuilding
the feature matrix, guaranteeing identical preprocessing across the pipeline.

Result JSONs are written to `results/study0/` and loaded by later notebooks as needed.
Each notebook checks that its prerequisites exist on disk and raises `FileNotFoundError`
with a clear message if you try to run out of order.

## Re-running experiments

Each notebook has a `FORCE_RERUN` flag (default `False`). When set to `True`, the
notebook re-trains models from scratch instead of loading cached results. NB 05 has
additional per-section flags (`FORCE_RERUN_TUNED_BASELINES`, etc.) so you can
selectively re-run individual robustness analyses.

**After changing any code that affects training**, set `FORCE_RERUN = True` in the
relevant notebook and all downstream notebooks, then finish with NB 06 to verify
seed-42 consistency.

## Result files

All results land in `results/study0/`:

| File | Produced by |
|------|-------------|
| `benchmark_results.json` | NB 01 |
| `ft_sweep_results.json` | NB 02 |
| `ssl_experiment_results_v2.json` | NB 03 |
| `final_benchmark.json` | NB 04 |
| `multiseed_results.json` | NB 05 (section 5c) |
| `multiseed_ssl.json` | NB 05 (section 5d) |
| `walk_forward_results.json` | NB 05 (section 5e) |
| `ft_transformer_tuning.json` | NB 05 (section 5b) |
| `verification_report.json` | NB 06 |

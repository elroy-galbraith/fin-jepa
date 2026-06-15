# Reproducing Study 0

This document lists the exact, ordered steps to reproduce every result and table in
the Study 0 paper (`paper/study0/`) starting from the published HuggingFace dataset.
It is the canonical pipeline; the notebooks under `experiments/study0/` are the
original exploratory drivers and are not required (except where noted for figures).

## Environment

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e '.[hf]'
```

## 0. Data

```bash
python scripts/download_data.py
```

Pulls the four source parquets from `elroyg/fin-jepa-study0` into `data/raw/` and
`data/processed/`. (Rebuilding from EDGAR instead is documented in `README.md`; the
HF snapshot is the supported reproduction path.)

## 1. Results

Run in order. Each writes JSON under `results/study0/`. **Compute warning:** the
FT-Transformer / SSL steps are written for a GPU. On CPU, FT tuning is ~15 h, the
consistency re-run ~12 h, and SSL ~50 h — see notes below.

| # | Command | Writes | Feeds (paper) |
|---|---------|--------|---------------|
| 1 | `python scripts/run_baseline_pipeline.py` | `baseline_results.json` | default LR / XGB / GBT |
| 2 | `python scripts/run_ssl_experiment.py` | `ssl_experiment_results.json` | SSL table, FT-scratch (**GPU**) |
| 3 | `python scripts/generate_final_benchmark.py` | `final_benchmark.json` | Table 1, default-baseline gate (4/4), bootstrap CIs |
| 4 | `python scripts/run_robustness.py close-out` | `corrected/ft_transformer_tuning.json`, `corrected/benchmark_results.json` | Optuna FT params, **equal-budget gate (3/4)** (**GPU**) |
| 5 | `python scripts/rerun_consistency.py` | `corrected/walk_forward_results.json`, `corrected/multiseed_results.json` | walk-forward table/figure, multi-seed table (**GPU**) |
| 6 | `python scripts/plot_walk_forward.py` | `figures/walk_forward_delta.png` | walk-forward figure |

Notes:
- **Step 4** (`close-out`) tunes the FT-Transformer with Optuna and runs the
  corrected, equal-budget benchmark that underpins the 3/4 gate. It is resumable
  (Optuna SQLite). Add `--smoke` for a fast CPU sanity run.
- **Step 5** (`rerun_consistency`) reads the tuned FT params from step 4 and produces
  the *canonical* walk-forward (leak-free, per-fold preprocessing) and multi-seed
  (Optuna architecture) numbers used in the paper. These supersede the walk-forward
  emitted as a side effect of step 4. Add `--smoke` for a fast CPU sanity run.
- **SSL** (step 2) stays at the sweep architecture (`d=192`); it is a self-contained
  scratch-vs-pretrained comparison and is not re-run on the Optuna architecture.

## 2. Figures

`plot_walk_forward.py` (step 6 above) regenerates the walk-forward figure from the
committed JSON. The remaining five figures — `sweep_heatmap`, `baseline_auroc`,
`final_benchmark_all_models`, `bootstrap_ci_forest_plot`, `ssl_loss_curves` — are
currently produced by the notebooks (`02`–`04`) and are **not yet scripted**
(tracked as a tidy-up item). `sweep_heatmap` additionally requires
`ft_sweep_results.json` (the 36-config grid sweep, produced by `02_hp_sweep.ipynb`).

## Artifact → paper map (summary)

| Artifact | Producer | Paper |
|----------|----------|-------|
| `baseline_results.json` | `run_baseline_pipeline.py` | default baselines |
| `ssl_experiment_results.json` | `run_ssl_experiment.py` | SSL table |
| `final_benchmark.json` | `generate_final_benchmark.py` | Table 1, gate-default, bootstrap |
| `corrected/benchmark_results.json` | `run_robustness.py close-out` | tuned gate (3/4), tuned-baseline table |
| `corrected/ft_transformer_tuning.json` | `run_robustness.py close-out` | Optuna FT hyperparameters |
| `corrected/walk_forward_results.json` | `rerun_consistency.py` | walk-forward table + figure |
| `corrected/multiseed_results.json` | `rerun_consistency.py` | multi-seed table |
| `ft_sweep_results.json` | `02_hp_sweep.ipynb` | sweep heatmap |

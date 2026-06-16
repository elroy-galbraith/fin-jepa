# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased] — Study 0 pre-arXiv revision — 2026-06-16

### Changed
- Test-split label counts corrected to measured values (`sec_enforcement` 96/14,064; `bankruptcy` 113/14,047; `earnings_restate` 2,952/11,208; `stock_decline` 8,334/5,821); rare-event framing fixed (only `sec_enforcement` and `bankruptcy` are low-base-rate).
- 36-config hyperparameter sweep (Fig. 1, Table 3) demoted to a test-selected exploratory diagnostic; the Optuna/CV-tuned configuration (d=128) is now the basis for every reported result.
- Go/no-go gate reframed onto the two robust wins (`earnings_restate`, `bankruptcy`); the SEC-enforcement win is flagged as within seed noise.

### Fixed
- Table 4 `stock_decline` bolding; SSL gain consistency (+0.093); 2012–2024 universe vs 2012–2023 modeling-window reconciliation; forward-returns-are-labels-only clarification (no market look-ahead).
- `scripts/generate_final_benchmark.py` no longer overwrites the real paired-bootstrap CIs (n=1000) and now uses measured sample sizes; it refuses to overwrite `final_benchmark.json` unless `--force` is passed.

### Added
- Credibility notes: SEC-enforcement label right-censoring (24–36-month lag), UCLA–LoPucki BRD Dec-2022 freeze and large-firm bias, the XGBoost-only class-imbalance asymmetry, and the baseline Optuna search spaces.
- Related-work citations: SAINT, TabTransformer, and Karpoff et al. (2017).

_No models were retrained in this revision; all changes are documentation, analysis, and reproducibility corrections._

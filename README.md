# Fin-JEPA

**Financial JEPA: Self-Supervised Representation Learning on XBRL Data**

This repository contains the research code for the Fin-JEPA project, structured as a
sequence of studies. Each study gates the next.

---

## Project Roadmap

| Study | Name | Gate Criterion |
|-------|------|----------------|
| **Study 0** | Financial Encoder Validation | FT-Transformer beats XGBoost on ≥3/5 distress outcomes |
| Study 1 | JEPA Pretraining on XBRL | Latent prediction loss converges; transfer > from-scratch |
| Study 2 | Cross-Modal Grounding (Text) | 8-K/MD&A alignment improves downstream AUC |
| Study 3 | Multi-Horizon Forecasting | Encoder generalizes across forecast horizons |
| Study 4 | Full Fin-JEPA Evaluation | End-to-end benchmark vs. prior work |

---

## Study 0 — Financial Encoder Validation

**Goal:** Validate the FT-Transformer as a standalone financial encoder on XBRL data,
benchmarking it against XGBoost and logistic regression on five distress outcome types.
Also tests whether masked-feature self-supervised pretraining helps.

### Workstreams

- [x] **Company Universe** — Define SEC filer universe, 2012–2024 (ATS-161)
- [x] **XBRL Features** — Extract 16 canonical financial features from EDGAR (ATS-162)
- [x] **Market Data** — Collect and align prices, volumes, corporate actions (ATS-163)
- [x] **Label DB** — Build distress event label database, 5 outcome types (ATS-164)
- [x] **FT-Transformer** — Encoder for XBRL tabular features
- [x] **SSL Pretraining** — Masked feature reconstruction experiment
- [x] **Benchmark** — FT-Transformer vs. XGBoost vs. logistic regression, multi-seed variance estimation (go/no-go gate)
- [x] **Data Splits** — Reproducible time-based train/val/test splits
- [x] **Ablations** — Ablation studies and scaling curves
- [x] **Paper** — Study 0 technical report (`paper/study0/`)

---

## Repository Layout

```
fin-jepa/
├── src/fin_jepa/
│   ├── data/               # Ingestion, cleaning, label construction, splits
│   ├── models/             # FT-Transformer, baselines, SSL pretraining head
│   ├── training/           # Train loops, evaluation, metrics
│   └── utils/              # Logging, config helpers, reproducibility
├── configs/                # Hydra experiment configs (YAML)
│   └── study0/
├── scripts/                # Standalone pipeline runners
│   ├── run_market_pipeline.py
│   ├── prepare_hf_dataset.py
│   ├── audit_earnings_restate.py
│   └── diff_label_versions.py
├── notebooks/              # Exploratory analysis
│   └── explore_market_data.ipynb
├── experiments/            # Notebooks and scripts for each study
│   └── study0/
├── tests/                  # Unit + integration tests
├── results/                # Saved metrics, figures, paper assets
│   └── study0/
├── paper/                  # LaTeX source for study reports
│   └── study0/
├── data/                   # Data directory (NOT committed)
│   ├── raw/                #   EDGAR caches, parquets, market prices
│   ├── processed/          #   Label database, engineered features
│   └── splits/             #   Train/val/test splits
├── logs/                   # Pipeline log files (NOT committed)
└── models/                 # Saved checkpoints (NOT committed)
```

---

## Setup

```bash
# Create virtualenv
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Install (editable + dev extras)
pip install -e ".[dev,sec]"
```

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `EDGAR_USER_AGENT` | Recommended | Your identity for SEC EDGAR fair-use compliance (`"Name email@example.com"`). A default placeholder is used if unset. |

---

## Building the Dataset

The data pipeline has four sequential steps. Each step caches its HTTP responses
to disk, so interrupted runs resume where they left off.

```
Step 1              Step 2              Step 3                Step 4
Company Universe -> XBRL Features  ->  Market Data       ->  Distress Labels
(EDGAR index +     (Company Facts     (yfinance prices +    (5 binary outcomes
 submissions API)   API, 16 features)  filing-date align)    per company-year)

Output:            Output:            Output:               Output:
company_universe   xbrl_features      market_aligned        label_database
.parquet           .parquet           .parquet              .parquet
```

### Step 1 — Build Company Universe

Constructs a survivorship-bias-free universe of all SEC 10-K filers (2012–2024)
from EDGAR index files and the submissions API.

```bash
python -c "
from fin_jepa.data.universe import build_company_universe
build_company_universe(raw_dir='data/raw')
"
```

- **Output:** `data/raw/company_universe.parquet` (one row per company, 13 columns)
- **Time:** ~20 min on first run; cached after
- **Source:** EDGAR quarterly index files + per-company submissions JSON

### Step 2 — Extract XBRL Features

Downloads structured financial data from the EDGAR Company Facts API and extracts
16 canonical features (balance sheet, income statement, cash flow) per company-year.

```bash
python -c "
from pathlib import Path
from fin_jepa.data.xbrl_pipeline import build_xbrl_dataset
build_xbrl_dataset(raw_dir=Path('data/raw'), universe_path=Path('data/raw/company_universe.parquet'))
"
```

- **Output:** `data/raw/xbrl_features.parquet` (one row per company-year)
- **Requires:** Step 1 output
- **Features:** total_assets, total_liabilities, total_equity, current_assets,
  current_liabilities, retained_earnings, cash_equivalents, total_debt,
  total_revenue, cost_of_sales, operating_income, net_income, interest_expense,
  operating_cash_flow, capex, depreciation

### Step 3 — Collect and Align Market Data

Fetches adjusted OHLCV prices via yfinance, computes forward returns at multiple
horizons (21/63/126/252 trading days), and aligns everything to 10-K filing dates.

```bash
# Uses the dedicated CLI script (recommended)
python scripts/run_market_pipeline.py --skip-universe

# Key options:
#   --skip-universe    Use existing company_universe.parquet (skip Step 1 rebuild)
#   --with-actions     Also fetch corporate actions (splits/dividends; adds ~1-2 h)
#   --raw-dir DIR      Root data directory (default: data/raw)
#   --start-date DATE  Earliest price date (default: 2010-01-01)
#   --end-date DATE    Latest price date (default: 2024-12-31)
#   --batch-size N     Tickers per yfinance batch (default: 100)
```

- **Output:** `data/raw/market/market_aligned.parquet`
- **Requires:** Step 1 output
- **Time:** ~20–30 min for ~8k tickers; fully resumable
- **Log:** `logs/market_pipeline.log`

### Step 4 — Build Distress Labels

Constructs binary distress labels for five adverse outcome types per (company, year).

```bash
python -c "
from pathlib import Path
from fin_jepa.data.labels import build_label_database
build_label_database(raw_dir=Path('data/raw'))
"
```

- **Output:** `data/processed/label_database.parquet`
- **Requires:** Step 3 output (`market_aligned.parquet`)
- **Label columns** (nullable Int8 — 0/1/NaN where NaN = data unavailable):

| Outcome | Source | Description |
|---------|--------|-------------|
| `stock_decline` | market_aligned.parquet | >20% market-adjusted decline within 12 months |
| `earnings_restate` | EDGAR filing index + XBRL registry | Reconciled 10-K/A amendment + XBRL multi-filing detection |
| `audit_qualification` | External CSV (optional) | Going-concern or adverse opinion |
| `sec_enforcement` | External CSV (optional) | AAERs / litigation releases |
| `bankruptcy` | Compustat or external CSV (optional) | Chapter 7/11 filing |

### Optional External Data

Three of the five label types can use external CSV files for ground-truth data.
Place them in `data/raw/` and configure paths via `LabelConfig`:

- **Audit Analytics** — going-concern opinions (for `audit_qualification`)
- **Dechow et al. AAER database** — SEC enforcement actions (for `sec_enforcement`)
- **Bankruptcy filings** — Chapter 7/11 records (for `bankruptcy`)

Without these files, the corresponding label columns will be all-NaN. The pipeline
still runs and produces the two EDGAR-derived labels (`stock_decline`, `earnings_restate`).

---

## Running Experiments

```bash
# Study 0 benchmark (FT-Transformer vs baselines)
python -m fin_jepa.training.train_study0 experiment=study0/benchmark

# Study 0 SSL pretraining
python -m fin_jepa.training.pretrain_ssl experiment=study0/pretrain
```

Configuration lives in `configs/study0/` (Hydra YAML). Key files:
- `benchmark.yaml` — data paths, model hyperparameters, go/no-go gate thresholds
- `pretrain.yaml` — SSL pretraining (mask ratio, warmup, checkpoint dir)
- `ssl_experiment.yaml` — full SSL experiment (pretrain + fine-tune vs. scratch baseline)
- `ablations.yaml` — ablation study configurations

---

## Running Tests

```bash
pytest                     # all tests
pytest tests/ -x           # stop on first failure
pytest tests/ -k universe  # run only universe-related tests
pytest --co -q             # list tests without running
```

---

## References

- Gorishniy et al. (2021). *Revisiting Deep Learning Models for Tabular Data* (FT-Transformer)
- Assran et al. (2023). *Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture* (I-JEPA)

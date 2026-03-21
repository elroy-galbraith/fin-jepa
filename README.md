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

1. **Label DB** — Build distress event label database (5 outcome types)
2. **Market Data** — Collect and align prices, volumes, corporate actions
3. **FT-Transformer** — Implement encoder for XBRL tabular features
4. **SSL Pretraining** — Masked feature reconstruction experiment
5. **Benchmark** — FT-Transformer vs. XGBoost vs. logistic regression (go/no-go gate)
6. **Data Splits** — Reproducible train/test splits and data specification doc
7. **Ablations** — Ablation studies and scaling curves
8. **Paper** — Study 0 technical report

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
├── experiments/            # Notebooks and scripts for each study
│   └── study0/
├── tests/                  # Unit + integration tests
├── results/                # Saved metrics, figures, paper assets
│   └── study0/
├── data/                   # Data directory (raw/processed NOT committed)
│   ├── raw/
│   ├── processed/
│   └── splits/
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

## Running Experiments

```bash
# Study 0 benchmark (FT-Transformer vs baselines)
python -m fin_jepa.training.train_study0 experiment=study0/benchmark

# Study 0 SSL pretraining
python -m fin_jepa.training.pretrain_ssl experiment=study0/pretrain
```

---

## References

- Gorishniy et al. (2021). *Revisiting Deep Learning Models for Tabular Data* (FT-Transformer)
- Assran et al. (2023). *Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture* (I-JEPA)

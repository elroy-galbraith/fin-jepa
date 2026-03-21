"""
Ablation studies and scaling curves.

Workstream: Run ablation studies and produce scaling curves.

Ablations planned for Study 0
------------------------------
- Effect of SSL pretraining (pretrained vs. from-scratch)
- Number of Transformer layers (1, 2, 3, 6)
- Token dimension (64, 128, 192, 256)
- Training set size scaling curve (10%, 25%, 50%, 75%, 100%)
- Masking ratio for SSL pretraining (0.10, 0.15, 0.20, 0.30)
- Feature set: XBRL raw vs. engineered ratios only vs. both

TODO:
  - Implement sweep runner using Optuna or manual grid
  - Produce scaling curve plots (training size vs. AUROC)
  - Save results to results/study0/ablations/
"""

from __future__ import annotations

import logging

log = logging.getLogger(__name__)


def run_ablations(config: dict) -> None:
    """Entry point for ablation study sweeps."""
    raise NotImplementedError("Implement ablation study runner.")

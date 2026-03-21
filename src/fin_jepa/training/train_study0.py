"""
Study 0 benchmark training script.

Workstream: Run FT-Transformer vs. baselines benchmark (go/no-go gate).

Usage:
    python -m fin_jepa.training.train_study0 experiment=study0/benchmark

TODO:
  - Load processed dataset and splits
  - Train logistic regression and XGBoost baselines per outcome
  - Train FT-Transformer per outcome (or multi-task)
  - Evaluate all models on test split
  - Run go/no-go gate check
  - Save results to results/study0/benchmark_results.json
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

log = logging.getLogger(__name__)


def run_benchmark(config: dict) -> None:
    """Entry point for the Study 0 benchmark experiment."""
    raise NotImplementedError(
        "Implement Study 0 benchmark. See module docstring for workstream steps."
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_benchmark({})

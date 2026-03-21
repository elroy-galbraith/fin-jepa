"""
Self-supervised pretraining script (masked feature reconstruction).

Workstream: Self-supervised pretraining experiment.

Usage:
    python -m fin_jepa.training.pretrain_ssl experiment=study0/pretrain

TODO:
  - Load unlabeled XBRL feature dataset (train split only)
  - Initialize MaskedFeatureSSL wrapper around FTTransformer backbone
  - Run pretraining loop with AdamW + cosine LR schedule
  - Save pretrained encoder checkpoint to models/checkpoints/
  - Compare fine-tuned pretrained encoder vs. from-scratch on downstream tasks
"""

from __future__ import annotations

import logging

log = logging.getLogger(__name__)


def run_pretraining(config: dict) -> None:
    """Entry point for SSL pretraining experiment."""
    raise NotImplementedError(
        "Implement SSL pretraining loop. See module docstring for workstream steps."
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_pretraining({})

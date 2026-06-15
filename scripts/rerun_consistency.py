#!/usr/bin/env python
"""Regenerate the walk-forward (per-fold scaler) and multi-seed results on the
Optuna-tuned FT-Transformer architecture.

Writes to ``results/study0/corrected/``:
  * ``walk_forward_results.json`` — now leak-free (each fold's scaler is fit on
    that fold's training split only; see ``run_walk_forward`` / PR #32).
  * ``multiseed_results.json`` — FT-Transformer variance on the Optuna arch.

SSL is intentionally NOT re-run here: at the Optuna architecture it is ~50h of
pretraining on CPU, and the scratch-vs-SSL comparison is self-consistent at its
own (d=192) architecture, so re-running buys no scientific gain.

Usage::

    python scripts/rerun_consistency.py            # full run
    python scripts/rerun_consistency.py --smoke     # fast CPU sanity
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from fin_jepa.training.train_study0 import (  # noqa: E402
    run_multiseed_benchmark,
    run_walk_forward,
)

log = logging.getLogger("rerun_consistency")
CORRECTED = ROOT / "results" / "study0" / "corrected"


def _load_cfg() -> dict:
    cfg_path = ROOT / "configs" / "study0" / "benchmark.yaml"
    try:
        from omegaconf import OmegaConf

        return OmegaConf.to_container(OmegaConf.load(cfg_path), resolve=True)
    except ImportError:
        import yaml

        with open(cfg_path) as f:
            return yaml.safe_load(f)


def main(smoke: bool = False) -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )

    ft = json.loads((CORRECTED / "ft_transformer_tuning.json").read_text())["best_params"]
    log.info("Optuna FT params: %s", ft)

    cfg = _load_cfg()
    cfg["results_dir"] = str(CORRECTED)
    # Inject the Optuna-tuned FT architecture everywhere.
    cfg["ft_transformer"] = {**cfg["ft_transformer"],
                             "d_token": int(ft["d_token"]),
                             "n_layers": int(ft["n_layers"])}
    cfg["training"] = {**cfg["training"], "learning_rate": float(ft["learning_rate"])}

    if smoke:
        # Never overwrite the canonical corrected/ artifacts during a smoke.
        cfg["results_dir"] = str(CORRECTED / "_smoke_rerun")
        cfg["outcomes"] = ["stock_decline"]
        cfg["training"] = {**cfg["training"], "epochs": 3, "patience": 2,
                           "seeds": [42, 123]}
        cfg["rolling_split"] = {
            "first_train_end": "2017-12-31", "val_window_years": 1,
            "test_window_years": 2, "step_years": 1, "last_test_end": "2021-12-31",
        }

    # ── Walk-forward (per-fold scaler, tuned FT) ─────────────────────────
    log.info("=" * 64)
    log.info("STEP 1/2  Walk-forward (per-fold scaler, tuned FT)")
    t0 = time.time()
    wf = run_walk_forward(cfg, tuned_ft_params=ft)
    log.info("STEP 1 done: %d folds in %.1f min", wf["n_folds"], (time.time() - t0) / 60)

    # ── Multi-seed (Optuna arch) ─────────────────────────────────────────
    log.info("=" * 64)
    log.info("STEP 2/2  Multi-seed FT-Transformer (Optuna arch)")
    t0 = time.time()
    ms = run_multiseed_benchmark(cfg, seeds=list(cfg["training"]["seeds"]))
    n = len(ms.get("multiseed", {}))
    log.info("STEP 2 done: %d outcomes in %.1f min", n, (time.time() - t0) / 60)

    log.info("=" * 64)
    log.info("CONSISTENCY RE-RUN COMPLETE -> %s", CORRECTED)


if __name__ == "__main__":
    main(smoke="--smoke" in sys.argv)

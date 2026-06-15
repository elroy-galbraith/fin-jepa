#!/usr/bin/env python
"""Download the Study 0 dataset from HuggingFace into the local ``data/`` layout.

Pulls the four source parquets published at ``elroyg/fin-jepa-study0`` (the
``raw/`` folder) and places them where the training/evaluation code expects them,
so the reproduction pipeline (see ``REPRODUCE.md``) can run without rebuilding the
data from EDGAR.

This is the *download* counterpart to ``prepare_hf_dataset.py`` (which uploads).

Usage::

    python scripts/download_data.py
    python scripts/download_data.py --repo-id elroyg/fin-jepa-study0 --force

Requires ``huggingface_hub`` (``pip install -e '.[hf]'`` or ``pip install huggingface_hub``).
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# HF repo path  ->  local destination (relative to repo root).
# These destinations match the loaders: load_xbrl_features(data/raw),
# load_label_database(data/processed/label_database.parquet),
# company_universe at data/raw, market_aligned at data/raw/market.
FILE_MAP: dict[str, str] = {
    "raw/xbrl_features.parquet": "data/raw/xbrl_features.parquet",
    "raw/company_universe.parquet": "data/raw/company_universe.parquet",
    "raw/label_database.parquet": "data/processed/label_database.parquet",
    "raw/market_aligned.parquet": "data/raw/market/market_aligned.parquet",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default="elroyg/fin-jepa-study0",
                        help="HuggingFace dataset repo id")
    parser.add_argument("--force", action="store_true",
                        help="re-download even if the destination already exists")
    args = parser.parse_args()

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise SystemExit(
            "huggingface_hub is not installed. Run: pip install -e '.[hf]'"
        )

    for src, rel_dst in FILE_MAP.items():
        dst = ROOT / rel_dst
        if dst.exists() and not args.force:
            print(f"skip (exists)  {rel_dst}")
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        cached = hf_hub_download(args.repo_id, src, repo_type="dataset")
        shutil.copyfile(cached, dst)
        print(f"downloaded     {rel_dst}  ({dst.stat().st_size / 1e6:.1f} MB)")

    print("\nDone. Data is ready under data/. See REPRODUCE.md for the next steps.")


if __name__ == "__main__":
    main()

"""
Self-supervised pretraining script (masked feature reconstruction).

Workstream: Self-supervised pretraining experiment.

Usage:
    python -m fin_jepa.training.pretrain_ssl experiment=study0/pretrain
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from fin_jepa.data.feature_engineering import FeatureConfig, build_feature_matrix
from fin_jepa.data.splits import SplitConfig
from fin_jepa.data.xbrl_loader import load_xbrl_features
from fin_jepa.models.ft_transformer import FTTransformer
from fin_jepa.models.ssl_head import MaskedFeatureSSL
from fin_jepa.training.dataset import make_dataloader
from fin_jepa.utils.reproducibility import seed_everything

log = logging.getLogger(__name__)


def run_pretraining(config) -> Path:
    """Entry point for SSL pretraining experiment.

    Parameters
    ----------
    config : dict or DictConfig
        Hydra configuration (from ``configs/study0/pretrain.yaml``).

    Returns
    -------
    Path to the saved encoder checkpoint.
    """
    from fin_jepa.training.train_study0 import _cfg

    seed = int(_cfg(config, "training.seed", 42))
    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    # ── Load data (train split only) ─────────────────────────────────
    raw_dir = Path(_cfg(config, "data.raw_dir", "data/raw"))

    xbrl_df = load_xbrl_features(raw_dir)

    split_cfg = SplitConfig(
        train_end=_cfg(config, "data.split.train_end", "2017-12-31"),
        val_end=_cfg(config, "data.split.val_end", "2019-12-31"),
        test_end=_cfg(config, "data.split.test_end", "2023-12-31"),
    )
    feat_cfg = FeatureConfig(
        use_raw=_cfg(config, "features.use_raw", True),
        use_ratios=_cfg(config, "features.use_ratios", True),
        use_yoy=_cfg(config, "features.use_yoy", True),
        use_missingness_flags=_cfg(config, "features.use_missingness_flags", True),
        coverage_threshold=_cfg(config, "features.coverage_threshold", 0.50),
        normalization_method=_cfg(config, "features.normalization_method", "quantile"),
        median_impute=_cfg(config, "features.median_impute", True),
    )
    splits, scaler, feature_cols = build_feature_matrix(xbrl_df, split_cfg, feat_cfg)

    train_loader = make_dataloader(
        splits["train"],
        feature_cols,
        label_col=None,
        batch_size=int(_cfg(config, "training.batch_size", 512)),
        shuffle=True,
    )

    n_features = len(feature_cols)
    log.info("Training SSL on %d features, %d train rows.", n_features, len(splits["train"]))

    # ── Model ────────────────────────────────────────────────────────
    encoder = FTTransformer(
        n_features=n_features,
        d_token=int(_cfg(config, "ft_transformer.d_token", 192)),
        n_heads=int(_cfg(config, "ft_transformer.n_heads", 8)),
        n_layers=int(_cfg(config, "ft_transformer.n_layers", 3)),
        d_ffn_factor=int(_cfg(config, "ft_transformer.d_ffn_factor", 4)),
        dropout=float(_cfg(config, "ft_transformer.dropout", 0.0)),
        n_outputs=1,
    )
    mask_ratio = float(_cfg(config, "ssl.mask_ratio", 0.15))
    ssl_model = MaskedFeatureSSL(encoder, mask_ratio=mask_ratio).to(device)

    # ── Optimizer + LR schedule ──────────────────────────────────────
    lr = float(_cfg(config, "training.learning_rate", 1e-4))
    wd = float(_cfg(config, "training.weight_decay", 1e-5))
    epochs = int(_cfg(config, "training.epochs", 200))
    warmup_epochs = int(_cfg(config, "training.warmup_epochs", 10))

    optimizer = torch.optim.AdamW(ssl_model.parameters(), lr=lr, weight_decay=wd)

    warmup_scheduler = LinearLR(optimizer, start_factor=1e-3, total_iters=warmup_epochs)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=max(epochs - warmup_epochs, 1))
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_epochs],
    )

    # ── Training loop ────────────────────────────────────────────────
    for epoch in range(1, epochs + 1):
        ssl_model.train()
        total_loss = 0.0
        n_batches = 0

        for (x_batch,) in train_loader:
            x_batch = x_batch.to(device)
            optimizer.zero_grad()
            loss, _x_hat, _mask = ssl_model(x_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ssl_model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = total_loss / max(n_batches, 1)
        current_lr = optimizer.param_groups[0]["lr"]

        if epoch <= 5 or epoch % 10 == 0 or epoch == epochs:
            log.info(
                "Epoch %3d/%d | loss=%.6f | lr=%.2e",
                epoch, epochs, avg_loss, current_lr,
            )

    # ── Save checkpoint ──────────────────────────────────────────────
    ckpt_dir = Path(_cfg(config, "checkpoint_dir", "models/checkpoints"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "encoder_pretrained.pt"
    torch.save(ssl_model.encoder.state_dict(), ckpt_path)
    log.info("Pretrained encoder saved to %s", ckpt_path)

    return ckpt_path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    try:
        import hydra
        from omegaconf import DictConfig

        @hydra.main(config_path="../../../configs/study0", config_name="pretrain", version_base=None)
        def main(cfg: DictConfig) -> None:
            run_pretraining(cfg)

        main()
    except ImportError:
        log.warning("Hydra not available — running with empty config.")
        run_pretraining({})

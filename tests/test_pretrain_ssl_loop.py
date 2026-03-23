"""Tests for the SSL pretraining loop."""

import tempfile
from pathlib import Path

import numpy as np
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from fin_jepa.models.ft_transformer import FTTransformer
from fin_jepa.models.ssl_head import MaskedFeatureSSL
from fin_jepa.training.dataset import TabularDataset


class TestSSLPretrainingSmoke:
    def test_loss_is_finite(self):
        """SSL pretraining on synthetic data should produce finite loss."""
        n_features = 8
        X = np.random.randn(50, n_features).astype(np.float32)
        ds = TabularDataset(X)
        loader = torch.utils.data.DataLoader(ds, batch_size=16, shuffle=True)

        encoder = FTTransformer(
            n_features=n_features, d_token=16, n_heads=2, n_layers=1, n_outputs=1,
        )
        ssl_model = MaskedFeatureSSL(encoder, mask_ratio=0.15)
        optimizer = torch.optim.AdamW(ssl_model.parameters(), lr=1e-3)

        losses = []
        for epoch in range(3):
            ssl_model.train()
            epoch_loss = 0.0
            n = 0
            for (x_batch,) in loader:
                optimizer.zero_grad()
                loss, _x_hat, _mask = ssl_model(x_batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                n += 1
            avg = epoch_loss / n
            losses.append(avg)
            assert np.isfinite(avg), f"Non-finite loss at epoch {epoch}: {avg}"

    def test_loss_does_not_increase_monotonically(self):
        """After enough epochs, loss should generally trend downward."""
        torch.manual_seed(42)
        np.random.seed(42)

        n_features = 8
        # Use structured data (repeated pattern) so reconstruction is learnable
        base = np.random.randn(10, n_features).astype(np.float32)
        X = np.tile(base, (20, 1))  # 200 rows of repeated patterns
        ds = TabularDataset(X)
        loader = torch.utils.data.DataLoader(ds, batch_size=64, shuffle=True)

        encoder = FTTransformer(
            n_features=n_features, d_token=32, n_heads=2, n_layers=1, n_outputs=1,
        )
        ssl_model = MaskedFeatureSSL(encoder, mask_ratio=0.15)
        optimizer = torch.optim.AdamW(ssl_model.parameters(), lr=5e-4)

        losses = []
        for epoch in range(20):
            ssl_model.train()
            epoch_loss = 0.0
            n = 0
            for (x_batch,) in loader:
                optimizer.zero_grad()
                loss, _, _ = ssl_model(x_batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                n += 1
            losses.append(epoch_loss / n)

        # Average of last 3 epochs should be lower than average of first 3
        early_avg = np.mean(losses[:3])
        late_avg = np.mean(losses[-3:])
        assert late_avg < early_avg, (
            f"Loss did not decrease: early_avg={early_avg:.4f} -> late_avg={late_avg:.4f}"
        )

    def test_checkpoint_roundtrip(self):
        """Encoder weights can be saved and reloaded."""
        n_features = 5
        encoder = FTTransformer(
            n_features=n_features, d_token=16, n_heads=2, n_layers=1, n_outputs=1,
        )
        ssl_model = MaskedFeatureSSL(encoder, mask_ratio=0.15)

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "encoder.pt"
            torch.save(ssl_model.encoder.state_dict(), ckpt_path)

            # Load into a fresh encoder
            new_encoder = FTTransformer(
                n_features=n_features, d_token=16, n_heads=2, n_layers=1, n_outputs=1,
            )
            new_encoder.load_state_dict(torch.load(ckpt_path, weights_only=True))

            # Verify weights match
            for (k1, v1), (k2, v2) in zip(
                encoder.state_dict().items(),
                new_encoder.state_dict().items(),
            ):
                assert k1 == k2
                assert torch.equal(v1, v2), f"Mismatch in {k1}"

    def test_lr_schedule(self):
        """LinearLR + CosineAnnealingLR schedule produces valid learning rates."""
        encoder = FTTransformer(n_features=3, d_token=8, n_heads=1, n_layers=1, n_outputs=1)
        optimizer = torch.optim.AdamW(encoder.parameters(), lr=1e-3)

        warmup = LinearLR(optimizer, start_factor=1e-3, total_iters=5)
        cosine = CosineAnnealingLR(optimizer, T_max=15)
        scheduler = SequentialLR(optimizer, [warmup, cosine], milestones=[5])

        lrs = []
        for _ in range(20):
            lrs.append(optimizer.param_groups[0]["lr"])
            scheduler.step()

        # LR should increase during warmup then decrease
        assert lrs[4] > lrs[0], "LR should increase during warmup"
        assert all(lr > 0 for lr in lrs), "All LRs should be positive"

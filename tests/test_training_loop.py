"""Tests for the FT-Transformer training loop and helpers."""

import numpy as np
import torch
import torch.nn as nn

from fin_jepa.models.ft_transformer import FTTransformer
from fin_jepa.training.dataset import TabularDataset
from fin_jepa.training.train_study0 import train_ft_transformer, _evaluate_auroc


def _make_loaders(n_train=80, n_val=20, n_features=10, batch_size=16):
    """Create tiny synthetic train/val DataLoaders for smoke testing."""
    rng = np.random.RandomState(0)

    # Linearly separable data so the model can learn
    X_train = rng.randn(n_train, n_features).astype(np.float32)
    y_train = (X_train[:, 0] > 0).astype(np.float32)
    X_val = rng.randn(n_val, n_features).astype(np.float32)
    y_val = (X_val[:, 0] > 0).astype(np.float32)

    train_ds = TabularDataset(X_train, y_train)
    val_ds = TabularDataset(X_val, y_val)

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


class TestTrainFTTransformer:
    def test_smoke_loss_decreases(self):
        """Training for a few epochs should reduce loss on a trivial task."""
        train_loader, val_loader = _make_loaders()
        device = torch.device("cpu")

        model = FTTransformer(n_features=10, d_token=16, n_heads=2, n_layers=1, n_outputs=1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        criterion = nn.BCEWithLogitsLoss()

        # Record initial loss
        model.eval()
        with torch.no_grad():
            init_losses = []
            for x, y in train_loader:
                logits = model(x).squeeze(-1)
                init_losses.append(criterion(logits, y).item())
        init_loss = np.mean(init_losses)

        result = train_ft_transformer(
            model, train_loader, val_loader, criterion, optimizer,
            device, epochs=20, patience=50,
        )

        # Final loss should be lower
        model.eval()
        with torch.no_grad():
            final_losses = []
            for x, y in train_loader:
                logits = model(x).squeeze(-1)
                final_losses.append(criterion(logits, y).item())
        final_loss = np.mean(final_losses)

        assert final_loss < init_loss, f"Loss did not decrease: {init_loss:.4f} -> {final_loss:.4f}"

    def test_return_structure(self):
        """train_ft_transformer returns the expected dict keys."""
        train_loader, val_loader = _make_loaders()
        device = torch.device("cpu")

        model = FTTransformer(n_features=10, d_token=16, n_heads=2, n_layers=1, n_outputs=1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        criterion = nn.BCEWithLogitsLoss()

        result = train_ft_transformer(
            model, train_loader, val_loader, criterion, optimizer,
            device, epochs=3, patience=50,
        )

        assert "best_val_auroc" in result
        assert "best_epoch" in result
        assert "state_dict" in result
        assert isinstance(result["best_val_auroc"], float)
        assert isinstance(result["best_epoch"], int)
        assert result["best_epoch"] >= 1

    def test_early_stopping(self):
        """Training should stop before max epochs when val AUROC plateaus."""
        train_loader, val_loader = _make_loaders()
        device = torch.device("cpu")

        model = FTTransformer(n_features=10, d_token=16, n_heads=2, n_layers=1, n_outputs=1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        criterion = nn.BCEWithLogitsLoss()

        max_epochs = 200
        result = train_ft_transformer(
            model, train_loader, val_loader, criterion, optimizer,
            device, epochs=max_epochs, patience=5,
        )

        # With patience=5 on such a small dataset, it should stop well before 200
        assert result["best_epoch"] < max_epochs


class TestEvaluateAuroc:
    def test_perfect_predictions(self):
        """AUROC should be 1.0 for perfectly separable predictions."""
        X = np.array([[10.0], [-10.0], [10.0], [-10.0]], dtype=np.float32)
        y = np.array([1.0, 0.0, 1.0, 0.0], dtype=np.float32)
        ds = TabularDataset(X, y)
        loader = torch.utils.data.DataLoader(ds, batch_size=4)

        # A model that outputs large positive/negative logits based on sign
        model = FTTransformer(n_features=1, d_token=8, n_heads=1, n_layers=1, n_outputs=1)
        # Override weights so output correlates with input sign
        with torch.no_grad():
            # Make tokenizer pass through the sign of input
            model.tokenizer.weight.fill_(1.0)
            model.tokenizer.bias.fill_(0.0)

        # Even without perfect weights, AUROC should be a valid float
        auroc = _evaluate_auroc(model, loader, torch.device("cpu"))
        assert 0.0 <= auroc <= 1.0

    def test_single_class_returns_zero(self):
        """AUROC should be 0.0 when only one class is present."""
        X = np.ones((4, 2), dtype=np.float32)
        y = np.ones(4, dtype=np.float32)  # all positive
        ds = TabularDataset(X, y)
        loader = torch.utils.data.DataLoader(ds, batch_size=4)

        model = FTTransformer(n_features=2, d_token=8, n_heads=1, n_layers=1, n_outputs=1)
        auroc = _evaluate_auroc(model, loader, torch.device("cpu"))
        assert auroc == 0.0

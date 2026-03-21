"""Unit tests for evaluation metrics."""

import numpy as np
import pytest

from fin_jepa.training.metrics import compute_all_metrics, go_no_go_gate


def test_compute_all_metrics_perfect():
    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([0.1, 0.2, 0.8, 0.9])
    m = compute_all_metrics(y_true, y_score)
    assert m["auroc"] == pytest.approx(1.0)
    assert 0 <= m["auprc"] <= 1
    assert 0 <= m["brier"] <= 1


def test_go_no_go_gate_passes():
    outcomes = ["a", "b", "c", "d", "e"]
    ft = {o: {"auroc": 0.80} for o in outcomes}
    xgb = {o: {"auroc": 0.75} for o in outcomes}
    passed, wins, detail = go_no_go_gate(ft, xgb, outcomes, margin=0.01)
    assert passed
    assert wins == 5


def test_go_no_go_gate_fails():
    outcomes = ["a", "b", "c", "d", "e"]
    ft = {o: {"auroc": 0.70} for o in outcomes}
    xgb = {o: {"auroc": 0.75} for o in outcomes}
    passed, wins, detail = go_no_go_gate(ft, xgb, outcomes, margin=0.01)
    assert not passed
    assert wins == 0

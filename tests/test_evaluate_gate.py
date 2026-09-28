"""
tests/test_evaluate_gate.py — Tests for src/evaluate_gate.py.

Covers:
  - pred_converging / actual_converging label helpers
  - compute_clf_metrics with known inputs
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from evaluate_gate import (
    actual_converging,
    compute_clf_metrics,
    gate_score,
    pred_converging,
)


class TestGateLabels:
    def test_pred_converging_true(self):
        preds = [0.1, 0.2, 0.1, 0.1, 0.1]   # mean|pred| = 0.12 < 2.0
        assert pred_converging(preds, z_t=2.0) == 1

    def test_pred_converging_false(self):
        preds = [1.5, 2.0, 2.5, 2.0, 1.8]   # mean|pred| = 1.96 > 0.5
        assert pred_converging(preds, z_t=0.5) == 0

    def test_pred_converging_boundary_is_false(self):
        # mean|pred| == |z_t| → not STRICTLY less → 0
        preds = [1.0, 1.0, 1.0, 1.0, 1.0]
        assert pred_converging(preds, z_t=1.0) == 0

    def test_actual_converging_true(self):
        future_z = [0.3, 0.2, 0.1, 0.05, 0.0]   # mean|fut| = 0.13 < 2.0
        assert actual_converging(future_z, z_t=2.0) == 1

    def test_actual_converging_false(self):
        future_z = [2.5, 3.0, 3.5, 3.0, 2.5]    # mean|fut| = 2.9 > 1.0
        assert actual_converging(future_z, z_t=1.0) == 0

    def test_gate_score_positive_when_converging(self):
        # score = |z_t| - mean|pred|; positive means gate predicts convergence
        score = gate_score(z_t=2.0, preds=[0.5, 0.4, 0.3, 0.2, 0.1])
        assert score > 0

    def test_gate_score_negative_when_diverging(self):
        score = gate_score(z_t=0.5, preds=[1.5, 2.0, 2.5, 2.0, 1.8])
        assert score < 0


class TestComputeClfMetrics:
    def test_perfect_classifier(self):
        y_true = [0, 0, 1, 1]
        y_pred = [0, 0, 1, 1]
        scores = [0.0, 0.0, 1.0, 1.0]
        m = compute_clf_metrics(y_true, y_pred, scores)
        assert m["accuracy"]  == pytest.approx(1.0)
        assert m["precision"] == pytest.approx(1.0)
        assert m["recall"]    == pytest.approx(1.0)
        assert m["f1"]        == pytest.approx(1.0)
        assert m["roc_auc"]   == pytest.approx(1.0)

    def test_all_wrong(self):
        y_true = [0, 0, 1, 1]
        y_pred = [1, 1, 0, 0]
        scores = [1.0, 1.0, 0.0, 0.0]
        m = compute_clf_metrics(y_true, y_pred, scores)
        assert m["accuracy"]  == pytest.approx(0.0)
        assert m["roc_auc"]   == pytest.approx(0.0)

    def test_empty_input_returns_nan(self):
        m = compute_clf_metrics([], [], [])
        assert np.isnan(m["accuracy"])
        assert np.isnan(m["roc_auc"])

    def test_n_matches_input_length(self):
        y = [0, 1, 0, 1, 0]
        m = compute_clf_metrics(y, y, [0.0]*5)
        assert m["n"] == 5

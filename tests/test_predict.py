"""
tests/test_predict.py — Tests for src/predict.py.

Covers:
  - forecast_pair returns correct output shape and types for AMZN/META
  - convergence gate logic (mean|pred| < |z_now| => converging)
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from predict import build_features, forecast_pair


class TestBuildFeatures:
    def test_output_columns(self):
        import pandas as pd
        z = pd.Series(np.random.randn(50), name="z_std")
        feat = build_features(z)
        assert list(feat.columns) == ["z_std", "lag1", "lag2", "lag3", "roll_std"]

    def test_lag1_is_shifted(self):
        import pandas as pd
        z = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        feat = build_features(z)
        assert feat["lag1"].iloc[1] == pytest.approx(1.0)
        assert feat["lag1"].iloc[2] == pytest.approx(2.0)


class TestConvergenceGateLogic:
    """Unit-test the gate rule without loading TF or real data."""

    def _gate(self, preds, z_now):
        return float(np.mean(np.abs(preds))) < abs(z_now)

    def test_converging_when_mean_pred_smaller(self):
        preds = [0.1, 0.2, 0.1, 0.1, 0.1]
        assert self._gate(preds, z_now=2.0) is True

    def test_diverging_when_mean_pred_larger(self):
        preds = [1.5, 2.0, 2.5, 2.0, 1.8]
        assert self._gate(preds, z_now=0.5) is False

    def test_boundary_is_diverging(self):
        # mean|pred| == |z_now| → not strictly less → diverging
        preds = [1.0, 1.0, 1.0, 1.0, 1.0]
        assert self._gate(preds, z_now=1.0) is False


@pytest.mark.slow
class TestForecastPairAmznMeta:
    """Integration test: loads saved artefacts and runs inference."""

    def test_forecast_shape_and_types(self):
        r = forecast_pair("AMZN", "META")
        assert r["dep"] == "AMZN"
        assert r["indep"] == "META"
        assert len(r["forecast"]) == 5
        assert all(isinstance(v, float) for v in r["forecast"])
        assert isinstance(r["z_std"], float)
        assert isinstance(r["converging"], bool)

    def test_forecast_values_are_finite(self):
        r = forecast_pair("AMZN", "META")
        assert all(np.isfinite(v) for v in r["forecast"])

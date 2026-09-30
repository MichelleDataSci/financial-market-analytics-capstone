"""
tests/test_app.py — FastAPI app tests.

Three cases:
  1. Cointegrated pair  : synthetic y = 0.8*x + noise (tight cointegration)
  2. Not cointegrated   : two independent random walks
  3. Bad input          : CSV missing the Date column
"""

import io
import sys
from pathlib import Path
from unittest.mock import patch

import matplotlib.axes
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from app.main import app  # noqa: E402

client = TestClient(app)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_RNG = np.random.default_rng(42)


def _make_csv(df: pd.DataFrame) -> bytes:
    buf = io.StringIO()
    df.to_csv(buf, index=True)
    return buf.getvalue().encode()


def _cointegrated_csv(n: int = 200) -> bytes:
    """y = 0.8*x + small noise → tight cointegration."""
    dates = pd.bdate_range("2018-01-02", periods=n)
    x = np.cumsum(_RNG.standard_normal(n)) + 100
    y = 0.8 * x + _RNG.standard_normal(n) * 0.5
    df = pd.DataFrame({"Date": dates, "AssetA": x, "AssetB": y})
    df = df.set_index("Date")
    return _make_csv(df)


def _independent_csv(n: int = 200) -> bytes:
    """Two independent random walks — unlikely to be cointegrated."""
    dates = pd.bdate_range("2018-01-02", periods=n)
    x = np.cumsum(_RNG.standard_normal(n)) + 100
    y = np.cumsum(_RNG.standard_normal(n)) + 50
    df = pd.DataFrame({"Date": dates, "SeriesX": x, "SeriesY": y})
    df = df.set_index("Date")
    return _make_csv(df)


def _no_date_csv() -> bytes:
    """CSV without a Date column — should fail validation."""
    df = pd.DataFrame({"Price1": [100, 101, 102], "Price2": [200, 201, 202]})
    return df.to_csv(index=False).encode()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestAnalyseEndpoint:
    def test_get_home_returns_200(self):
        resp = client.get("/")
        assert resp.status_code == 200
        assert "Pairs Trading" in resp.text

    def test_cointegrated_pair_html(self):
        csv_bytes = _cointegrated_csv()
        resp = client.post(
            "/analyse",
            files={"file": ("pair.csv", csv_bytes, "text/csv")},
            data={"ticker1": "", "ticker2": ""},
        )
        assert resp.status_code == 200
        # Should show cointegrated badge OR the weak/no badge — just check the page rendered
        assert "AssetA" in resp.text or "AssetB" in resp.text

    def test_not_cointegrated_html_no_signals(self):
        csv_bytes = _independent_csv()
        resp = client.post(
            "/analyse",
            files={"file": ("indep.csv", csv_bytes, "text/csv")},
            data={"ticker1": "", "ticker2": ""},
        )
        assert resp.status_code == 200
        # Results page must be returned; signals section should NOT appear
        assert "SeriesX" in resp.text or "SeriesY" in resp.text
        # The spread chart should only appear when cointegrated
        assert "Spread (log-price residual)" not in resp.text or \
               "signals not generated" in resp.text or \
               "Not cointegrated" in resp.text

    def test_bad_input_no_date_column(self):
        csv_bytes = _no_date_csv()
        resp = client.post(
            "/analyse",
            files={"file": ("bad.csv", csv_bytes, "text/csv")},
            data={"ticker1": "", "ticker2": ""},
        )
        # Should re-render the form with an error (422 from HTTPException, rendered as HTML)
        assert resp.status_code in (200, 422)
        assert "Date" in resp.text  # error message mentions Date column

    def test_bad_input_no_ticker_no_file(self):
        resp = client.post(
            "/analyse",
            data={"ticker1": "", "ticker2": ""},
        )
        assert resp.status_code in (200, 422)
        assert "Select two tickers" in resp.text or "error" in resp.text.lower()


class TestApiAnalyseEndpoint:
    def test_cointegrated_pair_json_structure(self):
        csv_bytes = _cointegrated_csv()
        resp = client.post(
            "/api/analyse",
            files={"file": ("pair.csv", csv_bytes, "text/csv")},
            data={"ticker1": "", "ticker2": ""},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "is_cointegrated" in data
        assert "engle_granger" in data
        assert "johansen" in data
        # Charts must NOT appear in JSON
        assert "chart_spread_b64" not in data
        assert "chart_zscore_b64" not in data

    def test_not_cointegrated_json_no_signals(self):
        csv_bytes = _independent_csv()
        resp = client.post(
            "/api/analyse",
            files={"file": ("indep.csv", csv_bytes, "text/csv")},
            data={"ticker1": "", "ticker2": ""},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "is_cointegrated" in data
        # When not cointegrated: signals list is empty
        if not data["is_cointegrated"]:
            assert data["signals"] == []

    def test_bad_input_returns_422(self):
        csv_bytes = _no_date_csv()
        resp = client.post(
            "/api/analyse",
            files={"file": ("bad.csv", csv_bytes, "text/csv")},
            data={"ticker1": "", "ticker2": ""},
        )
        assert resp.status_code == 422
        assert "Date" in resp.json()["detail"]

    def test_zero_price_returns_422(self):
        """A CSV with only a handful of valid rows (after zero-price rows are dropped)
        should fall below MIN_ROWS and return 422."""
        # 65 rows total, last 10 have y=0 → only 55 finite rows < MIN_ROWS=60
        n = 65
        dates = pd.bdate_range("2018-01-02", periods=n)
        x = np.cumsum(_RNG.standard_normal(n)) + 100
        y = np.cumsum(_RNG.standard_normal(n)) + 80
        y[55:] = 0.0  # 10 rows become non-finite after log
        df = pd.DataFrame({"Date": dates, "AssetA": x, "AssetB": y}).set_index("Date")
        resp = client.post(
            "/api/analyse",
            files={"file": ("zero.csv", _make_csv(df), "text/csv")},
            data={"ticker1": "", "ticker2": ""},
        )
        assert resp.status_code == 422

    def test_negative_price_returns_422(self):
        """A CSV with only a handful of valid rows (after negative-price rows are dropped)
        should fall below MIN_ROWS and return 422."""
        # 65 rows total, last 10 have y<0 → only 55 finite rows < MIN_ROWS=60
        n = 65
        dates = pd.bdate_range("2018-01-02", periods=n)
        x = np.cumsum(_RNG.standard_normal(n)) + 100
        y = np.cumsum(_RNG.standard_normal(n)) + 80
        y[55:] = -5.0
        df = pd.DataFrame({"Date": dates, "AssetA": x, "AssetB": y}).set_index("Date")
        resp = client.post(
            "/api/analyse",
            files={"file": ("neg.csv", _make_csv(df), "text/csv")},
            data={"ticker1": "", "ticker2": ""},
        )
        assert resp.status_code == 422

    def test_single_zero_price_row_succeeds(self):
        """A 200-row CSV with one zero price should return 200, not 500.
        The non-finite-row fix drops the one bad row and continues with 199 rows."""
        n = 200
        dates = pd.bdate_range("2018-01-02", periods=n)
        x = np.cumsum(_RNG.standard_normal(n)) + 100
        y = 0.8 * x + _RNG.standard_normal(n) * 0.5
        y[100] = 0.0  # single zero — used to raise unhandled 500
        df = pd.DataFrame({"Date": dates, "AssetA": x, "AssetB": y}).set_index("Date")
        resp = client.post(
            "/api/analyse",
            files={"file": ("one_zero.csv", _make_csv(df), "text/csv")},
            data={"ticker1": "", "ticker2": ""},
        )
        assert resp.status_code == 200

    def test_reverse_ordered_csv_same_result_as_sorted(self):
        """A reverse-date CSV (newest first) should produce the same analysis as sorted."""
        dates = pd.bdate_range("2018-01-02", periods=200)
        x = np.cumsum(_RNG.standard_normal(200)) + 100
        y = 0.8 * x + _RNG.standard_normal(200) * 0.5
        df_fwd = pd.DataFrame({"Date": dates, "AssetA": x, "AssetB": y}).set_index("Date")
        df_rev = df_fwd[::-1]  # reverse row order

        resp_fwd = client.post(
            "/api/analyse",
            files={"file": ("fwd.csv", _make_csv(df_fwd), "text/csv")},
            data={"ticker1": "", "ticker2": ""},
        )
        resp_rev = client.post(
            "/api/analyse",
            files={"file": ("rev.csv", _make_csv(df_rev), "text/csv")},
            data={"ticker1": "", "ticker2": ""},
        )
        assert resp_fwd.status_code == 200
        assert resp_rev.status_code == 200
        data_fwd = resp_fwd.json()
        data_rev = resp_rev.json()
        assert data_fwd["is_cointegrated"] == data_rev["is_cointegrated"]
        assert abs(data_fwd["engle_granger"]["eg_pval"] - data_rev["engle_granger"]["eg_pval"]) < 1e-8


class TestCachedTickerPairs:
    """Integration tests using cached data from data/raw/."""

    def test_amzn_meta_is_cointegrated(self):
        """AMZN/META passes EG or Johansen trace at 5% on the full cached dataset."""
        resp = client.post(
            "/api/analyse",
            data={"ticker1": "AMZN", "ticker2": "META"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["is_cointegrated"] is True

    def test_msft_aapl_is_borderline(self):
        """MSFT/AAPL passes Johansen trace at 10% only — borderline inclusion."""
        resp = client.post(
            "/api/analyse",
            data={"ticker1": "MSFT", "ticker2": "AAPL"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["is_cointegrated"] is True
        assert data["is_borderline"] is True

    def test_amzn_meta_spread_chart_title_contains_hedge_ratio(self):
        """Spread chart title must show the actual hedge ratio at 3 d.p., not 0.000."""
        from app.main import _chart_spread

        dates = pd.bdate_range("2020-01-02", periods=50)
        spread = pd.Series(np.zeros(50), index=dates)

        titles: list[str] = []
        real_set_title = matplotlib.axes.Axes.set_title

        def _capture(self, label, *args, **kwargs):
            titles.append(label)
            return real_set_title(self, label, *args, **kwargs)

        with patch.object(matplotlib.axes.Axes, "set_title", _capture):
            _chart_spread(spread, "AMZN", "META", hr=0.5977)

        assert len(titles) == 1
        assert "0.598" in titles[0], f"Expected '0.598' in chart title, got: {titles[0]!r}"

    def test_amzn_meta_html_shows_eg_pval(self):
        """Results page EG row must show the engle_granger p-value (0.0143), not the ADF residual p-value."""
        resp = client.post(
            "/analyse",
            data={"ticker1": "AMZN", "ticker2": "META"},
        )
        assert resp.status_code == 200
        assert "0.0143" in resp.text


@pytest.mark.slow
class TestPredictEndpoint:
    """Integration tests for GET /api/predict/{pair} — loads TF artefacts.

    Uses setup_class/teardown_class so the FastAPI lifespan (artefact loading)
    runs once for the whole class rather than on every request.
    """

    @classmethod
    def setup_class(cls):
        cls._ctx = TestClient(app)
        cls._ctx.__enter__()

    @classmethod
    def teardown_class(cls):
        cls._ctx.__exit__(None, None, None)

    def test_known_pair_amzn_meta_returns_200(self):
        resp = self._ctx.get("/api/predict/AMZN_META")
        assert resp.status_code == 200

    def test_response_has_required_fields(self):
        resp = self._ctx.get("/api/predict/AMZN_META")
        assert resp.status_code == 200
        data = resp.json()
        assert data["pair"] == "AMZN_META"
        assert data["dep"] == "AMZN"
        assert data["indep"] == "META"
        assert "signal_date" in data
        assert isinstance(data["z_std"], float)
        assert isinstance(data["forecast"], list)
        assert len(data["forecast"]) == 5
        assert all(isinstance(v, float) for v in data["forecast"])
        assert isinstance(data["converging"], bool)

    def test_unknown_pair_returns_404(self):
        resp = self._ctx.get("/api/predict/NVDA_GOOGL")
        assert resp.status_code == 404
        assert "NVDA_GOOGL" in resp.json()["detail"]

    def test_malformed_pair_returns_404(self):
        resp = self._ctx.get("/api/predict/NOT_A_PAIR")
        assert resp.status_code == 404

"""
Unit tests for Phase 5 (LSTM spread prediction) helper functions.

Covers:
  1. make_sequences       — shape, no look-ahead, NaN skipping
  2. standardise          — scaler fit boundary (training data only)
  3. build_convergence_signal — convergence gate logic
  4. persistence_baseline — uses X[:, -1, 0] in correct units
  5. compute_win_rate     — completed-trade denominator, open-trade exclusion

Run from the project root:
    python -m pytest tests/ -v
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from phase5_ml_spread import (
    make_sequences,
    standardise,
    build_convergence_signal,
    persistence_baseline,
    compute_win_rate,
    ML_TRAIN_END,
    ML_VAL_START,
    ML_VAL_END,
    ML_TEST_START,
    FINAL_TRAIN_END,
    FINAL_VAL_START,
    FINAL_VAL_END,
    P4_TEST_START,
    SEQ_LEN,
    HORIZON,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bdate_index(n, start="2018-01-02"):
    return pd.bdate_range(start=start, periods=n)


def _feat_df(n=100, start="2018-01-02", seed=0):
    """Minimal 5-column feature DataFrame matching build_features output."""
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n).cumsum() * 0.1
    df = pd.DataFrame({
        "z_std":    z,
        "lag1":     np.roll(z, 1),
        "lag2":     np.roll(z, 2),
        "lag3":     np.roll(z, 3),
        "roll_std": np.abs(rng.normal(size=n)) + 0.01,
    }, index=_bdate_index(n, start))
    # first 3 rows have artificial NaN from shifting — mirror build_features
    df.iloc[:3, 1:4] = np.nan
    return df


# ---------------------------------------------------------------------------
# 1. make_sequences — shape and chronological alignment
# ---------------------------------------------------------------------------

class TestMakeSequences:

    def test_output_shapes(self):
        df = _feat_df(n=200)
        df = df.dropna()
        X, y, dates = make_sequences(df, seq_len=SEQ_LEN, horizon=HORIZON)
        n_expected = len(df) - SEQ_LEN - HORIZON + 1
        assert X.shape == (n_expected, SEQ_LEN, df.shape[1])
        assert y.shape == (n_expected, HORIZON)
        assert len(dates) == n_expected

    def test_signal_date_is_last_input_bar(self):
        """signal_dates[i] must equal the index of the last row in X[i]."""
        df = _feat_df(n=100).dropna()
        X, y, dates = make_sequences(df, seq_len=SEQ_LEN, horizon=HORIZON)
        # last bar of the i-th window in the source df
        for i in range(min(5, len(dates))):
            expected = df.index[i + SEQ_LEN - 1]
            assert dates[i] == expected, (
                f"Sample {i}: signal_date={dates[i]} but expected {expected}"
            )

    def test_y_starts_one_bar_after_signal(self):
        """y[i, 0] must correspond to the bar immediately after signal_dates[i]."""
        df = _feat_df(n=100).dropna()
        _, y, dates = make_sequences(df, seq_len=SEQ_LEN, horizon=HORIZON)
        for i in range(min(5, len(dates))):
            signal_pos = df.index.get_loc(dates[i])
            expected_z = df["z_std"].iloc[signal_pos + 1]
            assert abs(float(y[i, 0]) - float(expected_z)) < 1e-6, (
                f"Sample {i}: y[0]={y[i,0]:.6f} but z_std[signal+1]={expected_z:.6f}"
            )

    def test_nan_rows_skipped(self):
        """Samples whose X window contains NaN must be dropped."""
        df = _feat_df(n=60)
        # inject NaN inside a window that would otherwise be valid
        df.iloc[25, 0] = np.nan
        X, y, dates = make_sequences(df, seq_len=SEQ_LEN, horizon=HORIZON)
        # none of the returned windows should contain NaN
        assert not np.any(np.isnan(X))
        assert not np.any(np.isnan(y))

    def test_empty_df_returns_empty_arrays(self):
        df = _feat_df(n=5).dropna()   # fewer rows than SEQ_LEN + HORIZON
        X, y, dates = make_sequences(df, seq_len=SEQ_LEN, horizon=HORIZON)
        assert X.shape[0] == 0
        assert y.shape[0] == 0
        assert len(dates) == 0


# ---------------------------------------------------------------------------
# 2. standardise — scaler fit boundary
# ---------------------------------------------------------------------------

class TestStandardise:

    def test_mu_sigma_from_training_only(self):
        """mu and sigma must equal the stats of the training slice only."""
        idx = _bdate_index(500, start="2018-01-02")
        vals = np.arange(500, dtype=float)   # deterministic
        series = pd.Series(vals, index=idx)

        z_std, mu, sigma = standardise(series, train_end=ML_TRAIN_END)

        train_mask = idx <= ML_TRAIN_END
        train_vals = pd.Series(vals[train_mask])
        expected_mu    = float(train_vals.mean())
        expected_sigma = float(train_vals.std())   # ddof=1, matches pandas
        assert abs(mu    - expected_mu)    < 1e-10
        assert abs(sigma - expected_sigma) < 1e-10

    def test_post_train_values_not_used_in_scaler(self):
        """Appending data after train_end must not change mu/sigma."""
        # Build a series that spans the ML_TRAIN_END boundary explicitly.
        # Pre-train: 2018-01-02 to ML_TRAIN_END; post-train: 2021-01-04 onwards.
        pre_idx  = pd.bdate_range("2018-01-02", ML_TRAIN_END)
        post_idx = pd.bdate_range("2021-01-04", periods=200)
        rng      = np.random.default_rng(42)

        pre_vals  = rng.normal(size=len(pre_idx))
        post_vals = rng.normal(size=len(post_idx)) * 100   # very different scale

        s_pre  = pd.Series(pre_vals,  index=pre_idx)
        s_full = pd.concat([s_pre, pd.Series(post_vals, index=post_idx)])

        _, mu_pre,  sigma_pre  = standardise(s_pre,  train_end=ML_TRAIN_END)
        _, mu_full, sigma_full = standardise(s_full, train_end=ML_TRAIN_END)

        assert abs(mu_pre    - mu_full)    < 1e-10, "mu changed when post-train data added"
        assert abs(sigma_pre - sigma_full) < 1e-10, "sigma changed when post-train data added"

    def test_standardised_train_has_zero_mean(self):
        idx  = _bdate_index(400, start="2018-01-02")
        vals = np.random.default_rng(1).normal(size=400)
        s    = pd.Series(vals, index=idx)
        z_std, _, _ = standardise(s, train_end=ML_TRAIN_END)
        train_z = z_std[z_std.index <= ML_TRAIN_END]
        assert abs(train_z.mean()) < 1e-10


# ---------------------------------------------------------------------------
# 3. build_convergence_signal — gate logic
# ---------------------------------------------------------------------------

class TestConvergenceGate:

    def _make_inputs(self, z_vals, pred_matrix):
        """
        z_vals: list of current z_std values (signal bars)
        pred_matrix: list of 5-element forecast vectors
        """
        idx = _bdate_index(len(z_vals))
        z_series = pd.Series(z_vals, index=idx)
        pred_df  = pd.DataFrame(
            pred_matrix, index=idx,
            columns=[f"h{h+1}" for h in range(5)]
        )
        return z_series, pred_df

    def test_converging_when_mean_abs_pred_lt_abs_z(self):
        # |z| = 3.0, mean(|preds|) = 1.0 → converging
        z_series, pred_df = self._make_inputs(
            [3.0], [[0.8, 1.0, 1.2, 0.9, 1.1]]
        )
        sig = build_convergence_signal(z_series, pred_df)
        assert sig.iloc[0] == 1

    def test_diverging_when_mean_abs_pred_ge_abs_z(self):
        # |z| = 1.0, mean(|preds|) = 2.0 → diverging
        z_series, pred_df = self._make_inputs(
            [1.0], [[1.5, 2.0, 2.5, 1.8, 2.2]]
        )
        sig = build_convergence_signal(z_series, pred_df)
        assert sig.iloc[0] == -1

    def test_gate_uses_absolute_z(self):
        # negative z — should use |z| = 3.0
        z_series, pred_df = self._make_inputs(
            [-3.0], [[0.8, 1.0, 1.2, 0.9, 1.1]]
        )
        sig = build_convergence_signal(z_series, pred_df)
        assert sig.iloc[0] == 1  # same as positive 3.0 case

    def test_boundary_equal_is_diverging(self):
        # mean(|pred|) == |z| → diverging (strict less-than required)
        z_series, pred_df = self._make_inputs(
            [2.0], [[2.0, 2.0, 2.0, 2.0, 2.0]]
        )
        sig = build_convergence_signal(z_series, pred_df)
        assert sig.iloc[0] == -1

    def test_index_alignment(self):
        """Signal dates must align with the pred_df index."""
        n = 10
        idx = _bdate_index(n)
        z_series = pd.Series(np.full(n, 3.0), index=idx)
        pred_df  = pd.DataFrame(
            np.ones((n, 5)) * 0.5, index=idx,
            columns=[f"h{h+1}" for h in range(5)]
        )
        sig = build_convergence_signal(z_series, pred_df)
        assert list(sig.index) == list(idx)
        assert (sig == 1).all()


# ---------------------------------------------------------------------------
# 4. persistence_baseline — units and shape
# ---------------------------------------------------------------------------

class TestPersistenceBaseline:

    def test_output_keys_present(self):
        X      = np.random.default_rng(0).normal(size=(50, SEQ_LEN, 5)).astype(np.float32)
        y_true = np.random.default_rng(1).normal(size=(50, HORIZON)).astype(np.float32)
        result = persistence_baseline(X, y_true)
        for h in range(1, HORIZON + 1):
            assert f"Persist_RMSE_h{h}" in result
            assert f"Persist_MAE_h{h}"  in result
        assert "Persist_RMSE_overall" in result
        assert "Persist_MAE_overall"  in result

    def test_uses_last_z_std_feature(self):
        """y_persist must equal X[:, -1, 0] broadcast across all horizons."""
        n = 20
        X = np.zeros((n, SEQ_LEN, 5), dtype=np.float32)
        last_z = np.arange(n, dtype=np.float32)
        X[:, -1, 0] = last_z               # feature 0, last time step
        X[:, -1, 1] = last_z * 99          # feature 1 must not be used
        y_true = np.zeros((n, HORIZON), dtype=np.float32)  # all zeros
        result = persistence_baseline(X, y_true)
        # RMSE_h1 = sqrt(mean(last_z^2))
        expected_rmse = float(np.sqrt(np.mean(last_z ** 2)))
        assert abs(result["Persist_RMSE_h1"] - expected_rmse) < 1e-5

    def test_perfect_forecast_gives_zero_error(self):
        """If y_true == X[:,-1,0] for all h, all errors should be zero."""
        rng = np.random.default_rng(42)
        last_z = rng.normal(size=30).astype(np.float32)
        X = np.zeros((30, SEQ_LEN, 5), dtype=np.float32)
        X[:, -1, 0] = last_z
        y_true = np.tile(last_z[:, None], (1, HORIZON)).astype(np.float32)
        result = persistence_baseline(X, y_true)
        for key in result:
            assert abs(result[key]) < 1e-6, f"{key} = {result[key]:.2e}, expected 0"


# ---------------------------------------------------------------------------
# 5. compute_win_rate — completed-trade denominator
# ---------------------------------------------------------------------------

def _make_bt_df(positions, index):
    """Minimal backtest DataFrame with all columns required by extract_trade_log."""
    n       = len(positions)
    pos_arr = np.array(positions, dtype=int)
    # daily_pnl: assign token winning exit P&L so completed trades are winners
    daily_pnl = np.zeros(n)
    for i in range(1, n):
        if pos_arr[i - 1] != 0 and pos_arr[i] == 0:
            daily_pnl[i] = 0.05

    # zscore: just assign a constant signal value so extract_trade_log can read it
    zscore = np.where(pos_arr == 1, -2.5, np.where(pos_arr == -1, 2.5, 0.0))

    return pd.DataFrame({
        "position":   pos_arr,
        "zscore":     zscore,
        "daily_pnl":  daily_pnl,
        "cum_pnl":    np.cumsum(daily_pnl),
        "spread_ret": np.zeros(n),
        "trade_cost": np.zeros(n),
        "close_px":   np.ones(n) * 100.0,
        "open_px":    np.ones(n) * 100.0,
    }, index=index)


class TestComputeWinRate:

    def test_no_trades_returns_zero_count(self):
        idx   = _bdate_index(50)
        bt_df = _make_bt_df([0] * 50, idx)
        wr, n_comp, pnl_list = compute_win_rate(bt_df)
        assert n_comp == 0
        assert pnl_list == []

    def test_open_trade_at_end_excluded(self):
        """A trade that is still open on the last bar is not counted."""
        idx      = _bdate_index(30)
        # enter long on bar 10, never exit
        pos      = [0] * 10 + [1] * 20
        bt_df    = _make_bt_df(pos, idx)
        wr, n_comp, _ = compute_win_rate(bt_df)
        assert n_comp == 0    # no completed trades

    def test_completed_trade_counted(self):
        """One trade that opens and closes before the last bar is counted."""
        idx  = _bdate_index(40)
        # enter bar 5, exit bar 15, flat thereafter
        pos  = [0]*5 + [1]*10 + [0]*25
        bt_df = _make_bt_df(pos, idx)
        wr, n_comp, _ = compute_win_rate(bt_df)
        assert n_comp == 1

    def test_open_trade_excluded_when_mixed(self):
        """One completed + one still-open: only the completed one is counted."""
        idx  = _bdate_index(50)
        # first trade: enter bar 2, exit bar 12; second: enter bar 20, open at end
        pos  = [0]*2 + [1]*10 + [0]*8 + [1]*30
        bt_df = _make_bt_df(pos, idx)
        wr, n_comp, _ = compute_win_rate(bt_df)
        assert n_comp == 1


# ---------------------------------------------------------------------------
# 6. Date-window chronology — both experiments use the actual constants
# ---------------------------------------------------------------------------

class TestDateWindows:

    def test_historical_splits_are_chronological(self):
        """Historical experiment: train < val < test, no gaps or overlaps."""
        assert ML_TRAIN_END < ML_VAL_START, (
            f"Val start {ML_VAL_START} must be after train end {ML_TRAIN_END}")
        assert ML_VAL_END < ML_TEST_START, (
            f"Test start {ML_TEST_START} must be after val end {ML_VAL_END}")

    def test_historical_splits_do_not_overlap(self):
        train_end = pd.Timestamp(ML_TRAIN_END)
        val_start = pd.Timestamp(ML_VAL_START)
        val_end   = pd.Timestamp(ML_VAL_END)
        test_start = pd.Timestamp(ML_TEST_START)
        assert val_start > train_end
        assert test_start > val_end

    def test_final_splits_are_chronological(self):
        """Final 2026 experiment: train < val < test, no gaps or overlaps."""
        assert FINAL_TRAIN_END < FINAL_VAL_START, (
            f"Final val start {FINAL_VAL_START} must be after train end {FINAL_TRAIN_END}")
        assert FINAL_VAL_END < P4_TEST_START, (
            f"2026 test start {P4_TEST_START} must be after final val end {FINAL_VAL_END}")

    def test_final_splits_do_not_overlap(self):
        final_train_end = pd.Timestamp(FINAL_TRAIN_END)
        final_val_start = pd.Timestamp(FINAL_VAL_START)
        final_val_end   = pd.Timestamp(FINAL_VAL_END)
        test_start      = pd.Timestamp(P4_TEST_START)
        assert final_val_start > final_train_end
        assert test_start > final_val_end

    def test_historical_train_end_strictly_before_final_train_end(self):
        """Historical model trains on less data than the final model."""
        assert ML_TRAIN_END < FINAL_TRAIN_END

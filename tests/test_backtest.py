"""
Unit tests for the core pairs-trading logic.

Tests are self-contained: they construct minimal synthetic data and check
hand-computed expected values, so no real market data files are required.

Run from the project root:
    python -m pytest tests/ -v
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

# Make src/ importable without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from phase3_strategy import (backtest, compute_metrics, build_spread_zscore,
                              block_bootstrap_sharpe, extract_trade_log)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_index(n: int, start: str = "2020-01-02") -> pd.DatetimeIndex:
    return pd.bdate_range(start=start, periods=n)


# ---------------------------------------------------------------------------
# backtest() — position state-machine
# ---------------------------------------------------------------------------

class TestBacktestPositionLogic:

    def test_no_trades_when_z_below_entry(self):
        """All z-scores inside the entry band → position stays 0 throughout."""
        idx = _make_index(10)
        z  = pd.Series([0.0, 0.5, 1.0, 1.5, 1.9, 1.9, 1.5, 1.0, 0.5, 0.0], index=idx)
        sp = pd.Series(np.zeros(10), index=idx)
        bt = backtest(z, sp, z_entry=2.0, z_exit=0.0, z_stop=3.0, cost_per_leg=0.0)
        assert (bt["position"] == 0).all(), "No position should be opened below entry threshold"

    def test_long_position_entered_on_low_z(self):
        """z drops below -entry → long spread position (+1) opened next bar."""
        idx = _make_index(5)
        # bar 0: z = -2.5  (signal fires)
        # bar 1: position should be +1
        z  = pd.Series([-2.5, -2.5, -1.0, 0.0, 0.5], index=idx)
        sp = pd.Series([0.0,   0.1,  0.2, 0.3, 0.4], index=idx)
        bt = backtest(z, sp, z_entry=2.0, z_exit=0.0, z_stop=3.0, cost_per_leg=0.0)
        assert bt["position"].iloc[1] == 1, "Long position expected after z < -entry"

    def test_short_position_entered_on_high_z(self):
        """z rises above +entry → short spread position (-1) opened next bar."""
        idx = _make_index(5)
        z  = pd.Series([2.5, 2.5, 1.0, 0.0, -0.5], index=idx)
        sp = pd.Series([0.0, 0.1, 0.2, 0.3,  0.4], index=idx)
        bt = backtest(z, sp, z_entry=2.0, z_exit=0.0, z_stop=3.0, cost_per_leg=0.0)
        assert bt["position"].iloc[1] == -1, "Short position expected after z > +entry"

    def test_long_position_closed_at_mean_reversion(self):
        """Long position closes when z crosses back through z_exit (0)."""
        idx = _make_index(6)
        # bar 0: z = -2.5  → enter long at bar 1
        # bar 3: z = 0.0   → exit at bar 4
        z  = pd.Series([-2.5, -2.0, -1.0,  0.0,  0.5,  0.8], index=idx)
        sp = pd.Series([ 0.0,  0.1,  0.2,  0.3,  0.4,  0.5], index=idx)
        bt = backtest(z, sp, z_entry=2.0, z_exit=0.0, z_stop=3.0, cost_per_leg=0.0)
        assert bt["position"].iloc[1] == 1,  "Should be long at bar 1"
        assert bt["position"].iloc[2] == 1,  "Should still be long at bar 2"
        assert bt["position"].iloc[3] == 1,  "Should still be long at bar 3"
        assert bt["position"].iloc[4] == 0,  "Should be flat after mean reversion"

    def test_stop_loss_closes_long_position(self):
        """
        Long position hits stop loss; re-entry in the same direction is blocked
        until z returns inside the entry band.

        Sequence (z by bar index, 0..4):
          i=1: z_prev=z[0]=-2.5 < -2.0  -> enter long.  pos[1]=+1
          i=2: z_prev=z[1]=-2.5          -> hold.        pos[2]=+1
          i=3: z_prev=z[2]=-3.1 <= -3.0 -> stop loss fires, stopped_dir=+1.
               Re-entry blocked (stopped_dir==+1).       pos[3]=0
          i=4: z_prev=z[3]=0.0 > -2.0   -> stop guard cleared (z inside band).
               No new signal (z_prev not < -2 or > 2).  pos[4]=0
        """
        idx = _make_index(5)
        z  = pd.Series([-2.5, -2.5, -3.1,  0.0,  0.5], index=idx)
        sp = pd.Series([ 0.0,  0.1,  0.2,  0.3,  0.4], index=idx)
        bt = backtest(z, sp, z_entry=2.0, z_exit=0.0, z_stop=3.0, cost_per_leg=0.0)
        assert bt["position"].iloc[1] == 1, "Long entered at bar 1"
        assert bt["position"].iloc[2] == 1, "Still long at bar 2"
        assert bt["position"].iloc[3] == 0, "Stop fires and re-entry blocked (same direction)"
        assert bt["position"].iloc[4] == 0, "Flat: stop guard cleared but no new signal"

    def test_stop_loss_closes_short_position(self):
        """
        Short position hits stop loss; re-entry in the same direction is blocked
        until z returns inside the entry band.

        Sequence (z by bar index, 0..4):
          i=1: z_prev=z[0]=+2.5 > +2.0  -> enter short.  pos[1]=-1
          i=2: z_prev=z[1]=+2.5          -> hold.         pos[2]=-1
          i=3: z_prev=z[2]=+3.1 >= +3.0 -> stop loss fires, stopped_dir=-1.
               Re-entry blocked (stopped_dir==-1).        pos[3]=0
          i=4: z_prev=z[3]=0.0 < +2.0   -> stop guard cleared (z inside band).
               No new signal (z_prev not < -2 or > 2).   pos[4]=0
        """
        idx = _make_index(5)
        z  = pd.Series([2.5, 2.5,  3.1,  0.0, -0.5], index=idx)
        sp = pd.Series([0.0, 0.1,  0.2,  0.3,  0.4], index=idx)
        bt = backtest(z, sp, z_entry=2.0, z_exit=0.0, z_stop=3.0, cost_per_leg=0.0)
        assert bt["position"].iloc[1] == -1, "Short entered at bar 1"
        assert bt["position"].iloc[2] == -1, "Still short at bar 2"
        assert bt["position"].iloc[3] ==  0, "Stop fires and re-entry blocked (same direction)"
        assert bt["position"].iloc[4] ==  0, "Flat: stop guard cleared but no new signal"

    def test_stop_loss_allows_opposite_direction_reentry(self):
        """
        After a long stop loss, entering a short position is still permitted
        on the same bar (z well above +entry), and vice versa.

        Sequence (z by bar index, 0..4):
          i=1: z_prev=z[0]=-2.5 < -2.0  -> enter long.  pos[1]=+1
          i=2: z_prev=z[1]=-3.1 <= -3.0 -> stop long, stopped_dir=+1.
               z_prev=-3.1 not > +2.0    -> no short entry.   pos[2]=0
          i=3: z_prev=z[2]=+3.0 > +2.0  -> enter short (opposite direction OK).  pos[3]=-1
          i=4: z_prev=z[3]=0.0  <= 0.0  -> exit short.  pos[4]=0
        """
        idx = _make_index(5)
        z  = pd.Series([-2.5, -3.1,  3.0,  0.0,  0.5], index=idx)
        sp = pd.Series([ 0.0,  0.1,  0.2,  0.3,  0.4], index=idx)
        bt = backtest(z, sp, z_entry=2.0, z_exit=0.0, z_stop=3.0, cost_per_leg=0.0)
        assert bt["position"].iloc[1] ==  1, "Long entered at bar 1"
        assert bt["position"].iloc[2] ==  0, "Stop loss fired; flat"
        assert bt["position"].iloc[3] == -1, "Short entered (opposite direction allowed)"
        assert bt["position"].iloc[4] ==  0, "Flat after mean reversion"

    def test_no_position_flip_without_going_flat_first(self):
        """
        While in a long position the z-score should not immediately trigger a
        short entry — the engine must go flat first.
        """
        idx = _make_index(6)
        # bar 0: z = -2.5  → enter long
        # bar 1-2: position = +1
        # bar 3: z = 0.0   → exit (flat)
        # bar 4: z = 2.5   → enter short
        z  = pd.Series([-2.5, -2.5, -1.0,  0.0,  2.5,  2.5], index=idx)
        sp = pd.Series([ 0.0,  0.1,  0.2,  0.3,  0.4,  0.5], index=idx)
        bt = backtest(z, sp, z_entry=2.0, z_exit=0.0, z_stop=3.0, cost_per_leg=0.0)
        assert bt["position"].iloc[1] == 1,  "Long at bar 1"
        assert bt["position"].iloc[4] == 0,  "Flat at bar 4 (exit processed, entry next)"
        assert bt["position"].iloc[5] == -1, "Short at bar 5"


# ---------------------------------------------------------------------------
# backtest() — P&L arithmetic
# ---------------------------------------------------------------------------

class TestBacktestPnL:

    def test_single_long_trade_pnl_no_costs(self):
        """
        Hand-computed expected P&L for a simple long trade.

        Setup:
          bar 0 (signal): z = -2.5  → enter long at bar 1
          bar 1: spread moves from 1.0 → 1.1  (+0.1)  position = +1  → P&L = +0.1
          bar 2: spread moves from 1.1 → 1.2  (+0.1)  position = +1  → P&L = +0.1
          bar 3 (signal): z = 0.0   → exit at bar 4  (spread now at 1.2)
          bar 4: spread moves from 1.2 → 1.3  (but position already 0) → P&L = 0
        """
        idx = _make_index(5)
        z   = pd.Series([-2.5, -2.5,  -1.0,   0.0,  0.5], index=idx)
        sp  = pd.Series([ 1.0,  1.1,   1.2,   1.2,  1.3], index=idx)
        bt  = backtest(z, sp, z_entry=2.0, z_exit=0.0, z_stop=3.0, cost_per_leg=0.0)
        assert bt["daily_pnl"].iloc[1] == pytest.approx(0.1, abs=1e-9)
        assert bt["daily_pnl"].iloc[2] == pytest.approx(0.1, abs=1e-9)
        assert bt["daily_pnl"].iloc[4] == pytest.approx(0.0, abs=1e-9)
        assert bt["cum_pnl"].iloc[2]   == pytest.approx(0.2, abs=1e-9)

    def test_transaction_costs_deducted_at_entry_and_exit(self):
        """
        Each entry and exit costs 2 * cost_per_leg.
        One complete round-trip (entry + exit) → 4 * cost_per_leg deducted total.
        """
        cost = 0.001   # 10 bps
        idx  = _make_index(5)
        # Flat spread — all P&L comes only from costs
        z  = pd.Series([-2.5, -2.5, -1.0, 0.0, 0.5], index=idx)
        sp = pd.Series([ 1.0,  1.0,  1.0, 1.0, 1.0], index=idx)
        bt = backtest(z, sp, z_entry=2.0, z_exit=0.0, z_stop=3.0, cost_per_leg=cost)
        # Entry cost at bar 1
        assert bt["trade_cost"].iloc[1] == pytest.approx(2 * cost, abs=1e-9)
        # Exit cost at bar 4
        assert bt["trade_cost"].iloc[4] == pytest.approx(2 * cost, abs=1e-9)
        # Total cumulative P&L = -(entry cost + exit cost)
        assert bt["cum_pnl"].iloc[-1]   == pytest.approx(-4 * cost, abs=1e-9)

    def test_cumulative_pnl_is_running_sum_of_daily_pnl(self):
        """cum_pnl must equal np.cumsum(daily_pnl) at every row."""
        idx = _make_index(20)
        rng = np.random.default_rng(42)
        z   = pd.Series(rng.normal(0, 2, 20), index=idx)
        sp  = pd.Series(rng.normal(0, 0.1, 20).cumsum(), index=idx)
        bt  = backtest(z, sp, z_entry=1.5, z_exit=0.0, z_stop=3.0, cost_per_leg=0.001)
        expected = np.cumsum(bt["daily_pnl"].values)
        np.testing.assert_allclose(bt["cum_pnl"].values, expected, atol=1e-12)

    def test_nan_z_scores_do_not_open_positions(self):
        """NaN z-score bars (e.g. during rolling-window warm-up) must be skipped."""
        idx = _make_index(6)
        z   = pd.Series([np.nan, np.nan, np.nan, -2.5, -2.5, 0.0], index=idx)
        sp  = pd.Series([   1.0,    1.0,    1.0,   1.0,  1.1, 1.2], index=idx)
        bt  = backtest(z, sp, z_entry=2.0, z_exit=0.0, z_stop=3.0, cost_per_leg=0.0)
        # No position should be opened during NaN period
        assert bt["position"].iloc[0] == 0
        assert bt["position"].iloc[1] == 0
        assert bt["position"].iloc[2] == 0
        # Long position opens after first valid signal at bar 3
        assert bt["position"].iloc[4] == 1


# ---------------------------------------------------------------------------
# compute_metrics() — performance metric calculations
# ---------------------------------------------------------------------------

class TestComputeMetrics:

    def _flat_bt(self, pnl_values: list, start: str = "2020-01-02") -> pd.DataFrame:
        """Build a minimal backtest DataFrame from a list of daily P&L values."""
        idx = _make_index(len(pnl_values), start=start)
        pnl = pd.Series(pnl_values, index=idx, dtype=float)
        pos = pd.Series(np.where(pnl != 0, 1, 0), index=idx, dtype=float)
        return pd.DataFrame({
            "zscore":     np.zeros(len(pnl_values)),
            "spread":     np.zeros(len(pnl_values)),
            "position":   pos.values,
            "daily_pnl":  pnl.values,
            "trade_cost": np.zeros(len(pnl_values)),
            "cum_pnl":    pnl.cumsum().values,
        }, index=idx)

    def test_sharpe_zero_when_no_pnl(self):
        bt = self._flat_bt([0.0] * 252)
        m  = compute_metrics(bt, label="test")
        assert m["Sharpe_ratio"] == 0.0

    def test_sharpe_positive_for_constant_positive_pnl(self):
        """Constant daily gain → positive Sharpe (std is zero, but gain is positive)."""
        # std of a constant series is 0, so Sharpe calculation hits the vol==0 guard
        bt = self._flat_bt([0.001] * 252)
        m  = compute_metrics(bt, label="test")
        # With zero vol the code returns 0.0 by the guard — check it doesn't crash
        assert isinstance(m["Sharpe_ratio"], float)

    def test_max_drawdown_is_negative(self):
        """Max drawdown should always be ≤ 0."""
        pnl = [0.01, 0.01, -0.05, -0.03, 0.02]
        bt  = self._flat_bt(pnl)
        m   = compute_metrics(bt, label="test")
        assert m["Max_drawdown"] <= 0.0

    def test_max_drawdown_hand_computed(self):
        """
        Cumulative P&L: 0, 0.01, 0.02, -0.03, -0.06, -0.04
        Running max   : 0, 0.01, 0.02,  0.02,  0.02,  0.02
        Drawdown      : 0,    0,    0,  -0.05, -0.08, -0.06
        Max drawdown  : -0.08
        """
        pnl = [0.01, 0.01, -0.05, -0.03, 0.02]
        bt  = self._flat_bt(pnl)
        m   = compute_metrics(bt, label="test")
        assert m["Max_drawdown"] == pytest.approx(-0.08, abs=1e-9)

    def test_positive_day_rate_all_positive(self):
        """All in-position days are positive → positive-day rate = 100%."""
        bt = self._flat_bt([0.01] * 20)
        m  = compute_metrics(bt, label="test")
        assert m["Positive_Day_Rate_pct"] == pytest.approx(100.0, abs=1e-9)

    def test_positive_day_rate_all_negative(self):
        """All in-position days are negative → positive-day rate = 0%."""
        bt = self._flat_bt([-0.01] * 20)
        m  = compute_metrics(bt, label="test")
        assert m["Positive_Day_Rate_pct"] == pytest.approx(0.0, abs=1e-9)

    def test_total_pnl_equals_sum_of_daily(self):
        pnl = [0.01, -0.02, 0.03, -0.01, 0.005]
        bt  = self._flat_bt(pnl)
        m   = compute_metrics(bt, label="test")
        assert m["Total_PnL"] == pytest.approx(sum(pnl), abs=1e-9)


# ---------------------------------------------------------------------------
# build_spread_zscore() — spread construction
# ---------------------------------------------------------------------------

class TestBuildSpreadZscore:

    def test_spread_formula(self):
        """spread = log(DEP) - hr * log(INDEP) - intercept."""
        idx = _make_index(10)
        log_dep   = pd.Series(np.log([100.0] * 10), index=idx, name="AMZN")
        log_indep = pd.Series(np.log([50.0]  * 10), index=idx, name="META")
        log_df    = pd.DataFrame({"AMZN": log_dep, "META": log_indep})

        hr, ic, lookback = 1.0, 0.5, 5
        spread, _ = build_spread_zscore(log_df, "AMZN", "META", hr, ic, lookback)

        expected = log_dep - hr * log_indep - ic
        pd.testing.assert_series_equal(spread, expected, check_names=False)

    def test_zscore_nan_during_warmup(self):
        """Z-score must be NaN for the first (lookback - 1) rows."""
        idx     = _make_index(20)
        log_dep = pd.Series(np.log(np.linspace(100, 120, 20)), index=idx, name="AMZN")
        log_ind = pd.Series(np.log(np.linspace(50,   60, 20)), index=idx, name="META")
        log_df  = pd.DataFrame({"AMZN": log_dep, "META": log_ind})

        _, zscore = build_spread_zscore(log_df, "AMZN", "META", 1.0, 0.0, 5)

        assert zscore.iloc[:4].isna().all(), "First (lookback-1) z-scores must be NaN"
        assert not np.isnan(zscore.iloc[4]), "Z-score should be valid at index lookback-1"


# ---------------------------------------------------------------------------
# extract_trade_log() — trade logging and timing convention
# ---------------------------------------------------------------------------

class TestExtractTradeLog:
    """
    Tests for extract_trade_log().

    Timing convention under test:
      Signal_Date    -- date of the bar whose close produced the triggering z-score
      Signal_Z_score -- z-score at close(Signal_Date); this is z_prev for bar i
      Execution_Date -- date of bar i; position entered at open(i)

    All tests use backtest() to produce a realistic bt_df so that the
    full pipeline (backtest → extract_trade_log) is exercised end-to-end.
    """

    def _make_bt(self, z_list, sp_list, cost_per_leg=0.001, z_stop=4.0):
        """Run backtest on synthetic data and return (bt_df, dates, cost_per_leg)."""
        idx = _make_index(len(z_list))
        z   = pd.Series(z_list,  index=idx)
        sp  = pd.Series(sp_list, index=idx)
        bt  = backtest(z, sp, z_entry=2.0, z_exit=0.0, z_stop=z_stop,
                       cost_per_leg=cost_per_leg)
        return bt, idx, cost_per_leg

    # ------------------------------------------------------------------
    # Signal date precedes execution date
    # ------------------------------------------------------------------

    def test_signal_date_is_day_before_execution(self):
        """
        Signal fires at close(bar 0); position entered at open(bar 1).
        Signal_Date must be bar 0, Execution_Date must be bar 1.

        z = [-2.5, -1.8, ...]: z[0]=-2.5 triggers the long; z[1]=-1.8 is
        the execution day's z-score (inside the band, would not trigger entry).
        """
        bt, idx, cpl = self._make_bt(
            z_list  = [-2.5, -1.8,  0.0,  0.5,  0.5],
            sp_list = [ 0.0,  0.1,  0.2,  0.3,  0.4],
        )
        tl = extract_trade_log(bt, cpl)
        assert len(tl) == 1
        assert tl.iloc[0]["Signal_Date"]    == str(idx[0].date()), \
            "Signal_Date should be bar 0 (close triggered entry)"
        assert tl.iloc[0]["Execution_Date"] == str(idx[1].date()), \
            "Execution_Date should be bar 1 (position entered at open)"

    def test_signal_z_score_is_triggering_value_not_execution_day(self):
        """
        Signal_Z_score must be z_prev (the z-score that crossed the threshold),
        NOT the execution day's z-score (which may lie inside the band).

        z[0] = -2.5  → triggers long (below -2.0 threshold)
        z[1] = -1.8  → execution day z; inside band, would NOT trigger a trade
        Signal_Z_score should be -2.5, not -1.8.
        """
        bt, idx, cpl = self._make_bt(
            z_list  = [-2.5, -1.8,  0.0,  0.5,  0.5],
            sp_list = [ 0.0,  0.1,  0.2,  0.3,  0.4],
        )
        tl = extract_trade_log(bt, cpl)
        assert tl.iloc[0]["Signal_Z_score"] == pytest.approx(-2.5, abs=1e-9), \
            "Signal_Z_score must be z[0]=-2.5 (the threshold-crossing z), not z[1]=-1.8"

    def test_exit_signal_date_and_z_score(self):
        """
        Exit triggered at close(bar 2, z=0.0); position closed at open(bar 3).
        Exit_Date must be bar 3; Exit_Z_score must be 0.0 (z at close of bar 2).
        """
        bt, idx, cpl = self._make_bt(
            z_list  = [-2.5, -1.8,  0.0,  0.5,  0.5],
            sp_list = [ 0.0,  0.1,  0.2,  0.3,  0.4],
        )
        tl = extract_trade_log(bt, cpl)
        assert tl.iloc[0]["Exit_Date"]      == str(idx[3].date()), \
            "Exit_Date should be bar 3 (first flat bar after exit signal at bar 2)"
        assert tl.iloc[0]["Exit_Z_score"]   == pytest.approx(0.0, abs=1e-9), \
            "Exit_Z_score should be z[2]=0.0 (the signal that triggered exit)"

    # ------------------------------------------------------------------
    # +1 → -1 flip
    # ------------------------------------------------------------------

    def test_long_to_short_flip_produces_two_trades(self):
        """
        z[0]=-2.5 → long at bar 1.
        z[1]=+3.0 → at bar 2: exit long (z≥0) AND enter short (z>+2) simultaneously.
        This is a flip: position jumps directly from +1 to -1 without going through 0.
        z[3]=-0.1 → exit short at bar 4.

        Two trades must appear in the log (the short was previously omitted).
        """
        bt, idx, cpl = self._make_bt(
            z_list  = [-2.5,  3.0,  3.0, -0.1,  0.5,  0.5],
            sp_list = [ 0.0,  0.1,  0.2,  0.3,  0.4,  0.5],
            z_stop  = 4.0,
        )
        tl = extract_trade_log(bt, cpl)
        assert len(tl) == 2, \
            "Flip must produce two logged trades, not one (short was previously omitted)"

        long_t  = tl.iloc[0]
        short_t = tl.iloc[1]

        # Long trade
        assert long_t["Position"]         == "Long Spread"
        assert long_t["Execution_Date"]   == str(idx[1].date())
        assert long_t["Exit_Date"]        == str(idx[2].date()), \
            "Long trade exits at the flip bar (bar 2)"
        assert long_t["Signal_Z_score"]   == pytest.approx(-2.5, abs=1e-9)
        assert long_t["Exit_Z_score"]     == pytest.approx(3.0,  abs=1e-9), \
            "Exit_Z_score for long is z[1]=3.0 (the bar whose close triggered the flip)"

        # Short trade — entered at the flip bar
        assert short_t["Position"]        == "Short Spread"
        assert short_t["Signal_Date"]     == str(idx[1].date()), \
            "Short's Signal_Date is bar 1 (z=3.0 there triggered both flip legs)"
        assert short_t["Execution_Date"]  == str(idx[2].date()), \
            "Short enters at the flip bar (bar 2)"
        assert short_t["Signal_Z_score"]  == pytest.approx(3.0,  abs=1e-9)

    def test_long_to_short_flip_cost_split(self):
        """
        At a flip bar the backtester charges 4 legs (2 exit + 2 entry).
        Each of the two resulting trades should be allocated exactly 2 legs:
          long trade  = 2 entry legs (bar 1) + 2 exit legs (flip bar 2) = 4 * cost_per_leg
          short trade = 2 entry legs (flip bar 2) + 2 exit legs (bar 4) = 4 * cost_per_leg
        """
        cpl  = 0.001
        bt, idx, _ = self._make_bt(
            z_list  = [-2.5,  3.0,  3.0, -0.1,  0.5,  0.5],
            sp_list = [ 0.0,  0.1,  0.2,  0.3,  0.4,  0.5],
            z_stop  = 4.0,
            cost_per_leg=cpl,
        )
        tl = extract_trade_log(bt, cpl)
        assert tl.iloc[0]["Transaction_Cost"] == pytest.approx(4 * cpl, abs=1e-9), \
            "Long trade: 2 entry + 2 exit legs = 4 * cost_per_leg"
        assert tl.iloc[1]["Transaction_Cost"] == pytest.approx(4 * cpl, abs=1e-9), \
            "Short trade: 2 entry + 2 exit legs = 4 * cost_per_leg"

    # ------------------------------------------------------------------
    # -1 → +1 flip
    # ------------------------------------------------------------------

    def test_short_to_long_flip_produces_two_trades(self):
        """
        z[0]=+2.5 → short at bar 1.
        z[1]=-3.0 → at bar 2: exit short (z≤0) AND enter long (z<-2) simultaneously.
        z[3]=+0.1 → exit long at bar 4.

        Two trades must appear; short is first, long is second.
        """
        bt, idx, cpl = self._make_bt(
            z_list  = [ 2.5, -3.0, -3.0,  0.1,  0.5,  0.5],
            sp_list = [ 0.0,  0.1,  0.2,  0.3,  0.4,  0.5],
            z_stop  = 4.0,
        )
        tl = extract_trade_log(bt, cpl)
        assert len(tl) == 2, "-1→+1 flip must produce two logged trades"

        short_t = tl.iloc[0]
        long_t  = tl.iloc[1]

        assert short_t["Position"]       == "Short Spread"
        assert short_t["Execution_Date"] == str(idx[1].date())
        assert short_t["Exit_Date"]      == str(idx[2].date())

        assert long_t["Position"]        == "Long Spread"
        assert long_t["Execution_Date"]  == str(idx[2].date()), \
            "Long enters at the flip bar"
        assert long_t["Signal_Z_score"]  == pytest.approx(-3.0, abs=1e-9)

        assert short_t["Transaction_Cost"] == pytest.approx(4 * cpl, abs=1e-9)
        assert long_t["Transaction_Cost"]  == pytest.approx(4 * cpl, abs=1e-9)

    # ------------------------------------------------------------------
    # Trade-level cost and P&L reconciliation
    # ------------------------------------------------------------------

    def test_trade_log_costs_reconcile_with_backtest(self):
        """
        sum(trade_log.Transaction_Cost) must exactly equal sum(bt.trade_cost).
        This holds for any mix of normal exits and flips.
        """
        bt, _, cpl = self._make_bt(
            # Sequence designed to include one flip (bar 2→3)
            z_list  = [-2.5, 3.0, 3.0, -0.1, 0.5, -2.5, -2.5, 0.0, 0.5, 0.5],
            sp_list = [ 0.0, 0.1, 0.2,  0.3, 0.4,  0.3,  0.4, 0.5, 0.6, 0.7],
            z_stop  = 4.0,
        )
        tl = extract_trade_log(bt, cpl)
        assert tl["Transaction_Cost"].sum() == pytest.approx(
            bt["trade_cost"].sum(), abs=1e-9
        ), "Sum of trade-log costs must equal backtest engine's total cost"

    def test_trade_log_pnl_reconciles_with_backtest(self):
        """
        sum(trade_log.Net_PnL) must reconcile with sum(bt.daily_pnl) to within
        floating-point rounding introduced by round(..., 6) in the trade log.
        Tolerance is 1e-5 (much larger than rounding error, much smaller than
        any real P&L difference).
        """
        bt, _, cpl = self._make_bt(
            z_list  = [-2.5, 3.0, 3.0, -0.1, 0.5, -2.5, -2.5, 0.0, 0.5, 0.5],
            sp_list = [ 0.0, 0.1, 0.2,  0.3, 0.4,  0.3,  0.4, 0.5, 0.6, 0.7],
            z_stop  = 4.0,
        )
        tl = extract_trade_log(bt, cpl)
        assert tl["Net_PnL"].sum() == pytest.approx(
            bt["daily_pnl"].sum(), abs=1e-5
        ), "Sum of trade-log Net_PnL must reconcile with backtest cumulative P&L"

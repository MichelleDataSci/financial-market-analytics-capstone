"""
Phase 3b -- Pairs Trading Strategy
Reads selected_pairs.csv (from Phase 3a) and runs the full strategy for
every selected pair: spread construction, z-score signals, backtesting,
walk-forward validation, sensitivity analysis, and benchmark comparison.

Falls back to AMZN/META if selected_pairs.csv has not yet been generated.

Execution convention
--------------------
Signal is generated from close(t).  Trade executes at open(t+1).

P&L is computed using open prices for the execution bar:
  - Entry bar i  : pos * (spread_close[i] - spread_open[i])
  - Exit  bar i  : prev_pos * (spread_open[i] - spread_close[i-1])
  - Hold  bar i  : pos * (spread_close[i] - spread_close[i-1])
  - Flip  bar i  : both components at open

Stop-loss re-entry guard
------------------------
After a stop loss, re-entry in the same direction is blocked until the
z-score returns inside the entry band (|z| < z_entry).  Re-entry in the
opposite direction remains permitted.
"""

import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import statsmodels.api as sm
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import DATA_RAW, CHARTS_DIR, REPORTS_DIR

# ---------------------------------------------------------------------------
# Shared strategy parameters
# ---------------------------------------------------------------------------
Z_ENTRY      = 2.0      # open position when |z| > Z_ENTRY
Z_EXIT       = 0.0      # close position when z crosses Z_EXIT
Z_STOP       = 3.0      # stop loss when |z| exceeds Z_STOP in the loss direction
LOOKBACK     = 30       # rolling window (trading days) for z-score
COST_PER_LEG = 0.001    # 10 bps transaction cost per leg
TRAIN_END    = "2021-12-31"
TEST_START   = "2022-01-01"

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def load_prices(dep, indep):
    """
    Load Close and Open prices for two tickers from raw CSVs.
    Returns (close_df, open_df, log_close_df, log_open_df), all indexed on
    the same trading dates.
    """
    close_px = {}
    open_px  = {}
    for ticker in [dep, indep]:
        path = DATA_RAW / f"{ticker}_raw.csv"
        df   = pd.read_csv(path, index_col="Date", parse_dates=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        close_px[ticker] = df["Close"]
        open_px[ticker]  = df["Open"]
    close_df = pd.DataFrame(close_px).dropna()
    open_df  = pd.DataFrame(open_px).reindex(close_df.index).dropna()
    common   = close_df.index.intersection(open_df.index)
    close_df = close_df.loc[common]
    open_df  = open_df.loc[common]
    return close_df, open_df, np.log(close_df), np.log(open_df)



def estimate_hr(y_log, x_log):
    """OLS with constant. Returns (hedge_ratio, intercept, R-squared)."""
    X   = sm.add_constant(x_log)
    mod = sm.OLS(y_log, X).fit()
    return float(mod.params.iloc[1]), float(mod.params.iloc[0]), float(mod.rsquared)


def build_spread_zscore(log_prices, dep, indep, hedge_ratio, intercept, lookback):
    """Compute close-price spread and rolling z-score given OLS parameters."""
    spread    = log_prices[dep] - hedge_ratio * log_prices[indep] - intercept
    roll_mean = spread.rolling(lookback).mean()
    roll_std  = spread.rolling(lookback).std()
    zscore    = (spread - roll_mean) / roll_std
    return spread, zscore


def build_open_spread(log_open, dep, indep, hedge_ratio, intercept):
    """
    Compute the spread using open prices.
    Used as the execution price for entries and exits (signal at close,
    execute at next-day open).
    """
    return log_open[dep] - hedge_ratio * log_open[indep] - intercept


def backtest(zscore_series, spread_series, z_entry, z_exit, z_stop, cost_per_leg,
             spread_open_series=None):
    """
    Simulate pairs trading on z-score signal.

    Position rules:
      +1 (long spread)  : entered when z < -z_entry  (buy DEP, sell INDEP)
      -1 (short spread) : entered when z > +z_entry  (sell DEP, buy INDEP)
       0 (flat)

    Exit rules:
      Long  (pos=+1): close when z >= z_exit  OR  z <= -z_stop (stop loss)
      Short (pos=-1): close when z <= z_exit  OR  z >= +z_stop (stop loss)

    Stop-loss re-entry guard:
      After a stop loss, re-entry in the same direction is blocked until z
      returns inside the entry band (|z| < z_entry).  This prevents the
      strategy from immediately re-entering the same adverse trade on the
      stop bar.  Re-entry in the opposite direction is still permitted.

    All entry/exit decisions use z_prev (signal from previous day's close),
    executed at the next bar.

    P&L calculation:
      If spread_open_series is provided (real market data):
        - Entry bar i  : pos * (spread_close[i] - spread_open[i])
        - Exit  bar i  : prev_pos * (spread_open[i] - spread_close[i-1])
        - Hold  bar i  : pos * (spread_close[i] - spread_close[i-1])
        - Flip  bar i  : exit-at-open + entry-at-open
      If spread_open_series is None (unit tests with synthetic data):
        - All bars: pos * (spread_close[i] - spread_close[i-1])
    """
    z_arr   = zscore_series.values
    sp_cl   = spread_series.values
    sp_op   = spread_open_series.values if spread_open_series is not None else None
    use_open = sp_op is not None
    n       = len(z_arr)

    position    = np.zeros(n)
    daily_pnl   = np.zeros(n)
    trade_cost  = np.zeros(n)
    pos         = 0
    stopped_dir = 0   # direction most recently stopped (+1 or -1); 0 = none

    for i in range(1, n):
        z_prev = z_arr[i - 1]
        if np.isnan(z_prev) or np.isnan(sp_cl[i]) or np.isnan(sp_cl[i - 1]):
            continue
        if use_open and np.isnan(sp_op[i]):
            continue

        prev_pos = pos

        # Reset stopped direction when z returns inside the entry band.
        # Must happen before exit/entry checks so a return inside the band
        # on the same bar as a new signal allows re-entry immediately.
        if stopped_dir == 1 and z_prev > -z_entry:
            stopped_dir = 0
        elif stopped_dir == -1 and z_prev < z_entry:
            stopped_dir = 0

        # --- Exit check (signal = z_prev) ---
        if pos == 1:
            if z_prev >= z_exit or z_prev <= -z_stop:
                trade_cost[i] += 2 * cost_per_leg
                if z_prev <= -z_stop:       # stop loss
                    stopped_dir = 1
                pos = 0
        elif pos == -1:
            if z_prev <= z_exit or z_prev >= z_stop:
                trade_cost[i] += 2 * cost_per_leg
                if z_prev >= z_stop:        # stop loss
                    stopped_dir = -1
                pos = 0

        # --- Entry check (blocked if stopped in same direction) ---
        if pos == 0:
            if z_prev < -z_entry and stopped_dir != 1:
                trade_cost[i] += 2 * cost_per_leg
                pos = 1
            elif z_prev > z_entry and stopped_dir != -1:
                trade_cost[i] += 2 * cost_per_leg
                pos = -1

        # --- P&L ---
        if use_open:
            if prev_pos == 0 and pos != 0:
                # Pure entry: earn spread change from open to close
                pnl = pos * (sp_cl[i] - sp_op[i])
            elif prev_pos != 0 and pos == 0:
                # Pure exit: earn spread change from prev close to open
                pnl = prev_pos * (sp_op[i] - sp_cl[i - 1])
            elif prev_pos != 0 and pos == prev_pos:
                # Hold unchanged: earn full close-to-close
                pnl = pos * (sp_cl[i] - sp_cl[i - 1])
            elif prev_pos != 0 and pos != 0 and pos != prev_pos:
                # Flip (exit one side, enter other at same open):
                # earn overnight move at old direction, then intraday at new
                pnl = (prev_pos * (sp_op[i] - sp_cl[i - 1])
                       + pos    * (sp_cl[i]  - sp_op[i]))
            else:
                pnl = 0.0
        else:
            # Close-to-close P&L (used by unit tests with synthetic data)
            pnl = pos * (sp_cl[i] - sp_cl[i - 1])

        position[i]  = pos
        daily_pnl[i] = pnl - trade_cost[i]

    return pd.DataFrame({
        "zscore":     z_arr,
        "spread":     sp_cl,
        "position":   position,
        "daily_pnl":  daily_pnl,
        "trade_cost": trade_cost,
        "cum_pnl":    np.cumsum(daily_pnl),
    }, index=zscore_series.index)


def compute_metrics(bt_df, label=""):
    """Compute performance metrics from a backtest DataFrame."""
    pnl = bt_df["daily_pnl"]
    cum = bt_df["cum_pnl"]
    n   = len(pnl)

    total_pnl = float(cum.iloc[-1])
    ann_pnl   = total_pnl / (n / 252)
    ann_vol   = float(pnl.std() * np.sqrt(252))
    sharpe    = ann_pnl / ann_vol if ann_vol > 0 else 0.0

    downside  = pnl[pnl < 0]
    down_std  = float(downside.std() * np.sqrt(252)) if len(downside) > 1 else 0.0
    sortino   = ann_pnl / down_std if down_std > 0 else 0.0

    running_max = cum.cummax()
    max_dd      = float((cum - running_max).min())
    calmar      = ann_pnl / abs(max_dd) if max_dd < 0 else 0.0

    in_pos      = bt_df["position"] != 0
    pos_pnl     = bt_df.loc[in_pos, "daily_pnl"]
    # Positive_Day_Rate: fraction of in-position days with positive daily P&L.
    # This is a day-level hit rate, NOT a per-trade win rate.
    pos_day_rate = float((pos_pnl > 0).sum() / len(pos_pnl)) if len(pos_pnl) > 0 else 0.0
    pct_in_mkt  = float(in_pos.sum() / n * 100)

    # Exact trade count: every 0→±1 transition is a new entry; every ±1→∓1 flip
    # is also a new entry (it closes one trade and opens the next at the same bar).
    # fillna(0) treats the bar before the series as flat, which is always true here.
    pos          = bt_df["position"]
    prev_pos     = pos.shift(1).fillna(0)
    n_trades     = int(
        ((prev_pos == 0) & (pos != 0)).sum()   # flat → non-flat: new entry
        + (pos.diff().abs() >= 2).sum()         # direct flip: also a new entry
    )
    avg_trade    = total_pnl / n_trades if n_trades > 0 else 0.0

    pos_changes  = pos.diff().abs()
    n_pos_chg    = int((pos_changes > 0).sum())

    return {
        "Period":                label,
        "Days":                  n,
        "Total_PnL":             round(total_pnl, 6),
        "Ann_PnL":               round(ann_pnl, 6),
        "Ann_Vol":               round(ann_vol, 6),
        "Sharpe_ratio":          round(sharpe, 4),
        "Sortino_ratio":         round(sortino, 4),
        "Max_drawdown":          round(max_dd, 6),
        "Calmar_ratio":          round(calmar, 4),
        "Positive_Day_Rate_pct": round(pos_day_rate * 100, 2),
        "Num_position_chg":      n_pos_chg,
        "Num_trades":            n_trades,
        "Avg_trade_PnL":         round(avg_trade, 6),
        "Pct_in_market":         round(pct_in_mkt, 2),
    }


def block_bootstrap_sharpe(daily_pnl_series, n_boot=2000, block_size=20, seed=42):
    """Block bootstrap 95% CI on annualised Sharpe ratio."""
    rng = np.random.default_rng(seed)
    pnl = daily_pnl_series.dropna().values
    n   = len(pnl)

    n_blocks    = n // block_size
    pnl_trimmed = pnl[:n_blocks * block_size].reshape(n_blocks, block_size)

    boot_sharpes = []
    for _ in range(n_boot):
        idx      = rng.integers(0, n_blocks, size=n_blocks)
        boot_pnl = pnl_trimmed[idx].ravel()
        ann_vol  = boot_pnl.std() * np.sqrt(252)
        if ann_vol > 0:
            boot_sharpes.append((boot_pnl.mean() * 252) / ann_vol)

    return (round(float(np.percentile(boot_sharpes, 2.5)),  4),
            round(float(np.percentile(boot_sharpes, 97.5)), 4))


def extract_trade_log(bt_df, cost_per_leg):
    """
    Extract individual round-trip trades from a backtest DataFrame.

    Timing convention (consistent with Phase 3b and Phase 4):
      Signal_Date     -- bar i-1; close of this bar produced the z-score that
                         triggered the entry decision.
      Signal_Z_score  -- z_arr[i-1]; the z-score that actually triggered entry.
      Execution_Date  -- bar i; open of this bar is when the position is entered.

    Flip trades (position jumps from +1 to -1 or vice versa on the same bar):
      The flip bar's combined cost (4 legs = exit + entry) is split evenly:
        - exiting trade gets 2 legs (its exit cost)
        - entering trade gets 2 legs (its entry cost)
      The flip bar's P&L is attributed in full to the entering trade, because
      the backtest engine computes daily_pnl[flip] as the net of the overnight
      long exit and the intraday short entry combined, and there is no per-leg
      breakdown available in the stored columns.  This is an approximation for
      the trade log only; the P&L engine is exact.
    """
    pos_arr  = bt_df["position"].values
    z_arr    = bt_df["zscore"].values
    pnl_arr  = bt_df["daily_pnl"].values
    cost_arr = bt_df["trade_cost"].values
    dates    = bt_df.index
    n        = len(pos_arr)
    trades   = []
    i        = 0

    while i < n:
        if pos_arr[i] != 0:
            entry_i    = i
            entry_date = dates[i]
            entry_pos  = pos_arr[i]

            # Signal date/z: the PREVIOUS bar's close triggered this entry.
            # Guard against i=0 (no prior bar; use current bar as fallback).
            signal_date = dates[entry_i - 1] if entry_i > 0 else dates[entry_i]
            entry_z     = (float(z_arr[entry_i - 1]) if entry_i > 0
                           else float(z_arr[entry_i]))

            # Is this entry at a flip bar?  (previous bar held opposite position)
            is_flip_entry = (entry_i > 0
                             and pos_arr[entry_i - 1] != 0
                             and pos_arr[entry_i - 1] != entry_pos)

            running = float(pnl_arr[i])
            i += 1

            # Accumulate P&L while position is unchanged
            while i < n and pos_arr[i] == entry_pos:
                running += float(pnl_arr[i])
                i += 1

            # Is the exit a flip (→ opposite) or a normal exit (→ 0)?
            is_flip_exit = (i < n and pos_arr[i] != 0)

            if is_flip_exit:
                # Flip exit: bar i is the FIRST bar of the new trade.
                # Do NOT add pnl_arr[i] here -- it belongs to the new trade.
                # Do NOT advance i -- outer loop will start new trade at bar i.
                # Charge only 2 exit legs from bar i's combined 4-leg cost.
                exit_date = dates[i]
                exit_z    = float(z_arr[i - 1])   # signal z that triggered the flip
                cost      = (float(cost_arr[entry_i : i].sum())
                             + 2.0 * cost_per_leg
                             - (2.0 * cost_per_leg if is_flip_entry else 0.0))
                # i is NOT advanced; outer loop picks up the new trade at bar i

            elif i < n:
                # Normal exit: position → 0 at bar i.
                # daily_pnl[i] = 0 for a flat bar so adding it is harmless.
                running  += float(pnl_arr[i])
                exit_date = dates[i]
                exit_z    = float(z_arr[i - 1])   # signal z that triggered the exit
                cost      = (float(cost_arr[entry_i : i + 1].sum())
                             - (2.0 * cost_per_leg if is_flip_entry else 0.0))
                i += 1

            else:
                # End of data while still in position (no exit bar)
                exit_date = dates[i - 1]
                exit_z    = float(z_arr[i - 1])
                cost      = (float(cost_arr[entry_i : i].sum())
                             - (2.0 * cost_per_leg if is_flip_entry else 0.0))

            gross = running + cost
            trades.append({
                "Signal_Date":      str(signal_date.date()),
                "Execution_Date":   str(entry_date.date()),
                "Exit_Date":        str(exit_date.date()),
                "Position":         "Long Spread" if entry_pos == 1 else "Short Spread",
                "Signal_Z_score":   round(entry_z, 4),
                "Exit_Z_score":     round(exit_z, 4),
                "Holding_Days":     (exit_date - entry_date).days,
                "Gross_PnL":        round(gross, 6),
                "Transaction_Cost": round(cost, 6),
                "Net_PnL":          round(running, 6),
            })
        else:
            i += 1
    return pd.DataFrame(trades)


# ---------------------------------------------------------------------------
# Per-pair runner
# ---------------------------------------------------------------------------

def run_pair(dep, indep, tests_passed):
    """
    Run the full pairs trading strategy for one (dep, indep) pair.
    Returns a summary dict for the cross-pair comparison table.
    """
    tag = f"{dep}_{indep}"

    print(f"\n{'='*65}")
    print(f"PAIR: {dep} / {indep}  [{tests_passed}]")
    print(f"{'='*65}")
    print(f"  Entry threshold : |z| > {Z_ENTRY}")
    print(f"  Exit threshold  : z crosses {Z_EXIT}")
    print(f"  Stop loss       : |z| > {Z_STOP}")
    print(f"  Z-score window  : {LOOKBACK} days")
    print(f"  Cost per leg    : {COST_PER_LEG*10_000:.1f} bps")
    print(f"  Train period    : 2018-01-01 to {TRAIN_END}")
    print(f"  Test period     : {TEST_START} to 2025-12-31")
    print(f"  Execution       : signal at close(t), execute at open(t+1)")

    # Load close and open prices
    price_df, open_df, log_df, log_open_df = load_prices(dep, indep)
    print(f"\n  Prices loaded: {len(price_df)} days  "
          f"({price_df.index[0].date()} to {price_df.index[-1].date()})")

    train_log = log_df.loc[:TRAIN_END]
    test_log  = log_df.loc[TEST_START:]
    print(f"  Train : {len(train_log)} days  "
          f"({train_log.index[0].date()} to {train_log.index[-1].date()})")
    print(f"  Test  : {len(test_log)} days  "
          f"({test_log.index[0].date()} to {test_log.index[-1].date()})")

    # Estimate hedge ratio on training data (close prices)
    hr, ic, r2 = estimate_hr(train_log[dep], train_log[indep])
    print(f"\n  OLS ({train_log.index[0].date()} - {train_log.index[-1].date()}):")
    print(f"    Hedge ratio : {hr:.4f}")
    print(f"    Intercept   : {ic:.4f}")
    print(f"    R-squared   : {r2:.4f}")

    # Build close-price spread/zscore and open-price spread for execution
    spread_full,      zscore_full      = build_spread_zscore(log_df,      dep, indep, hr, ic, LOOKBACK)
    spread_open_full                   = build_open_spread(log_open_df,   dep, indep, hr, ic)

    train_spread_mean = float(spread_full.loc[:TRAIN_END].mean())
    train_spread_std  = float(spread_full.loc[:TRAIN_END].std())
    print(f"\n  Training spread:  mean={train_spread_mean:.4f}  std={train_spread_std:.4f}")

    # --- Out-of-sample test backtest ---
    bt_test  = backtest(
        zscore_full.loc[TEST_START:], spread_full.loc[TEST_START:],
        Z_ENTRY, Z_EXIT, Z_STOP, COST_PER_LEG,
        spread_open_series=spread_open_full.loc[TEST_START:],
    )

    # --- Training period backtest ---
    bt_train = backtest(
        zscore_full.loc[:TRAIN_END], spread_full.loc[:TRAIN_END],
        Z_ENTRY, Z_EXIT, Z_STOP, COST_PER_LEG,
        spread_open_series=spread_open_full.loc[:TRAIN_END],
    )
    train_m  = compute_metrics(bt_train, label="Train 2018-2021")
    test_m   = compute_metrics(bt_test,  label="Test 2022-2025")

    train_ci = block_bootstrap_sharpe(bt_train["daily_pnl"])
    test_ci  = block_bootstrap_sharpe(bt_test["daily_pnl"])

    print(f"\n  {'Period':<20}  {'Sharpe':>8}  {'95% CI lower':>13}  {'95% CI upper':>13}")
    print(f"  {'-'*60}")
    for lbl, m, ci in [("Train 2018-2021", train_m, train_ci),
                        ("Test  2022-2025", test_m,  test_ci)]:
        print(f"  {lbl:<20}  {m['Sharpe_ratio']:>8.4f}  "
              f"{ci[0]:>13.4f}  {ci[1]:>13.4f}")

    # Save train/test comparison
    comparison_rows = [
        {**train_m, "Sharpe_CI_lo": train_ci[0], "Sharpe_CI_hi": train_ci[1]},
        {**test_m,  "Sharpe_CI_lo": test_ci[0],  "Sharpe_CI_hi": test_ci[1]},
    ]
    comparison_df = pd.DataFrame(comparison_rows)
    cmp_csv = REPORTS_DIR / f"strategy_{tag}_train_test_comparison.csv"
    comparison_df.to_csv(cmp_csv, index=False)
    print(f"\n  Train/test comparison saved -> {cmp_csv.name}")

    # --- Walk-forward validation ---
    print(f"\n  Walk-forward (annual, expanding window):")
    print(f"  {'Year':<6} {'HR':>7} {'IC':>9} {'R2':>6}  "
          f"{'Sharpe':>8}  {'Total PnL':>10}  {'PosRate':>9}")
    print(f"  {'-'*68}")

    wf_records  = []
    wf_segments = {}
    for year in range(2019, 2026):
        train_wf = log_df.loc[:f"{year-1}-12-31"]
        test_wf  = log_df.loc[f"{year}-01-01":f"{year}-12-31"]
        if len(test_wf) == 0:
            continue
        hr_wf, ic_wf, r2_wf = estimate_hr(train_wf[dep], train_wf[indep])
        # Close spread for signal z-score
        spread_wf = log_df[dep] - hr_wf * log_df[indep] - ic_wf
        spread_to = spread_wf.loc[:f"{year}-12-31"]
        z_to = ((spread_to - spread_to.rolling(LOOKBACK).mean())
                / spread_to.rolling(LOOKBACK).std())
        z_yr    = z_to.loc[f"{year}-01-01":f"{year}-12-31"]
        sp_yr   = spread_wf.loc[f"{year}-01-01":f"{year}-12-31"]
        # Open spread for execution pricing
        sp_op_yr = (log_open_df[dep] - hr_wf * log_open_df[indep] - ic_wf
                    ).loc[f"{year}-01-01":f"{year}-12-31"]
        bt_yr = backtest(z_yr, sp_yr, Z_ENTRY, Z_EXIT, Z_STOP, COST_PER_LEG,
                         spread_open_series=sp_op_yr)
        m = compute_metrics(bt_yr, label=str(year))
        m["HR"] = round(hr_wf, 4)
        m["R2"] = round(r2_wf, 4)
        wf_records.append(m)
        wf_segments[year] = bt_yr
        print(f"  {year:<4}  {hr_wf:>7.4f}  {ic_wf:>9.4f}  {r2_wf:>6.4f}  "
              f"{m['Sharpe_ratio']:>8.4f}  {m['Total_PnL']:>+10.6f}  "
              f"{m['Positive_Day_Rate_pct']:>8.2f}%")

    wf_df = pd.DataFrame(wf_records)

    # Stress test: 2022 bear market
    if 2022 in wf_segments:
        m_2022 = compute_metrics(wf_segments[2022], label="2022 Bear Market")
        print(f"\n  2022 stress:  Sharpe={m_2022['Sharpe_ratio']:.4f}  "
              f"PnL={m_2022['Total_PnL']:+.6f}  "
              f"MaxDD={m_2022['Max_drawdown']:.4f}")

    # Save all strategy results
    all_records = [test_m] + wf_records
    results_df  = pd.DataFrame(all_records)
    csv_out = REPORTS_DIR / f"strategy_{tag}_results.csv"
    results_df.to_csv(csv_out, index=False)
    print(f"  Results saved -> {csv_out.name}")

    # --- Full period and trade log ---
    bt_full = backtest(
        zscore_full, spread_full, Z_ENTRY, Z_EXIT, Z_STOP, COST_PER_LEG,
        spread_open_series=spread_open_full,
    )
    full_m  = compute_metrics(bt_full, label="Full 2018-2025")

    comp_df = pd.DataFrame([train_m, test_m, full_m])
    display_cols = [
        "Period", "Days", "Total_PnL", "Ann_PnL", "Ann_Vol",
        "Sharpe_ratio", "Sortino_ratio", "Max_drawdown", "Calmar_ratio",
        "Positive_Day_Rate_pct", "Num_trades", "Avg_trade_PnL", "Pct_in_market",
    ]
    comp_csv = REPORTS_DIR / f"strategy_{tag}_period_comparison.csv"
    comp_df[display_cols].to_csv(comp_csv, index=False)

    trade_log = extract_trade_log(bt_test, COST_PER_LEG)
    if not trade_log.empty:
        trade_csv = REPORTS_DIR / f"strategy_{tag}_trade_log.csv"
        trade_log.to_csv(trade_csv, index=False)
        print(f"  Trade log ({len(trade_log)} trades) saved -> {trade_csv.name}")

    # --- Transaction-cost sensitivity ---
    cost_sweep_records = []
    for bps in [5, 10, 20, 30]:
        cost = bps / 10_000
        bt_c = backtest(
            zscore_full.loc[TEST_START:], spread_full.loc[TEST_START:],
            Z_ENTRY, Z_EXIT, Z_STOP, cost,
            spread_open_series=spread_open_full.loc[TEST_START:],
        )
        m_c = compute_metrics(bt_c, label=f"{bps}bps")
        cost_sweep_records.append({
            "Cost_bps": bps, "Sharpe": m_c["Sharpe_ratio"],
            "Total_PnL": m_c["Total_PnL"], "Max_drawdown": m_c["Max_drawdown"],
            "Num_trades": m_c["Num_trades"],
        })
    cost_sweep_df  = pd.DataFrame(cost_sweep_records)
    cost_sweep_csv = REPORTS_DIR / f"strategy_{tag}_cost_sensitivity.csv"
    cost_sweep_df.to_csv(cost_sweep_csv, index=False)

    # --- Sensitivity grid ---
    def _sens(z_entry_val, lb, z_stop_val):
        sp_v, z_v  = build_spread_zscore(log_df, dep, indep, hr, ic, lb)
        sp_op_v    = build_open_spread(log_open_df, dep, indep, hr, ic)
        bt_v = backtest(
            z_v.loc[TEST_START:], sp_v.loc[TEST_START:],
            z_entry_val, Z_EXIT, z_stop_val, COST_PER_LEG,
            spread_open_series=sp_op_v.loc[TEST_START:],
        )
        return compute_metrics(bt_v)

    sens_records = []
    for z_e in [1.5, 2.0, 2.5]:
        for lb in [20, 30, 60, 90]:
            for z_s in [3.0, 3.5, 4.0]:
                m_s = _sens(z_e, lb, z_s)
                sens_records.append({
                    "Entry_threshold": z_e, "Window": lb, "Stop_loss": z_s,
                    "Sharpe": m_s["Sharpe_ratio"], "Total_PnL": m_s["Total_PnL"],
                    "Max_drawdown": m_s["Max_drawdown"], "Num_trades": m_s["Num_trades"],
                })
    sens_df  = pd.DataFrame(sens_records)
    sens_csv = REPORTS_DIR / f"strategy_{tag}_sensitivity.csv"
    sens_df.to_csv(sens_csv, index=False)
    best_sens = sens_df.loc[sens_df["Sharpe"].idxmax()]
    print(f"  Best sensitivity Sharpe: {best_sens['Sharpe']:.4f}  "
          f"(entry={best_sens['Entry_threshold']}, window={int(best_sens['Window'])}d, "
          f"stop={best_sens['Stop_loss']})")

    # --- Charts ---
    # Chart 1: Z-score + spread + equity curve (test period)
    fig = plt.figure(figsize=(16, 13))
    gs  = gridspec.GridSpec(3, 1, hspace=0.45, figure=fig)

    ax1 = fig.add_subplot(gs[0])
    z_test = zscore_full.loc[TEST_START:]
    ax1.plot(z_test.index, z_test.values, color="steelblue", lw=0.85, label="Z-score")
    ax1.axhline( Z_ENTRY, color="red",     ls="--", lw=1.0, label=f"+{Z_ENTRY} short entry")
    ax1.axhline(-Z_ENTRY, color="green",   ls="--", lw=1.0, label=f"-{Z_ENTRY} long entry")
    ax1.axhline( Z_EXIT,  color="black",   ls="-",  lw=0.5)
    ax1.axhline( Z_STOP,  color="darkred", ls=":",  lw=0.9, label=f"+/-{Z_STOP} stop loss")
    ax1.axhline(-Z_STOP,  color="darkred", ls=":",  lw=0.9)
    pos_arr = bt_test["position"].values
    dates   = bt_test.index
    for i in range(1, len(pos_arr)):
        if   pos_arr[i] ==  1:
            ax1.axvspan(dates[i-1], dates[i], alpha=0.13, color="green",  lw=0)
        elif pos_arr[i] == -1:
            ax1.axvspan(dates[i-1], dates[i], alpha=0.13, color="tomato", lw=0)
    ax1.set_title(f"{dep}/{indep} Z-score -- Test {TEST_START[:4]}-2025  "
                  "(green=long, red=short spread)", fontsize=10)
    ax1.set_ylabel("Z-score")
    ax1.legend(fontsize=7.5, loc="upper right", ncol=3)
    ax1.set_xlim(bt_test.index[0], bt_test.index[-1])

    ax2 = fig.add_subplot(gs[1])
    sp_test = spread_full.loc[TEST_START:]
    ax2.plot(sp_test.index, sp_test.values, color="darkorange", lw=0.85)
    ax2.axhline(train_spread_mean, color="black", ls="--", lw=0.8,
                label=f"Training mean ({train_spread_mean:.4f})")
    ax2.set_title(f"Spread = log({dep}) - {hr:.4f}*log({indep}) - {ic:.4f}", fontsize=10)
    ax2.set_ylabel("Spread (log-price units)")
    ax2.legend(fontsize=8)
    ax2.set_xlim(bt_test.index[0], bt_test.index[-1])

    ax3 = fig.add_subplot(gs[2])
    ax3.plot(bt_test.index, bt_test["cum_pnl"].values, color="steelblue", lw=1.2)
    ax3.fill_between(bt_test.index, bt_test["cum_pnl"].values, 0,
                     where=bt_test["cum_pnl"].values >= 0, alpha=0.25, color="green")
    ax3.fill_between(bt_test.index, bt_test["cum_pnl"].values, 0,
                     where=bt_test["cum_pnl"].values <  0, alpha=0.25, color="tomato")
    ax3.axhline(0, color="black", lw=0.6)
    metrics_str = (f"Sharpe={test_m['Sharpe_ratio']:.2f}  "
                   f"Sortino={test_m['Sortino_ratio']:.2f}  "
                   f"MaxDD={test_m['Max_drawdown']:.4f}  "
                   f"PosRate={test_m['Positive_Day_Rate_pct']:.1f}%")
    ax3.set_title(f"Cumulative P&L -- Test Period  |  {metrics_str}", fontsize=9.5)
    ax3.set_ylabel("Cumulative log-return P&L")
    ax3.set_xlabel("Date")
    ax3.set_xlim(bt_test.index[0], bt_test.index[-1])

    chart1 = CHARTS_DIR / f"strategy_{tag}_zscore_equity.png"
    plt.savefig(chart1, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Chart saved -> {chart1.name}")

    # Chart 2: Walk-forward Sharpe + P&L
    fig2, (ax_s, ax_p) = plt.subplots(1, 2, figsize=(14, 5))
    yr_labels = [r["Period"] for r in wf_records]
    sharpes   = [r["Sharpe_ratio"] for r in wf_records]
    pnls      = [r["Total_PnL"]    for r in wf_records]
    ax_s.bar(yr_labels, sharpes,
             color=["steelblue" if s >= 0 else "tomato" for s in sharpes],
             edgecolor="white", lw=0.5)
    ax_s.axhline(0, color="black", lw=0.8)
    ax_s.set_title(f"{dep}/{indep} Walk-Forward Annual Sharpe", fontsize=11)
    ax_s.set_ylabel("Annualised Sharpe")
    ax_p.bar(yr_labels, pnls,
             color=["steelblue" if p >= 0 else "tomato" for p in pnls],
             edgecolor="white", lw=0.5)
    ax_p.axhline(0, color="black", lw=0.8)
    ax_p.set_title(f"{dep}/{indep} Walk-Forward Annual P&L", fontsize=11)
    ax_p.set_ylabel("Total P&L (log-return)")
    plt.tight_layout()
    chart2 = CHARTS_DIR / f"strategy_{tag}_walkforward.png"
    plt.savefig(chart2, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Chart saved -> {chart2.name}")

    # Chart 3: Full-period z-score context
    fig3, ax_full = plt.subplots(figsize=(16, 4))
    ax_full.plot(zscore_full.index, zscore_full.values,
                 color="steelblue", lw=0.7, alpha=0.9)
    for h, c, ls in [(Z_ENTRY, "red", "--"), (-Z_ENTRY, "green", "--"),
                     (Z_STOP, "darkred", ":"), (-Z_STOP, "darkred", ":")]:
        ax_full.axhline(h, color=c, ls=ls, lw=0.9)
    ax_full.axhline(0, color="black", lw=0.5)
    ax_full.axvspan(pd.Timestamp(TEST_START), zscore_full.index[-1],
                    alpha=0.07, color="orange", label=f"Test ({TEST_START[:4]}-2025)")
    ax_full.axvspan(zscore_full.index[0], pd.Timestamp(TRAIN_END),
                    alpha=0.07, color="skyblue", label=f"Train (2018-{TRAIN_END[:4]})")
    ax_full.set_title(f"{dep}/{indep} Spread Z-score -- Full Period 2018-2025", fontsize=11)
    ax_full.set_ylabel("Z-score")
    ax_full.legend(fontsize=9)
    plt.tight_layout()
    chart3 = CHARTS_DIR / f"strategy_{tag}_zscore_full.png"
    plt.savefig(chart3, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Chart saved -> {chart3.name}")

    avg_sharpe   = float(wf_df["Sharpe_ratio"].mean())
    positive_yrs = sum(1 for r in wf_records if r["Total_PnL"] > 0)

    return {
        "Pair":          f"{dep}/{indep}",
        "Tests_passed":  tests_passed,
        "HR":            round(hr, 4),
        "R2_train":      round(r2, 4),
        "Train_Sharpe":  train_m["Sharpe_ratio"],
        "Train_CI_lo":   train_ci[0],
        "Train_CI_hi":   train_ci[1],
        "Test_Sharpe":   test_m["Sharpe_ratio"],
        "Test_CI_lo":    test_ci[0],
        "Test_CI_hi":    test_ci[1],
        "Test_TotalPnL": test_m["Total_PnL"],
        "Test_MaxDD":    test_m["Max_drawdown"],
        "Test_PosRate":  test_m["Positive_Day_Rate_pct"],
        "WF_AvgSharpe":  round(avg_sharpe, 4),
        "WF_ProfitableYears": positive_yrs,
        "BestSens_Sharpe": round(float(best_sens["Sharpe"]), 4),
    }


# ---------------------------------------------------------------------------
# Main -- read selected pairs and run strategy for each
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 65)
    print("PHASE 3B -- PAIRS TRADING STRATEGY")
    print("=" * 65)

    selected_csv = REPORTS_DIR / "selected_pairs.csv"
    if selected_csv.exists():
        sel_df = pd.read_csv(selected_csv)
        pair_list = [
            (row["OLS_direction"].split("~")[0],
             row["OLS_direction"].split("~")[1],
             row["Tests_passed"])
            for _, row in sel_df.iterrows()
        ]
        print(f"\nLoaded {len(pair_list)} pair(s) from {selected_csv.name}:")
        for dep, indep, tp in pair_list:
            print(f"  {dep}/{indep}  [{tp}]")
    else:
        pair_list = [("AMZN", "META", "Both")]
        print(f"\nWARNING: {selected_csv.name} not found -- falling back to AMZN/META.")
        print("Run phase3_cointegration.py first to generate selected_pairs.csv.")

    print(f"\nStrategy parameters: entry=+-{Z_ENTRY}  exit={Z_EXIT}  "
          f"stop=+-{Z_STOP}  window={LOOKBACK}d  cost={COST_PER_LEG*10_000:.1f}bps/leg")
    print("Execution: signal at close(t), execute at open(t+1) [open-price P&L]")
    print("Stop-loss guard: re-entry in same direction blocked until z returns inside band")

    # Run all pairs
    summary_rows = []
    for dep, indep, tests_passed in pair_list:
        try:
            row = run_pair(dep, indep, tests_passed)
            summary_rows.append(row)
        except Exception as exc:
            print(f"\n  ERROR running {dep}/{indep}: {exc}")

    # Cross-pair summary
    if summary_rows:
        summary_df  = pd.DataFrame(summary_rows)
        summary_csv = REPORTS_DIR / "strategy_cross_pair_summary.csv"
        summary_df.to_csv(summary_csv, index=False)

        print(f"\n{'='*65}")
        print("CROSS-PAIR SUMMARY")
        print(f"{'='*65}")
        print(summary_df.to_string(index=False))
        print(f"\nCross-pair summary saved -> {summary_csv}")

    print(f"\n{'='*65}")
    print("PHASE 3B COMPLETE")
    print(f"{'='*65}")
    print(f"\nNext step: Phase 4 -- Testing mean reversion on unseen 2026 data.")

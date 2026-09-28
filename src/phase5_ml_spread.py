"""
Phase 5 – Machine Learning for Predicting Spread
LSTM-based prediction of standardised OLS residuals for AMZN/META and MSFT/AAPL.

Pipeline (per project brief)
------------------------------
1.  OLS spread computed using log-transformed prices.  Parameters are
    re-estimated on the training period (2018-2020) only, so no test-period
    data enters the spread construction.
2.  Residuals standardised using fixed mean/std estimated on the LSTM training
    set (2018-2020) only — validation (2021) and test (2022+) are never used
    to fit the scaler.
3.  Time-series features: Lag-1, Lag-2, Lag-3, rolling 20-day std.
4.  Sequential input: 20-trading-day lookback → 5-day-ahead output (LSTM).
5.  LSTM trained on 2018-2020, validated on 2021 (chronological split).
6.  Evaluated on unseen test data (2022-2025): RMSE and MAE per horizon step.
7.  Predicted standardised residuals converted to a convergence/divergence
    signal: converging if mean(|pred h1..h5|) < |z_std at signal bar|.
8.  Step 8 — two-part comparison:
    8a (supplementary historical holdout — kept for audit trail):
      Historical model (OLS + scaler + LSTM all trained on 2018-2020).
      Phase 4-style baseline with 2021 fixed anchor vs LSTM-enhanced on
      2022-2025 test period.  Labelled "Phase 4-style holdout" NOT the
      actual Phase 4 comparison.
    8b (definitive 2026 comparison):
      A SEPARATE final model is trained — OLS + scaler + LSTM all on
      2018-2024, validated on 2025.  Tested on 2026-01-01 to 2026-07-31.
      Phase 4 baseline (OLS 2018-2025, 2025 anchor, entry ±2, exit 0) vs
      LSTM-enhanced (same Phase 4 signals gated by the final 2026 LSTM).

Leakage corrections vs original implementation
------------------------------------------------
- OLS previously used pre-computed full-dataset params from selected_pairs.csv
  (estimated on 2018-2025).  Now re-estimated on the LSTM training set only
  (2018-2020), consistent with the scaler.  Using 2021 (the validation year)
  for OLS would allow the validation-period residual distribution to influence
  the spread definition seen during LSTM training — so both OLS and scaler
  are anchored to the same 2018-2020 window.
- Standardisation scaler previously used 2018-2021 (including val year).
  Now uses 2018-2020 (LSTM training set only).

Execution convention (inherited from Phase 3b / Phase 4)
---------------------------------------------------------
Signal at close(t-1) → execute at open(t).
LSTM features are constructed up to bar t-1 (the signal day), so the
convergence filter contains no look-ahead bias.

Outputs
-------
Historical experiment (2018-2020 model):
  outputs/reports/phase5_{tag}_lstm_metrics.csv
  outputs/reports/phase5_{tag}_holdout_comparison.csv   (Step 8a: 2022-2025 supplementary)
  outputs/charts/phase5_{tag}_predictions.png
  outputs/charts/phase5_{tag}_holdout_comparison.png

Final 2026 experiment (2018-2024 model):
  outputs/reports/phase5_{tag}_final_lstm_metrics.csv
  outputs/reports/phase5_{tag}_2026_comparison.csv      (Step 8b: definitive 2026)
  outputs/reports/phase5_{tag}_2026_trade_log.csv
  outputs/charts/phase5_{tag}_final_predictions.png
  outputs/charts/phase5_{tag}_2026_comparison.png

Cross-pair:
  outputs/reports/phase5_cross_pair_summary.csv
  outputs/reports/phase5_cross_pair_final_summary.csv
"""

import os
import sys
sys.stdout.reconfigure(encoding="utf-8")
import warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import statsmodels.api as sm
import yfinance as yf
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"]    = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"]   = "0"
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import DATA_RAW, CHARTS_DIR, REPORTS_DIR, MODELS_DIR, load_or_download_2026
from phase3_strategy import (
    load_prices, build_open_spread, backtest, compute_metrics,
    extract_trade_log,
)

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
SEED = 42
# tf.keras.utils.set_random_seed sets TF, NumPy, and Python random seeds.
# enable_op_determinism() is called once in main() before any computation.
tf.keras.utils.set_random_seed(SEED)

# ---------------------------------------------------------------------------
# LSTM chronological splits
# ---------------------------------------------------------------------------
ML_TRAIN_END  = "2020-12-31"   # LSTM training data
ML_VAL_START  = "2021-01-01"   # LSTM validation year (not used for OLS, scaler, or LSTM fitting)
ML_VAL_END    = "2021-12-31"
ML_TEST_START = "2022-01-01"   # out-of-sample test (same as Phase 3b)

# Final 2026 LSTM model — trained on all pre-2026 history.
# This is the model used for the definitive 2026 comparison (Step 8).
# OLS and scaler also fit on 2018-2024 so 2025 is a clean validation year.
FINAL_TRAIN_END = "2024-12-31"
FINAL_VAL_START = "2025-01-01"
FINAL_VAL_END   = "2025-12-31"

# OLS re-estimation period: same as LSTM training set.
# Using ML_TRAIN_END (2020-12-31) keeps OLS and scaler on identical data so
# neither the validation year (2021) nor the test period (2022+) influences
# the residual definition.
ML_OLS_END    = ML_TRAIN_END   # "2020-12-31"

# Phase 4-style anchor for 2022-2025 holdout comparison (Step 8a)
# Use 2021 (the validation year, immediately preceding the test window) as the
# fixed anchor — mirrors Phase 4's approach of anchoring to the last 12 months
# before the test period.
P4_HOLDOUT_ANCHOR_START = "2021-01-01"
P4_HOLDOUT_ANCHOR_END   = "2021-12-31"

# Phase 4 2026 unseen comparison (Step 8b) — mirrors phase4_unseen.py exactly
P4_OLS_END    = "2025-12-31"   # Phase 4 fits OLS on full 2018-2025 history
P4_NORM_START = "2025-01-01"   # fixed anchor = last 12 months of training
P4_NORM_END   = "2025-12-31"
P4_TEST_START = "2026-01-01"
P4_TEST_END   = "2026-07-31"
P4_TEST_END_EX= "2026-08-01"   # exclusive end for yfinance

# Phase 4 signal rules (no stop-loss)
Z_ENTRY_P4    = 2.0
Z_EXIT_P4     = 0.0

# ---------------------------------------------------------------------------
# Feature / sequence parameters
# ---------------------------------------------------------------------------
ROLLING_STD_WIN = 20    # window for roll_std feature
SEQ_LEN         = 20    # LSTM lookback window (trading days)
HORIZON         = 5     # prediction horizon (trading days ahead)

# ---------------------------------------------------------------------------
# LSTM hyper-parameters
# ---------------------------------------------------------------------------
LSTM_UNITS_1  = 64
LSTM_UNITS_2  = 32
DROPOUT       = 0.2
LEARNING_RATE = 1e-3
BATCH_SIZE    = 32
MAX_EPOCHS    = 100
PATIENCE      = 15

# ---------------------------------------------------------------------------
# Transaction cost (applied to both baseline and LSTM-enhanced strategies)
# ---------------------------------------------------------------------------
COST_PER_LEG = 0.001


# ---------------------------------------------------------------------------
# Step 1: OLS re-estimation on training data only
# ---------------------------------------------------------------------------

def fit_ols(log_prices, dep, indep, end_date):
    """
    OLS of log(dep) on log(indep) with constant, using data up to end_date.
    Returns (hedge_ratio, intercept, r_squared).
    Replicates Phase 3b's training-only OLS to avoid leaking test data.
    """
    subset = log_prices.loc[log_prices.index <= end_date]
    X      = sm.add_constant(subset[indep])
    model  = sm.OLS(subset[dep], X).fit()
    return float(model.params.iloc[1]), float(model.params.iloc[0]), float(model.rsquared)


def build_raw_spread(log_prices, dep, indep, hedge_ratio, intercept):
    """OLS log-price residual: dep_log - HR * indep_log - intercept."""
    return log_prices[dep] - hedge_ratio * log_prices[indep] - intercept


# ---------------------------------------------------------------------------
# Step 2: Standardisation — scaler fitted on LSTM training set only
# ---------------------------------------------------------------------------

def standardise(series, train_end=ML_TRAIN_END):
    """
    Standardise using the mean and std of data up to train_end only.
    Default train_end = ML_TRAIN_END (2020-12-31) so the scaler is never
    contaminated by validation-period (2021) or test-period (2022+) data.
    Returns (z_std, mu, sigma).
    """
    train_vals = series.loc[series.index <= train_end]
    mu         = float(train_vals.mean())
    sigma      = float(train_vals.std())
    z_std      = (series - mu) / sigma
    return z_std, mu, sigma


# ---------------------------------------------------------------------------
# Step 3: Feature construction
# ---------------------------------------------------------------------------

def build_features(z_std):
    """
    Build feature DataFrame from the standardised residual series.
    Columns at bar t: z_std, lag1 = z(t-1), lag2 = z(t-2), lag3 = z(t-3),
    roll_std = rolling 20-day std of z_std over [t-19, t].
    All features are causally correct (only past/current data used at t).
    """
    df             = pd.DataFrame({"z_std": z_std}, index=z_std.index)
    df["lag1"]     = df["z_std"].shift(1)
    df["lag2"]     = df["z_std"].shift(2)
    df["lag3"]     = df["z_std"].shift(3)
    df["roll_std"] = df["z_std"].rolling(ROLLING_STD_WIN).std()
    return df


# ---------------------------------------------------------------------------
# Step 4: Sequence builders
# ---------------------------------------------------------------------------

def make_sequences(feat_df, seq_len=SEQ_LEN, horizon=HORIZON):
    """
    Build (X, y, signal_dates) for training/validation/evaluation.
    X[i] = feature rows [i … i+seq_len-1]; y[i] = z_std values [i+seq_len … +horizon].
    signal_dates[i] = last input bar date (= signal bar in Phase 3b convention).
    Samples with any NaN in X or y are skipped.
    """
    feat_arr = feat_df.values.astype(np.float32)
    z_arr    = feat_df["z_std"].values.astype(np.float32)
    dates    = feat_df.index
    n        = len(feat_arr)
    X_list, y_list, sig_dates = [], [], []
    for i in range(n - seq_len - horizon + 1):
        xb = feat_arr[i : i + seq_len]
        yb = z_arr[i + seq_len : i + seq_len + horizon]
        if np.any(np.isnan(xb)) or np.any(np.isnan(yb)):
            continue
        X_list.append(xb)
        y_list.append(yb)
        sig_dates.append(dates[i + seq_len - 1])
    if not X_list:
        empty = np.empty((0, seq_len, feat_arr.shape[1]), dtype=np.float32)
        return empty, np.empty((0, horizon), dtype=np.float32), pd.DatetimeIndex([])
    return (np.array(X_list, dtype=np.float32),
            np.array(y_list,  dtype=np.float32),
            pd.DatetimeIndex(sig_dates))


def make_sequences_infer(feat_df, seq_len=SEQ_LEN):
    """
    Build X sequences for inference only (no y required).
    Used for the 2026 period where horizon bars ahead may not exist.
    signal_dates[i] = last input bar date.
    """
    feat_arr = feat_df.values.astype(np.float32)
    dates    = feat_df.index
    n        = len(feat_arr)
    X_list, sig_dates = [], []
    for i in range(n - seq_len + 1):
        xb = feat_arr[i : i + seq_len]
        if np.any(np.isnan(xb)):
            continue
        X_list.append(xb)
        sig_dates.append(dates[i + seq_len - 1])
    if not X_list:
        empty = np.empty((0, seq_len, feat_arr.shape[1]), dtype=np.float32)
        return empty, pd.DatetimeIndex([])
    return (np.array(X_list, dtype=np.float32),
            pd.DatetimeIndex(sig_dates))


# ---------------------------------------------------------------------------
# Step 5: LSTM model
# ---------------------------------------------------------------------------

def build_lstm(n_features, seq_len=SEQ_LEN, horizon=HORIZON):
    """Two-layer stacked LSTM with dropout. Loss: MSE."""
    inp = keras.Input(shape=(seq_len, n_features), name="input_seq")
    x   = layers.LSTM(LSTM_UNITS_1, return_sequences=True,  name="lstm1")(inp)
    x   = layers.Dropout(DROPOUT,                           name="drop1")(x)
    x   = layers.LSTM(LSTM_UNITS_2, return_sequences=False, name="lstm2")(x)
    x   = layers.Dropout(DROPOUT,                           name="drop2")(x)
    out = layers.Dense(horizon,                             name="output")(x)
    model = keras.Model(inp, out, name="SpreadLSTM")
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
                  loss="mse")
    return model


# ---------------------------------------------------------------------------
# Step 6: Evaluation metrics
# ---------------------------------------------------------------------------

def evaluate_predictions(y_true, y_pred):
    """RMSE and MAE per horizon step and overall."""
    result = {}
    for h in range(HORIZON):
        err = y_true[:, h] - y_pred[:, h]
        result[f"RMSE_h{h+1}"] = float(np.sqrt(np.mean(err ** 2)))
        result[f"MAE_h{h+1}"]  = float(np.mean(np.abs(err)))
    all_err = (y_true - y_pred).ravel()
    result["RMSE_overall"] = float(np.sqrt(np.mean(all_err ** 2)))
    result["MAE_overall"]  = float(np.mean(np.abs(all_err)))
    return result


def persistence_baseline(X, y_true):
    """
    Naive persistence baseline: predict z_t (last input value) for all h.

    X shape: (n_samples, seq_len, n_features); z_std is feature 0.
    y_true shape: (n_samples, horizon).
    y_persist[i, h] = X[i, -1, 0] for all h (last observed z_std).
    Returns dict with Persist_RMSE_h{1..5}, Persist_MAE_h{1..5},
    Persist_RMSE_overall, Persist_MAE_overall.
    """
    y_persist = np.repeat(X[:, -1, 0:1], y_true.shape[1], axis=1)
    result = {}
    for h in range(y_true.shape[1]):
        err = y_true[:, h] - y_persist[:, h]
        result[f"Persist_RMSE_h{h+1}"] = float(np.sqrt(np.mean(err ** 2)))
        result[f"Persist_MAE_h{h+1}"]  = float(np.mean(np.abs(err)))
    all_err = (y_true - y_persist).ravel()
    result["Persist_RMSE_overall"] = float(np.sqrt(np.mean(all_err ** 2)))
    result["Persist_MAE_overall"]  = float(np.mean(np.abs(all_err)))
    return result


# ---------------------------------------------------------------------------
# Model artefact persistence (Task 3)
# ---------------------------------------------------------------------------

def save_artifacts(tag, model, hr, ic, mu, sigma):
    """
    Save final LSTM model and fitted OLS/scaler parameters to models/.

    Files written:
      models/{tag}_lstm_final.keras
      models/{tag}_ols_params.joblib   — {"hedge_ratio": hr, "intercept": ic}
      models/{tag}_scaler_params.joblib — {"mu": mu, "sigma": sigma}
    """
    model_path  = MODELS_DIR / f"{tag}_lstm_final.keras"
    ols_path    = MODELS_DIR / f"{tag}_ols_params.joblib"
    scaler_path = MODELS_DIR / f"{tag}_scaler_params.joblib"
    model.save(model_path)
    joblib.dump({"hedge_ratio": hr, "intercept": ic}, ols_path)
    joblib.dump({"mu": mu, "sigma": sigma}, scaler_path)
    print(f"  Artefacts saved:")
    print(f"    {model_path}")
    print(f"    {ols_path}")
    print(f"    {scaler_path}")
    return model_path, ols_path, scaler_path


def load_artifacts(tag):
    """
    Load saved artefacts for a pair tag (e.g. 'AMZN_META').
    Returns (model, ols_params, scaler_params) where:
      ols_params    = {"hedge_ratio": float, "intercept": float}
      scaler_params = {"mu": float, "sigma": float}
    Raises FileNotFoundError if any file is missing.
    """
    model_path  = MODELS_DIR / f"{tag}_lstm_final.keras"
    ols_path    = MODELS_DIR / f"{tag}_ols_params.joblib"
    scaler_path = MODELS_DIR / f"{tag}_scaler_params.joblib"
    for p in (model_path, ols_path, scaler_path):
        if not p.exists():
            raise FileNotFoundError(
                f"Artefact not found: {p}\n"
                "Run phase5_ml_spread.py first to generate saved models.")
    model       = keras.models.load_model(model_path)
    ols_params    = joblib.load(ols_path)
    scaler_params = joblib.load(scaler_path)
    return model, ols_params, scaler_params


# ---------------------------------------------------------------------------
# Step 6b: Per-trade win rate (uses extract_trade_log from phase3_strategy)
# ---------------------------------------------------------------------------

def compute_win_rate(bt_df, cost_per_leg=COST_PER_LEG):
    """
    Compute completed-trade win rate (%) and per-trade net P&L list.
    Uses phase3_strategy.extract_trade_log so the trade boundary logic is
    consistent with Phase 3b.  Incomplete trades (position open at end of
    data) are excluded from the win rate denominator.
    Returns (win_rate_pct, n_completed, trade_pnl_list).
    n_completed is the denominator used for win_rate_pct.
    """
    trade_df = extract_trade_log(bt_df, cost_per_leg)
    if trade_df.empty:
        return 0.0, 0, []
    # Only count completed trades: those with an Exit_Date before the last bar
    last_bar = bt_df.index[-1]
    completed = trade_df[pd.to_datetime(trade_df["Exit_Date"]) < last_bar]
    if completed.empty:
        # All trades still open — no completed trades to assess
        return float("nan"), 0, []
    trade_pnls = completed["Net_PnL"].tolist()
    wins       = sum(1 for p in trade_pnls if p > 0)
    win_rate   = wins / len(trade_pnls) * 100
    return round(win_rate, 1), len(trade_pnls), trade_pnls


# ---------------------------------------------------------------------------
# Step 7: Convergence signal
# ---------------------------------------------------------------------------

def build_convergence_signal(z_std_series, pred_df):
    """
    +1 (converging)  if mean(|pred h1..h5|) < |z_std at signal bar|
    -1 (diverging)   otherwise
    pred_df indexed by signal bar (last input bar = bar t-1 in execution conv.)
    """
    mean_abs_pred = pred_df.abs().mean(axis=1)
    abs_current   = z_std_series.reindex(pred_df.index).abs()
    conv          = np.where(mean_abs_pred < abs_current, 1, -1)
    return pd.Series(conv, index=pred_df.index, name="conv_signal", dtype=int)


# ---------------------------------------------------------------------------
# Step 8 helpers: Phase 4-style signal counting and LSTM-enhanced backtest
# ---------------------------------------------------------------------------

def count_signals(z_series, z_entry=Z_ENTRY_P4, z_exit=Z_EXIT_P4):
    """
    Phase 4 signal counter.
    Long  entry: z_prev < -z_entry    exit: z_prev >= z_exit
    Short entry: z_prev >  z_entry    exit: z_prev <= z_exit
    Returns (n_long, n_short, position_array, long_entry_list, short_entry_list).
    """
    z_arr  = z_series.values
    dates  = z_series.index
    n      = len(z_arr)
    pos    = 0
    position      = np.zeros(n, dtype=int)
    long_entries  = []
    short_entries = []
    for i in range(1, n):
        z_prev = z_arr[i - 1]
        if np.isnan(z_prev):
            position[i] = pos
            continue
        if pos == 1 and z_prev >= z_exit:
            pos = 0
        elif pos == -1 and z_prev <= z_exit:
            pos = 0
        if pos == 0:
            if z_prev < -z_entry:
                long_entries.append({
                    "Execution_Date": str(dates[i].date()),
                    "Signal_Date":    str(dates[i - 1].date()),
                    "Signal_Z":       round(float(z_prev), 4),
                })
                pos = 1
            elif z_prev > z_entry:
                short_entries.append({
                    "Execution_Date": str(dates[i].date()),
                    "Signal_Date":    str(dates[i - 1].date()),
                    "Signal_Z":       round(float(z_prev), 4),
                })
                pos = -1
        position[i] = pos
    return len(long_entries), len(short_entries), position, long_entries, short_entries


def backtest_phase4_style(z_series, spread_cl, spread_op, cost_per_leg,
                           z_entry=Z_ENTRY_P4, z_exit=Z_EXIT_P4):
    """
    Phase 4-style backtest: fixed-anchor z-score, no stop-loss.
    Entry/exit rules identical to Phase 4; open-price execution matches Phase 3b.
    Equivalent to calling phase3_strategy.backtest(..., z_stop=inf).
    """
    return backtest(z_series, spread_cl, z_entry, z_exit, float("inf"),
                    cost_per_leg, spread_op)


def backtest_lstm_enhanced(zscore_series, spread_cl, spread_op, conv_dict,
                            z_entry, z_exit, cost_per_leg):
    """
    Phase 4-style backtest (no stop-loss) with LSTM convergence gate on entries.
    conv_dict: {date → +1 (converging) | -1 (diverging)}, keyed by signal bar.
    Entry allowed only when Phase 4 signal fires AND conv_dict[signal_date] == 1.
    Exit logic is identical to Phase 4-style (z_exit, no stop-loss).
    """
    z_arr  = zscore_series.values
    sp_cl  = spread_cl.values
    sp_op  = spread_op.values
    dates  = zscore_series.index
    n      = len(z_arr)

    position   = np.zeros(n)
    daily_pnl  = np.zeros(n)
    trade_cost = np.zeros(n)
    pos        = 0

    for i in range(1, n):
        z_prev = z_arr[i - 1]
        if (np.isnan(z_prev) or np.isnan(sp_cl[i])
                or np.isnan(sp_cl[i - 1]) or np.isnan(sp_op[i])):
            continue

        prev_pos = pos

        # Exit (same as Phase 4)
        if pos == 1 and z_prev >= z_exit:
            trade_cost[i] += 2 * cost_per_leg
            pos = 0
        elif pos == -1 and z_prev <= z_exit:
            trade_cost[i] += 2 * cost_per_leg
            pos = 0

        # Entry gated by LSTM convergence (no stop-loss guard needed; no stop).
        # Default to 1 (allow) when no prediction exists for this signal date
        # so missing coverage never silently blocks a trade.
        if pos == 0:
            signal_date = dates[i - 1]
            if conv_dict.get(signal_date, 1) == 1:
                if z_prev < -z_entry:
                    trade_cost[i] += 2 * cost_per_leg
                    pos = 1
                elif z_prev > z_entry:
                    trade_cost[i] += 2 * cost_per_leg
                    pos = -1

        # P&L (open-price execution — identical to Phase 3b / Phase 4)
        if prev_pos == 0 and pos != 0:
            pnl = pos * (sp_cl[i] - sp_op[i])
        elif prev_pos != 0 and pos == 0:
            pnl = prev_pos * (sp_op[i] - sp_cl[i - 1])
        elif prev_pos != 0 and pos == prev_pos:
            pnl = pos * (sp_cl[i] - sp_cl[i - 1])
        elif prev_pos != 0 and pos != 0 and pos != prev_pos:
            pnl = (prev_pos * (sp_op[i] - sp_cl[i - 1])
                   + pos    * (sp_cl[i]  - sp_op[i]))
        else:
            pnl = 0.0

        position[i]  = pos
        daily_pnl[i] = pnl - trade_cost[i]

    return pd.DataFrame({
        "zscore":    z_arr,
        "spread":    sp_cl,
        "position":  position,
        "daily_pnl": daily_pnl,
        "trade_cost": trade_cost,
        "cum_pnl":   np.cumsum(daily_pnl),
    }, index=zscore_series.index)


def download_2026_prices(dep, indep, start, end_exclusive, refresh=False):
    """
    Load 2026 Close and Open prices for both tickers.
    Reads from data/raw/{ticker}_2026.csv if it exists; downloads and caches
    on first call or when refresh=True.
    Returns (close_df, open_df, log_close, log_open) or None on failure.
    """
    close_px, open_px = {}, {}
    for ticker in [dep, indep]:
        try:
            df = load_or_download_2026(ticker, start=start,
                                        end_exclusive=end_exclusive,
                                        refresh=refresh)
        except RuntimeError as e:
            print(f"    WARNING: {e}")
            return None
        source = "cache" if not refresh else "download"
        print(f"    {ticker}: {len(df)} trading days ({source})")
        close_px[ticker] = df["Close"]
        open_px[ticker]  = df["Open"]

    close_df = pd.DataFrame(close_px).dropna()
    open_df  = pd.DataFrame(open_px).reindex(close_df.index).dropna()
    common   = close_df.index.intersection(open_df.index)
    close_df = close_df.loc[common]
    open_df  = open_df.loc[common]
    return close_df, open_df, np.log(close_df), np.log(open_df)


# ---------------------------------------------------------------------------
# Per-pair runner
# ---------------------------------------------------------------------------

def run_phase5_pair(dep, indep, tests_passed):
    tag = f"{dep}_{indep}"
    print(f"\n{'='*65}")
    print(f"PAIR: {dep} / {indep}  [{tests_passed}]")
    print(f"{'='*65}")

    # -----------------------------------------------------------------------
    # Step 1: Load prices; re-estimate OLS on training data only (2018-2020)
    # -----------------------------------------------------------------------
    close_df, open_df, log_close, log_open = load_prices(dep, indep)
    print(f"  Price data: {log_close.index[0].date()} to "
          f"{log_close.index[-1].date()}  ({len(log_close)} trading days)")

    hr5, ic5, r2_5 = fit_ols(log_close, dep, indep, end_date=ML_OLS_END)
    print(f"\n  Step 1 – OLS re-estimated on 2018-{ML_OLS_END[:4]} training data:")
    print(f"    Hedge ratio : {hr5:.6f}   Intercept : {ic5:.6f}   R² : {r2_5:.4f}")

    raw_spread = build_raw_spread(log_close, dep, indep, hr5, ic5)

    # -----------------------------------------------------------------------
    # Step 2: Standardise using LSTM training set (2018-2020) only
    # -----------------------------------------------------------------------
    z_std, mu5, sigma5 = standardise(raw_spread, train_end=ML_TRAIN_END)
    print(f"\n  Step 2 – standardisation (scaler fit on 2018-{ML_TRAIN_END[:4]}):")
    print(f"    mean = {mu5:.6f}   std = {sigma5:.6f}")

    # -----------------------------------------------------------------------
    # Step 3: Features
    # -----------------------------------------------------------------------
    feat_df       = build_features(z_std)
    feat_df_clean = feat_df.dropna()
    FEATURE_NAMES = list(feat_df.columns)
    n_features    = len(FEATURE_NAMES)
    print(f"\n  Step 3 – features ({n_features}): {FEATURE_NAMES}")

    # -----------------------------------------------------------------------
    # Step 4: Sequences
    # -----------------------------------------------------------------------
    train_feat = feat_df_clean.loc[feat_df_clean.index <= ML_TRAIN_END]
    val_feat   = feat_df_clean.loc[
        (feat_df_clean.index >= ML_VAL_START) & (feat_df_clean.index <= ML_VAL_END)]
    test_feat  = feat_df_clean.loc[feat_df_clean.index >= ML_TEST_START]

    X_tr, y_tr, idx_tr = make_sequences(train_feat)
    X_va, y_va, idx_va = make_sequences(val_feat)
    X_te, y_te, idx_te = make_sequences(test_feat)

    print(f"\n  Step 4 – sequences (lookback={SEQ_LEN}d, horizon={HORIZON}d):")
    print(f"    ML train (≤{ML_TRAIN_END[:4]}): {X_tr.shape[0]} samples  "
          f"({len(train_feat)} feature rows)")
    print(f"    ML val  ({ML_VAL_END[:4]}):     {X_va.shape[0]} samples  "
          f"({len(val_feat)} feature rows)")
    print(f"    ML test (≥{ML_TEST_START[:4]}): {X_te.shape[0]} samples  "
          f"({len(test_feat)} feature rows)")

    # -----------------------------------------------------------------------
    # Step 5: Train LSTM
    # -----------------------------------------------------------------------
    print(f"\n  Step 5 – LSTM training "
          f"(units={LSTM_UNITS_1}/{LSTM_UNITS_2}, dropout={DROPOUT}, "
          f"lr={LEARNING_RATE}, patience={PATIENCE})...")
    tf.keras.utils.set_random_seed(SEED)
    model = build_lstm(n_features)
    es    = callbacks.EarlyStopping(monitor="val_loss", patience=PATIENCE,
                                     restore_best_weights=True, verbose=0)
    history = model.fit(
        X_tr, y_tr,
        validation_data=(X_va, y_va),
        epochs=MAX_EPOCHS, batch_size=BATCH_SIZE,
        callbacks=[es], verbose=0, shuffle=False,
    )
    epochs_run = len(history.history["loss"])
    best_val   = float(min(history.history["val_loss"]))
    val_rmse = float(best_val ** 0.5)
    print(f"  Training complete: {epochs_run} epochs  |  "
          f"best val MSE = {best_val:.6f}  val RMSE = {val_rmse:.4f}  "
          f"[on 2021 validation set, in z_std units]")

    # -----------------------------------------------------------------------
    # Step 6: Evaluate on 2022-2025 test set
    # -----------------------------------------------------------------------
    y_pred_te = model.predict(X_te, verbose=0)
    eval_dict   = evaluate_predictions(y_te, y_pred_te)
    persist_dict = persistence_baseline(X_te, y_te)
    print(f"\n  Step 6 – test evaluation on {ML_TEST_START}–{idx_te[-1].date()} "
          f"[out-of-sample, in z_std units]:")
    print(f"    {'Horizon':<10}  {'LSTM RMSE':>10}  {'LSTM MAE':>9}  "
          f"{'Persist RMSE':>13}  {'Persist MAE':>12}")
    for h in range(HORIZON):
        print(f"    h+{h+1:<7}  "
              f"{eval_dict[f'RMSE_h{h+1}']:>10.4f}  "
              f"{eval_dict[f'MAE_h{h+1}']:>9.4f}  "
              f"{persist_dict[f'Persist_RMSE_h{h+1}']:>13.4f}  "
              f"{persist_dict[f'Persist_MAE_h{h+1}']:>12.4f}")
    print(f"    {'Overall':<10}  "
          f"{eval_dict['RMSE_overall']:>10.4f}  "
          f"{eval_dict['MAE_overall']:>9.4f}  "
          f"{persist_dict['Persist_RMSE_overall']:>13.4f}  "
          f"{persist_dict['Persist_MAE_overall']:>12.4f}")

    # Save metrics CSV
    # Columns: Dataset distinguishes val (2021, early-stopping criterion) from
    # test (2022+, out-of-sample evaluation).  Val RMSE = sqrt(val MSE).
    metrics_rows = [{"Pair": f"{dep}/{indep}",
                     "Dataset": f"test ({ML_TEST_START}–{idx_te[-1].date()})",
                     "Horizon": f"h+{h+1}",
                     "RMSE": round(eval_dict[f"RMSE_h{h+1}"], 4),
                     "MAE":  round(eval_dict[f"MAE_h{h+1}"],  4),
                     "Persist_RMSE": round(persist_dict[f"Persist_RMSE_h{h+1}"], 4),
                     "Persist_MAE":  round(persist_dict[f"Persist_MAE_h{h+1}"],  4)}
                    for h in range(HORIZON)]
    metrics_rows.append({"Pair": f"{dep}/{indep}",
                         "Dataset": f"test ({ML_TEST_START}–{idx_te[-1].date()})",
                         "Horizon": "Overall",
                         "RMSE": round(eval_dict["RMSE_overall"], 4),
                         "MAE":  round(eval_dict["MAE_overall"],  4),
                         "Persist_RMSE": round(persist_dict["Persist_RMSE_overall"], 4),
                         "Persist_MAE":  round(persist_dict["Persist_MAE_overall"],  4)})
    metrics_rows.append({"Pair": f"{dep}/{indep}",
                         "Dataset": f"val ({ML_VAL_START}–{ML_VAL_END})",
                         "Horizon": "Overall",
                         "RMSE": round(val_rmse, 4),
                         "MAE":  None,
                         "Persist_RMSE": None,
                         "Persist_MAE":  None})
    pd.DataFrame(metrics_rows).to_csv(
        REPORTS_DIR / f"phase5_{tag}_lstm_metrics.csv", index=False)
    print(f"  Metrics CSV saved -> phase5_{tag}_lstm_metrics.csv")

    # Prediction chart (one subplot per horizon step)
    n_plot = min(252, len(y_te))
    fig, axes = plt.subplots(HORIZON, 1, figsize=(14, 2.8 * HORIZON), sharex=True)
    fig.suptitle(
        f"{dep}/{indep}  –  LSTM Predictions vs Actual (test period, "
        f"first {n_plot} samples)\n"
        f"Standardised OLS residual  |  Lookback={SEQ_LEN}d  Horizon={HORIZON}d  "
        f"|  RMSE={eval_dict['RMSE_overall']:.4f}",
        fontsize=10)
    for h in range(HORIZON):
        axes[h].plot(y_te[:n_plot, h],       color="steelblue", lw=0.8, label="Actual")
        axes[h].plot(y_pred_te[:n_plot, h],  color="darkorange", lw=0.8,
                     linestyle="--", label="LSTM prediction")
        axes[h].axhline(0, color="grey", lw=0.4, linestyle=":")
        axes[h].set_ylabel(f"h+{h+1}", fontsize=9)
        axes[h].tick_params(labelsize=8)
        if h == 0:
            axes[h].legend(fontsize=8, loc="upper right")
    axes[-1].set_xlabel("Sample index (test set, chronological)", fontsize=9)
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / f"phase5_{tag}_predictions.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Prediction chart saved -> phase5_{tag}_predictions.png")

    # -----------------------------------------------------------------------
    # Step 7: Convergence signal for 2022-2025 test period
    # -----------------------------------------------------------------------
    pred_cols  = {f"h{h+1}": y_pred_te[:, h] for h in range(HORIZON)}
    pred_df_te = pd.DataFrame(pred_cols, index=idx_te)
    conv_te    = build_convergence_signal(z_std, pred_df_te)
    conv_dict_te = conv_te.to_dict()
    conv_pct_te  = (conv_te == 1).mean() * 100
    print(f"\n  Step 7 – convergence signal ({ML_TEST_START} to "
          f"{idx_te[-1].date()}):")
    print(f"    Converging: {(conv_te==1).sum()} bars ({conv_pct_te:.1f}%)  |  "
          f"Diverging: {(conv_te==-1).sum()} bars ({100-conv_pct_te:.1f}%)")

    # -----------------------------------------------------------------------
    # Step 8a (supplementary): 2022-2025 historical holdout
    #   This uses the HISTORICAL model (OLS + scaler + LSTM all on 2018-2020).
    #   The baseline is Phase 4-style (fixed 2021 anchor, no stop-loss).
    #   Labelled clearly as supplementary holdout — NOT the definitive Phase 4
    #   comparison (see run_phase5_final_2026 for that).
    # -----------------------------------------------------------------------
    print(f"\n  Step 8a (supplementary) – 2022-2025 historical holdout "
          f"(Phase 4-style / 2021 anchor, not the actual Phase 4 comparison) ...")

    # Fixed anchor: mean/std from the validation year (2021), the year
    # immediately before the 2022-2025 test window
    anchor_slice = raw_spread.loc[P4_HOLDOUT_ANCHOR_START:P4_HOLDOUT_ANCHOR_END]
    p4h_mu       = float(anchor_slice.mean())
    p4h_sigma    = float(anchor_slice.std())
    z_p4h_full   = (raw_spread - p4h_mu) / p4h_sigma
    print(f"    Fixed anchor ({P4_HOLDOUT_ANCHOR_START[:4]}) — "
          f"mu={p4h_mu:.6f}  sigma={p4h_sigma:.6f}")

    # Build open spread using the same OLS (2018-2020)
    sp_open_full = build_open_spread(log_open, dep, indep, hr5, ic5)

    # Slice to test period
    test_mask  = z_p4h_full.index >= ML_TEST_START
    z_p4h_test = z_p4h_full.loc[test_mask]
    sp_cl_test = raw_spread.loc[test_mask]
    sp_op_test = sp_open_full.loc[test_mask]

    # Baseline: Phase 4-style (no stop-loss, fixed anchor)
    bt_p4h    = backtest_phase4_style(z_p4h_test, sp_cl_test, sp_op_test, COST_PER_LEG)
    m_p4h     = compute_metrics(bt_p4h, label=f"{tag} P4-style holdout baseline")

    # LSTM-enhanced: Phase 4-style z-score entries gated by LSTM
    bt_lstm_h = backtest_lstm_enhanced(
        z_p4h_test, sp_cl_test, sp_op_test, conv_dict_te,
        Z_ENTRY_P4, Z_EXIT_P4, COST_PER_LEG)
    m_lstm_h  = compute_metrics(bt_lstm_h, label=f"{tag} LSTM+P4-style holdout")

    # Print comparison
    print(f"\n    Supplementary comparison ({ML_TEST_START} to "
          f"{z_p4h_test.index[-1].date()}, Phase 4-style holdout):")
    COMPARE_KEYS = [
        ("Total_PnL",             "Total net P&L"),
        ("Ann_PnL",               "Annual P&L"),
        ("Ann_Vol",               "Annual volatility"),
        ("Sharpe_ratio",          "Sharpe ratio"),
        ("Sortino_ratio",         "Sortino ratio"),
        ("Max_drawdown",          "Max drawdown"),
        ("Calmar_ratio",          "Calmar ratio"),
        ("Positive_Day_Rate_pct", "Positive day rate %"),
        ("Num_trades",            "Number of trades"),
        ("Avg_trade_PnL",         "Avg P&L per trade"),
        ("Pct_in_market",         "Time in market %"),
    ]
    row_fmt = "    {:<28}  {:>12}  {:>14}"
    print(row_fmt.format("Metric", "P4-style baseline", "LSTM-enhanced"))
    print("    " + "-" * 56)
    for key, label in COMPARE_KEYS:
        vb = m_p4h.get(key); vl = m_lstm_h.get(key)
        print(row_fmt.format(label,
                             f"{vb:.4f}" if isinstance(vb, float) else str(vb),
                             f"{vl:.4f}" if isinstance(vl, float) else str(vl)))

    # Save holdout comparison CSV
    cmp_rows = [{"Pair": f"{dep}/{indep}", "Metric": key,
                 "Phase4_style_baseline": m_p4h.get(key),
                 "LSTM_enhanced":         m_lstm_h.get(key)}
                for key, _ in COMPARE_KEYS]
    pd.DataFrame(cmp_rows).to_csv(
        REPORTS_DIR / f"phase5_{tag}_holdout_comparison.csv", index=False)
    print(f"\n    CSV saved -> phase5_{tag}_holdout_comparison.csv")

    # Holdout comparison chart
    fig2 = plt.figure(figsize=(14, 10))
    gs2  = gridspec.GridSpec(3, 1, figure=fig2,
                              height_ratios=[2.5, 2.0, 1.0], hspace=0.40)
    ax0  = fig2.add_subplot(gs2[0])
    ax0.plot(bt_p4h.index,   bt_p4h["cum_pnl"],   color="steelblue", lw=1.1,
             label=(f"Phase 4-style baseline: Sharpe={m_p4h['Sharpe_ratio']:.2f}  "
                    f"Trades={m_p4h['Num_trades']}  P&L={m_p4h['Total_PnL']:.4f}"))
    ax0.plot(bt_lstm_h.index, bt_lstm_h["cum_pnl"], color="darkorange", lw=1.1, ls="--",
             label=(f"LSTM-enhanced: Sharpe={m_lstm_h['Sharpe_ratio']:.2f}  "
                    f"Trades={m_lstm_h['Num_trades']}  P&L={m_lstm_h['Total_PnL']:.4f}"))
    ax0.axhline(0, color="grey", lw=0.5, ls=":")
    ax0.set_ylabel("Cumulative net P&L (log-price units)", fontsize=9)
    ax0.set_title(
        f"{dep}/{indep}  –  Supplementary Holdout: Phase 4-style (2021 anchor) "
        f"vs LSTM-Enhanced  (2022-2025)\n"
        f"Historical model: OLS + scaler + LSTM all on 2018-2020  |  "
        f"Fixed anchor: 2021 mean/std  |  No stop-loss  |  "
        f"Cost={COST_PER_LEG*10_000:.0f} bps/leg",
        fontsize=9)
    ax0.legend(fontsize=8)
    ax1 = fig2.add_subplot(gs2[1])
    ax1.plot(z_p4h_test.index, z_p4h_test.values, color="dimgray", lw=0.7,
             label="Phase 4-style z-score (fixed 2021 anchor)")
    ax1.axhline( Z_ENTRY_P4, color="firebrick",   lw=0.8, ls="--", label=f"±{Z_ENTRY_P4} entry")
    ax1.axhline(-Z_ENTRY_P4, color="forestgreen", lw=0.8, ls="--")
    ax1.axhline(Z_EXIT_P4,   color="black",       lw=0.5, ls=":",  label="0 exit")
    ax1.fill_between(bt_lstm_h.index, bt_lstm_h["position"], 0,
                     where=(bt_lstm_h["position"] == 1),
                     alpha=0.15, color="forestgreen", label="LSTM long")
    ax1.fill_between(bt_lstm_h.index, bt_lstm_h["position"], 0,
                     where=(bt_lstm_h["position"] == -1),
                     alpha=0.15, color="firebrick", label="LSTM short")
    ax1.set_ylabel("Z-score", fontsize=9)
    ax1.legend(fontsize=7, ncol=3)
    ax2 = fig2.add_subplot(gs2[2], sharex=ax1)
    ax2.step(conv_te.index, conv_te.values, where="post", color="purple", lw=0.8)
    ax2.fill_between(conv_te.index, conv_te.values, 0,
                     where=(conv_te == 1),  alpha=0.2, color="mediumslateblue", label="Converging")
    ax2.fill_between(conv_te.index, conv_te.values, 0,
                     where=(conv_te == -1), alpha=0.15, color="sienna", label="Diverging")
    ax2.set_yticks([-1, 1])
    ax2.set_yticklabels(["Diverging (−1)", "Converging (+1)"], fontsize=8)
    ax2.set_ylabel("LSTM signal", fontsize=9)
    ax2.set_title(f"LSTM convergence filter ({conv_pct_te:.1f}% converging)",
                  fontsize=9)
    ax2.legend(fontsize=8, loc="upper right")
    plt.savefig(CHARTS_DIR / f"phase5_{tag}_holdout_comparison.png",
                dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Chart saved -> phase5_{tag}_holdout_comparison.png")
    print(f"  [Definitive 2026 comparison is in run_phase5_final_2026 "
          f"using the 2018-2024 model.]")

    # Return cross-pair summary row (historical experiment only)
    return {
        "Pair":               f"{dep}/{indep}",
        "Tests_passed":       tests_passed,
        "OLS_end":            ML_OLS_END,
        "Scaler_end":         ML_TRAIN_END,
        "LSTM_epochs":        epochs_run,
        "Val_RMSE_2021":      round(val_rmse, 4),
        "Test_RMSE_2022_2025": round(eval_dict["RMSE_overall"], 4),
        "MAE_2022_2025":       round(eval_dict["MAE_overall"],  4),
        "Conv_pct_2022_2025": round(conv_pct_te, 1),
        "P4style_Sharpe":     m_p4h.get("Sharpe_ratio"),
        "P4style_Trades":     m_p4h.get("Num_trades"),
        "P4style_PnL":        m_p4h.get("Total_PnL"),
        "LSTM_Sharpe":        m_lstm_h.get("Sharpe_ratio"),
        "LSTM_Trades":        m_lstm_h.get("Num_trades"),
        "LSTM_PnL":           m_lstm_h.get("Total_PnL"),
    }


# ---------------------------------------------------------------------------
# Step 8b: Final 2026 evaluation — separate LSTM model trained on 2018-2024
# ---------------------------------------------------------------------------

def run_phase5_final_2026(dep, indep, tests_passed, refresh=False):
    """
    Train a fresh LSTM on 2018-2024 (val 2025) and evaluate on 2026.
    All pre-processing (OLS, scaler, feature construction) uses only data
    up to 2024-12-31.  The 2025 validation year is used for early stopping
    only.  The 2026 period (2026-01-01 to 2026-07-31) is the unseen test.

    Step 8 comparison:
      Phase 4 baseline: OLS 2018-2025, fixed 2025 anchor, entry ±2, exit 0.
      LSTM-enhanced: same Phase 4 signals, entries gated by convergence filter
        from this final LSTM.  The filter rule (|mean_pred| < |z_current|) is
        fixed — not tuned on 2026 data.
    """
    tag = f"{dep}_{indep}"
    print(f"\n{'='*65}")
    print(f"FINAL 2026 EVALUATION: {dep} / {indep}  [{tests_passed}]")
    print(f"  Final model: OLS + scaler + LSTM all fit on 2018-{FINAL_TRAIN_END[:4]}")
    print(f"  Validation: {FINAL_VAL_START}–{FINAL_VAL_END}  "
          f"(early stopping only, not used for fitting)")
    print(f"  Unseen test: {P4_TEST_START}–{P4_TEST_END}")
    print(f"{'='*65}")

    # ------------------------------------------------------------------
    # 1. Load prices and fit OLS on 2018-2024 only
    # ------------------------------------------------------------------
    close_df, open_df, log_close, log_open = load_prices(dep, indep)
    # Restrict to pre-2026 training data for all fitting
    log_close_pre26 = log_close.loc[log_close.index <= FINAL_VAL_END]
    log_open_pre26  = log_open.loc[log_open.index <= FINAL_VAL_END]

    hr_f, ic_f, r2_f = fit_ols(log_close_pre26, dep, indep,
                                 end_date=FINAL_TRAIN_END)
    print(f"\n  OLS (2018-{FINAL_TRAIN_END[:4]}): "
          f"HR={hr_f:.6f}  IC={ic_f:.6f}  R²={r2_f:.4f}")

    raw_spread_pre26 = build_raw_spread(log_close_pre26, dep, indep, hr_f, ic_f)

    # ------------------------------------------------------------------
    # 2. Scaler on 2018-2024 only
    # ------------------------------------------------------------------
    z_pre26, mu_f, sigma_f = standardise(raw_spread_pre26,
                                          train_end=FINAL_TRAIN_END)
    print(f"  Scaler (2018-{FINAL_TRAIN_END[:4]}): "
          f"mean={mu_f:.6f}  sigma={sigma_f:.6f}")

    # ------------------------------------------------------------------
    # 3. Features
    # ------------------------------------------------------------------
    feat_pre26  = build_features(z_pre26).dropna()
    n_features  = feat_pre26.shape[1]

    # ------------------------------------------------------------------
    # 4. Sequences
    # ------------------------------------------------------------------
    train_f = feat_pre26.loc[feat_pre26.index <= FINAL_TRAIN_END]
    val_f   = feat_pre26.loc[(feat_pre26.index >= FINAL_VAL_START) &
                               (feat_pre26.index <= FINAL_VAL_END)]

    X_tr_f, y_tr_f, _ = make_sequences(train_f)
    X_va_f, y_va_f, _ = make_sequences(val_f)
    print(f"\n  Sequences — train: {X_tr_f.shape[0]}  val: {X_va_f.shape[0]}")

    # ------------------------------------------------------------------
    # 5. Train LSTM
    # ------------------------------------------------------------------
    print(f"  Training final LSTM (patience={PATIENCE}) ...")
    tf.keras.utils.set_random_seed(SEED)
    model_f = build_lstm(n_features)
    es_f    = callbacks.EarlyStopping(monitor="val_loss", patience=PATIENCE,
                                        restore_best_weights=True, verbose=0)
    hist_f  = model_f.fit(
        X_tr_f, y_tr_f,
        validation_data=(X_va_f, y_va_f),
        epochs=MAX_EPOCHS, batch_size=BATCH_SIZE,
        callbacks=[es_f], verbose=0, shuffle=False,
    )
    epochs_f   = len(hist_f.history["loss"])
    best_val_f = float(min(hist_f.history["val_loss"]))
    val_rmse_f = best_val_f ** 0.5
    print(f"  Done: {epochs_f} epochs  |  "
          f"val MSE={best_val_f:.6f}  val RMSE={val_rmse_f:.4f} "
          f"[2025 validation, z_std units]")

    # ------------------------------------------------------------------
    # 5b. Save artefacts and verify round-trip load
    # ------------------------------------------------------------------
    save_artifacts(tag, model_f, hr_f, ic_f, mu_f, sigma_f)

    # Verify: load back and compare predictions on one val sample
    _model_chk, _ols_chk, _scl_chk = load_artifacts(tag)
    _pred_orig = model_f.predict(X_va_f[:1], verbose=0)
    _pred_load = _model_chk.predict(X_va_f[:1], verbose=0)
    _max_diff  = float(np.max(np.abs(_pred_orig - _pred_load)))
    assert _max_diff < 1e-5, f"Load round-trip mismatch: max diff={_max_diff}"
    assert _ols_chk["hedge_ratio"] == hr_f and _ols_chk["intercept"] == ic_f
    assert _scl_chk["mu"] == mu_f and _scl_chk["sigma"] == sigma_f
    print(f"  Load round-trip verified (max pred diff={_max_diff:.2e})")
    del _model_chk, _ols_chk, _scl_chk, _pred_orig, _pred_load

    # ------------------------------------------------------------------
    # 6. Download 2026 data and evaluate LSTM on 2026
    # ------------------------------------------------------------------
    print(f"\n  Downloading 2026 data ...")
    data_2026 = download_2026_prices(dep, indep, P4_TEST_START, P4_TEST_END_EX,
                                      refresh=refresh)
    if data_2026 is None:
        print("  ERROR: 2026 data unavailable — aborting final 2026 evaluation.")
        return None

    close_2026, open_2026, log_close_2026, log_open_2026 = data_2026
    n_days_2026 = len(close_2026)
    print(f"  {n_days_2026} trading days: "
          f"{close_2026.index[0].date()} to {close_2026.index[-1].date()}")

    # Build Phase 5 spread and z_std for 2026 using the FINAL model's OLS/scaler
    sp_2026_f   = build_raw_spread(log_close_2026, dep, indep, hr_f, ic_f)
    z_2026_f    = (sp_2026_f - mu_f) / sigma_f

    # Warm-up: prepend ROLLING_STD_WIN + SEQ_LEN - 1 pre-2026 z_std bars.
    # This ensures:
    #   (a) rolling/lag features are fully populated at the very first 2026 bar
    #       (ROLLING_STD_WIN - 1 pre-2026 bars needed for roll_std to be valid),
    #   (b) there are SEQ_LEN valid pre-2026 feature rows so the LSTM can form
    #       a complete input window ending at the first 2026 bar — giving
    #       predictions from day 1 of 2026 onward.
    # Sequences are built on the FULL combined feature frame; signal dates are
    # filtered to >= P4_TEST_START only AFTER sequence construction.
    WARMUP_LEN = ROLLING_STD_WIN + SEQ_LEN - 1   # 20 + 20 - 1 = 39 bars
    warmup_z   = z_pre26.iloc[-WARMUP_LEN:]
    z_combo    = pd.concat([warmup_z, z_2026_f])
    feat_combo = build_features(z_combo).dropna()

    # Sanity check: at least HORIZON 2026 rows required for y labels
    feat_2026_check = feat_combo.loc[feat_combo.index >= P4_TEST_START]
    if len(feat_2026_check) < HORIZON:
        print("  WARNING: insufficient 2026 feature rows for sequences.")
        return None

    # RMSE/MAE evaluation: build sequences on full feat_combo, then keep only
    # those whose signal date falls inside 2026.
    X_te_all, y_te_all, idx_te_all = make_sequences(feat_combo)
    mask_te  = idx_te_all >= pd.Timestamp(P4_TEST_START)
    X_te_f   = X_te_all[mask_te]
    y_te_f   = y_te_all[mask_te]
    idx_te_f = idx_te_all[mask_te]
    y_pred_f     = model_f.predict(X_te_f, verbose=0)
    eval_f       = evaluate_predictions(y_te_f, y_pred_f)
    persist_f    = persistence_baseline(X_te_f, y_te_f)

    test_rmse_f  = eval_f["RMSE_overall"]
    test_mae_f   = eval_f["MAE_overall"]
    val_rmse_f_r = round(val_rmse_f, 4)

    print(f"\n  2026 forecast evaluation (in z_std units):")
    print(f"    Val RMSE (2025, early-stopping):   {val_rmse_f_r}")
    print(f"    {'Horizon':<10}  {'LSTM RMSE':>10}  {'LSTM MAE':>9}  "
          f"{'Persist RMSE':>13}  {'Persist MAE':>12}")
    for h in range(HORIZON):
        print(f"    h+{h+1:<7}  "
              f"{eval_f[f'RMSE_h{h+1}']:>10.4f}  "
              f"{eval_f[f'MAE_h{h+1}']:>9.4f}  "
              f"{persist_f[f'Persist_RMSE_h{h+1}']:>13.4f}  "
              f"{persist_f[f'Persist_MAE_h{h+1}']:>12.4f}")
    print(f"    {'Overall':<10}  "
          f"{test_rmse_f:>10.4f}  "
          f"{test_mae_f:>9.4f}  "
          f"{persist_f['Persist_RMSE_overall']:>13.4f}  "
          f"{persist_f['Persist_MAE_overall']:>12.4f}")

    # Save final LSTM metrics CSV (distinguishes 2025-val vs 2026-test)
    fin_metrics_rows = [
        {"Pair": f"{dep}/{indep}",
         "Dataset": f"val ({FINAL_VAL_START}–{FINAL_VAL_END})",
         "Horizon": "Overall", "RMSE": val_rmse_f_r, "MAE": None,
         "Persist_RMSE": None, "Persist_MAE": None},
    ] + [
        {"Pair": f"{dep}/{indep}",
         "Dataset": f"test ({P4_TEST_START}–{P4_TEST_END})",
         "Horizon": f"h+{h+1}",
         "RMSE": round(eval_f[f"RMSE_h{h+1}"], 4),
         "MAE":  round(eval_f[f"MAE_h{h+1}"],  4),
         "Persist_RMSE": round(persist_f[f"Persist_RMSE_h{h+1}"], 4),
         "Persist_MAE":  round(persist_f[f"Persist_MAE_h{h+1}"],  4)}
        for h in range(HORIZON)
    ] + [
        {"Pair": f"{dep}/{indep}",
         "Dataset": f"test ({P4_TEST_START}–{P4_TEST_END})",
         "Horizon": "Overall",
         "RMSE": round(test_rmse_f, 4), "MAE": round(test_mae_f, 4),
         "Persist_RMSE": round(persist_f["Persist_RMSE_overall"], 4),
         "Persist_MAE":  round(persist_f["Persist_MAE_overall"],  4)},
    ]
    pd.DataFrame(fin_metrics_rows).to_csv(
        REPORTS_DIR / f"phase5_{tag}_final_lstm_metrics.csv", index=False)
    print(f"  Metrics CSV saved -> phase5_{tag}_final_lstm_metrics.csv")

    # Prediction chart (2026)
    n_plot = len(y_te_f)
    fig0, axes0 = plt.subplots(HORIZON, 1, figsize=(14, 2.8 * HORIZON),
                                sharex=True)
    fig0.suptitle(
        f"{dep}/{indep}  –  Final LSTM Predictions vs Actual (2026 test period)\n"
        f"Model trained 2018-{FINAL_TRAIN_END[:4]}  |  Val RMSE={val_rmse_f_r} (2025)  "
        f"|  Test RMSE={test_rmse_f:.4f} (2026, z_std units)",
        fontsize=10)
    for h in range(HORIZON):
        axes0[h].plot(y_te_f[:n_plot, h],    color="steelblue", lw=0.8, label="Actual")
        axes0[h].plot(y_pred_f[:n_plot, h],  color="darkorange", lw=0.8,
                      linestyle="--", label="LSTM prediction")
        axes0[h].axhline(0, color="grey", lw=0.4, linestyle=":")
        axes0[h].set_ylabel(f"h+{h+1}", fontsize=9)
        axes0[h].tick_params(labelsize=8)
        if h == 0:
            axes0[h].legend(fontsize=8, loc="upper right")
    axes0[-1].set_xlabel("Sample index (2026 test set, chronological)", fontsize=9)
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / f"phase5_{tag}_final_predictions.png",
                dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Chart saved -> phase5_{tag}_final_predictions.png")

    # ------------------------------------------------------------------
    # 7. Convergence signal for 2026 (inference only — no y labels needed)
    #    Build on full feat_combo (same warm-up rationale as RMSE eval above),
    #    then filter signal dates to 2026 after sequence construction.
    # ------------------------------------------------------------------
    X_inf_all, idx_inf_all = make_sequences_infer(feat_combo)
    mask_inf   = idx_inf_all >= pd.Timestamp(P4_TEST_START)
    X_inf_f    = X_inf_all[mask_inf]
    idx_inf_f  = idx_inf_all[mask_inf]
    y_pred_inf = model_f.predict(X_inf_f, verbose=0)
    pred_df_inf        = pd.DataFrame(
        {f"h{h+1}": y_pred_inf[:, h] for h in range(HORIZON)},
        index=idx_inf_f)
    conv_f      = build_convergence_signal(z_combo, pred_df_inf)
    conv_dict_f = conv_f.to_dict()
    conv_pct_f  = (conv_f == 1).mean() * 100
    print(f"\n  LSTM convergence (2026): "
          f"{(conv_f==1).sum()} converging ({conv_pct_f:.1f}%)  |  "
          f"{(conv_f==-1).sum()} diverging ({100-conv_pct_f:.1f}%)")
    print(f"  [Convergence rule fixed before 2026 evaluation — no tuning on 2026]")

    # ------------------------------------------------------------------
    # 8. Phase 4 baseline — OLS 2018-2025, fixed 2025 anchor, entry ±2, exit 0
    # ------------------------------------------------------------------
    hr_p4, ic_p4, _ = fit_ols(log_close_pre26, dep, indep, end_date=P4_OLS_END)
    sp_p4_train      = build_raw_spread(log_close_pre26, dep, indep, hr_p4, ic_p4)
    norm_slice        = sp_p4_train.loc[P4_NORM_START:P4_NORM_END]
    p4_mu    = float(norm_slice.mean())
    p4_sigma = float(norm_slice.std())
    print(f"\n  Phase 4 OLS (2018-{P4_OLS_END[:4]}): "
          f"HR={hr_p4:.6f}  IC={ic_p4:.6f}")
    print(f"  Phase 4 anchor ({P4_NORM_START[:4]}): "
          f"mu={p4_mu:.6f}  sigma={p4_sigma:.6f}")

    sp_2026_p4 = build_raw_spread(log_close_2026, dep, indep, hr_p4, ic_p4)
    z_2026_p4  = (sp_2026_p4 - p4_mu) / p4_sigma
    sp_op_p4   = build_raw_spread(log_open_2026, dep, indep, hr_p4, ic_p4)

    print(f"  Phase 4 z-score (2026): mean={z_2026_p4.mean():.4f}  "
          f"std={z_2026_p4.std():.4f}  "
          f"min={z_2026_p4.min():.4f}  max={z_2026_p4.max():.4f}")

    bt_p4 = backtest_phase4_style(z_2026_p4, sp_2026_p4, sp_op_p4, COST_PER_LEG)
    m_p4  = compute_metrics(bt_p4, label=f"{tag} P4 2026")
    wr_p4, n_comp_p4, tpnl_p4 = compute_win_rate(bt_p4, COST_PER_LEG)

    # Phase 4 signal breakdown
    n_long_p4, n_short_p4, _, long_p4, short_p4 = count_signals(z_2026_p4)

    # Verify every Phase 4 candidate signal has an LSTM prediction
    p4_signal_dates = [pd.Timestamp(s["Signal_Date"])
                       for s in long_p4 + short_p4]
    missing_pred = [d for d in p4_signal_dates if d not in conv_dict_f]
    if missing_pred:
        print(f"  WARNING: {len(missing_pred)} Phase 4 signal date(s) lack an "
              f"LSTM prediction: {[str(d.date()) for d in missing_pred]}")
    else:
        n_p4_sigs = len(p4_signal_dates)
        print(f"  Verification: all {n_p4_sigs} Phase 4 signal date(s) have "
              f"LSTM predictions" if n_p4_sigs else
              f"  Verification: no Phase 4 signals fired in 2026")

    # ------------------------------------------------------------------
    # 9. LSTM-enhanced — same Phase 4 z-score signals, entries gated by
    #    the final LSTM convergence filter
    # ------------------------------------------------------------------
    bt_lstm = backtest_lstm_enhanced(
        z_2026_p4, sp_2026_p4, sp_op_p4, conv_dict_f,
        Z_ENTRY_P4, Z_EXIT_P4, COST_PER_LEG)
    m_lstm  = compute_metrics(bt_lstm, label=f"{tag} LSTM+P4 2026")
    wr_lstm, n_comp_lstm, tpnl_lstm = compute_win_rate(bt_lstm, COST_PER_LEG)

    n_long_lstm, n_short_lstm, _, long_lstm, short_lstm = \
        count_signals_lstm_gated(z_2026_p4, conv_dict_f)

    # ------------------------------------------------------------------
    # 10. Report comparison
    # ------------------------------------------------------------------
    REPORT_KEYS = [
        ("Num_trades",            "Entries (incl. open)"),
        ("Total_PnL",             "Total net P&L"),
        ("Ann_PnL",               "Annual P&L"),
        ("Sharpe_ratio",          "Sharpe ratio"),
        ("Sortino_ratio",         "Sortino ratio"),
        ("Max_drawdown",          "Max drawdown"),
        ("Calmar_ratio",          "Calmar ratio"),
        ("Avg_trade_PnL",         "Avg P&L per trade"),
        ("Pct_in_market",         "Time in market %"),
        ("Positive_Day_Rate_pct", "Positive day rate %"),
    ]

    print(f"\n  Step 8 – Phase 4 baseline vs LSTM-enhanced (2026):")
    row_fmt = "    {:<28}  {:>16}  {:>16}"
    print(row_fmt.format("Metric", "Phase 4 baseline", "LSTM-enhanced"))
    print("    " + "-" * 62)
    for key, label in REPORT_KEYS:
        vb = m_p4.get(key); vl = m_lstm.get(key)
        fmt_b = f"{vb:.4f}" if isinstance(vb, float) else str(vb)
        fmt_l = f"{vl:.4f}" if isinstance(vl, float) else str(vl)
        print(row_fmt.format(label, fmt_b, fmt_l))
    # Win rate: show "X% (n_completed/n_entries)" so the denominator is always visible
    n_entries_p4   = m_p4.get("Num_trades",   0)
    n_entries_lstm = m_lstm.get("Num_trades", 0)
    def _wr_str(wr, n_comp, n_entries):
        if n_comp == 0 or (isinstance(wr, float) and np.isnan(wr)):
            return f"n/a ({n_comp}/{n_entries} completed)"
        return f"{wr:.1f}% ({n_comp}/{n_entries} completed)"
    wr_p4_s   = _wr_str(wr_p4,   n_comp_p4,   n_entries_p4)
    wr_lstm_s = _wr_str(wr_lstm, n_comp_lstm, n_entries_lstm)
    print(row_fmt.format("Win rate (completed only)", wr_p4_s, wr_lstm_s))
    print()
    print(f"    Signal counts:")
    blocked = (n_long_p4 + n_short_p4) - (n_long_lstm + n_short_lstm)
    print(f"      Phase 4 : {n_long_p4} long + {n_short_p4} short = "
          f"{n_long_p4+n_short_p4} total")
    print(f"      LSTM    : {n_long_lstm} long + {n_short_lstm} short = "
          f"{n_long_lstm+n_short_lstm} total")
    if (n_long_p4 + n_short_p4) > 0:
        print(f"      Blocked : {blocked} of {n_long_p4+n_short_p4} "
              f"({blocked/(n_long_p4+n_short_p4)*100:.0f}% filtered)")

    # ------------------------------------------------------------------
    # 11. Save comparison CSV
    # ------------------------------------------------------------------
    cmp_rows = [
        {"Pair": f"{dep}/{indep}", "Metric": key,
         "Phase4_baseline": m_p4.get(key),
         "LSTM_enhanced":   m_lstm.get(key)}
        for key, _ in REPORT_KEYS
    ]
    cmp_rows += [
        {"Pair": f"{dep}/{indep}", "Metric": "Completed_trades",
         "Phase4_baseline": n_comp_p4,
         "LSTM_enhanced":   n_comp_lstm},
        {"Pair": f"{dep}/{indep}", "Metric": "Win_rate_pct",
         "Phase4_baseline": round(wr_p4, 2) if not np.isnan(wr_p4) else None,
         "LSTM_enhanced":   round(wr_lstm, 2) if not np.isnan(wr_lstm) else None},
        {"Pair": f"{dep}/{indep}", "Metric": "Long_signals",
         "Phase4_baseline": n_long_p4,  "LSTM_enhanced": n_long_lstm},
        {"Pair": f"{dep}/{indep}", "Metric": "Short_signals",
         "Phase4_baseline": n_short_p4, "LSTM_enhanced": n_short_lstm},
        {"Pair": f"{dep}/{indep}", "Metric": "Total_signals",
         "Phase4_baseline": n_long_p4 + n_short_p4,
         "LSTM_enhanced":   n_long_lstm + n_short_lstm},
        {"Pair": f"{dep}/{indep}", "Metric": "LSTM_conv_pct",
         "Phase4_baseline": None, "LSTM_enhanced": round(conv_pct_f, 1)},
    ]
    cmp_df = pd.DataFrame(cmp_rows)
    cmp_df.to_csv(REPORTS_DIR / f"phase5_{tag}_2026_comparison.csv", index=False)
    print(f"\n  CSV saved -> phase5_{tag}_2026_comparison.csv")

    # Trade log for both strategies
    tlog_p4   = extract_trade_log(bt_p4,   COST_PER_LEG)
    tlog_lstm = extract_trade_log(bt_lstm,  COST_PER_LEG)
    if not tlog_p4.empty:
        tlog_p4["Strategy"] = "Phase4_baseline"
    if not tlog_lstm.empty:
        tlog_lstm["Strategy"] = "LSTM_enhanced"
    tlog_all = pd.concat([tlog_p4, tlog_lstm], ignore_index=True)
    if not tlog_all.empty:
        tlog_all = tlog_all.sort_values(["Strategy", "Execution_Date"])
    tlog_all.to_csv(REPORTS_DIR / f"phase5_{tag}_2026_trade_log.csv", index=False)
    print(f"  Trade log saved -> phase5_{tag}_2026_trade_log.csv")

    # ------------------------------------------------------------------
    # 12. Chart: 3-panel (cumulative P&L / z-score + positions / LSTM filter)
    # ------------------------------------------------------------------
    fig3 = plt.figure(figsize=(14, 10))
    gs3  = gridspec.GridSpec(3, 1, figure=fig3,
                              height_ratios=[2.5, 2.0, 1.0], hspace=0.42)
    ax3a = fig3.add_subplot(gs3[0])
    ax3a.plot(bt_p4.index,   bt_p4["cum_pnl"],  color="steelblue", lw=1.2,
              label=(f"Phase 4 baseline: Sharpe={m_p4['Sharpe_ratio']:.2f}  "
                     f"Entries={m_p4['Num_trades']}  "
                     f"WinRate(completed)={wr_p4_s}  "
                     f"P&L={m_p4['Total_PnL']:.4f}"))
    ax3a.plot(bt_lstm.index, bt_lstm["cum_pnl"], color="darkorange", lw=1.2, ls="--",
              label=(f"LSTM-enhanced: Sharpe={m_lstm['Sharpe_ratio']:.2f}  "
                     f"Entries={m_lstm['Num_trades']}  "
                     f"WinRate(completed)={wr_lstm_s}  "
                     f"P&L={m_lstm['Total_PnL']:.4f}"))
    ax3a.axhline(0, color="grey", lw=0.5, ls=":")
    ax3a.set_ylabel("Cumulative net P&L (log-price units)", fontsize=9)
    ax3a.set_title(
        f"{dep}/{indep}  –  Phase 4 Baseline vs LSTM-Enhanced  "
        f"({P4_TEST_START}–{P4_TEST_END})\n"
        f"Final LSTM: OLS+scaler+train on 2018-{FINAL_TRAIN_END[:4]}  |  "
        f"Val 2025 RMSE={val_rmse_f_r}  |  Test 2026 RMSE={test_rmse_f:.4f}  |  "
        f"Phase 4: OLS 2018-2025, 2025 anchor",
        fontsize=9)
    ax3a.legend(fontsize=8)

    ax3b = fig3.add_subplot(gs3[1])
    ax3b.plot(z_2026_p4.index, z_2026_p4.values,
              color="dimgray", lw=0.8, label="Phase 4 z-score (2025 anchor)")
    ax3b.axhline( Z_ENTRY_P4, color="firebrick",   lw=0.8, ls="--",
                  label=f"±{Z_ENTRY_P4} entry")
    ax3b.axhline(-Z_ENTRY_P4, color="forestgreen", lw=0.8, ls="--")
    ax3b.axhline(Z_EXIT_P4,   color="black",       lw=0.5, ls=":",
                 label="0 exit")
    ax3b.fill_between(bt_lstm.index, bt_lstm["position"], 0,
                      where=(bt_lstm["position"] == 1),
                      alpha=0.15, color="forestgreen", label="LSTM long")
    ax3b.fill_between(bt_lstm.index, bt_lstm["position"], 0,
                      where=(bt_lstm["position"] == -1),
                      alpha=0.15, color="firebrick", label="LSTM short")
    ax3b.fill_between(bt_p4.index, bt_p4["position"], 0,
                      where=(bt_p4["position"] != bt_lstm["position"]) &
                             (bt_p4["position"] != 0),
                      alpha=0.08, color="navy",
                      label="P4-only positions (LSTM blocked)")
    ax3b.set_ylabel("Z-score", fontsize=9)
    ax3b.set_title(
        f"Phase 4: {n_long_p4}L / {n_short_p4}S signals  |  "
        f"LSTM allowed: {n_long_lstm}L / {n_short_lstm}S  |  "
        f"Blocked: {blocked}",
        fontsize=9)
    ax3b.legend(fontsize=7, ncol=3)

    ax3c = fig3.add_subplot(gs3[2], sharex=ax3b)
    if len(conv_f) > 0:
        ax3c.step(conv_f.index, conv_f.values, where="post",
                  color="purple", lw=0.8)
        ax3c.fill_between(conv_f.index, conv_f.values, 0,
                          where=(conv_f == 1),
                          alpha=0.2, color="mediumslateblue", label="Converging")
        ax3c.fill_between(conv_f.index, conv_f.values, 0,
                          where=(conv_f == -1),
                          alpha=0.15, color="sienna", label="Diverging")
    ax3c.set_yticks([-1, 1])
    ax3c.set_yticklabels(["Diverging (−1)", "Converging (+1)"], fontsize=8)
    ax3c.set_ylabel("LSTM signal", fontsize=9)
    ax3c.set_title(f"LSTM convergence filter ({conv_pct_f:.1f}% converging)  "
                   f"|  Rule fixed before 2026 evaluation",
                   fontsize=9)
    ax3c.legend(fontsize=8, loc="upper right")

    plt.savefig(CHARTS_DIR / f"phase5_{tag}_2026_comparison.png",
                dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Chart saved -> phase5_{tag}_2026_comparison.png")

    return {
        "Pair":             f"{dep}/{indep}",
        "Final_OLS_end":    FINAL_TRAIN_END,
        "Final_val_RMSE":   val_rmse_f_r,
        "Test_RMSE_2026":   round(test_rmse_f, 4),
        "Test_MAE_2026":    round(test_mae_f,  4),
        **{f"Test_RMSE_h{h+1}": round(eval_f[f"RMSE_h{h+1}"], 4)
           for h in range(HORIZON)},
        "Conv_pct_2026":    round(conv_pct_f, 1),
        "P4_long_signals":  n_long_p4,
        "P4_short_signals": n_short_p4,
        "LSTM_long_signals":  n_long_lstm,
        "LSTM_short_signals": n_short_lstm,
        "P4_Sharpe":              m_p4.get("Sharpe_ratio"),
        "P4_Entries":             m_p4.get("Num_trades"),
        "P4_PnL":                 m_p4.get("Total_PnL"),
        "P4_MaxDD":               m_p4.get("Max_drawdown"),
        "P4_CompletedWinRate":    round(wr_p4, 1) if not np.isnan(wr_p4) else None,
        "P4_CompletedTrades":     n_comp_p4,
        "P4_AvgTrade":            m_p4.get("Avg_trade_PnL"),
        "P4_InMkt":               m_p4.get("Pct_in_market"),
        "LSTM_Sharpe":            m_lstm.get("Sharpe_ratio"),
        "LSTM_Entries":           m_lstm.get("Num_trades"),
        "LSTM_PnL":               m_lstm.get("Total_PnL"),
        "LSTM_MaxDD":             m_lstm.get("Max_drawdown"),
        "LSTM_CompletedWinRate":  round(wr_lstm, 1) if not np.isnan(wr_lstm) else None,
        "LSTM_CompletedTrades":   n_comp_lstm,
        "LSTM_AvgTrade":          m_lstm.get("Avg_trade_PnL"),
        "LSTM_InMkt":             m_lstm.get("Pct_in_market"),
    }


# ---------------------------------------------------------------------------
# LSTM-gated signal counter (for 2026 signal comparison, Step 8b)
# ---------------------------------------------------------------------------

def count_signals_lstm_gated(z_series, conv_dict,
                               z_entry=Z_ENTRY_P4, z_exit=Z_EXIT_P4):
    """
    Phase 4 signal counting with LSTM convergence gate on entries.
    Only enters a long/short when conv_dict[signal_date] == 1.
    conv_dict keys must be Timestamps matching z_series.index.
    """
    z_arr  = z_series.values
    dates  = z_series.index
    n      = len(z_arr)
    pos    = 0
    position      = np.zeros(n, dtype=int)
    long_entries  = []
    short_entries = []
    for i in range(1, n):
        z_prev = z_arr[i - 1]
        if np.isnan(z_prev):
            position[i] = pos
            continue
        if pos == 1 and z_prev >= z_exit:
            pos = 0
        elif pos == -1 and z_prev <= z_exit:
            pos = 0
        if pos == 0:
            signal_date = dates[i - 1]
            # Default to 1 (allow) when no prediction exists so missing
            # coverage never silently blocks a trade.
            if conv_dict.get(signal_date, 1) == 1:
                if z_prev < -z_entry:
                    long_entries.append({
                        "Execution_Date": str(dates[i].date()),
                        "Signal_Date":    str(dates[i - 1].date()),
                        "Signal_Z":       round(float(z_prev), 4),
                    })
                    pos = 1
                elif z_prev > z_entry:
                    short_entries.append({
                        "Execution_Date": str(dates[i].date()),
                        "Signal_Date":    str(dates[i - 1].date()),
                        "Signal_Z":       round(float(z_prev), 4),
                    })
                    pos = -1
        position[i] = pos
    return len(long_entries), len(short_entries), position, long_entries, short_entries


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Phase 5: LSTM spread prediction")
    parser.add_argument("--refresh", action="store_true",
                        help="Re-download 2026 data and overwrite cached files in data/raw/")
    args = parser.parse_args()

    # Force deterministic TF ops so reruns produce identical weights.
    # Must be called before any TF computation.
    tf.config.experimental.enable_op_determinism()

    print("=" * 65)
    print("PHASE 5 – MACHINE LEARNING FOR PREDICTING SPREAD")
    print("=" * 65)
    print(f"  Framework    : TensorFlow {tf.__version__}")
    print(f"  Architecture : LSTM({LSTM_UNITS_1}→{LSTM_UNITS_2})  "
          f"Dropout({DROPOUT})  Dense({HORIZON})")
    print(f"  Lookback / horizon  : {SEQ_LEN}d / {HORIZON}d   Seed : {SEED}")
    print()
    print(f"  Experiment A — Historical chronological holdout")
    print(f"    OLS + scaler + LSTM train : 2018-01-01 to {ML_TRAIN_END}")
    print(f"    LSTM val                  : {ML_VAL_START}–{ML_VAL_END}")
    print(f"    Test (supplementary)      : {ML_TEST_START}–2025-12-31")
    print(f"    Step 8a comparison        : Phase 4-style baseline "
          f"(2021 anchor) vs LSTM-enhanced")
    print()
    print(f"  Experiment B — Final 2026 evaluation (Step 8, definitive)")
    print(f"    OLS + scaler + LSTM train : 2018-01-01 to {FINAL_TRAIN_END}")
    print(f"    LSTM val                  : {FINAL_VAL_START}–{FINAL_VAL_END}")
    print(f"    Unseen test               : {P4_TEST_START}–{P4_TEST_END}")
    print(f"    Phase 4 baseline          : OLS 2018-2025, 2025 anchor, "
          f"entry ±{Z_ENTRY_P4}, exit {Z_EXIT_P4}")
    print(f"    LSTM-enhanced             : same Phase 4 signals, "
          f"entries gated by final LSTM convergence")

    selected_csv = REPORTS_DIR / "selected_pairs.csv"
    if not selected_csv.exists():
        raise FileNotFoundError(
            f"selected_pairs.csv not found at {selected_csv}.\n"
            "Run phase3_cointegration.py first.")
    sel_df = pd.read_csv(selected_csv)
    print(f"\n  Loaded {len(sel_df)} selected pair(s) from {selected_csv.name}")

    # ------------------------------------------------------------------
    # Experiment A: historical holdout (2018-2020 / 2021 / 2022-2025)
    # ------------------------------------------------------------------
    print(f"\n{'='*65}")
    print("EXPERIMENT A — Historical chronological holdout")
    print(f"{'='*65}")
    hist_rows = []
    for _, row in sel_df.iterrows():
        dep, indep = row["OLS_direction"].split("~")
        result = run_phase5_pair(
            dep=dep, indep=indep, tests_passed=row["Tests_passed"])
        hist_rows.append(result)

    hist_df  = pd.DataFrame(hist_rows)
    hist_csv = REPORTS_DIR / "phase5_cross_pair_summary.csv"
    hist_df.to_csv(hist_csv, index=False)

    col_w = 14
    print(f"\n  Supplementary 2022-2025 holdout summary:")
    hdr = (f"    {'Pair':<12}  {'TestRMSE':>{col_w}}  {'Conv%':>{col_w}}  "
           f"{'P4style Sharpe':>{col_w}}  {'LSTM Sharpe':>{col_w}}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for r in hist_rows:
        print(f"    {r['Pair']:<12}  "
              f"{r['Test_RMSE_2022_2025']:>{col_w}.4f}  "
              f"{r['Conv_pct_2022_2025']:>{col_w}.1f}  "
              f"{r['P4style_Sharpe']:>{col_w}.4f}  "
              f"{r['LSTM_Sharpe']:>{col_w}.4f}")
    print(f"  Saved -> {hist_csv.name}")

    # ------------------------------------------------------------------
    # Experiment B: final 2026 evaluation (2018-2024 / 2025 / 2026)
    # ------------------------------------------------------------------
    print(f"\n{'='*65}")
    print("EXPERIMENT B — Final 2026 evaluation (Step 8, definitive)")
    print(f"{'='*65}")
    final_rows = []
    for _, row in sel_df.iterrows():
        dep, indep = row["OLS_direction"].split("~")
        result = run_phase5_final_2026(
            dep=dep, indep=indep, tests_passed=row["Tests_passed"],
            refresh=args.refresh)
        if result is not None:
            final_rows.append(result)

    if final_rows:
        final_df  = pd.DataFrame(final_rows)
        final_csv = REPORTS_DIR / "phase5_cross_pair_final_summary.csv"
        final_df.to_csv(final_csv, index=False)

        print(f"\n{'='*65}")
        print("FINAL 2026 CROSS-PAIR SUMMARY")
        print(f"{'='*65}")
        col_w = 12
        hdr2 = (f"  {'Pair':<12}  {'TestRMSE':>{col_w}}  {'P4 Sharpe':>{col_w}}  "
                f"{'LSTM Sharpe':>{col_w}}  {'P4 Entries':>{col_w}}  "
                f"{'LSTM Entries':>{col_w}}  {'P4 WinRate*':>{col_w}}  "
                f"{'LSTM WinRate*':>{col_w}}")
        print(hdr2)
        print("-" * len(hdr2))
        for r in final_rows:
            wr_p = (f"{r['P4_CompletedWinRate']:.1f}%"
                    f"({r['P4_CompletedTrades']}/{r['P4_Entries']})"
                    if r.get("P4_CompletedWinRate") is not None
                    and r.get("P4_CompletedTrades", 0) > 0 else "n/a")
            wr_l = (f"{r['LSTM_CompletedWinRate']:.1f}%"
                    f"({r['LSTM_CompletedTrades']}/{r['LSTM_Entries']})"
                    if r.get("LSTM_CompletedWinRate") is not None
                    and r.get("LSTM_CompletedTrades", 0) > 0 else "n/a")
            print(f"  {r['Pair']:<12}  "
                  f"{r['Test_RMSE_2026']:>{col_w}.4f}  "
                  f"{r['P4_Sharpe']:>{col_w}.4f}  "
                  f"{r['LSTM_Sharpe']:>{col_w}.4f}  "
                  f"{r['P4_Entries']:>{col_w}}  "
                  f"{r['LSTM_Entries']:>{col_w}}  "
                  f"{wr_p:>{col_w}}  {wr_l:>{col_w}}")
        print("  * Win rate over completed (closed) trades only; open trades excluded.")
        print(f"\n  Saved -> {final_csv.name}")

    print(f"\n{'='*65}")
    print("PHASE 5 COMPLETE")
    print(f"{'='*65}")
    print(f"\nOutputs written to {REPORTS_DIR} and {CHARTS_DIR}")


if __name__ == "__main__":
    main()

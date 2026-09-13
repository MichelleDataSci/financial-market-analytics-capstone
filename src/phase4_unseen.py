"""
Phase 4 -- Testing Mean Reversion on Unseen Data
Reads selected_pairs.csv (from Phase 3a) and tests each pair on genuinely
unseen 2026 data using a fixed Z-score anchor from the last year of training.

Procedure (per project brief):
  Step 1  Cointegrated pairs identified in Phase 3a (from selected_pairs.csv)
  Step 2  Fit OLS on full training period log prices (2018-01-01 to 2025-12-31)
  Step 3  Compute FIXED mean and std of residuals from 2025 only (last year)
  Step 4  Download unseen data (2026-01-01 to 2026-07-31) and apply same model
  Step 5  Standardise testing-period residuals using the 2025 mean and std
  Step 6  Plot Z-scores over the testing period -- assess mean-reversion behaviour
  Step 7  Count long (Z < -2, exit Z >= 0) and short (Z > +2, exit Z <= 0) signals

Key differences from Phase 3b:
  - Training covers the FULL 2018-2025 history (not just 2018-2021)
  - Z-score uses a FIXED anchor (2025 mean/std) -- not a rolling window
  - Test data is genuinely unseen: 2026 was not available when the model was built
"""

import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import statsmodels.api as sm
import yfinance as yf
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import DATA_RAW, CHARTS_DIR, REPORTS_DIR

# ---------------------------------------------------------------------------
# Shared parameters
# ---------------------------------------------------------------------------
TRAIN_END  = "2025-12-31"
NORM_START = "2025-01-01"   # last 12 months of training used for fixed mu/sigma
TEST_START = "2026-01-01"
TEST_END   = "2026-07-31"
Z_ENTRY    = 2.0
Z_EXIT     = 0.0


# ---------------------------------------------------------------------------
# Per-pair runner
# ---------------------------------------------------------------------------

def run_phase4_pair(dep, indep, tests_passed):
    """
    Run the full Phase 4 mean-reversion test for one (dep, indep) pair.
    Returns a summary dict for the cross-pair comparison table.
    """
    tag = f"{dep}_{indep}"

    print(f"\n{'='*65}")
    print(f"PAIR: {dep} / {indep}  [{tests_passed}]")
    print(f"{'='*65}")
    print(f"  Training period : 2018-01-01 to {TRAIN_END}")
    print(f"  Norm window     : {NORM_START} to {TRAIN_END}")
    print(f"  Test period     : {TEST_START} to {TEST_END} (unseen data)")
    print(f"  Entry signal    : |Z| > {Z_ENTRY}")
    print(f"  Exit signal     : Z crosses {Z_EXIT}")
    print(f"  Execution       : signal observed at close(t-1); position entered at open(t)")

    # ── Step 2: Load full training history and fit OLS ──
    print(f"\n  [Step 2] Fitting OLS on full training period 2018-2025 ...")
    train_prices = {}
    for ticker in [dep, indep]:
        path = DATA_RAW / f"{ticker}_raw.csv"
        df = pd.read_csv(path, index_col="Date", parse_dates=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        train_prices[ticker] = df["Close"]

    train_df  = pd.DataFrame(train_prices).dropna().loc[:TRAIN_END]
    log_train = np.log(train_df)
    print(f"  Training data: {len(train_df)} days  "
          f"({train_df.index[0].date()} to {train_df.index[-1].date()})")

    X     = sm.add_constant(log_train[indep])
    model = sm.OLS(log_train[dep], X).fit()
    beta  = float(model.params.iloc[1])
    alpha = float(model.params.iloc[0])
    r2    = float(model.rsquared)

    print(f"  OLS: log({dep}) ~ log({indep}) + constant")
    print(f"    Hedge ratio (beta) : {beta:.6f}")
    print(f"    Intercept (alpha)  : {alpha:.6f}")
    print(f"    R-squared          : {r2:.6f}")

    train_spread = log_train[dep] - beta * log_train[indep] - alpha

    # ── Step 3: Fixed mu and sigma from NORM_START to TRAIN_END ──
    print(f"\n  [Step 3] Computing fixed mean/std from {NORM_START} to {TRAIN_END} ...")
    norm_spread = train_spread.loc[NORM_START:TRAIN_END]
    mu    = float(norm_spread.mean())
    sigma = float(norm_spread.std())
    print(f"    Normalisation window : {len(norm_spread)} days")
    print(f"    Fixed mu             : {mu:.6f}")
    print(f"    Fixed sigma          : {sigma:.6f}")
    print(f"    Z = (spread - {mu:.4f}) / {sigma:.4f}")

    # ── Step 4: Download unseen 2026 data ──
    print(f"\n  [Step 4] Downloading unseen data ({TEST_START} to {TEST_END}) ...")
    test_end_exclusive = "2026-08-01"
    test_prices = {}
    for ticker in [dep, indep]:
        print(f"    Downloading {ticker} ...", end=" ")
        df_t = yf.download(
            ticker, start=TEST_START, end=test_end_exclusive,
            auto_adjust=True, progress=False
        )
        if df_t.empty:
            print("WARNING: no data returned.")
            continue
        if isinstance(df_t.columns, pd.MultiIndex):
            df_t.columns = df_t.columns.get_level_values(0)
        df_t.index.name = "Date"
        test_prices[ticker] = df_t["Close"]
        print(f"{len(df_t)} trading days")

    if len(test_prices) < 2:
        print(f"  ERROR: could not download test data for {dep}/{indep}. Skipping pair.")
        return None

    test_df  = pd.DataFrame(test_prices).dropna()
    log_test = np.log(test_df)
    print(f"  Test data: {len(test_df)} days  "
          f"({test_df.index[0].date()} to {test_df.index[-1].date()})")

    test_spread = log_test[dep] - beta * log_test[indep] - alpha
    print(f"  Out-of-sample spread:  mean={test_spread.mean():.6f}  "
          f"(training mu={mu:.6f})   std={test_spread.std():.6f}  "
          f"(training sigma={sigma:.6f})")

    # ── Step 5: Standardise using fixed mu/sigma ──
    zscore = (test_spread - mu) / sigma
    dates  = zscore.index
    z_arr  = zscore.values
    n      = len(z_arr)

    print(f"\n  [Step 5] Z-score statistics:")
    print(f"    Min  : {zscore.min():.4f}")
    print(f"    Max  : {zscore.max():.4f}")
    print(f"    Mean : {zscore.mean():.4f}  (near 0 if stationary)")
    print(f"    Std  : {zscore.std():.4f}   (near 1 if variance stable)")
    print(f"    Days above +{Z_ENTRY} : {(zscore > Z_ENTRY).sum()}")
    print(f"    Days below -{Z_ENTRY} : {(zscore < -Z_ENTRY).sum()}")

    # ── Step 7: Trading signals ──
    # Signal convention: z_prev (previous bar's close) drives the position on
    # the current bar, consistent with Phase 3b (signal at close(t), execute
    # at open(t+1)).  Bar 0 is always flat -- no prior z-score is available.
    #
    # Signal log columns:
    #   Execution_Date  -- bar i; the date the position is entered/held
    #   Signal_Date     -- bar i-1; the bar whose close produced Signal_Z_score
    #   Signal_Z_score  -- z_prev; the z-score that triggered the action
    #   Current_Z_score -- z_curr; today's z-score (for observation, not decision)
    #   Position / Signal -- position held and action label after execution
    print(f"\n  [Step 7] Trading signals "
          f"(Signal_Z_score = z at close(t-1); Execution_Date = open(t)) ...")
    position      = np.zeros(n, dtype=int)   # position[0] = 0 (flat)
    pos           = 0
    long_entries  = []
    short_entries = []
    signal_log    = []

    # Bar 0: no prior signal -- record observed z-score, stay flat
    signal_log.append({
        "Execution_Date":  str(dates[0].date()),
        "Signal_Date":     None,               # no prior bar
        "Signal_Z_score":  None,               # no prior z-score
        "Current_Z_score": round(float(z_arr[0]), 4) if not np.isnan(z_arr[0]) else None,
        "Position":        0,
        "Signal":          "warm-up",
    })

    for i in range(1, n):
        z_prev = z_arr[i - 1]   # signal from previous bar's close
        z_curr = z_arr[i]       # current bar z-score (for logging only)

        if np.isnan(z_prev):
            position[i] = pos
            signal_log.append({
                "Execution_Date":  str(dates[i].date()),
                "Signal_Date":     str(dates[i - 1].date()),
                "Signal_Z_score":  None,
                "Current_Z_score": round(float(z_curr), 4) if not np.isnan(z_curr) else None,
                "Position":        pos,
                "Signal":          "NaN z_prev",
            })
            continue

        action = "hold"
        if pos == 1 and z_prev >= Z_EXIT:
            pos    = 0
            action = "exit long"
        elif pos == -1 and z_prev <= Z_EXIT:
            pos    = 0
            action = "exit short"

        if pos == 0:
            if z_prev < -Z_ENTRY:
                long_entries.append({
                    "Execution_Date": dates[i],
                    "Signal_Date":    dates[i - 1],
                    "Z_score":        round(float(z_prev), 4),
                })
                pos    = 1
                action = "enter long"
            elif z_prev > Z_ENTRY:
                short_entries.append({
                    "Execution_Date": dates[i],
                    "Signal_Date":    dates[i - 1],
                    "Z_score":        round(float(z_prev), 4),
                })
                pos    = -1
                action = "enter short"

        position[i] = pos
        signal_log.append({
            "Execution_Date":  str(dates[i].date()),
            "Signal_Date":     str(dates[i - 1].date()),
            "Signal_Z_score":  round(float(z_prev), 4),
            "Current_Z_score": round(float(z_curr), 4) if not np.isnan(z_curr) else None,
            "Position":        pos,
            "Signal":          action,
        })

    n_long  = len(long_entries)
    n_short = len(short_entries)
    n_total = n_long + n_short
    n_in_mkt = int((position != 0).sum())

    print(f"    Long  entries (Z < -{Z_ENTRY}) : {n_long}")
    print(f"    Short entries (Z >  {Z_ENTRY}) : {n_short}")
    print(f"    Total signals                  : {n_total}")
    print(f"    Days in market                 : {n_in_mkt} of {n} "
          f"({n_in_mkt / n * 100:.1f}%)")

    if long_entries:
        print(f"    Long execution dates (Signal_Z_score = z at prior close):")
        for e in long_entries:
            print(f"      exec {e['Execution_Date'].date()}  "
                  f"[signal {e['Signal_Date'].date()}: Z={e['Z_score']:+.4f}]")
    if short_entries:
        print(f"    Short execution dates (Signal_Z_score = z at prior close):")
        for e in short_entries:
            print(f"      exec {e['Execution_Date'].date()}  "
                  f"[signal {e['Signal_Date'].date()}: Z={e['Z_score']:+.4f}]")

    # Save signal log
    signals_df  = pd.DataFrame(signal_log)
    signals_csv = REPORTS_DIR / f"phase4_{tag}_signals.csv"
    signals_df.to_csv(signals_csv, index=False)
    print(f"\n  Signal log saved -> {signals_csv.name}")

    # Mean-reversion verdict
    z_mean = float(zscore.mean())
    z_std  = float(zscore.std())
    if abs(z_mean) < 0.5 and 0.5 < z_std < 2.0:
        verdict = "SUPPORTS mean reversion"
    elif abs(z_mean) > 1.0:
        verdict = "WEAK mean reversion — spread has drifted from equilibrium"
    else:
        verdict = "MODERATE mean reversion"

    # Per-pair summary CSV
    summary = {
        "Pair":          f"{dep}/{indep}",
        "Tests_passed":  tests_passed,
        "Hedge_ratio":   round(beta, 6),
        "Intercept":     round(alpha, 6),
        "R2_training":   round(r2, 6),
        "Norm_mu":       round(mu, 6),
        "Norm_sigma":    round(sigma, 6),
        "Test_days":     n,
        "Z_min":         round(float(zscore.min()), 4),
        "Z_max":         round(float(zscore.max()), 4),
        "Z_mean":        round(z_mean, 4),
        "Z_std":         round(z_std, 4),
        "Long_signals":  n_long,
        "Short_signals": n_short,
        "Total_signals": n_total,
        "Days_in_mkt":   n_in_mkt,
        "Pct_in_mkt":    round(n_in_mkt / n * 100, 2),
        "Verdict":       verdict,
    }
    summary_df  = pd.DataFrame([summary])
    summary_csv = REPORTS_DIR / f"phase4_{tag}_summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"  Summary saved   -> {summary_csv.name}")

    # ── Step 6: Chart ──
    print(f"\n  [Step 6] Generating Z-score chart ...")
    fig = plt.figure(figsize=(16, 11))
    gs  = gridspec.GridSpec(3, 1, hspace=0.45, figure=fig)

    # Panel 1: Z-score with entry/exit bands and position shading
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(dates, z_arr, color="steelblue", lw=1.0, label="Z-score", zorder=3)
    ax1.axhline( Z_ENTRY, color="tomato",   ls="--", lw=1.1, label=f"+{Z_ENTRY} short entry")
    ax1.axhline(-Z_ENTRY, color="seagreen", ls="--", lw=1.1, label=f"-{Z_ENTRY} long entry")
    ax1.axhline( Z_EXIT,  color="black",    ls="-",  lw=0.6, label="0 exit", alpha=0.6)
    for i in range(1, n):
        if   position[i] ==  1:
            ax1.axvspan(dates[i-1], dates[i], alpha=0.14, color="seagreen", lw=0)
        elif position[i] == -1:
            ax1.axvspan(dates[i-1], dates[i], alpha=0.14, color="tomato",   lw=0)
    for e in long_entries:
        ax1.axvline(e["Execution_Date"], color="seagreen", lw=1.0, ls=":", alpha=0.7)
    for e in short_entries:
        ax1.axvline(e["Execution_Date"], color="tomato",   lw=1.0, ls=":", alpha=0.7)

    ax1.set_title(
        f"{dep}/{indep} Spread Z-score — Unseen Test Period ({TEST_START} to {TEST_END})\n"
        f"Z = (spread − mu) / sigma  |  mu={mu:.4f}, sigma={sigma:.4f}  |  "
        f"Green = long, Red = short spread  [{tests_passed}]",
        fontsize=9.5
    )
    ax1.set_ylabel("Z-score")
    ax1.legend(fontsize=8, loc="upper right", ncol=2)
    ax1.set_xlim(dates[0], dates[-1])
    ax1.text(0.01, 0.97,
             f"Z stats: mean={z_mean:.3f}, std={z_std:.3f} | "
             f"Signals: {n_long} long, {n_short} short",
             transform=ax1.transAxes, fontsize=8, va="top",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                       edgecolor="lightgrey", alpha=0.85))

    # Panel 2: Raw spread
    ax2 = fig.add_subplot(gs[1])
    ax2.plot(dates, test_spread.values, color="darkorange", lw=0.9)
    ax2.axhline(mu, color="black", ls="--", lw=0.8, label=f"2025 mean ({mu:.4f})")
    ax2.fill_between(dates, test_spread.values, mu,
                     where=test_spread.values >= mu, alpha=0.18, color="tomato")
    ax2.fill_between(dates, test_spread.values, mu,
                     where=test_spread.values <  mu, alpha=0.18, color="seagreen")
    ax2.set_title(
        f"Out-of-Sample Spread = log({dep}) − {beta:.4f}·log({indep}) − {alpha:.4f}",
        fontsize=9.5)
    ax2.set_ylabel("Spread (log units)")
    ax2.legend(fontsize=8)
    ax2.set_xlim(dates[0], dates[-1])

    # Panel 3: Log prices (dual y-axis)
    ax3       = fig.add_subplot(gs[2])
    ax3_right = ax3.twinx()
    ax3.plot(log_test.index, log_test[dep].values,
             color="steelblue", lw=1.0, label=f"log({dep})")
    ax3_right.plot(log_test.index, log_test[indep].values,
                   color="darkorange", lw=1.0, ls="--", label=f"log({indep})")
    ax3.set_ylabel(f"log({dep})", color="steelblue")
    ax3_right.set_ylabel(f"log({indep})", color="darkorange")
    ax3.tick_params(axis="y", labelcolor="steelblue")
    ax3_right.tick_params(axis="y", labelcolor="darkorange")
    lines1, labels1 = ax3.get_legend_handles_labels()
    lines2, labels2 = ax3_right.get_legend_handles_labels()
    ax3.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="lower right")
    ax3.set_title(f"log({dep}) and log({indep}) — Test Period", fontsize=9.5)
    ax3.set_xlim(dates[0], dates[-1])

    chart_path = CHARTS_DIR / f"phase4_{tag}_zscore_2026.png"
    plt.savefig(chart_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Chart saved     -> {chart_path.name}")

    print(f"\n  Verdict: {verdict}")

    return summary


# ---------------------------------------------------------------------------
# Main — read selected pairs and run Phase 4 for each
# ---------------------------------------------------------------------------
print("=" * 65)
print("PHASE 4 -- TESTING MEAN REVERSION ON UNSEEN DATA")
print("=" * 65)
print(f"\n  Training period : 2018-01-01 to {TRAIN_END}")
print(f"  Norm window     : {NORM_START} to {TRAIN_END}")
print(f"  Test period     : {TEST_START} to {TEST_END}")
print(f"  Entry |Z| > {Z_ENTRY}, exit Z crosses {Z_EXIT}")

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
    print(f"\nWARNING: {selected_csv.name} not found — falling back to AMZN/META.")
    print("Run phase3_cointegration.py first to generate selected_pairs.csv.")

# Run all pairs
summary_rows = []
for dep, indep, tests_passed in pair_list:
    result = run_phase4_pair(dep, indep, tests_passed)
    if result is not None:
        summary_rows.append(result)

# Cross-pair summary
if summary_rows:
    cross_df  = pd.DataFrame(summary_rows)
    cross_csv = REPORTS_DIR / "phase4_cross_pair_summary.csv"
    cross_df.to_csv(cross_csv, index=False)

    print(f"\n{'='*65}")
    print("PHASE 4 -- CROSS-PAIR SUMMARY")
    print(f"{'='*65}")
    display_cols = ["Pair", "Tests_passed", "Z_mean", "Z_std",
                    "Long_signals", "Short_signals", "Total_signals",
                    "Pct_in_mkt", "Verdict"]
    print(cross_df[display_cols].to_string(index=False))
    print(f"\nCross-pair summary saved -> {cross_csv}")

print(f"\n{'='*65}")
print("PHASE 4 COMPLETE")
print(f"{'='*65}")
print(f"\nNext step: Phase 5 -- Machine Learning for Predicting Spread (brief pending).")

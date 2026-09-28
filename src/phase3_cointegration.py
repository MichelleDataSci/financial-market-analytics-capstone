"""
Phase 3 -- Cointegration Screening
Tests all 15 pairs of 6 tech stocks for cointegration using log prices.

Methods (per project scope):
  1. OLS regression in both directions + ADF on residuals -- pick direction
     with stronger stationarity evidence (lower ADF p-value).
  2. Engle-Granger test via statsmodels.tsa.stattools.coint (auto regression
     + residual ADF with correct non-standard critical values).
  3. Johansen multivariate cointegration test.

Selection criterion:
  Primary   -- at least one of EG or Johansen trace passes at 5%.
  Secondary -- Johansen trace passes at 10% but not 5%; pair carried
               forward as a borderline candidate per supervisor instruction
               (Vinayak, project brief).  Documented explicitly so the
               selection basis is transparent in all downstream files.

Output:
  outputs/reports/cointegration_results.csv  -- full results table, all 15 pairs
  outputs/reports/selected_pairs.csv         -- pairs taken forward to strategy
  outputs/charts/spread_series_all_pairs.png -- OLS residual plots, all 15 pairs
  outputs/charts/cointegration_ranking.png   -- EG p-value ranking bar chart
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from statsmodels.tsa.vector_ar.var_model import VAR
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import ALL_TICKERS, DATA_RAW, CHARTS_DIR, REPORTS_DIR, TRAIN_END

# ---------------------------------------------------------------------------
# Helper functions -- module-level so they can be imported by other phases
# ---------------------------------------------------------------------------

def ols_adf(y_log, x_log):
    """
    OLS of y on x (with constant) using log prices.
    Returns hedge_ratio, intercept, residuals, ADF stat, ADF p-value.
    ADF uses autolag='AIC' to select lag length automatically.
    """
    X = sm.add_constant(x_log)
    model = sm.OLS(y_log, X).fit()
    resid = model.resid
    adf = adfuller(resid, autolag="AIC", regression="c")
    # params: [const, slope]
    return float(model.params.iloc[1]), float(model.params.iloc[0]), resid, float(adf[0]), float(adf[1])


def pick_direction(log_a, log_b, ticker_a, ticker_b):
    """
    Run OLS in both directions and return the direction whose residuals
    are more stationary (lower ADF p-value = stronger evidence against
    a unit root in the spread).

    Returns:
        chosen_dir  : str  e.g. "AMZN~META"
        hedge_ratio : float
        intercept   : float
        resid       : pd.Series
        adf_stat    : float
        adf_pval    : float
        dep         : str  -- the dependent ticker in the chosen direction
        indep       : str  -- the independent ticker in the chosen direction
    """
    hr_ab, ic_ab, resid_ab, adf_ab, pval_ab = ols_adf(log_a, log_b)
    hr_ba, ic_ba, resid_ba, adf_ba, pval_ba = ols_adf(log_b, log_a)

    if pval_ab <= pval_ba:
        return (f"{ticker_a}~{ticker_b}", hr_ab, ic_ab,
                resid_ab, adf_ab, pval_ab, ticker_a, ticker_b)
    else:
        return (f"{ticker_b}~{ticker_a}", hr_ba, ic_ba,
                resid_ba, adf_ba, pval_ba, ticker_b, ticker_a)


def engle_granger(y_log, x_log):
    """
    Engle-Granger test via statsmodels coint().
    Uses non-standard critical values appropriate for regression residuals.
    Returns (test stat, p-value).
    """
    stat, pval, _ = coint(y_log, x_log, autolag="AIC")
    return float(stat), float(pval)


def johansen(log_pair_df, det_order=0, k_ar_diff=1):
    """
    Johansen cointegration test on a 2-column log price DataFrame.
    det_order=0: constant in the cointegrating relationship.
    k_ar_diff  : lags in differences.  Determined by the VAR LAG SELECTION
                 block below, which runs VAR.select_order() on all 15 pairs
                 and takes the median BIC-selected lag as a shared K_AR_DIFF
                 applied uniformly to every Johansen test.  This converts the
                 lag choice from an assertion into a reproducible, data-driven
                 decision.  The function accepts k_ar_diff as a parameter so
                 callers can override it; the actual value is printed to stdout.
    Critical value columns: ci=0 -> 10%, ci=1 -> 5%, ci=2 -> 1%.
    Returns trace and max-eigenvalue stats at both 5% and 10%, plus pass flags.
    """
    res = coint_johansen(log_pair_df, det_order=det_order, k_ar_diff=k_ar_diff)
    ci5  = 1   # 5%  significance column
    ci10 = 0   # 10% significance column
    return {
        "trace_stat_r0":          round(float(res.lr1[0]), 4),
        "trace_crit_r0_5pct":     round(float(res.cvt[0, ci5]),  4),
        "trace_crit_r0_10pct":    round(float(res.cvt[0, ci10]), 4),
        "trace_stat_r1":          round(float(res.lr1[1]), 4),
        "trace_crit_r1_5pct":     round(float(res.cvt[1, ci5]),  4),
        "maxeig_stat_r0":         round(float(res.lr2[0]), 4),
        "maxeig_crit_r0_5pct":    round(float(res.cvm[0, ci5]),  4),
        "maxeig_crit_r0_10pct":   round(float(res.cvm[0, ci10]), 4),
        "maxeig_stat_r1":         round(float(res.lr2[1]), 4),
        "maxeig_crit_r1_5pct":    round(float(res.cvm[1, ci5]),  4),
        "trace_pass_5pct":        bool(res.lr1[0] > res.cvt[0, ci5]),
        "trace_pass_10pct":       bool(res.lr1[0] > res.cvt[0, ci10]),
        "maxeig_pass_5pct":       bool(res.lr2[0] > res.cvm[0, ci5]),
        "maxeig_pass_10pct":      bool(res.lr2[0] > res.cvm[0, ci10]),
    }


def classify_pair(
    eg_pass_5pct: bool,
    jo_trace_pass_5pct: bool,
    jo_trace_pass_10pct: bool,
) -> str:
    """
    Single source of truth for the Phase 3 / app selection rule.

    Returns:
      'primary'   -- EG 5% or Johansen trace 5% passes
      'borderline' -- Johansen trace 10% only (supervisor-approved secondary)
      'none'      -- neither
    """
    if eg_pass_5pct or jo_trace_pass_5pct:
        return "primary"
    if jo_trace_pass_10pct:
        return "borderline"
    return "none"


def select_johansen_lag(log_pair_df: "pd.DataFrame", maxlags: int = 5) -> int:
    """BIC-selected VAR lag for Johansen test (mirrors Phase 3 lag selection).

    Runs VAR.select_order() on differenced log prices and returns the BIC-
    chosen lag, floored at 1.  Phase 3 takes the median of this across all 15
    pairs; the app calls it per-pair — both yield k_ar_diff=1 for the current
    dataset because all BIC selections are 0 or 1 and max(1, median)=1.
    """
    diff_df = log_pair_df.diff().dropna()
    sel = VAR(diff_df).select_order(maxlags=maxlags)
    return max(1, int(sel.selected_orders["bic"]))


# ---------------------------------------------------------------------------
# Main execution
# ---------------------------------------------------------------------------

def main():
    print("=" * 65)
    print("PHASE 3 -- COINTEGRATION SCREENING (LOG PRICES)")
    print("=" * 65)

    # -------------------------------------------------------------------------
    # 1. Load close prices from raw CSVs and compute log prices
    # -------------------------------------------------------------------------
    prices = {}
    for ticker in ALL_TICKERS:
        path = DATA_RAW / f"{ticker}_raw.csv"
        df = pd.read_csv(path, index_col="Date", parse_dates=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        prices[ticker] = df["Close"]

    price_df = pd.DataFrame(prices).dropna()
    log_price_df = np.log(price_df)

    print(f"\nClose prices loaded: {price_df.shape[0]} trading days, {price_df.shape[1]} tickers")
    print(f"Date range: {price_df.index[0].date()} to {price_df.index[-1].date()}")
    print(f"Log prices computed for: {ALL_TICKERS}")

    # -------------------------------------------------------------------------
    # 2. Generate all 15 pairs
    # -------------------------------------------------------------------------
    pairs = list(combinations(ALL_TICKERS, 2))
    print(f"\nAll {len(pairs)} pairs:")
    for i, (a, b) in enumerate(pairs, 1):
        print(f"  {i:2d}. {a} / {b}")

    # -------------------------------------------------------------------------
    # 3. VAR lag selection -- validate k_ar_diff used in Johansen test
    #    Runs VAR.select_order() on differenced log prices for all 15 pairs.
    #    AIC, BIC, HQC, and FPE are reported; the median BIC-selected lag across
    #    all 15 pairs determines K_AR_DIFF, which is then applied uniformly to
    #    every Johansen test.  Using the median rather than a per-pair lag keeps
    #    the comparison consistent across pairs and prevents overfitting the lag
    #    to any individual pair's residuals.
    # -------------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("VAR LAG SELECTION -- JOHANSEN k_ar_diff VALIDATION (maxlags=5)")
    print("=" * 65)
    print(f"  {'Pair':<12}  {'AIC':>5}  {'BIC':>5}  {'HQC':>5}  {'FPE':>5}")
    print(f"  {'-'*38}")

    _lag_rows = []
    for _a, _b in pairs:
        _pair_diff = log_price_df[[_a, _b]].diff().dropna()
        _sel = VAR(_pair_diff).select_order(maxlags=5)
        _aic = int(_sel.selected_orders["aic"])
        _bic = int(_sel.selected_orders["bic"])
        _hqc = int(_sel.selected_orders["hqic"])
        _fpe = int(_sel.selected_orders["fpe"])
        print(f"  {_a}/{_b:<9}  {_aic:>5}  {_bic:>5}  {_hqc:>5}  {_fpe:>5}")
        _lag_rows.append({"Pair": f"{_a}/{_b}", "AIC": _aic, "BIC": _bic,
                          "HQC": _hqc, "FPE": _fpe})

    _lag_df      = pd.DataFrame(_lag_rows)
    _bic_median  = int(_lag_df["BIC"].median())
    _bic_max     = int(_lag_df["BIC"].max())
    K_AR_DIFF    = max(1, _bic_median)   # never go below 1

    print(f"\n  BIC-selected lag summary: median={_bic_median}, max={_bic_max}")
    print(f"  -> Using k_ar_diff={K_AR_DIFF} (median BIC across all 15 pairs) "
          f"for all Johansen tests")
    print(f"  Note: a shared lag is applied uniformly to all pairs rather than "
          f"fitting a separate lag per pair.")

    # -------------------------------------------------------------------------
    # 4. Run all tests for all 15 pairs
    # -------------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("RUNNING TESTS FOR ALL 15 PAIRS")
    print("=" * 65)

    records = []
    spreads = {}  # chosen OLS residuals for plotting

    for a, b in pairs:
        log_a = log_price_df[a]
        log_b = log_price_df[b]

        # OLS in both directions -- pick the direction whose residuals are more
        # stationary (lower ADF p-value = stronger evidence against unit root).
        (chosen_dir, hedge_ratio, intercept,
         resid, adf_stat, adf_pval, dep, indep) = pick_direction(log_a, log_b, a, b)

        # Engle-Granger on chosen direction
        eg_stat, eg_pval = engle_granger(log_price_df[dep], log_price_df[indep])

        # Johansen (direction-independent)
        joh = johansen(log_price_df[[a, b]], k_ar_diff=K_AR_DIFF)

        pair_label = f"{a}/{b}"
        spreads[pair_label] = resid

        eg_pass     = eg_pval < 0.05
        joh_pass_5  = joh["trace_pass_5pct"]
        joh_pass_10 = joh["trace_pass_10pct"]

        # Classify by which test(s) each pair satisfies.
        # "Johansen 10%" means the Johansen trace test passes at 10% but not 5%;
        # the pair is carried forward as a secondary candidate per supervisor
        # instruction.
        if eg_pass and joh_pass_5:
            tests_passed = "Both"
        elif eg_pass:
            tests_passed = "EG only"
        elif joh_pass_5:
            tests_passed = "Johansen only"
        elif joh_pass_10:
            tests_passed = "Johansen 10%"
        else:
            tests_passed = "Neither"

        # Selected = True for primary or borderline criterion (classify_pair).
        selected = classify_pair(eg_pass, joh_pass_5, joh_pass_10) != "none"

        records.append({
            "Pair":                       pair_label,
            "OLS_direction":              chosen_dir,
            "Hedge_ratio":                round(hedge_ratio, 4),
            "Intercept":                  round(intercept, 4),
            "ADF_stat_on_residuals":      round(adf_stat, 4),
            "ADF_pval_on_residuals":      round(adf_pval, 4),
            "EG_stat":                    round(eg_stat, 4),
            "EG_pval":                    round(eg_pval, 4),
            "EG_cointegrated_5pct":       eg_pass,
            "Johansen_trace_stat_r0":     joh["trace_stat_r0"],
            "Johansen_trace_crit_r0":     joh["trace_crit_r0_5pct"],
            "Johansen_trace_crit_r0_10pct": joh["trace_crit_r0_10pct"],
            "Johansen_trace_stat_r1":     joh["trace_stat_r1"],
            "Johansen_trace_crit_r1":     joh["trace_crit_r1_5pct"],
            "Johansen_maxeig_stat_r0":    joh["maxeig_stat_r0"],
            "Johansen_maxeig_crit_r0":    joh["maxeig_crit_r0_5pct"],
            "Johansen_maxeig_crit_r0_10pct": joh["maxeig_crit_r0_10pct"],
            "Johansen_trace_pass_5pct":   joh_pass_5,
            "Johansen_trace_pass_10pct":  joh_pass_10,
            "Johansen_maxeig_pass_5pct":  joh["maxeig_pass_5pct"],
            "Johansen_maxeig_pass_10pct": joh["maxeig_pass_10pct"],
            "Both_methods_agree":         eg_pass and joh_pass_5,
            "Tests_passed":               tests_passed,
            "Selected":                   selected,
        })

        eg_flag  = "PASS" if eg_pass   else "FAIL"
        joh_flag = "PASS" if joh_pass_5 else ("10%" if joh_pass_10 else "FAIL")
        print(f"  {pair_label:<12}  dir={chosen_dir:<13}  "
              f"EG p={eg_pval:.4f} [{eg_flag}]  "
              f"Johansen trace={joh['trace_stat_r0']:.2f} vs "
              f"{joh['trace_crit_r0_5pct']:.2f} (5%) / "
              f"{joh['trace_crit_r0_10pct']:.2f} (10%) [{joh_flag}]")

    # -------------------------------------------------------------------------
    # 4b. Rank by EG p-value and Johansen ratio; compute combined rank
    # -------------------------------------------------------------------------
    results_df = pd.DataFrame(records)

    # EG rank: ascending p-value (rank 1 = strongest cointegration evidence)
    results_df["EG_rank"] = results_df["EG_pval"].rank(method="min").astype(int)

    # Johansen rank: trace_stat / 5% CV; descending (rank 1 = highest ratio)
    results_df["Johansen_ratio"] = (
        results_df["Johansen_trace_stat_r0"] / results_df["Johansen_trace_crit_r0"]
    ).round(4)
    results_df["Johansen_rank"] = (
        results_df["Johansen_ratio"].rank(method="min", ascending=False).astype(int)
    )

    # Combined rank: average of EG_rank and Johansen_rank, then re-ranked
    results_df["Combined_rank_score"] = (
        (results_df["EG_rank"] + results_df["Johansen_rank"]) / 2
    )
    results_df["Combined_rank"] = (
        results_df["Combined_rank_score"].rank(method="min").astype(int)
    )

    results_df = results_df.sort_values("EG_pval").reset_index(drop=True)

    # Bonferroni-corrected flag stored alongside the nominal 5% flag so it
    # appears in the saved CSV and is easy to query downstream.
    _bonf_threshold = 0.05 / len(pairs)
    results_df["EG_cointegrated_bonferroni"] = results_df["EG_pval"] < _bonf_threshold

    # -------------------------------------------------------------------------
    # 5. Save full results table
    # -------------------------------------------------------------------------
    out_csv = REPORTS_DIR / "cointegration_results.csv"
    results_df.to_csv(out_csv, index=False)
    print(f"\nFull results saved -> {out_csv}")

    # -------------------------------------------------------------------------
    # 6. Print ranking summary to console
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("RANKING TABLE -- EG P-VALUE | JOHANSEN RATIO | COMBINED")
    print("=" * 80)
    hdr = (f"{'EGr':>4} {'JOr':>4} {'Cr':>4}  {'Pair':<12}  "
           f"{'EG p':>7}  {'EG':>6}  {'JO ratio':>9}  {'JO':>6}  {'Comb':>5}")
    print(hdr)
    print("-" * len(hdr))
    for _, row in results_df.iterrows():
        eg_tag  = "PASS" if row["EG_cointegrated_5pct"] else "FAIL"
        if row["Johansen_trace_pass_5pct"]:
            joh_tag = "PASS"
        elif row["Johansen_trace_pass_10pct"]:
            joh_tag = "10%"
        else:
            joh_tag = "FAIL"
        print(f"  {row['EG_rank']:>3} {row['Johansen_rank']:>4} {row['Combined_rank']:>4}  "
              f"{row['Pair']:<12}  "
              f"{row['EG_pval']:>7.4f}  "
              f"{eg_tag:>6}  "
              f"{row['Johansen_ratio']:>9.3f}  "
              f"{joh_tag:>6}  "
              f"{row['Combined_rank_score']:>5.1f}")

    # -------------------------------------------------------------------------
    # 7. Shortlist -- selected pairs
    # -------------------------------------------------------------------------
    shortlist = results_df[results_df["Selected"]]
    print(f"\n{'='*65}")
    print("SHORTLISTED PAIRS")
    print(f"{'='*65}")
    print("  Primary criterion  : EG OR Johansen trace passes at 5%")
    print("  Secondary criterion: Johansen trace passes at 10% (supervisor-approved)")
    print()
    if len(shortlist) == 0:
        print("  No pairs selected.")
    else:
        for _, row in shortlist.iterrows():
            basis = ("(secondary -- Johansen 10%, supervisor-approved)"
                     if row["Tests_passed"] == "Johansen 10%"
                     else "(primary)")
            print(f"  EG rank {row['EG_rank']} / JO rank {row['Johansen_rank']} / Combined {row['Combined_rank']}: "
              f"{row['Pair']}  [{row['Tests_passed']}]  {basis}")
            print(f"    OLS dir={row['OLS_direction']}  HR={row['Hedge_ratio']}  "
                  f"EG p={row['EG_pval']:.4f}  "
                  f"Johansen trace={row['Johansen_trace_stat_r0']} "
                  f"vs {row['Johansen_trace_crit_r0']} (5%) "
                  f"/ {row['Johansen_trace_crit_r0_10pct']} (10%)")

    # -------------------------------------------------------------------------
    # Multiple-comparisons note (Bonferroni correction)
    # -------------------------------------------------------------------------
    n_tests        = len(pairs)           # 15
    alpha_nominal  = 0.05
    alpha_bonf     = alpha_nominal / n_tests   # 0.0033...
    shortlist_bonf = results_df[results_df["EG_pval"] < alpha_bonf]

    print(f"\n{'='*65}")
    print("MULTIPLE-COMPARISONS WARNING (Bonferroni correction)")
    print(f"{'='*65}")
    print(f"  Tests run        : {n_tests} pairs at nominal alpha={alpha_nominal}")
    print(f"  Expected false positives by chance: "
          f"{n_tests * alpha_nominal:.2f}  (i.e. ~1 false positive is likely)")
    print(f"  Bonferroni threshold : alpha / {n_tests} = {alpha_bonf:.4f}")
    if len(shortlist_bonf) == 0:
        print(f"  Result: NO pair survives Bonferroni correction.")
        print(f"  The pair(s) shortlisted at 5% may be statistical false positives.")
    else:
        print(f"  Pairs that also survive Bonferroni (EG p < {alpha_bonf:.4f}):")
        for _, row in shortlist_bonf.iterrows():
            print(f"    {row['Pair']}  EG p={row['EG_pval']:.4f}")
    print(f"  NOTE: Johansen confirmation reduces (but does not eliminate) the")
    print(f"  false-positive risk when the EG test alone is marginal.")

    # -------------------------------------------------------------------------
    # 8. Plot OLS residual (spread) series for all 15 pairs
    #    Blue     = both EG + Johansen pass at 5%
    #    Orange   = one of EG / Johansen passes at 5%
    #    Goldenrod = Johansen trace passes at 10% only (supervisor-approved)
    #    Grey     = neither
    # -------------------------------------------------------------------------
    n_cols = 3
    n_rows = (len(pairs) + n_cols - 1) // n_cols   # 5 rows for 15 pairs

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, n_rows * 3.4))
    fig.suptitle(
        "OLS Residual (Spread) Series -- All 15 Pairs (Log Prices, 2018-2025)\n"
        "Blue = Both EG+Johansen 5% | Orange = One method 5% | "
        "Gold = Johansen 10% only | Grey = Neither",
        fontsize=11, y=1.01
    )

    axes_flat = axes.flatten()
    for i, (pair_label, resid) in enumerate(spreads.items()):
        ax = axes_flat[i]
        row = results_df[results_df["Pair"] == pair_label].iloc[0]
        eg_p     = row["EG_cointegrated_5pct"]
        joh_p5   = row["Johansen_trace_pass_5pct"]
        joh_p10  = row["Johansen_trace_pass_10pct"]

        if eg_p and joh_p5:
            color, tag = "steelblue", "EG + JOH"
        elif eg_p or joh_p5:
            color, tag = "darkorange", "EG" if eg_p else "JOH"
        elif joh_p10:
            color, tag = "goldenrod", "JOH 10%"
        else:
            color, tag = "slategray", "none"

        ax.plot(resid.index, resid.values, color=color, linewidth=0.75, alpha=0.9)
        ax.axhline(0, color="black", linewidth=0.6, linestyle="--")
        ax.set_title(
            f"{pair_label}  [{tag}]  EG p={row['EG_pval']:.4f}  HR={row['Hedge_ratio']:.3f}",
            fontsize=8.5
        )
        ax.tick_params(axis="x", labelsize=6.5, rotation=30)
        ax.tick_params(axis="y", labelsize=7)
        ax.set_xlabel("")

    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)

    plt.tight_layout()
    chart1 = CHARTS_DIR / "spread_series_all_pairs.png"
    plt.savefig(chart1, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSpread chart saved -> {chart1}")

    # -------------------------------------------------------------------------
    # 9. Ranking chart -- EG p-value (top) and Johansen ratio (bottom)
    # -------------------------------------------------------------------------
    from matplotlib.patches import Patch

    # Sort by combined rank for a consistent y-axis order on both panels
    _rank_df = results_df.sort_values("Combined_rank", ascending=False).reset_index(drop=True)

    _bar_colors = [
        "steelblue"   if (eg and joh5) else
        "darkorange"  if (eg or joh5)  else
        "goldenrod"   if joh10         else
        "lightcoral"
        for eg, joh5, joh10 in zip(
            _rank_df["EG_cointegrated_5pct"],
            _rank_df["Johansen_trace_pass_5pct"],
            _rank_df["Johansen_trace_pass_10pct"],
        )
    ]

    fig2, (ax2a, ax2b) = plt.subplots(1, 2, figsize=(16, 7))
    fig2.suptitle(
        "Cointegration Ranking — All 15 Pairs\n"
        "Blue = Both 5% | Orange = One 5% | Gold = Johansen 10% | Red = Neither",
        fontsize=11
    )

    # Top panel: EG p-value
    ax2a.barh(_rank_df["Pair"], _rank_df["EG_pval"], color=_bar_colors,
              edgecolor="white", linewidth=0.5)
    ax2a.axvline(0.05, color="red", linestyle="--", linewidth=1.3)
    ax2a.set_xlabel("EG p-value  (lower → stronger)", fontsize=10)
    ax2a.set_title("EG rank  (ascending p-value)", fontsize=10)
    ax2a.invert_yaxis()

    # Bottom panel: Johansen ratio (trace stat / 5% CV)
    ax2b.barh(_rank_df["Pair"], _rank_df["Johansen_ratio"], color=_bar_colors,
              edgecolor="white", linewidth=0.5)
    ax2b.axvline(1.0, color="red", linestyle="--", linewidth=1.3,
                 label="Ratio = 1.0  (5% CV)")
    ax2b.set_xlabel("Johansen ratio  (trace stat / 5% CV,  higher → stronger)", fontsize=10)
    ax2b.set_title("Johansen rank  (descending ratio)", fontsize=10)
    ax2b.invert_yaxis()

    legend_items = [
        Patch(facecolor="steelblue",  label="Both EG + Johansen 5%"),
        Patch(facecolor="darkorange", label="One method at 5%"),
        Patch(facecolor="goldenrod",  label="Johansen 10% only"),
        Patch(facecolor="lightcoral", label="Neither"),
        plt.Line2D([0], [0], color="red", linestyle="--", label="5% threshold"),
    ]
    ax2b.legend(handles=legend_items, fontsize=8, loc="lower right")

    plt.tight_layout()
    chart2 = CHARTS_DIR / "cointegration_ranking.png"
    plt.savefig(chart2, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Ranking chart saved -> {chart2}")

    # -------------------------------------------------------------------------
    # 10. Comparison summary -- EG vs Johansen agreement
    # -------------------------------------------------------------------------
    print(f"\n{'='*65}")
    print("EG vs JOHANSEN AGREEMENT SUMMARY")
    print(f"{'='*65}")
    both      = results_df["Both_methods_agree"].sum()
    eg_only   = (results_df["EG_cointegrated_5pct"] & ~results_df["Johansen_trace_pass_5pct"]).sum()
    joh_only  = (~results_df["EG_cointegrated_5pct"] & results_df["Johansen_trace_pass_5pct"]).sum()
    joh_10    = (results_df["Tests_passed"] == "Johansen 10%").sum()
    neither   = (~results_df["EG_cointegrated_5pct"] & ~results_df["Johansen_trace_pass_5pct"]
                 & ~results_df["Johansen_trace_pass_10pct"]).sum()
    print(f"  Both EG and Johansen 5% pass  : {both:2d} pair(s)")
    print(f"  EG 5% only                    : {eg_only:2d} pair(s)")
    print(f"  Johansen 5% only              : {joh_only:2d} pair(s)")
    print(f"  Johansen 10% only (borderline): {joh_10:2d} pair(s)  [supervisor-approved secondary]")
    print(f"  Neither                       : {neither:2d} pair(s)")

    # -------------------------------------------------------------------------
    # 11. NVDA restricted window check (2018-01-01 to 2022-12-31)
    #     Phase 2 EDA flagged that NVDA decoupled from peers from mid-2023 onward
    #     due to the AI/GPU-driven price surge. Restricting to 2018-2022 removes
    #     that structural break and gives the NVDA pairs the best possible chance
    #     of passing cointegration tests.
    # -------------------------------------------------------------------------
    NVDA_END = "2022-12-31"
    nvda_pairs = [p for p in pairs if "NVDA" in p]
    log_nvda_window = log_price_df.loc[:NVDA_END]

    print(f"\n{'='*65}")
    print(f"NVDA SUB-PERIOD CHECK (2018-01-01 to {NVDA_END})")
    print(f"{'='*65}")
    print(f"  Window : {log_nvda_window.index[0].date()} to {log_nvda_window.index[-1].date()}  "
          f"({len(log_nvda_window)} days)")
    print(f"  Pairs  : {[f'{a}/{b}' for a,b in nvda_pairs]}\n")

    nvda_records = []
    for a, b in nvda_pairs:
        log_a = log_nvda_window[a]
        log_b = log_nvda_window[b]

        (chosen_dir, hedge_ratio, _intercept,
         resid, adf_stat, adf_pval, dep, indep) = pick_direction(log_a, log_b, a, b)

        eg_stat, eg_pval = engle_granger(log_nvda_window[dep], log_nvda_window[indep])
        joh = johansen(log_nvda_window[[a, b]], k_ar_diff=K_AR_DIFF)

        eg_pass  = eg_pval < 0.05
        joh_pass = joh["trace_pass_5pct"]
        eg_flag  = "PASS" if eg_pass  else "FAIL"
        joh_flag = "PASS" if joh_pass else "FAIL"

        print(f"  {a}/{b:<12}  dir={chosen_dir:<13}  "
              f"EG p={eg_pval:.4f} [{eg_flag}]  "
              f"Johansen trace={joh['trace_stat_r0']:.2f} vs {joh['trace_crit_r0_5pct']:.2f} [{joh_flag}]")

        nvda_records.append({
            "Pair": f"{a}/{b}", "Window": f"2018-{NVDA_END[:4]}",
            "OLS_direction": chosen_dir, "Hedge_ratio": round(hedge_ratio, 4),
            "ADF_pval": round(adf_pval, 4), "EG_pval": round(eg_pval, 4),
            "EG_pass": eg_pass, "Johansen_trace_pass": joh_pass,
        })

    nvda_df = pd.DataFrame(nvda_records)
    nvda_csv = REPORTS_DIR / "nvda_subperiod_results.csv"
    nvda_df.to_csv(nvda_csv, index=False)

    any_nvda_pass = nvda_df["EG_pass"].any() or nvda_df["Johansen_trace_pass"].any()
    print(f"\n  Result: {'At least one NVDA pair passes in the restricted window.' if any_nvda_pass else 'All NVDA pairs still fail in the restricted 2018-2022 window.'}")
    print(f"  Saved -> {nvda_csv}")
    print(f"\n  Interpretation: Restricting to 2018-2022 removes the AI-driven structural")
    print(f"  break in NVDA from mid-2023, but the cointegrating relationship with peer")
    print(f"  stocks was also not established in the earlier period.")

    # -------------------------------------------------------------------------
    # 11b. TRAINING-WINDOW COINTEGRATION SCREEN (2018-2021 only)
    #
    #      Phase 3b splits data at TRAIN_END = 2021-12-31 / TEST_START = 2022-01-01.
    #      The full-period screen above uses 2018-2025, which overlaps the test
    #      window (2022-2025) — a look-ahead issue: pair selection has already
    #      "seen" the test data.  This block reruns the same EG + Johansen tests
    #      using only the training window so the selection decision is independent
    #      of out-of-sample data.
    #
    #      Results are saved to cointegration_results_train_only.csv.
    #      selected_pairs.csv (the downstream contract) is NOT changed here;
    #      see the comparison table printed below.
    # -------------------------------------------------------------------------
    log_train = log_price_df.loc[:TRAIN_END]

    print(f"\n{'='*65}")
    print(f"TRAINING-WINDOW SCREEN (2018-01-01 to {TRAIN_END})")
    print(f"{'='*65}")
    print(f"  Window: {log_train.index[0].date()} to {log_train.index[-1].date()}  "
          f"({len(log_train)} days)")

    _lag_rows_train = []
    for _a, _b in pairs:
        _pair_diff = log_train[[_a, _b]].diff().dropna()
        _sel = VAR(_pair_diff).select_order(maxlags=5)
        _lag_rows_train.append(int(_sel.selected_orders["bic"]))
    _bic_median_train = int(pd.Series(_lag_rows_train).median())
    K_AR_DIFF_TRAIN   = max(1, _bic_median_train)
    print(f"  BIC-median lag for training window: {K_AR_DIFF_TRAIN}")

    train_records = []
    for a, b in pairs:
        log_a = log_train[a]
        log_b = log_train[b]

        (chosen_dir, hedge_ratio, intercept,
         resid, adf_stat, adf_pval, dep, indep) = pick_direction(log_a, log_b, a, b)

        eg_stat, eg_pval = engle_granger(log_train[dep], log_train[indep])
        joh = johansen(log_train[[a, b]], k_ar_diff=K_AR_DIFF_TRAIN)

        eg_pass     = eg_pval < 0.05
        joh_pass_5  = joh["trace_pass_5pct"]
        joh_pass_10 = joh["trace_pass_10pct"]

        if eg_pass and joh_pass_5:
            tests_passed = "Both"
        elif eg_pass:
            tests_passed = "EG only"
        elif joh_pass_5:
            tests_passed = "Johansen only"
        elif joh_pass_10:
            tests_passed = "Johansen 10%"
        else:
            tests_passed = "Neither"

        selected_train = classify_pair(eg_pass, joh_pass_5, joh_pass_10) != "none"

        train_records.append({
            "Pair":                  f"{a}/{b}",
            "OLS_direction":         chosen_dir,
            "Hedge_ratio":           round(hedge_ratio, 4),
            "EG_pval":               round(eg_pval, 4),
            "EG_pass":               eg_pass,
            "Johansen_trace_stat":   joh["trace_stat_r0"],
            "Johansen_trace_crit_5": joh["trace_crit_r0_5pct"],
            "Johansen_trace_pass_5": joh_pass_5,
            "Johansen_trace_pass_10": joh_pass_10,
            "Tests_passed":          tests_passed,
            "Selected":              selected_train,
        })

    train_df     = pd.DataFrame(train_records)
    train_csv    = REPORTS_DIR / "cointegration_results_train_only.csv"
    train_df.to_csv(train_csv, index=False)
    print(f"  Saved -> {train_csv}")

    # Print comparison: full-period selection vs training-window selection
    print(f"\n  {'Pair':<12}  {'Full 2018-2025':>15}  {'Train 2018-2021':>16}  {'Agreement':>10}")
    print(f"  {'-'*60}")
    for _, tr in train_df.iterrows():
        pair = tr["Pair"]
        full_row   = results_df[results_df["Pair"] == pair].iloc[0]
        full_sel   = full_row["Selected"]
        train_sel  = tr["Selected"]
        full_tag   = full_row["Tests_passed"] if full_sel  else "—"
        train_tag  = tr["Tests_passed"]       if train_sel else "—"
        agree      = "SAME" if full_sel == train_sel else "DIFFERS"
        print(f"  {pair:<12}  {full_tag:>15}  {train_tag:>16}  {agree:>10}")

    train_selected = train_df[train_df["Selected"]]
    full_selected_pairs  = set(results_df[results_df["Selected"]]["Pair"])
    train_selected_pairs = set(train_selected["Pair"])
    added   = train_selected_pairs - full_selected_pairs
    dropped = full_selected_pairs  - train_selected_pairs
    same    = full_selected_pairs  == train_selected_pairs

    print(f"\n  Full-period selected : {sorted(full_selected_pairs)}")
    print(f"  Train-only selected  : {sorted(train_selected_pairs)}")
    if same:
        print(f"\n  RESULT: Both screens select the SAME pairs.")
        print(f"  The look-ahead does not change which pairs are taken forward.")
    else:
        print(f"\n  RESULT: Selection DIFFERS between the two windows.")
        if added:
            print(f"  Pairs added by training-only screen   : {sorted(added)}")
        if dropped:
            print(f"  Pairs dropped by training-only screen : {sorted(dropped)}")
        print(f"  See comparison table above for details.")

    # -------------------------------------------------------------------------
    # 12. Export selected_pairs.csv -- contract for downstream phases
    #
    #     Selection basis:
    #       Primary   -- at least one of EG or Johansen trace passes at 5%.
    #       Secondary -- Johansen trace passes at 10% (carried forward per
    #                    supervisor instruction); Tests_passed = "Johansen 10%".
    #
    #     The Tests_passed column records which criterion applies.
    # -------------------------------------------------------------------------
    selected_cols = [
        "Pair", "OLS_direction", "Hedge_ratio", "Intercept",
        "EG_pval", "EG_cointegrated_5pct",
        "Johansen_trace_stat_r0",
        "Johansen_trace_crit_r0",       # 5% critical value
        "Johansen_trace_crit_r0_10pct", # 10% critical value (for borderline pairs)
        "Johansen_trace_pass_5pct",
        "Johansen_trace_pass_10pct",
        "Tests_passed",
    ]
    selected_df  = results_df[results_df["Selected"]][selected_cols].copy()
    selected_csv = REPORTS_DIR / "selected_pairs.csv"
    selected_df.to_csv(selected_csv, index=False)
    print(f"\nSelected pairs ({len(selected_df)}) saved -> {selected_csv}")
    for _, row in selected_df.iterrows():
        basis = ("secondary (Johansen 10%, supervisor-approved)"
                 if row["Tests_passed"] == "Johansen 10%"
                 else "primary (5%)")
        print(f"  {row['Pair']:<12}  [{row['Tests_passed']}]  {basis}  "
              f"dir={row['OLS_direction']}  HR={row['Hedge_ratio']}")

    print(f"\n{'='*65}")
    print("PHASE 3 -- COINTEGRATION SCREENING COMPLETE")
    print(f"{'='*65}")
    print(f"\nOutputs:")
    print(f"  {out_csv}")
    print(f"  {selected_csv}")
    print(f"  {nvda_csv}")
    print(f"  {chart1}")
    print(f"  {chart2}")
    print(f"\nSelection:")
    print(f"  Primary   -- EG or Johansen trace at 5%  -> AMZN/META")
    print(f"  Secondary -- Johansen trace at 10% (supervisor-approved) -> MSFT/AAPL")
    print(f"\nNext step: Phase 3b -- pairs trading strategy on selected pairs.")


if __name__ == "__main__":
    main()

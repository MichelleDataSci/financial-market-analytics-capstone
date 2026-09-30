"""
evaluate_gate.py — Classify the LSTM convergence gate as a binary predictor.

For each selected pair and each evaluation period (2022-2025 holdout, 2026 test):
  Predicted label  = gate says converging:  mean|pred_h1..h5| < |z_t|
  Actual label     = spread actually conv.: mean|z_{t+1}..z_{t+5}| < |z_t|

Reports accuracy, precision, recall, F1, ROC-AUC for the LSTM gate and a
naive "always converging" baseline.

Also reports permutation feature importance (mean increase in 5-step RMSE
against ground-truth z-scores when each feature is shuffled across sequences)
using the 2026 test data.

Model assignment:
  2022-2025 holdout — historical model (hist_v1): OLS + scaler + LSTM all
                       trained on 2018-2020; 2022-2025 is genuinely unseen.
  2026 test         — final model (v1): OLS + scaler + LSTM all trained on
                       2018-2024; 2026 is genuinely unseen.

Saves:
  outputs/reports/gate_metrics.csv
  outputs/charts/confusion_matrix_{pair}_{period}.png  (one per pair/period)
  outputs/reports/feature_importance_{pair}.csv        (one per pair)
  outputs/charts/feature_importance_{pair}.png         (one per pair)

Usage:
  python src/evaluate_gate.py
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import REPORTS_DIR, CHARTS_DIR
from predict import load_artefacts, build_features, load_log_close

# Must match phase5_ml_spread.py / predict.py
SEQ_LEN         = 20
HORIZON         = 5
ROLLING_STD_WIN = 20
FEATURE_NAMES   = ["z_std", "lag1", "lag2", "lag3", "roll_std"]

# Each period maps to (date_start, date_end, artefact_suffix).
# holdout uses the historical model (trained 2018-2020) to avoid leakage;
# 2026 test uses the final model (trained 2018-2024).
EVAL_PERIODS = {
    "holdout_2022_2025": ("2022-01-01", "2025-12-31", "hist_v1"),
    "test_2026":         ("2026-01-01", "2026-12-31", "v1"),
}


# ---------------------------------------------------------------------------
# Testable label and score helpers
# ---------------------------------------------------------------------------

def pred_converging(preds, z_t):
    """Gate predicts convergence when mean|pred| < |z_t|. Returns 0 or 1."""
    return int(float(np.mean(np.abs(preds))) < abs(z_t))


def actual_converging(future_z, z_t):
    """Spread actually converges when mean|future_z| < |z_t|. Returns 0 or 1."""
    return int(float(np.mean(np.abs(future_z))) < abs(z_t))


def gate_score(z_t, preds):
    """Continuous score for ROC-AUC. Higher = model more confident in convergence."""
    return abs(z_t) - float(np.mean(np.abs(preds)))


# ---------------------------------------------------------------------------
# Evaluation array builder
# ---------------------------------------------------------------------------

def build_eval_arrays(dep, indep, model, ols, scaler):
    """
    Build all sequences and labels for classification evaluation.

    Returns:
      eval_df  — DataFrame indexed by date with z_std, pred_mean_abs, score,
                 pred_label, actual_label, _row_idx (row position in X/y arrays)
      X        — (N, SEQ_LEN, n_features) float32 input sequences
      y_actual — (N, HORIZON) ground-truth z_std values for h1..h5
    """
    log_close = load_log_close(dep, indep)
    spread    = log_close[dep] - ols["hedge_ratio"] * log_close[indep] - ols["intercept"]
    z_std_ser = (spread - scaler["mu"]) / scaler["sigma"]

    feat_df = build_features(z_std_ser).dropna()
    z_vals  = feat_df["z_std"].values
    dates   = feat_df.index
    n       = len(feat_df)

    # Bar i: sequence = feat_df[i-SEQ_LEN : i], current bar = feat_df[i-1]
    eval_bars = list(range(SEQ_LEN, n - HORIZON))
    X = np.stack([
        feat_df.iloc[i - SEQ_LEN : i].values.astype("float32")
        for i in eval_bars
    ])  # (N, SEQ_LEN, n_features)

    preds_all = model.predict(X, verbose=0)  # (N, HORIZON)

    y_actual = np.stack([
        z_vals[i : i + HORIZON]
        for i in eval_bars
    ])  # (N, HORIZON)

    rows = []
    for k, i in enumerate(eval_bars):
        z_t    = float(z_vals[i - 1])
        pred_k = preds_all[k]
        fut_k  = y_actual[k]
        rows.append({
            "date":          dates[i - 1],
            "z_std":         z_t,
            "pred_mean_abs": float(np.mean(np.abs(pred_k))),
            "score":         gate_score(z_t, pred_k),
            "pred_label":    pred_converging(pred_k, z_t),
            "actual_label":  actual_converging(fut_k, z_t),
            "_row_idx":      k,
        })

    eval_df = pd.DataFrame(rows).set_index("date")
    return eval_df, X, y_actual


# ---------------------------------------------------------------------------
# Classification metrics
# ---------------------------------------------------------------------------

def compute_clf_metrics(y_true, y_pred, scores, label="LSTM gate"):
    """Return dict with accuracy, precision, recall, F1, ROC-AUC."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    scores = np.asarray(scores, dtype=float)
    n = len(y_true)
    if n == 0:
        return {k: float("nan") for k in
                ["label", "n", "accuracy", "precision", "recall", "f1", "roc_auc"]}
    try:
        roc = float(roc_auc_score(y_true, scores))
    except ValueError:
        roc = float("nan")  # only one class in y_true
    return {
        "label":     label,
        "n":         n,
        "accuracy":  float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall":    float(recall_score(y_true, y_pred, zero_division=0)),
        "f1":        float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc":   roc,
    }


def naive_metrics(y_true):
    """Naive baseline: always predict 'converging' (label = 1)."""
    y_true = np.asarray(y_true)
    y_pred = np.ones(len(y_true), dtype=int)
    scores = np.ones(len(y_true), dtype=float)  # constant score → ROC-AUC = 0.5
    return compute_clf_metrics(y_true, y_pred, scores, label="Naive (always conv.)")


# ---------------------------------------------------------------------------
# Confusion matrix chart
# ---------------------------------------------------------------------------

def save_confusion_matrix_chart(y_true, y_pred, pair, period, n):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues", vmin=0)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Not conv.", "Converging"])
    ax.set_yticklabels(["Not conv.", "Converging"])
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("Actual label")
    period_str = period.replace("_", " ").replace("holdout ", "Holdout ").replace("test ", "Test ").title()
    ax.set_title(
        f"LSTM Gate — Confusion Matrix\n"
        f"{pair.replace('_', '/')}  {period_str}  (n={n})"
    )
    thresh = cm.max() * 0.55
    for i in range(2):
        for j in range(2):
            val = cm[i, j]
            ax.text(j, i, str(val), ha="center", va="center",
                    color="white" if val > thresh else "black",
                    fontsize=14, fontweight="bold")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    out = CHARTS_DIR / f"confusion_matrix_{pair}_{period}.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {out.name}")


# ---------------------------------------------------------------------------
# Permutation feature importance
# ---------------------------------------------------------------------------

def compute_permutation_importance(model, X, y_actual, n_repeats=5, rng_seed=42):
    """
    For each feature, shuffle its values across all sequences and measure the
    mean increase in 5-step RMSE against ground-truth z-scores over n_repeats.

    X        : (N, SEQ_LEN, n_features)
    y_actual : (N, HORIZON) ground-truth z_std values h1..h5

    Returns (importance_dict, baseline_rmse).
    """
    rng = np.random.default_rng(rng_seed)

    baseline_preds = model.predict(X, verbose=0)
    baseline_rmse  = float(np.sqrt(np.mean((baseline_preds - y_actual) ** 2)))
    print(f"    Baseline 5-step RMSE (2026): {baseline_rmse:.4f}")

    importance = {}
    for f_idx, fname in enumerate(FEATURE_NAMES):
        deltas = []
        for _ in range(n_repeats):
            X_s = X.copy()
            X_s[:, :, f_idx] = X[rng.permutation(len(X)), :, f_idx]
            shuf_preds = model.predict(X_s, verbose=0)
            shuf_rmse  = float(np.sqrt(np.mean((shuf_preds - y_actual) ** 2)))
            deltas.append(shuf_rmse - baseline_rmse)
        importance[fname] = float(np.mean(deltas))

    return importance, baseline_rmse


def save_importance_chart(importance, pair, baseline_rmse):
    features = list(importance.keys())
    values   = [importance[f] for f in features]
    colours  = ["steelblue" if v >= 0 else "lightcoral" for v in values]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(features, values, color=colours, edgecolor="white", linewidth=0.5)
    ax.axhline(0, color="black", lw=0.8, ls="--")
    ax.set_xlabel("Feature")
    ax.set_ylabel("Mean increase in 5-step RMSE")
    ax.set_title(
        f"Permutation Feature Importance — {pair.replace('_', '/')}\n"
        f"Baseline RMSE = {baseline_rmse:.4f} · 2026 test · {5} repeats"
    )
    for bar, val in zip(bars, values):
        offset = max(abs(v) for v in values) * 0.03
        ypos   = bar.get_height() + offset if val >= 0 else bar.get_height() - offset * 3
        ax.text(bar.get_x() + bar.get_width() / 2, ypos,
                f"{val:+.4f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    out = CHARTS_DIR / f"feature_importance_{pair}.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {out.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    sel_csv = REPORTS_DIR / "selected_pairs.csv"
    if not sel_csv.exists():
        sys.exit("selected_pairs.csv not found. Run phase3_cointegration.py first.")

    sel_df = pd.read_csv(sel_csv)
    all_metrics: list[dict] = []

    for _, sel_row in sel_df.iterrows():
        dep, indep = sel_row["OLS_direction"].split("~")
        tag = f"{dep}_{indep}"
        print(f"\n{'='*60}")
        print(f"Pair: {dep}/{indep}  [{sel_row['Tests_passed']}]")
        print("="*60)

        # Load both models upfront; cache by suffix to avoid reloading per period
        loaded_artefacts: dict[str, tuple] = {}

        # Per-period classification evaluation
        for period_name, (start, end, suffix) in EVAL_PERIODS.items():
            if suffix not in loaded_artefacts:
                print(f"  Loading {suffix} artefacts ...")
                loaded_artefacts[suffix] = load_artefacts(tag, suffix=suffix)

            model, ols, scaler = loaded_artefacts[suffix]
            print(f"  Building eval arrays for {period_name} "
                  f"(model={suffix}) ...")
            eval_df, X_all, y_actual_all = build_eval_arrays(
                dep, indep, model, ols, scaler
            )

            prd = eval_df.loc[start:end].copy()
            if len(prd) < 5:
                print(f"  [{period_name}] Too few bars ({len(prd)}), skipping.")
                continue

            y_true = prd["actual_label"].values
            y_pred = prd["pred_label"].values
            scores = prd["score"].values
            n      = len(prd)

            lstm_m  = compute_clf_metrics(y_true, y_pred, scores, label="LSTM gate")
            naive_m = naive_metrics(y_true)

            print(f"\n  [{period_name}]  model={suffix}  n={n}  "
                  f"convergence_rate={y_true.mean():.2f}")
            hdr = f"  {'Model':<30}  {'acc':>5}  {'prec':>5}  {'rec':>5}  {'f1':>5}  {'roc_auc':>7}"
            print(hdr)
            for m in [lstm_m, naive_m]:
                print(
                    f"  {m['label']:<30}  "
                    f"{m['accuracy']:5.3f}  "
                    f"{m['precision']:5.3f}  "
                    f"{m['recall']:5.3f}  "
                    f"{m['f1']:5.3f}  "
                    f"{m['roc_auc']:7.3f}"
                )
                all_metrics.append({
                    "pair": tag, "period": period_name,
                    "model_suffix": suffix, **m,
                })

            save_confusion_matrix_chart(y_true, y_pred, tag, period_name, n)

        # Permutation feature importance on 2026 test using the final v1 model
        print(f"\n  Permutation feature importance "
              f"(2026 test · v1 model · {5} repeats × {len(FEATURE_NAMES)} features) ...")
        v1_model, v1_ols, v1_scaler = loaded_artefacts.get("v1") or load_artefacts(tag, suffix="v1")
        eval_df_v1, X_all_v1, y_actual_v1 = build_eval_arrays(
            dep, indep, v1_model, v1_ols, v1_scaler
        )
        test_prd = eval_df_v1.loc["2026-01-01":"2026-12-31"]
        if len(test_prd) < 20:
            print(f"  Too few 2026 bars ({len(test_prd)}), skipping importance.")
        else:
            t_idx   = test_prd["_row_idx"].values.astype(int)
            X_t     = X_all_v1[t_idx]
            y_t     = y_actual_v1[t_idx]
            imp, bl = compute_permutation_importance(v1_model, X_t, y_t)
            save_importance_chart(imp, tag, bl)

            imp_csv = REPORTS_DIR / f"feature_importance_{tag}.csv"
            pd.DataFrame([
                {"feature": k, "mean_rmse_increase": v}
                for k, v in imp.items()
            ]).to_csv(imp_csv, index=False)
            print(f"    Saved: {imp_csv.name}")
            for k, v in imp.items():
                print(f"    {k:<12}  ΔRMSE = {v:+.4f}")

    # Save consolidated gate metrics CSV
    metrics_csv = REPORTS_DIR / "gate_metrics.csv"
    cols = ["pair", "period", "model_suffix", "label", "n",
            "accuracy", "precision", "recall", "f1", "roc_auc"]
    (
        pd.DataFrame(all_metrics)[cols]
        .round(4)
        .to_csv(metrics_csv, index=False)
    )
    print(f"\n  Gate metrics saved -> {metrics_csv.name}")
    print("\n" + "="*60)
    print("EVALUATION COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()

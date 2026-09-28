"""
predict.py — Load saved LSTM artefacts and print the latest 5-day z-score
forecast and convergence gate decision for each selected pair.

No retraining. Reads:
    models/{TAG}_lstm_v1.keras
    models/{TAG}_ols_v1.joblib
    models/{TAG}_scaler_v1.joblib
    data/raw/{dep}_raw.csv
    data/raw/{indep}_raw.csv
    outputs/reports/selected_pairs.csv

Usage:
    python src/predict.py
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import DATA_RAW, REPORTS_DIR, MODELS_DIR, load_or_download_2026

# Must match phase5_ml_spread.py
SEQ_LEN         = 20
HORIZON         = 5
ROLLING_STD_WIN = 20


def load_artefacts(tag):
    """Return (model, ols_params, scaler_params) for the given pair tag."""
    import tensorflow as tf  # deferred: avoid TF startup cost at import time
    model  = tf.keras.models.load_model(MODELS_DIR / f"{tag}_lstm_v1.keras")
    ols    = joblib.load(MODELS_DIR / f"{tag}_ols_v1.joblib")
    scaler = joblib.load(MODELS_DIR / f"{tag}_scaler_v1.joblib")
    return model, ols, scaler


def load_log_close(dep, indep):
    """
    Load log close prices for dep and indep.

    Concatenates the main raw CSV (2018-2025) with the cached 2026 CSV so the
    forecast uses the latest available date (July 2026).  The 2026 file is read
    from cache only (no network call); if it is absent the function falls back
    to 2025-end data and prints a warning.
    """
    frames = {}
    for ticker in [dep, indep]:
        path = DATA_RAW / f"{ticker}_raw.csv"
        df = pd.read_csv(path, index_col="Date", parse_dates=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        series = df["Close"]

        # Append cached 2026 data if available
        path_2026 = DATA_RAW / f"{ticker}_2026.csv"
        if path_2026.exists():
            df26 = pd.read_csv(path_2026, index_col="Date", parse_dates=True)
            new_rows = df26["Close"].loc[df26.index > series.index[-1]]
            if len(new_rows) > 0:
                series = pd.concat([series, new_rows])
        else:
            print(f"  Warning: {path_2026.name} not found — using data up to {series.index[-1].date()}")

        frames[ticker] = series

    close_df = pd.DataFrame(frames).dropna()
    return np.log(close_df)


def build_features(z_std):
    """Build feature DataFrame: z_std, lag1, lag2, lag3, roll_std."""
    df             = pd.DataFrame({"z_std": z_std})
    df["lag1"]     = df["z_std"].shift(1)
    df["lag2"]     = df["z_std"].shift(2)
    df["lag3"]     = df["z_std"].shift(3)
    df["roll_std"] = df["z_std"].rolling(ROLLING_STD_WIN).std()
    return df


def forecast_with_artefacts(dep, indep, model, ols, scaler):
    """
    Run inference for one pair using pre-loaded artefacts.

    Returns a dict with:
      signal_date  — last date in the feature window
      z_std        — standardised spread at signal_date
      forecast     — list of HORIZON predicted z_std values (h1 … h5)
      converging   — True if mean(|pred|) < |z_std|  (gate allows a trade)
    """
    log_close = load_log_close(dep, indep)
    spread    = log_close[dep] - ols["hedge_ratio"] * log_close[indep] - ols["intercept"]
    z_std     = (spread - scaler["mu"]) / scaler["sigma"]

    feat_df = build_features(z_std).dropna()
    if len(feat_df) < SEQ_LEN:
        raise RuntimeError(
            f"Not enough data for {dep}/{indep}: need {SEQ_LEN} rows, got {len(feat_df)}"
        )

    seq   = feat_df.iloc[-SEQ_LEN:].values.astype("float32")[np.newaxis]  # (1, 20, 5)
    preds = model.predict(seq, verbose=0)[0]                               # (5,)

    z_now      = float(feat_df["z_std"].iloc[-1])
    converging = float(np.mean(np.abs(preds))) < abs(z_now)

    return {
        "dep":         dep,
        "indep":       indep,
        "signal_date": feat_df.index[-1],
        "z_std":       z_now,
        "forecast":    preds.tolist(),
        "converging":  converging,
    }


def forecast_pair(dep, indep):
    """Load artefacts then run inference. Convenience wrapper around forecast_with_artefacts."""
    model, ols, scaler = load_artefacts(f"{dep}_{indep}")
    return forecast_with_artefacts(dep, indep, model, ols, scaler)


def main():
    selected_csv = REPORTS_DIR / "selected_pairs.csv"
    if not selected_csv.exists():
        sys.exit(
            f"selected_pairs.csv not found at {selected_csv}. "
            "Run phase3_cointegration.py first."
        )

    sel_df = pd.read_csv(selected_csv)
    print("=" * 60)
    print("LSTM SPREAD FORECAST (latest 5 trading days ahead)")
    print("=" * 60)

    for _, row in sel_df.iterrows():
        dep, indep = row["OLS_direction"].split("~")

        print(f"\nPair: {dep} / {indep}  [{row['Tests_passed']}]")
        r = forecast_pair(dep, indep)

        print(f"  Signal date      : {r['signal_date'].date()}")
        print(f"  z_std (current)  : {r['z_std']:+.3f}")
        print(
            f"  5-day forecast   : "
            + "  ".join(f"h{i+1}={v:+.3f}" for i, v in enumerate(r["forecast"]))
        )
        mean_abs = float(np.mean(np.abs(r["forecast"])))
        gate     = "ALLOWS TRADE" if r["converging"] else "BLOCKS TRADE"
        label    = "converging" if r["converging"] else "diverging"
        print(f"  Convergence gate : {gate}  ({label})")
        print(f"  mean|pred|={mean_abs:.3f}  |z_now|={abs(r['z_std']):.3f}")

    print("\n" + "=" * 60)
    print("FORECAST COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()

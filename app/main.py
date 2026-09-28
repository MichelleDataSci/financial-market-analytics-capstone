"""
Pairs Trading Analyser — FastAPI application.

Run locally (from project root):
    uvicorn app.main:app --reload

Endpoints:
    GET  /                  — form: select cached tickers or upload CSV
    POST /analyse           — HTML results page (Jinja2)
    POST /api/analyse       — same analysis as JSON (no charts)

Input options (mutually exclusive — file takes priority):
    A) ticker1 + ticker2 form fields  → reads data/raw/{ticker}_raw.csv
    B) CSV file upload                → Date column + two numeric price columns

No live downloads; no credentials required.
"""

import base64
import io
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from phase3_cointegration import (  # noqa: E402
    classify_pair, engle_granger, johansen, pick_direction, select_johansen_lag,
)
from phase3_strategy import (  # noqa: E402
    Z_ENTRY, Z_EXIT, Z_STOP, LOOKBACK, COST_PER_LEG,
    backtest, build_open_spread, build_spread_zscore,
    compute_metrics, extract_trade_log,
)
from utils import DATA_RAW  # noqa: E402

app = FastAPI(
    title="Pairs Trading Analyser",
    description="Cointegration test, spread analysis and signal generation.",
)
templates = Jinja2Templates(directory=Path(__file__).parent / "templates")

MIN_ROWS = 60   # rolling z-score needs LOOKBACK + enough bars to trade

# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _available_tickers() -> list[str]:
    return sorted(
        p.stem.replace("_raw", "")
        for p in DATA_RAW.glob("*_raw.csv")
        if not p.stem.startswith(("sp500", "vix"))
    )


def _load_cached(ticker1: str, ticker2: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load Close and Open from {ticker}_raw.csv files."""
    if ticker1 == ticker2:
        raise ValueError("Both tickers are the same — choose two different assets.")
    available = _available_tickers()
    for t in [ticker1, ticker2]:
        if t not in available:
            raise ValueError(
                f"Ticker '{t}' not in cached data. "
                f"Available: {', '.join(available)}"
            )
    close, opens = {}, {}
    for t in [ticker1, ticker2]:
        df = pd.read_csv(DATA_RAW / f"{t}_raw.csv", index_col="Date", parse_dates=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        close[t] = df["Close"]
        opens[t] = df["Open"]
    close_df = pd.DataFrame(close).dropna()
    open_df  = pd.DataFrame(opens).reindex(close_df.index).ffill().dropna()
    return close_df, open_df


def _parse_uploaded_csv(contents: bytes) -> tuple[pd.DataFrame, str, str]:
    """Parse user-uploaded CSV. Returns (close_df, ticker1_name, ticker2_name)."""
    try:
        raw = pd.read_csv(io.BytesIO(contents))
    except Exception as exc:
        raise ValueError(f"Cannot read CSV: {exc}")
    if "Date" not in raw.columns:
        raise ValueError("CSV must have a 'Date' column.")
    try:
        raw["Date"] = pd.to_datetime(raw["Date"])
    except Exception:
        raise ValueError("Could not parse 'Date' column as dates.")
    raw = raw.set_index("Date")
    price_cols = raw.select_dtypes(include="number").columns.tolist()
    if len(price_cols) < 2:
        raise ValueError(
            "CSV must have at least two numeric price columns besides 'Date'. "
            f"Found: {price_cols or 'none'}."
        )
    t1, t2 = str(price_cols[0]), str(price_cols[1])
    close_df = raw[[t1, t2]].dropna()
    return close_df, t1, t2

# ---------------------------------------------------------------------------
# Chart helpers
# ---------------------------------------------------------------------------

def _fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=90, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def _chart_spread(spread: pd.Series, dep: str, indep: str) -> str:
    fig, ax = plt.subplots(figsize=(11, 3))
    ax.plot(spread.index, spread.values, lw=0.9, color="steelblue")
    ax.axhline(0, color="grey", lw=0.5, ls="--")
    ax.set_title(f"Spread  (log {dep}  −  {'{:.3f}'.format(0)} · log {indep})")
    ax.set_ylabel("Log-price residual")
    return _fig_to_b64(fig)


def _chart_zscore(zscore: pd.Series, trade_df: pd.DataFrame | None) -> str:
    fig, ax = plt.subplots(figsize=(11, 3))
    ax.plot(zscore.index, zscore.values, lw=0.9, color="dimgray", label="Z-score")
    for level, colour, ls, lbl in [
        ( Z_ENTRY, "red",       "--", f"+{Z_ENTRY} (short entry)"),
        (-Z_ENTRY, "green",     "--", f"−{Z_ENTRY} (long entry)"),
        ( Z_STOP,  "darkred",   ":",  f"+{Z_STOP} (stop)"),
        (-Z_STOP,  "darkgreen", ":",  f"−{Z_STOP} (stop)"),
        ( Z_EXIT,  "black",     "-",  "0 (exit)"),
    ]:
        ax.axhline(level, color=colour, lw=0.8, ls=ls, label=lbl)
    if trade_df is not None and not trade_df.empty and "Execution_Date" in trade_df.columns:
        for _, row in trade_df.iterrows():
            dt = pd.to_datetime(row["Execution_Date"])
            if dt in zscore.index:
                z_val = float(zscore.loc[dt])
                col   = "green" if str(row.get("Direction", "")).lower() == "long" else "red"
                ax.scatter([dt], [z_val], color=col, s=40, zorder=5)
    ax.set_title("Z-score with entry / exit thresholds")
    ax.set_ylabel("Z-score")
    ax.legend(fontsize=7, ncol=3, loc="upper left")
    return _fig_to_b64(fig)

# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------

def run_analysis(
    close_df: pd.DataFrame,
    open_df: pd.DataFrame | None,
    ticker1: str,
    ticker2: str,
) -> dict:
    if len(close_df) < MIN_ROWS:
        raise ValueError(
            f"Need at least {MIN_ROWS} rows after alignment; got {len(close_df)}."
        )

    log_df = np.log(close_df)
    log_a, log_b = log_df[ticker1], log_df[ticker2]

    # pick_direction: choose the OLS direction with more stationary residuals.
    chosen_dir, hr, ic, _resid, adf_stat, adf_pval, dep, indep = \
        pick_direction(log_a, log_b, ticker1, ticker2)

    # engle_granger: coint() p-value on the chosen direction — matches Phase 3.
    # pick_direction's adf_pval (ADF on OLS residuals) selects the direction;
    # engle_granger's eg_pval (coint()) drives the 5%/10% pass/fail decision.
    eg_stat, eg_pval = engle_granger(log_df[dep], log_df[indep])
    eg = {
        "direction":   chosen_dir,
        "hedge_ratio": round(hr, 6),
        "intercept":   round(ic, 6),
        "adf_stat":    round(adf_stat, 4),
        "adf_pval":    round(adf_pval, 4),
        "eg_stat":     round(eg_stat, 4),
        "eg_pval":     round(eg_pval, 4),
        "pass_5pct":   bool(eg_pval < 0.05),
        "pass_10pct":  bool(eg_pval < 0.10),
    }

    # Johansen: BIC-selected lag, matching Phase 3's VAR.select_order() approach.
    k_ar_diff = select_johansen_lag(log_df[[dep, indep]])
    jo = johansen(log_df[[dep, indep]], det_order=0, k_ar_diff=k_ar_diff)

    # Classification via the shared Phase 3 rule (single source of truth).
    _label = classify_pair(eg["pass_5pct"], jo["trace_pass_5pct"], jo["trace_pass_10pct"])
    is_cointegrated = _label != "none"
    is_borderline   = _label == "borderline"

    if eg["pass_5pct"] and jo["trace_pass_5pct"]:
        summary = "Both EG and Johansen trace pass at 5% — strong cointegration evidence."
    elif eg["pass_5pct"]:
        summary = "EG passes at 5%; Johansen trace inconclusive — included."
    elif jo["trace_pass_5pct"]:
        summary = "Johansen trace passes at 5%; EG inconclusive — included."
    elif is_borderline:
        summary = (
            "Johansen trace passes at 10% only — borderline. "
            "Included per supervisor-approved selection rule; interpret results cautiously."
        )
    else:
        summary = "No cointegration detected (EG or Johansen trace at 5% or 10%) — signals not generated."

    result: dict = {
        "ticker1": ticker1,
        "ticker2": ticker2,
        "dep":     dep,
        "indep":   indep,
        "n_obs":   len(close_df),
        "date_range": {
            "start": str(close_df.index[0].date()),
            "end":   str(close_df.index[-1].date()),
        },
        "engle_granger": eg,
        "johansen":      jo,
        "is_cointegrated":   is_cointegrated,
        "is_borderline":     is_borderline,
        "cointegration_summary": summary,
        "signals":   [],
        "n_signals": 0,
    }

    if is_cointegrated:
        spread, zscore = build_spread_zscore(log_df, dep, indep, hr, ic, LOOKBACK)

        spread_open = None
        if open_df is not None:
            log_open    = np.log(open_df)
            spread_open = build_open_spread(log_open, dep, indep, hr, ic)

        bt_df    = backtest(zscore, spread, Z_ENTRY, Z_EXIT, Z_STOP,
                            COST_PER_LEG, spread_open_series=spread_open)
        metrics  = compute_metrics(bt_df, label=f"{dep}/{indep}")
        trade_df = extract_trade_log(bt_df, COST_PER_LEG)

        result.update({
            "metrics": {
                k: metrics[k]
                for k in ("Total_PnL", "Ann_PnL", "Sharpe_ratio",
                          "Max_drawdown", "Num_trades", "Pct_in_market")
            },
            "signals":   trade_df.to_dict("records") if not trade_df.empty else [],
            "n_signals": len(trade_df),
            "chart_spread_b64":  _chart_spread(spread.dropna(), dep, indep),
            "chart_zscore_b64":  _chart_zscore(
                zscore.dropna(),
                trade_df if not trade_df.empty else None,
            ),
        })

    return result

# ---------------------------------------------------------------------------
# Shared input handler
# ---------------------------------------------------------------------------

async def _process_inputs(
    ticker1: str, ticker2: str, file: UploadFile | None
) -> dict:
    try:
        if file and file.filename:
            contents = await file.read()
            close_df, t1, t2 = _parse_uploaded_csv(contents)
            open_df = None
        elif ticker1 and ticker2:
            t1 = ticker1.upper().strip()
            t2 = ticker2.upper().strip()
            close_df, open_df = _load_cached(t1, t2)
        else:
            raise ValueError(
                "Select two tickers from the list or upload a CSV file."
            )
        return run_analysis(close_df, open_df, t1, t2)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(request, "index.html", {
        "tickers": _available_tickers(),
        "error":   None,
    })


@app.post("/analyse", response_class=HTMLResponse)
async def analyse_html(
    request: Request,
    ticker1: str = Form(""),
    ticker2: str = Form(""),
    file: UploadFile = File(None),
):
    try:
        result = await _process_inputs(ticker1, ticker2, file)
    except HTTPException as exc:
        return templates.TemplateResponse(request, "index.html", {
            "tickers": _available_tickers(),
            "error":   exc.detail,
        }, status_code=exc.status_code)
    return templates.TemplateResponse(request, "results.html", {
        "result": result,
    })


@app.post("/api/analyse")
async def api_analyse(
    ticker1: str = Form(""),
    ticker2: str = Form(""),
    file: UploadFile = File(None),
):
    result = await _process_inputs(ticker1, ticker2, file)
    # Strip binary chart data from JSON response
    result.pop("chart_spread_b64", None)
    result.pop("chart_zscore_b64", None)
    return JSONResponse(content=result)

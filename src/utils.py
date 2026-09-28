from pathlib import Path
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Project root (resolves regardless of where the script is called from)
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Output paths
# ---------------------------------------------------------------------------
DATA_RAW       = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
CHARTS_DIR     = ROOT / "outputs" / "charts"
REPORTS_DIR    = ROOT / "outputs" / "reports"
MODELS_DIR     = ROOT / "models"

for _dir in (DATA_RAW, DATA_PROCESSED, CHARTS_DIR, REPORTS_DIR, MODELS_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Date range
# ---------------------------------------------------------------------------
START_DATE = "2018-01-01"
END_DATE   = "2026-01-01"  # yfinance end is exclusive; use 2026-01-01 to include 2025-12-31

# Phase 3b canonical train/test split (used in cointegration screening and strategy)
TRAIN_END  = "2021-12-31"
TEST_START = "2022-01-01"

# ---------------------------------------------------------------------------
# Ticker universe  — 6 large-cap S&P 500 tech stocks
# ---------------------------------------------------------------------------
TICKERS = {
    "Technology": [
        "MSFT", "GOOGL", "NVDA", "AAPL", "AMZN", "META",
    ],
}

# Flat list for convenience
ALL_TICKERS = TICKERS["Technology"]

# Full company names for each ticker
TICKER_NAMES = {
    "MSFT":  "Microsoft Corporation",
    "GOOGL": "Alphabet Inc. (Google)",
    "NVDA":  "NVIDIA Corporation",
    "AAPL":  "Apple Inc.",
    "AMZN":  "Amazon.com Inc.",
    "META":  "Meta Platforms Inc.",
}

# Benchmark / macro series downloaded alongside the universe
BENCHMARK_TICKERS = {
    "sp500": "^GSPC",
    "vix":   "^VIX",
}

# ---------------------------------------------------------------------------
# 2026 unseen data — cached download
# ---------------------------------------------------------------------------
# Phase 4 and Phase 5 both need 2026 prices.  Fetching from yfinance on every
# run gives slightly different adjusted prices on different days, which flips
# binary convergence-gate decisions even when the LSTM weights are fixed.
# This helper saves one authoritative copy per ticker to data/raw/ and reuses
# it on subsequent runs.  Pass refresh=True (or --refresh on the CLI) to
# overwrite with a fresh download.

def load_or_download_2026(ticker, start="2026-01-01", end_exclusive="2026-08-01",
                           refresh=False):
    """
    Return a DataFrame with Close and Open columns for `ticker` over the 2026
    window.  On the first call (or when refresh=True) it downloads from
    yfinance and writes data/raw/{ticker}_2026.csv.  Subsequent calls read
    from that file instead of hitting the network.

    Raises RuntimeError if the download is empty or the cached file is
    missing and refresh=False was not enough to recover.
    """
    import yfinance as yf  # local import to keep utils lightweight when unused

    cache_path = DATA_RAW / f"{ticker}_2026.csv"

    if cache_path.exists() and not refresh:
        df = pd.read_csv(cache_path, index_col="Date", parse_dates=True)
        if "Close" in df.columns and "Open" in df.columns and len(df) > 0:
            return df

    # Download from yfinance
    raw = yf.download(ticker, start=start, end=end_exclusive,
                      auto_adjust=True, progress=False)
    if raw.empty:
        raise RuntimeError(f"yfinance returned no data for {ticker} "
                           f"({start} – {end_exclusive})")
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    raw.index.name = "Date"
    df = raw[["Close", "Open"]].copy()
    df.to_csv(cache_path)
    return df

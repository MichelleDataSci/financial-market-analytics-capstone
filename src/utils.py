from pathlib import Path

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

for _dir in (DATA_RAW, DATA_PROCESSED, CHARTS_DIR, REPORTS_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Date range
# ---------------------------------------------------------------------------
START_DATE = "2018-01-01"
END_DATE   = "2026-01-01"  # yfinance end is exclusive; use 2026-01-01 to include 2025-12-31

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

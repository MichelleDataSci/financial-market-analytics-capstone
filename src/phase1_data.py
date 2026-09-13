"""
Phase 1 — Master Data Creation
Downloads OHLC + Volume for 6 tech stocks plus S&P 500 and VIX,
computes daily returns, adds time indicators, and saves a master CSV.
"""

import yfinance as yf
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import ALL_TICKERS, BENCHMARK_TICKERS, TICKER_NAMES, START_DATE, END_DATE, DATA_RAW, DATA_PROCESSED

# ---------------------------------------------------------------------------
# Step 1 — Download OHLC + Volume for each stock
# ---------------------------------------------------------------------------
print("=" * 60)
print("PHASE 1 — MASTER DATA CREATION")
print("=" * 60)
print(f"\nTickers  : {ALL_TICKERS}")
print(f"Period   : {START_DATE}  to  {END_DATE}")
print(f"Raw dir  : {DATA_RAW}")
print()

# ---------------------------------------------------------------------------
# Step 2 — Confirm ticker symbols and company names
# ---------------------------------------------------------------------------
print("Step 2 — Ticker symbols and company names:")
print(f"  {'Symbol':<8} {'Company'}")
print(f"  {'-'*8} {'-'*30}")
for ticker in ALL_TICKERS:
    print(f"  {ticker:<8} {TICKER_NAMES[ticker]}")
print(f"\n  Benchmarks:")
for label, symbol in BENCHMARK_TICKERS.items():
    print(f"  {symbol:<8} {label.upper()}")
print()

raw_frames = {}

for ticker in ALL_TICKERS:
    print(f"  Downloading {ticker} ...", end=" ")
    df = yf.download(ticker, start=START_DATE, end=END_DATE, auto_adjust=True, progress=False)
    if df.empty:
        raise RuntimeError(
            f"yfinance returned no data for {ticker} "
            f"({START_DATE} to {END_DATE}). "
            "Check the ticker symbol, your internet connection, and whether "
            "the market was open during the requested period."
        )
    # Flatten MultiIndex columns if present (yfinance ≥ 0.2.x)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.index.name = "Date"
    raw_path = DATA_RAW / f"{ticker}_raw.csv"
    df.to_csv(raw_path)
    raw_frames[ticker] = df
    print(f"{len(df)} rows saved -> {raw_path.name}")

print(f"\n[Step 3 complete] Downloaded {len(raw_frames)} tickers.\n")

# ---------------------------------------------------------------------------
# Step 4 — Calculate daily returns (Xt - Xt-1) / Xt-1 * 100
# ---------------------------------------------------------------------------
print("Step 4 — Computing daily returns: (Xt - Xt-1) / Xt-1 x 100 ...")

returns = {}
for ticker, df in raw_frames.items():
    close = df["Close"]
    ret = (close - close.shift(1)) / close.shift(1) * 100
    returns[ticker] = ret.rename(f"{ticker}_return")

returns_df = pd.concat(returns.values(), axis=1)
print(f"  Returns shape: {returns_df.shape}")
print(f"[Step 4 complete] Daily returns computed.\n")

# ---------------------------------------------------------------------------
# Step 5 — Download S&P 500 and VIX, compute their daily returns
# ---------------------------------------------------------------------------
print("Step 5 — Downloading S&P 500 and VIX benchmark series ...")

benchmark_series = {}
for label, symbol in BENCHMARK_TICKERS.items():
    print(f"  Downloading {symbol} ({label}) ...", end=" ")
    df_b = yf.download(symbol, start=START_DATE, end=END_DATE, auto_adjust=True, progress=False)
    if df_b.empty:
        raise RuntimeError(
            f"yfinance returned no data for benchmark {symbol} ({label}) "
            f"({START_DATE} to {END_DATE}). "
            "Check the ticker symbol and your internet connection."
        )
    if isinstance(df_b.columns, pd.MultiIndex):
        df_b.columns = df_b.columns.get_level_values(0)
    df_b.index.name = "Date"
    raw_path = DATA_RAW / f"{label}_raw.csv"
    df_b.to_csv(raw_path)
    close_b = df_b["Close"]
    ret_b = (close_b - close_b.shift(1)) / close_b.shift(1) * 100
    benchmark_series[f"{label}_return"] = ret_b.rename(f"{label}_return")
    # VIX level (raw index value) is needed by Phase 2's sensitivity analysis
    # and is part of the brief's Phase 1 item 5.  S&P 500 level is not required.
    if label == "vix":
        benchmark_series["vix_level"] = close_b.rename("vix_level")
    print(f"{len(df_b)} rows saved -> {raw_path.name}")

benchmark_df = pd.concat(benchmark_series.values(), axis=1)
print(f"  Benchmark returns shape: {benchmark_df.shape}")
print(f"[Step 5 complete] Benchmark series downloaded.\n")

# ---------------------------------------------------------------------------
# Steps 6, 7, 8 — Add Year, Quarter, Month indicator columns
# ---------------------------------------------------------------------------
print("Steps 6/7/8 — Building master dataset (OHLC + Volume + returns + benchmarks) ...")

# Interleave OHLC + Volume + Daily_Return per ticker in spec order:
#   {ticker}_Open, {ticker}_High, {ticker}_Low, {ticker}_Close,
#   {ticker}_Volume, {ticker}_return
price_parts = []
for ticker in ALL_TICKERS:
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        price_parts.append(raw_frames[ticker][col].rename(f"{ticker}_{col}"))
    price_parts.append(returns[ticker])   # already named {ticker}_return
ohlcv_returns_df = pd.concat(price_parts, axis=1)
ohlcv_returns_df.index = pd.to_datetime(ohlcv_returns_df.index)
ohlcv_returns_df.index.name = "Date"

master = ohlcv_returns_df.join(benchmark_df, how="outer")
master.index = pd.to_datetime(master.index)
master["Year"]    = master.index.year
master["Quarter"] = master.index.quarter
master["Month"]   = master.index.month

print(f"  Columns: OHLC + Volume + Return per ticker, S&P500/VIX returns, Year/Quarter/Month")
print(f"  Master shape : {master.shape}")
print(f"[Steps 6/7/8 complete] Master dataset built with full OHLC, Volume, and returns.\n")

# ---------------------------------------------------------------------------
# Steps 9 & 10 — Merge all series and finalise master CSV
# ---------------------------------------------------------------------------
print("Steps 9/10 — Merging all series and finalising master CSV ...")

master.dropna(how="all", inplace=True)

# Drop only the very first row that has NaN returns due to the shift
first_valid = master[list(returns.keys())[0] + "_return"].first_valid_index()
master = master.loc[first_valid:]

out_path = DATA_PROCESSED / "master_data.csv"
master.to_csv(out_path)

print(f"  Final shape : {master.shape}")
print(f"  Date range  : {master.index[0].date()}  to  {master.index[-1].date()}")
print(f"  Saved to    : {out_path}")
print(f"\n[Steps 9/10 complete] All series merged. Master CSV saved.")
print("\n" + "=" * 60)
print("PHASE 1 COMPLETE")
print("=" * 60)
print("\nColumns in master_data.csv:")
for col in master.columns:
    print(f"  {col}")

print("\nMaster data preview (first 3 rows):")
print(master.head(3).to_string())

print("\nMaster data summary statistics:")
print(master.describe().round(4).to_string())

print(f"\nNull value check:")
nulls = master.isnull().sum()
if nulls.sum() == 0:
    print("  No null values — master data is complete.")
else:
    print(nulls[nulls > 0].to_string())

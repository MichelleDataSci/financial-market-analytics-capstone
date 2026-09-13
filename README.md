# Pairs Trading Analytics

End-to-end pipeline for identifying, backtesting, and monitoring statistical arbitrage pairs
across six S&P 500 technology stocks (MSFT, GOOGL, NVDA, AAPL, AMZN, META).

## Project layout

```
.
|-- data/
|   |-- raw/          # downloaded OHLC / VIX / S&P 500 CSVs as-is
|   +-- processed/    # master merged CSV, cleaned data
|-- notebooks/        # optional Jupyter exploration
|-- src/
|   |-- phase1_data.py          # data ingestion (8 years, 6 stocks + benchmarks)
|   |-- phase2_eda.py           # exploratory analysis (15 pairs, beta, VIX)
|   |-- phase3_cointegration.py # pair screening: EG + Johansen, exports selected_pairs.csv
|   |-- phase3_strategy.py      # backtesting engine (loops over all selected pairs)
|   |-- phase4_unseen.py        # mean-reversion test on genuinely unseen 2026 data
|   +-- utils.py                # shared constants & paths
|-- outputs/
|   |-- charts/       # saved figures (named by pair where applicable)
|   +-- reports/      # exported tables / summary stats
|-- requirements.txt
+-- README.md
```

## Setup

```bash
pip install -r requirements.txt
```

## Running the pipeline

Run scripts in order — each phase depends on outputs from the previous:

```bash
python src/phase1_data.py          # download and build master_data.csv
python src/phase2_eda.py           # exploratory analysis (15 charts + tables)
python src/phase3_cointegration.py # cointegration screening → selected_pairs.csv
python src/phase3_strategy.py      # pairs trading backtest (all selected pairs)
python src/phase4_unseen.py        # mean-reversion test on unseen 2026 data
```

## Pair selection criterion

Phase 3 applies a two-tier selection:

- **Primary** — at least one of Engle-Granger or Johansen trace passes at 5%. Tracked as `Tests_passed` = "Both", "EG only", or "Johansen only" in `selected_pairs.csv`.
- **Secondary** — Johansen trace passes at 10% but not 5%; the pair is carried forward as a borderline candidate per supervisor instruction. Tracked as `Tests_passed` = "Johansen 10%".

Running `phase3_cointegration.py` selects **AMZN/META** (primary) and **MSFT/AAPL** (secondary, Johansen trace 10%).

## Key outputs

| File | Description |
|---|---|
| `data/processed/master_data.csv` | Master dataset (OHLCV + returns + benchmarks + time indicators) |
| `outputs/reports/cointegration_results.csv` | Full EG + Johansen results for all 15 pairs |
| `outputs/reports/selected_pairs.csv` | Pairs that pass at least one cointegration test — contract for downstream phases |
| `outputs/reports/strategy_{DEP}_{INDEP}_results.csv` | Per-pair strategy results (train/test/walk-forward) |
| `outputs/reports/strategy_cross_pair_summary.csv` | Cross-pair strategy comparison |
| `outputs/reports/phase4_{DEP}_{INDEP}_summary.csv` | Per-pair Phase 4 signal summary |
| `outputs/reports/phase4_cross_pair_summary.csv` | Cross-pair Phase 4 comparison (Z-stats, verdict) |

## Phase 5

Machine Learning for Predicting Spread — brief pending.

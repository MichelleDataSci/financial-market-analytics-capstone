# Pairs Trading Analytics

End-to-end pipeline for identifying, backtesting, and monitoring statistical arbitrage pairs
across six S&P 500 technology stocks (MSFT, GOOGL, NVDA, AAPL, AMZN, META).

## Project scope

The project brief defined five objectives:

1. Comparative study of technology stocks in S&P 500 — returns, volatility, correlations (EDA).
2. Identify pairs of stocks which are cointegrated over a period (8 years).
3. Design a pairs trading strategy and execute on train/test data.
4. ~~Sentiment analysis of technology stocks using X (Twitter) data or Google News.~~ **Removed from scope** by the project supervisor on 21 September 2026.
5. Web application development.

The work is delivered as **Phases 1–6** per the phase breakdown: Phase 1 (data ingestion), Phase 2 (EDA), Phase 3 (cointegration screening and backtesting), Phase 4 (unseen 2026 data), Phase 5 (LSTM spread prediction), Phase 6 (FastAPI web application).

## Project layout

```
.
├── data/
│   ├── raw/          # downloaded OHLC / VIX / S&P 500 CSVs (cached; never overwritten)
│   └── processed/    # master merged CSV
├── src/
│   ├── phase1_data.py          # data ingestion (8 years, 6 stocks + benchmarks)
│   ├── phase2_eda.py           # exploratory analysis (15 pairs, beta, VIX)
│   ├── phase3_cointegration.py # pair screening: EG + Johansen → selected_pairs.csv
│   ├── phase3_strategy.py      # backtesting engine (all selected pairs)
│   ├── phase4_unseen.py        # mean-reversion test on genuinely unseen 2026 data
│   ├── phase5_ml_spread.py     # LSTM spread prediction + Step 8 2026 comparison
│   └── utils.py                # shared constants & paths
├── app/
│   ├── main.py                 # FastAPI application
│   └── templates/              # Jinja2 HTML templates
├── models/                     # saved LSTM model weights and scalers
├── outputs/
│   ├── charts/                 # saved figures (named by pair)
│   └── reports/                # exported tables / summary CSVs
├── tests/                      # pytest suite (66 tests)
├── requirements.txt
├── requirements_exact.txt      # pinned versions for exact reproduction
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
```

For exact reproduction of all outputs, use the pinned versions:

```bash
pip install -r requirements_exact.txt
```

## End-to-end run guide

Run scripts in order — each phase depends on outputs from the previous:

```bash
# Phase 1 — download and build master_data.csv (requires internet)
python src/phase1_data.py

# Phase 2 — exploratory analysis: 15 charts + tables
python src/phase2_eda.py

# Phase 3a — cointegration screening → selected_pairs.csv
python src/phase3_cointegration.py

# Phase 3b — pairs trading backtest (all selected pairs)
python src/phase3_strategy.py

# Phase 4 — mean-reversion test on unseen 2026 data
#   (reads cached data/raw/{ticker}_2026.csv; use --refresh to re-download)
python src/phase4_unseen.py

# Phase 5 — LSTM spread prediction + Step 8 2026 comparison
python src/phase5_ml_spread.py
```

**Notes:**
- Phase 1 skips any ticker whose `data/raw/{ticker}_raw.csv` already exists. Pass `--refresh`
  to force a full re-download: `python src/phase1_data.py --refresh`. An internet connection
  is only required for tickers not already cached.
- Phase 4 and 5 read 2026 prices from `data/raw/{ticker}_2026.csv` (cached on first run).
  Pass `--refresh` to Phase 4 to re-download: `python src/phase4_unseen.py --refresh`.
- Phase 5 trains two LSTM models per pair and takes several minutes on CPU.
- All scripts print UTF-8 output; no `-X utf8` flag is required on Windows.

## Predict (inference only)

Print the latest 5-day z-score forecast and convergence gate decision for each selected pair
without retraining:

```bash
python src/predict.py
```

Reads saved artefacts from `models/` and cached raw prices from `data/raw/`. Requires
`selected_pairs.csv` (produced by Phase 3a).

## Run tests

```bash
python -m pytest tests/ -v
```

77 tests covering the backtesting engine (Phase 3b), Phase 5 utilities, predict utilities, and the FastAPI app.

## FastAPI deployment

### Start the app

```bash
uvicorn app.main:app --reload
```

Then open `http://127.0.0.1:8000` in your browser.

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET`  | `/` | Form page — select cached tickers or upload a CSV |
| `POST` | `/analyse` | Run analysis and return an HTML results page |
| `POST` | `/api/analyse` | Same analysis as JSON (charts omitted) |
| `GET`  | `/api/predict/{pair}` | Latest 5-day z-score forecast and convergence gate for a saved pair (e.g. `AMZN_META`, `MSFT_AAPL`). Artefacts loaded once at startup; no retraining. Returns 404 for unknown pairs. |

### What the user sees

**Input (either option):**
- **Cached data** — choose two tickers from the dropdown (AAPL, AMZN, GOOGL, META, MSFT, NVDA).
- **CSV upload** — a file with a `Date` column and two numeric price columns (minimum 60 rows).

**Output (HTML results page):**
- Cointegration table — Engle-Granger (`coint()` p-value) and Johansen trace / max-eigenvalue statistics with pass/fail flags at 5% and 10%.
- Spread chart — OLS log-price residual over time, with hedge ratio in the title.
- Z-score chart — rolling 30-day z-score with ±2.0 entry and ±3.0 stop-loss thresholds.
- Strategy metrics table — total P&L, annualised P&L, Sharpe ratio, max drawdown, number of entries, % time in market.
- Signals table — individual entry/exit dates, direction, z-scores, holding days, and net P&L per trade.

A badge indicates the classification:
- **Cointegrated (EG or Johansen trace at 5%)** — primary criterion, signals generated.
- **Borderline (trace 10% only)** — secondary criterion, signals generated but labelled.
- **Not cointegrated** — test results shown, signals not generated.

> **Note:** The hedge ratio is estimated in-sample on the data supplied. Signals are illustrative and not out-of-sample.

## Pair selection criterion

Phase 3 applies a two-tier rule (agreed with project supervisor, 2026-09-01):

| Criterion | Tests | Label |
|-----------|-------|-------|
| Primary | EG 5% **or** Johansen trace 5% | "Both" / "EG only" / "Johansen only" |
| Secondary | Johansen trace 10% only | "Johansen 10%" |

Max-eigenvalue statistics are reported but not used for selection.

Running `phase3_cointegration.py` selects **AMZN/META** (primary — both EG and Johansen trace pass at 5%) and **MSFT/AAPL** (secondary — Johansen trace at 10% only).

## Key findings

All numbers are from the current output CSVs (`outputs/reports/`).

### Pair selection (Phase 3a, full dataset 2018–2025)

| Pair | EG p-value | Johansen trace stat | CV 5% | Hedge ratio | Result |
|------|-----------|---------------------|-------|------------|--------|
| AMZN/META | 0.0143 | 17.54 | 15.49 | 0.60 | Primary (both) |
| MSFT/AAPL | 0.2031 | 14.56 | 15.49 | 0.87 | Secondary (trace 10%) |

AMZN/META ranks first by both the EG p-value (rank 1) and the Johansen ratio — trace stat / 5% CV = 1.13 (rank 1). The two methods diverge for lower-ranked pairs: MSFT/GOOGL ranks 15th by EG p-value but 4th by Johansen ratio, while NVDA/AMZN ranks 2nd by EG but 7th by Johansen.

### Backtesting strategy (Phase 3b, entry ±2σ, exit 0, stop ±3σ, cost 0.1%/leg)

| Pair | Period | Sharpe | Sharpe 95% CI | Total P&L | Max drawdown | WF profitable years |
|------|--------|--------|---------------|-----------|--------------|---------------------|
| AMZN/META | Train 2018–2021 | 0.56 | (−0.25, 1.37) | — | — | — |
| AMZN/META | Test 2022–2025 | 0.12 | (−0.74, 1.01) | +0.138 | −0.588 | 4/8 |
| MSFT/AAPL | Train 2018–2021 | 0.25 | (−0.71, 1.08) | — | — | — |
| MSFT/AAPL | Test 2022–2025 | −0.23 | (−1.16, 0.71) | −0.133 | −0.453 | 4/8 |

AMZN/META shows a positive out-of-sample Sharpe but the confidence interval spans zero. MSFT/AAPL is negative on the test set.

### Unseen 2026 data (Phase 4, 2026-01-01 to 2026-07-31, 145 days)

| Pair | Signals | P&L | Sharpe | Verdict |
|------|---------|-----|--------|---------|
| AMZN/META | 2 (short) | −0.046 | −0.26 | Weak — spread drifted from equilibrium |
| MSFT/AAPL | 1 (long) | +0.014 | 0.06 | Weak — spread drifted from equilibrium |

Both pairs show weak mean reversion in 2026; the spread has drifted materially from its training-period equilibrium.

### LSTM spread prediction (Phase 5)

**Historical experiment** (LSTM trained 2018–2020, tested 2022–2025):

| Pair | LSTM RMSE (overall) | Persistence RMSE | LSTM worse? | Val RMSE (2021) |
|------|---------------------|-----------------|------------|-----------------|
| AMZN/META | 1.82 | 0.36 | Yes (5× worse) | 0.27 |
| MSFT/AAPL | 0.39 | 0.25 | Yes (1.6× worse) | 0.22 |

The LSTM generalises poorly from 2018–2020 training to the 2022–2025 test regime; persistence (naive carry-forward) outperforms it on both pairs. In the 2022–2025 holdout the convergence gate did not change any trades for either pair (baseline and LSTM P&L identical), so the LSTM added no value to the strategy.

**Final 2026 comparison** (LSTM trained 2018–2024, tested 2026):

| Pair | P4 baseline Sharpe | LSTM Sharpe | LSTM entries | Note |
|------|-------------------|------------|--------------|------|
| AMZN/META | −0.26 | n/a (no trades) | 0 | LSTM gate blocked all 2 Phase 4 signals |
| MSFT/AAPL | 0.06 | 0.06 | 1 | Gate passed the 1 Phase 4 signal (78.6% converging) |

For AMZN/META, only 21.4% of 2026 bars were classified as converging, so the gate rejected both Phase 4 entries. This reflects that the LSTM predicted the spread would not converge on most bars — not a demonstration of LSTM skill, as the sample is too small to draw conclusions.

## Limitations

1. **Look-ahead in pair selection.** The primary cointegration screen uses the full 2018–2025 dataset, which overlaps the Phase 3b test window (2022–2025). On the training window alone (2018–2021): AMZN/META passes EG (p = 0.023) but not Johansen trace at 5%; MSFT/AAPL fails both EG (p = 0.293) and Johansen; AAPL/AMZN passes EG (p = 0.037) but is not selected in the full-dataset screen. The supervisor-approved two-tier rule (agreed 2026-09-01) retains MSFT/AAPL as a borderline case on the full dataset, but this selection contains a look-ahead relative to the backtest period. See `cointegration_results_train_only.csv`.

2. **Sharpe confidence intervals span zero.** All Sharpe ratios on the test set have 95% CIs that include zero. The sample is too short (≈4 years, 1 003 trading days) to draw statistically reliable conclusions about strategy profitability.

3. **Small 2026 sample.** Phase 4 and Phase 5's definitive evaluation covers only 145 trading days (2026-01-01 to 2026-07-31). Single-trade results (MSFT/AAPL: 1 open trade; AMZN/META: 2 trades) are not sufficient to assess the strategy in this regime.

4. **Hedge-ratio instability.** The OLS hedge ratio is estimated on the full training period and held fixed. Annual walk-forward shows the HR varying from 0.05 to 1.12 (AMZN/META) and 0.59 to 0.88 (MSFT/AAPL), indicating the linear relationship is non-stationary. The strategy does not adapt to this drift.

5. **LSTM loses to naive persistence.** On both pairs, the persistence baseline (carry-forward z-score) achieves lower RMSE than the stacked LSTM on the 2022–2025 test set. AMZN/META LSTM is 5× worse (RMSE 1.82 vs 0.36). This is consistent with a non-stationary spread that invalidates the historical training distribution.

6. **AMZN/META LSTM took no trades in 2026.** The convergence gate rejected both Phase 4 signals (only 21.4% of bars classified as converging). This is not evidence that the LSTM correctly avoided losses — it reflects a low convergence rate in a small sample and does not demonstrate forecasting skill.

7. **Sensitivity grid is diagnostic, not tuning.** Phase 3b sensitivity analysis over z-score thresholds and cost levels is reported for transparency; it was not used to select strategy parameters. Parameters (Z_entry = 2.0, Z_exit = 0.0, Z_stop = 3.0) were fixed before testing.

8. **LSTM seed sensitivity.** The LSTM uses a fixed global seed (`tf.keras.utils.set_random_seed` + op determinism) to ensure reproducibility. Across reruns made before determinism was enforced, MSFT/AAPL holdout RMSE ranges from 0.39 to 0.60, AMZN/META 2026 convergence rate ranges from 21% to 75%, and the LSTM loses to naive persistence in every run. The fixed-seed values reported here are representative but not uniquely determined.

9. **Cached 2026 data.** 2026 prices are cached to `data/raw/{ticker}_2026.csv` on first run (via `load_or_download_2026`). This prevents yfinance's retroactive price-adjustment mechanism from altering results across sessions. Use `--refresh` to re-download if needed.

## Key outputs

| File | Description |
|------|-------------|
| `data/processed/master_data.csv` | Master dataset (OHLCV + returns + benchmarks + time indicators) |
| `outputs/reports/cointegration_results.csv` | Full EG + Johansen results for all 15 pairs |
| `outputs/reports/cointegration_results_train_only.csv` | Same screen restricted to 2018–2021 training window (look-ahead check) |
| `outputs/reports/selected_pairs.csv` | Pairs taken forward to the strategy |
| `outputs/reports/strategy_cross_pair_summary.csv` | Cross-pair strategy comparison with Sharpe CIs |
| `outputs/reports/phase4_cross_pair_summary.csv` | Phase 4 signal summary and mean-reversion verdict |
| `outputs/reports/phase5_cross_pair_final_summary.csv` | Phase 5 definitive 2026 LSTM vs Phase 4 baseline |
| `models/{pair}_lstm_v1.keras` | Saved LSTM weights (versioned) |
| `models/{pair}_ols_v1.joblib` | OLS hedge ratio and intercept |
| `models/{pair}_scaler_v1.joblib` | Spread mean and std used for z-score normalisation |

## Data sources and citations

**Price data**  
All OHLCV data is sourced from [Yahoo Finance](https://finance.yahoo.com/) via the
[yfinance](https://github.com/ranaroussi/yfinance) library.

**Libraries**

| Library | Purpose | Link |
|---------|---------|------|
| [pandas](https://pandas.pydata.org/) | Data manipulation and time-series alignment | https://pandas.pydata.org/ |
| [NumPy](https://numpy.org/) | Numerical arrays and linear algebra | https://numpy.org/ |
| [statsmodels](https://www.statsmodels.org/) | Engle-Granger `coint()`, Johansen test, VAR lag selection | https://www.statsmodels.org/ |
| [TensorFlow / Keras](https://www.tensorflow.org/) | LSTM model training and inference | https://www.tensorflow.org/ |
| [scikit-learn / joblib](https://scikit-learn.org/) | Artefact serialisation (`.joblib`) | https://scikit-learn.org/ |
| [FastAPI](https://fastapi.tiangolo.com/) | Web application and REST API | https://fastapi.tiangolo.com/ |
| [matplotlib](https://matplotlib.org/) | All charts and figures | https://matplotlib.org/ |
| [seaborn](https://seaborn.pydata.org/) | Heatmap and styled plots | https://seaborn.pydata.org/ |

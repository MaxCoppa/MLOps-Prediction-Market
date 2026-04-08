# MLOps Prediction Market

LightGBM-based pipelines for Kalshi market prediction at two levels:

- **Series level**: train and backtest a model across all tickers in a Kalshi series
- **Ticker level**: train and backtest a model for one specific Kalshi ticker

## Project structure

```text
MLOps-Prediction-Market/
├── pyproject.toml
├── README.md
├── predict_series.py
├── predict_ticker.py
├── src/
│   └── kalshi_predictor/
│       ├── __init__.py
│       ├── series/
│       ├── ticker/
│       └── utils/
└── outputs/
````

## Setup

This project uses `uv`.

Install dependencies and the local package:

```bash
uv sync
```

## How to run

### 1. Series pipeline

Run the pipeline on a **series ticker** such as `KXGDP`:

```bash
uv run python predict_series.py KXGDP
```

### 2. Ticker pipeline

Run the pipeline on a **single market ticker** such as `KXGDP-26APR30-T1.5`:

```bash
uv run python predict_ticker.py KXGDP-26APR30-T1.5
```

### Outputs

All outputs are saved in the outputs/ directory. (This can be changes)

1. For Series pipeline

serie_name_backtest_results_opt.csv → optimized strategy results
serie_name_backtest_results_def.csv → baseline/default strategy results
serie_name_per_ticker_pnl.csv → per-ticker PnL breakdown
serie_name_report.json → aggregated performance report
serie_name_series_pnl.png → global PnL curve across the series

2. For Ticker pipeline

ticker_name_backtest_results.csv → detailed backtest results
ticker_name_report.json → performance metrics (MAE, etc.)
ticker_name_model.txt → trained LightGBM model
ticker_name_pnl_curve.png → PnL curve visualization


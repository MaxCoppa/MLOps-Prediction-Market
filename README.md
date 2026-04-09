[![ci](https://github.com/MaxCoppa/MLOps-Prediction-Market/actions/workflows/prod.yml/badge.svg)](https://github.com/MaxCoppa/MLOps-Prediction-Market/actions/workflows/prod.yml)

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

## Outputs

All outputs are saved in the `outputs/` directory (this can be changed via CLI arguments).

### 1. Series pipeline outputs

* `series_name_backtest_results_opt.csv` → optimized strategy results
* `series_name_backtest_results_def.csv` → baseline/default strategy results
* `series_name_per_ticker_pnl.csv` → per-ticker PnL breakdown
* `series_name_report.json` → aggregated performance report
* `series_name_series_pnl.png` → global PnL curve across the series

### 2. Ticker pipeline outputs

* `ticker_name_backtest_results.csv` → detailed backtest results
* `ticker_name_report.json` → performance metrics (MAE, etc.)
* `ticker_name_model.txt` → trained LightGBM model
* `ticker_name_pnl_curve.png` → PnL curve visualization


## Run API Local

```bash
uv run uvicorn app.api:app
```

THe X_example have been created from 
```bash
uv run python predict_ticker.py KXGDP-26APR30-T1.5
```
To test the api : 
```bash
curl "http://127.0.0.1:8000/predict/example"
{"prediction":-0.002294086224277482,"note":"This uses a default example from X_example.csv"}onyxia@vscode-python-gpu-784627-0:~/work/MLOps-Prediction-Market$ uv run test_api_local.py 
POST /predict
{'prediction': -0.002294086224277482}

GET /predict/example
{'prediction': -0.002294086224277482, 'note': 'This uses a default example from X_example.csv'}
```

## API 

kubectl run -it api-ml --env JETON_API='' --image=maxcoppa/application:latest

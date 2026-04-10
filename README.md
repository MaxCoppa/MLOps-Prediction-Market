[![ci](https://github.com/MaxCoppa/MLOps-Prediction-Market/actions/workflows/prod.yml/badge.svg)](https://github.com/MaxCoppa/MLOps-Prediction-Market/actions/workflows/prod.yml)

[![train](https://github.com/MaxCoppa/MLOps-Prediction-Market/actions/workflows/train.yml/badge.svg)](https://github.com/MaxCoppa/MLOps-Prediction-Market/actions/workflows/train.yml)

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

## Website Creation

### Local API Access

Start the API locally:

```bash
uv run uvicorn app.api:app
````

API:

```
http://127.0.0.1:8000
```

Docs:

```
http://127.0.0.1:8000/docs
```

Example prediction:

```bash
curl "http://127.0.0.1:8000/predict/example"
```

Or run tests:

```bash
uv run test_api_local.py
```

### Automated Deployment

Deployment is handled via a separate GitOps repository:

[https://github.com/MaxCoppa/kalshi-predictor-deployment](https://github.com/MaxCoppa/kalshi-predictor-deployment)

Changes to Kubernetes manifests in this repository are automatically applied in production.


### Update Production Version

To release a new version in production, first build and push a new Docker image:

```bash
docker build -t maxcoppa/application:v0.0.x .
docker push maxcoppa/application:v0.0.x
```

Then update the image tag in the deployment repository Kubernetes manifest:

```yaml
image: maxcoppa/application:v0.0.x
```

After commit and push, Argo CD automatically redeploys the application.


### Website

The project includes a Quarto website deployed with GitHub Pages:

[https://maxcoppa.github.io/MLOps-Prediction-Market/](https://maxcoppa.github.io/MLOps-Prediction-Market/)

It provides:

* API usage examples (same as `predict/example`)
* Live prediction demo
* Link to API documentation

### TO add

```bash
git tag -a v0.0.x -m "new version"
git push --tags
```



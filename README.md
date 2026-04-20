[![ci](https://github.com/MaxCoppa/MLOps-Prediction-Market/actions/workflows/prod.yml/badge.svg)](https://github.com/MaxCoppa/MLOps-Prediction-Market/actions/workflows/prod.yml)
[![train](https://github.com/MaxCoppa/MLOps-Prediction-Market/actions/workflows/train.yml/badge.svg)](https://github.com/MaxCoppa/MLOps-Prediction-Market/actions/workflows/train.yml)


# MLOps Prediction Market

Production-oriented LightGBM pipelines for prediction on Kalshi event markets.

This project was developed for the course **Mise en production des projets de Data Science** at **ENSAE Paris**. Its goal is to predict outcomes on Kalshi binary event markets while following the production and reproducibility best practices presented in the course.

In practice, the project provides the following outputs:

- **contract-level predictions** from a tuned **LightGBM** model through the [`/predict`](https://kalshi-predictor.lab.sspcloud.fr/predict/example) endpoint
- **daily PnL and signal outputs** for the latest trading session through the [`/predict/pnl`](https://kalshi-predictor.lab.sspcloud.fr/predict/pnl) endpoint
- **a Quarto website** presenting the project, the methodology, and API examples: [https://maxcoppa.github.io/MLOps-Prediction-Market/](https://maxcoppa.github.io/MLOps-Prediction-Market/)

API documentation is available at: [https://kalshi-predictor.lab.sspcloud.fr/docs](https://kalshi-predictor.lab.sspcloud.fr/docs)

The project implements an end-to-end machine learning pipeline for **Kalshi binary contracts**, from data ingestion and feature engineering to backtesting, experiment tracking, API serving, and deployment.

The current application is built around the **Kalshi KXCPI series**, but the pipeline can be extended to other Kalshi series by training the corresponding model and registering it in the MLflow environment used in production.

The current implementation relies on:

- **LightGBM** for prediction
- **Optuna** for hyperparameter tuning
- **MLflow** for experiment tracking
- **FastAPI** for model serving
- **Quarto** for project presentation
- **Docker**, **CI/CD**, and **GitOps** for deployment

The project currently focuses on:

- **Series level**: train and backtest a model across all tickers in a Kalshi series

It follows a production-style workflow inspired by the ENSAE reproducibility and deployment framework: modular code organization, experiment tracking, containerization, CI/CD, API serving, and GitOps-based deployment.

## Project structure

```text
MLOps-Prediction-Market/
├── .github/
├── src/
│   └── kalshi_predictor/
│       ├── __init__.py
│       ├── series/
│       ├── ticker/
│       └── utils/
├── predict_series.py
├── Dockerfile
├── pyproject.toml
├── requirements.txt
├── uv.lock
├── README.md
├── index.qmd
├── _quarto.yml
└── styles.css
````

### Main components

* `src/kalshi_predictor/`: core Python package

  * `series/`: series-level pipeline logic
  * `ticker/`: ticker-level pipeline logic
  * `utils/`: shared utilities
* `predict_series.py`: CLI entrypoint for the series pipeline
* `predict_ticker.py`: CLI entrypoint for the ticker pipeline
* `app/`: FastAPI application for serving predictions
* `index.qmd`, `_quarto.yml`, `styles.css`: Quarto website source
* `Dockerfile`: containerization for deployment

## Pipeline overview

The pipeline follows five steps:

1. **Data ingestion**
   Collect historical market data for Kalshi contracts.

2. **Feature engineering**
   Build lagged returns, rolling statistics, and liquidity-related features.

3. **Modeling**
   Train a LightGBM model with Optuna-based tuning and chronological validation.

4. **Backtesting**
   Evaluate predictive performance in a walk-forward setting.

5. **Tracking and deployment**
   Log experiments with MLflow and serve predictions through a FastAPI application.

## Setup

This project uses [`uv`](https://github.com/astral-sh/uv) for dependency management.

Install dependencies and the local package with:

```bash
uv sync
```

## Running

Run the series pipeline on a ticker such as `KXCPI`:

```bash
uv run python predict_series.py KXCPI
```

This pipeline:

* fetches all tickers in the selected Kalshi series
* downloads historical data
* builds lag-based features
* tunes a LightGBM model with Bayesian optimization
* runs walk-forward backtesting
* generates reports and output artifacts
* logs parameters, metrics, datasets, and models with MLflow


## MLflow experiment tracking

The project uses **MLflow** to track experiments, parameters, metrics, datasets, and trained models.

For the series pipeline, MLflow logs:

* pipeline parameters such as `window_days`, `n_lags`, `n_optuna_trials`, and `val_ratio`
* dataset metadata such as start and end dates
* number of tickers used
* validation and backtest performance
* generated CSV and JSON artifacts
* the final trained LightGBM model

The experiment name can be set from the command line:

```bash
uv run python predict_series.py KXCPI --experiment_name kalshi-series
```

Experiments were run on **SSPCloud MLflow** and have not been tested in other MLflow environments.

## API

### Local API access

Start the API locally:

```bash
uv run uvicorn app.api:app
```

API root:

```text
http://127.0.0.1:8000
```

Interactive documentation:

```text
http://127.0.0.1:8000/docs
```

Example prediction:

```bash
curl "http://127.0.0.1:8000/predict/example"
```

### Production API

The deployed API is available at:

```text
https://kalshi-predictor.lab.sspcloud.fr
```

Main endpoints:

* `/predict`: returns a prediction from one market observation
* `/predict/pnl`: returns the daily prediction, signal, and PnL results for the latest trading session

Examples:

```bash
curl "https://kalshi-predictor.lab.sspcloud.fr/predict?RET_1=-0.01636&VOL_YES_1=0.0&VOL_NO_1=0.0&DIST_50=0.4066&DIST_MIN_20=0.0&DIST_MAX_20=1.0"
```

```bash
curl "https://kalshi-predictor.lab.sspcloud.fr/predict/pnl"
```

## Deployment

Deployment is handled through a separate GitOps repository:

[MaxCoppa/kalshi-predictor-deployment](https://github.com/MaxCoppa/kalshi-predictor-deployment)

Changes to Kubernetes manifests in this repository are automatically applied in production.

### Update production version

To release a new production version, create and push a Git tag:

```bash
git tag -a v0.0.x -m "new version"
git push --tags
```

Then update the image version in the deployment manifest:

```yaml
spec:
  containers:
    - name: kalshi-predictor
      image: maxcoppa/application:v0.0.y
```

## Website

The project includes a Quarto website deployed with GitHub Pages:

[https://maxcoppa.github.io/MLOps-Prediction-Market/](https://maxcoppa.github.io/MLOps-Prediction-Market/)

The website provides:

* project presentation
* API usage examples
* a live prediction demo
* a daily prediction table
* a link to the API documentation

## Production features

* modular package structure
* `uv` dependency management
* MLflow experiment tracking
* Docker containerization
* CI/CD workflows
* FastAPI serving
* GitOps deployment
* Quarto website

## Future improvements

* add historical performance views to the website
* extend the pipeline to other Kalshi series
* compare LightGBM with alternative models


### Authors 
Maxime Coppa, Antoine Gilson, Marama Simoneau et Auguste Vautrin - ENSAE 2026


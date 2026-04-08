# MLOps-Prediction-Market

Train and evaluate a machine learning pipeline to predict **Kalshi** prediction-market price movements using the **Kalshi Elections Trade API v2**.

This repo includes:
- A small Kalshi API client (series/markets/orderbook/candlesticks/trades)
- Time-series construction (daily prices + daily traded value)
- Feature engineering for ML (lagged returns + lagged volume)
- Notebooks and scripts to run analysis / experiments

---

## Setup

Requires **Python 3.9+**.

Install dependencies with [uv](https://github.com/astral-sh/uv):

```bash
uv sync
````

---

## Run

Run the main pipeline / analysis:

```bash
uv python run main.py
```

---

## Notebooks

* `get_market_data_demo.ipynb`: explore series/markets/orderbook and download data
* `market_prediction.ipynb`: feature engineering + modeling experiments

---

## Project layout

- `market_data/`
  - `markets.py` — `KalshiClient`, `KalshiAnalyzer` (API + time series)
  - `feature_engineering.py` — `KalshiFeatureEngineer` (ML dataset builder)
  - `utils.py` — helpers (if any)
  - `__init__.py`
- `main.py` — entrypoint to run analysis / pipeline
- `exp_kalshi_api.py` — experiment / sandbox script (optional usage)
- `get_market_data_demo.ipynb` — demo: pulling market data
- `market_prediction.ipynb` — modeling experiments
- `data/` — local data artifacts (if used)
- `pyproject.toml`, `uv.lock`, `requirements.txt` — dependencies / env

---

## Kalshi API endpoints used

Base URL:

[https://api.elections.kalshi.com/trade-api/v2](https://api.elections.kalshi.com/trade-api/v2)



Endpoints mapped to code:

- **Series metadata**
  - `GET /series/{series_ticker}`
  - Used by: `KalshiClient.get_series_information()`

- **Markets in a series**
  - `GET /markets?series_ticker={series_ticker}&status={status}`
  - Used by: `KalshiClient.get_markets_data()`

- **Orderbook**
  - `GET /markets/{market_ticker}/orderbook`
  - Used by: `KalshiAnalyzer.get_orderbook_data()`

- **Candlesticks (historical prices)**
  - `GET /series/{series}/markets/{market}/candlesticks?start_ts&end_ts&period_interval`
  - Used by: `KalshiAnalyzer.get_price_data()`

- **Trades (historical executions, paginated)**
  - `GET /markets/trades?ticker&min_ts&max_ts&limit&cursor`
  - Used by: `KalshiAnalyzer.get_trades_data()`

---

## Data produced by the analyzer

### Daily price series
`KalshiAnalyzer.get_price_data()` converts candlesticks into:

- `pd.Series`
- index: daily UTC date
- value: mean price in **cents** (`mean_dollars * 100`)
- missing dates forward-filled (`ffill`) after sorting

### Daily traded value
`KalshiAnalyzer.get_trades_data()` downloads all trades (cursor pagination) and computes traded value per trade:

- if `taker_side == "yes"` → `count × yes_price`
- else → `count × no_price`

Then aggregates to:

- `pd.DataFrame`
- index: daily UTC date
- columns: `yes`, `no` (daily traded value per taker side)

---

## Feature engineering for prediction

`KalshiFeatureEngineer` (inherits from `KalshiAnalyzer`) builds a supervised ML dataset.

### Features (X)
From daily price returns and YES-side traded value:

1. Compute daily return:
   - `RET[t] = pct_change(price)[t]`

2. Define volume proxy:
   - `VOLUME[t] = daily YES traded value`

3. Create a 10-day lag window:
   - `RET_1 ... RET_10`
   - `VOLUME_1 ... VOLUME_10`

Final `X` columns:
- `TS` (numeric time identifier)
- `RET_1..RET_10`
- `VOLUME_1..VOLUME_10`

### Target (y)
Binary label:

- `y[t] = 1` if `RET[t] > 0`
- `y[t] = 0` otherwise

### Train/validation split
`split_data(train_size=0.8)` uses:

- chronological split (`shuffle=False`)
- returns `X_train, X_val, y_train, y_val`

---

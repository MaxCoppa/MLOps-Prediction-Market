import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta, timezone
from typing import Tuple
from ..utils import get_logger

log = get_logger()


def fetch_data(ticker: str, window_days: int = 365) -> pd.DataFrame:
    API_URL = "https://api.elections.kalshi.com/trade-api/v2"
    session = requests.Session()
    end_dt = datetime.now(timezone.utc)
    start_dt = end_dt - timedelta(days=window_days)
    start_ts = int(start_dt.timestamp())
    end_ts = int(end_dt.timestamp())

    """
    Fetch and merge candlestick + trade data for a single ticker, 
    returning a DataFrame indexed by date with features and target variable.
    """

    parts = ticker.split("-")
    series_ticker = parts[0]
    log.info(f"Fetching candlesticks for {ticker} (series={series_ticker}) …")

    resp = session.get(
        f"{API_URL}/series/{series_ticker}/markets/{ticker}/candlesticks",
        params={"start_ts": start_ts, "end_ts": end_ts, "period_interval": 1440},
    )
    resp.raise_for_status()
    candles = resp.json().get("candlesticks", [])

    if not candles:
        raise ValueError(
            f"No candlestick data returned for ticker '{ticker}'. Check the ticker name and date range."
        )

    rows = []
    for c in candles:
        if "price" not in c:
            continue
        rows.append(
            {
                "ts": pd.to_datetime(c["end_period_ts"], unit="s", utc=True),
                "open": c["price"].get("open_dollars", np.nan),
                "high": c["price"].get("high_dollars", np.nan),
                "low": c["price"].get("low_dollars", np.nan),
                "close": c["price"].get("close_dollars", np.nan),
                "mean": c["price"].get("mean_dollars", np.nan),
                "volume": c.get("volume", 0),
            }
        )

    df = pd.DataFrame(rows).set_index("ts").sort_index().astype(float).ffill()
    df["price"] = df["mean"] * 100

    log.info("Fetching trade history …")
    cursor, all_trades = "", []
    while True:
        params = {"ticker": ticker, "min_ts": start_ts, "max_ts": end_ts, "limit": 1000}
        if cursor:
            params["cursor"] = cursor
        data = session.get(f"{API_URL}/markets/trades", params=params).json()
        batch = data.get("trades", [])
        if not batch:
            break
        all_trades.extend(batch)
        cursor = data.get("cursor")
        if not cursor:
            break

    if all_trades:
        df_t = pd.DataFrame(all_trades)

        df_t["ts"] = pd.to_datetime(df_t["created_time"], utc=True).dt.normalize()

        for col in ["count_fp", "yes_price_dollars", "no_price_dollars"]:
            df_t[col] = df_t[col].astype(float) if col in df_t.columns else 0.0

        mask_yes = df_t["taker_side"] == "yes"
        df_t["val"] = np.where(
            mask_yes,
            df_t["count_fp"] * df_t["yes_price_dollars"],
            df_t["count_fp"] * df_t["no_price_dollars"],
        )

        daily_vol = (
            df_t.groupby(["ts", "taker_side"])["val"]
            .sum()
            .unstack(fill_value=0.0)
            .reindex(columns=["yes", "no"], fill_value=0.0)
        )

        df = df.join(
            daily_vol.rename(columns={"yes": "vol_yes", "no": "vol_no"}), how="left"
        ).fillna(0.0)
    else:
        df["vol_yes"] = df["vol_no"] = 0.0

    df["vol_total"] = df["vol_yes"] + df["vol_no"]
    log.info(
        f"Data fetched: {len(df)} daily bars from {df.index[0].date()} to {df.index[-1].date()}"
    )
    return df


def build_features(
    df: pd.DataFrame, n_lags: int = 10
) -> Tuple[pd.DataFrame, pd.Series]:
    f = pd.DataFrame(index=df.index)
    price = df["price"].replace(0, np.nan)
    ret = price.pct_change()

    """
    Build lagged features and rolling statistics from price and volume data,
    returning feature matrix X and target variable y (next-day return).
    """

    vol_total = df["vol_total"].replace(0, np.nan)
    vol_yes = df["vol_yes"]
    vol_no = df["vol_no"]

    for lag in range(1, n_lags + 1):
        f[f"RET_{lag}"] = ret.shift(lag)
        f[f"VOL_YES_{lag}"] = vol_yes.shift(lag)
        f[f"VOL_NO_{lag}"] = vol_no.shift(lag)
        f[f"VOL_TOTAL_{lag}"] = vol_total.shift(lag)

    for window in [5, 10]:
        f[f"RET_MEAN_{window}"] = ret.shift(1).rolling(window).mean()
        f[f"RET_STD_{window}"] = ret.shift(1).rolling(window).std()
        f[f"VOL_MEAN_{window}"] = vol_total.shift(1).rolling(window).mean()

    f["DIST_50"] = (price.shift(1) - 50.0) / 50.0
    roll_min = price.shift(1).rolling(20).min()
    roll_max = price.shift(1).rolling(20).max()
    f["DIST_MIN_20"] = (price.shift(1) - roll_min) / (roll_max - roll_min + 1e-9)
    f["DIST_MAX_20"] = (roll_max - price.shift(1)) / (roll_max - roll_min + 1e-9)
    f["VOL_IMBALANCE"] = ((vol_yes - vol_no) / (vol_total + 1e-9)).shift(1)

    y = ret.shift(-1)
    combined = pd.concat([f, y.rename("__target__")], axis=1)
    X = combined.drop(columns="__target__")
    y = combined["__target__"].fillna(0)

    log.info(f"Feature matrix: {X.shape[0]} rows × {X.shape[1]} features")
    return X, y

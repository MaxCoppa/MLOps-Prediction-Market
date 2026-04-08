import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta, timezone
import time
from typing import Tuple
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from ..utils import get_logger

log = get_logger()

API_URL = "https://api.elections.kalshi.com/trade-api/v2"


# ─────────────────────────────────────────────────────────────────────────────
# 1.  DISCOVER TICKERS IN A SERIES
# ─────────────────────────────────────────────────────────────────────────────


def fetch_tickers(series_ticker: str) -> list[str]:
    """
    Return all market tickers belonging to a series.
    """
    session = requests.Session()
    tickers, cursor = [], ""

    log.info(f"Discovering tickers for series '{series_ticker}'")
    while True:
        params = {"series_ticker": series_ticker, "limit": 200}
        if cursor:
            params["cursor"] = cursor
        resp = session.get(f"{API_URL}/markets", params=params)
        resp.raise_for_status()
        data = resp.json()
        batch = [m["ticker"] for m in data.get("markets", [])]
        tickers.extend(batch)
        cursor = data.get("cursor")
        if not cursor or not batch:
            break

    log.info(f"Found {len(tickers)} tickers in series '{series_ticker}'")
    return tickers


# ─────────────────────────────────────────────────────────────────────────────
# 2.  FETCH RAW DATA FOR ALL TICKERS  →  MultiIndex DataFrame
# ─────────────────────────────────────────────────────────────────────────────


def _fetch_trades_worker(ticker: str, start_ts: int, end_ts: int) -> list[dict]:
    session = requests.Session()
    cursor, trades = "", []

    while True:
        params = {"ticker": ticker, "min_ts": start_ts, "max_ts": end_ts, "limit": 1000}
        if cursor:
            params["cursor"] = cursor

        try:
            resp = session.get(f"{API_URL}/markets/trades", params=params)
            if resp.status_code == 429:
                time.sleep(2.0)
                continue
            resp.raise_for_status()

            data = resp.json()
            batch = data.get("trades", [])
            for t in batch:
                t["_ticker"] = ticker
            trades.extend(batch)

            cursor = data.get("cursor")
            time.sleep(0.06)
            if not cursor or not batch:
                break

        except requests.exceptions.RequestException as e:
            log.error(f"Worker network error on {ticker}: {e}")
            break

    return trades


def fetch_data(
    series_ticker: str, tickers: list[str], window_days: int = 365
) -> pd.DataFrame:
    session = requests.Session()
    end_dt = datetime.now(timezone.utc)
    start_ts = int((end_dt - timedelta(days=window_days)).timestamp())
    end_ts = int(end_dt.timestamp())

    log.info(f"Fetching candlesticks for {len(tickers)} tickers …")
    all_candles = []
    batch_size = max(1, int(10_000 / max(window_days, 1)))

    for i in tqdm(range(0, len(tickers), batch_size), desc="Candlesticks", ncols=80):
        batch = tickers[i : i + batch_size]
        params = {
            "market_tickers": ",".join(batch),
            "start_ts": start_ts,
            "end_ts": end_ts,
            "period_interval": 1440,
        }

        try:
            while True:
                resp = session.get(f"{API_URL}/markets/candlesticks", params=params)
                if resp.status_code == 429:
                    time.sleep(2.0)
                    continue
                resp.raise_for_status()
                break

            for m in resp.json().get("markets", []):
                t = m["market_ticker"]
                for c in m.get("candlesticks", []):
                    if "price" in c:
                        all_candles.append(
                            {
                                "ts": pd.to_datetime(
                                    c["end_period_ts"], unit="s", utc=True
                                ),
                                "ticker": t,
                                "open": c["price"].get("open_dollars", np.nan),
                                "high": c["price"].get("high_dollars", np.nan),
                                "low": c["price"].get("low_dollars", np.nan),
                                "close": c["price"].get("close_dollars", np.nan),
                                "mean": c["price"].get("mean_dollars", np.nan),
                                "volume": c.get("volume", 0),
                            }
                        )
            time.sleep(0.06)
        except requests.exceptions.RequestException as e:
            log.error(f"Candlesticks network error: {e}")
            break

    if not all_candles:
        raise ValueError(f"No candlestick data returned for series '{series_ticker}'.")

    df_candles = (
        pd.DataFrame(all_candles).set_index(["ts", "ticker"]).sort_index().astype(float)
    )
    df_candles["price"] = df_candles["mean"] * 100

    log.info(f"Fetching trade history in parallel (20 workers) …")
    all_trades = []

    with ThreadPoolExecutor(max_workers=20) as executor:
        future_to_ticker = {
            executor.submit(_fetch_trades_worker, t, start_ts, end_ts): t
            for t in tickers
        }

        for future in tqdm(
            as_completed(future_to_ticker), total=len(tickers), desc="Trades", ncols=80
        ):
            try:
                trades_batch = future.result()
                all_trades.extend(trades_batch)
            except Exception as e:
                ticker = future_to_ticker[future]
                log.error(f"Thread execution failed for {ticker}: {e}")

    if all_trades:
        df_t = pd.DataFrame(all_trades)
        df_t["ts"] = pd.to_datetime(
            df_t["created_time"], format="ISO8601", utc=True
        ).dt.normalize()
        df_t["ticker"] = df_t["_ticker"]

        for col in ["count_fp", "yes_price_dollars", "no_price_dollars"]:
            df_t[col] = df_t[col].astype(float) if col in df_t.columns else 0.0

        mask_yes = df_t["taker_side"] == "yes"
        df_t["val"] = np.where(
            mask_yes,
            df_t["count_fp"] * df_t["yes_price_dollars"],
            df_t["count_fp"] * df_t["no_price_dollars"],
        )

        daily_vol = (
            df_t.groupby(["ts", "ticker", "taker_side"])["val"]
            .sum()
            .unstack("taker_side", fill_value=0.0)
            .reindex(columns=["yes", "no"], fill_value=0.0)
            .rename(columns={"yes": "vol_yes", "no": "vol_no"})
        )

        normalized_ts_candles = pd.to_datetime(
            df_candles.index.get_level_values("ts")
        ).normalize()
        df_candles.index = pd.MultiIndex.from_arrays(
            [normalized_ts_candles, df_candles.index.get_level_values("ticker")],
            names=["ts", "ticker"],
        )

        normalized_ts_vol = pd.to_datetime(
            daily_vol.index.get_level_values("ts")
        ).normalize()
        daily_vol.index = pd.MultiIndex.from_arrays(
            [normalized_ts_vol, daily_vol.index.get_level_values("ticker")],
            names=["ts", "ticker"],
        )

        df_candles = df_candles.join(daily_vol, how="left").fillna(0.0)
    else:
        df_candles["vol_yes"] = 0.0
        df_candles["vol_no"] = 0.0

    df_candles["vol_total"] = df_candles["vol_yes"] + df_candles["vol_no"]
    n_tickers = df_candles.index.get_level_values("ticker").nunique()
    n_dates = df_candles.index.get_level_values("ts").nunique()
    log.info(
        f"Panel ready: {n_tickers} tickers × {n_dates} dates = {len(df_candles)} rows"
    )
    return df_candles


# ─────────────────────────────────────────────────────────────────────────────
# 3.  FEATURE ENGINEERING  →  panel (date, ticker) MultiIndex
# ─────────────────────────────────────────────────────────────────────────────


def build_features(
    df: pd.DataFrame,
    n_lags: int = 10,
    min_obs: int = 30,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Build a supervised ML dataset across all tickers simultaneously.

    Each ticker is treated as a separate time series; features are
    computed per-ticker (no cross-contamination), then stacked into a
    single panel DataFrame with MultiIndex (date, ticker).

    Features per ticker (same as single-ticker pipeline):
        RET_1…n, VOL_YES_1…n, VOL_NO_1…n, VOL_TOTAL_1…n,
        RET_MEAN_5/10, RET_STD_5/10, VOL_MEAN_5/10,
        DIST_50, DIST_MIN_20, DIST_MAX_20, VOL_IMBALANCE

    Target: next-day return per ticker.

    Tickers with fewer than `min_obs` valid rows are dropped.
    """
    frames = []
    tickers = df.index.get_level_values("ticker").unique()

    log.info(f"Building features for {len(tickers)} tickers …")

    for ticker in tqdm(tickers, desc="Features", ncols=80):
        sub = df.xs(ticker, level="ticker").sort_index()

        price = sub["price"].replace(0, np.nan)
        ret = price.pct_change()
        vol_total = sub["vol_total"].replace(0, np.nan)
        vol_yes = sub["vol_yes"]
        vol_no = sub["vol_no"]

        f = pd.DataFrame(index=sub.index)

        # lagged features
        for lag in range(1, n_lags + 1):
            f[f"RET_{lag}"] = ret.shift(lag)
            f[f"VOL_YES_{lag}"] = vol_yes.shift(lag)
            f[f"VOL_NO_{lag}"] = vol_no.shift(lag)
            f[f"VOL_TOTAL_{lag}"] = vol_total.shift(lag)

        # rolling stats
        for window in [5, 10]:
            f[f"RET_MEAN_{window}"] = ret.shift(1).rolling(window).mean()
            f[f"RET_STD_{window}"] = ret.shift(1).rolling(window).std()
            f[f"VOL_MEAN_{window}"] = vol_total.shift(1).rolling(window).mean()

        # price level features
        roll_min = price.shift(1).rolling(20).min()
        roll_max = price.shift(1).rolling(20).max()
        f["DIST_50"] = (price.shift(1) - 50.0) / 50.0
        f["DIST_MIN_20"] = (price.shift(1) - roll_min) / (roll_max - roll_min + 1e-9)
        f["DIST_MAX_20"] = (roll_max - price.shift(1)) / (roll_max - roll_min + 1e-9)
        f["VOL_IMBALANCE"] = ((vol_yes - vol_no) / (vol_total + 1e-9)).shift(1)

        # target
        y_ticker = ret.shift(-1).fillna(0).rename("__target__")

        combined = pd.concat([f, y_ticker], axis=1).dropna(subset=list(f.columns))
        if len(combined) < min_obs:
            continue  # skip illiquid / short-lived tickers

        combined.index = pd.MultiIndex.from_arrays(
            [combined.index, [ticker] * len(combined)],
            names=["ts", "ticker"],
        )
        frames.append(combined)

    if not frames:
        raise ValueError(
            "No tickers passed the min_obs filter. Try reducing min_obs or increasing window_days."
        )

    panel = pd.concat(frames).sort_index()
    X = panel.drop(columns="__target__")
    y = panel["__target__"]

    kept = X.index.get_level_values("ticker").nunique()
    log.info(
        f"Panel feature matrix: {X.shape[0]} rows × {X.shape[1]} features across {kept} tickers"
    )
    return X, y

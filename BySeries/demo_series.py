"""
demo_series.py
--------------
Validates the full series pipeline with synthetic multi-ticker data.
No API key required.

Run:
    python demo_series.py
"""
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

import data_series
import main_series

# ─────────────────────────────────────────────────────────────────────────────
# MOCK:  replace fetch_series_tickers + fetch_series_data
# ─────────────────────────────────────────────────────────────────────────────

N_TICKERS   = 15    # number of synthetic contracts
WINDOW_DAYS = 300   # days of history per ticker

def _mock_fetch_tickers(series_ticker: str, status: str = "closed") -> list[str]:
    tickers = [f"{series_ticker}-MOCK{i:02d}" for i in range(N_TICKERS)]
    print(f"[MOCK] Generated {len(tickers)} synthetic tickers for series '{series_ticker}'")
    return tickers


def _mock_fetch_data(
    series_ticker: str,
    tickers: list[str],
    window_days: int = WINDOW_DAYS,
) -> pd.DataFrame:
    """
    Generate independent mean-reverting price + volume series for each ticker.
    Returns a MultiIndex (ts, ticker) DataFrame matching the real schema.
    """
    np.random.seed(42)
    dates = pd.date_range(end="2025-01-01", periods=window_days, freq="D", tz="UTC")
    frames = []

    for i, ticker in enumerate(tickers):
        rng   = np.random.RandomState(i * 7 + 3)
        price = 50.0
        prices = []
        for _ in range(window_days):
            shock  = rng.normal(0, 1.8)
            price  = float(np.clip(price + 0.25 * (50 - price) + shock, 1, 99))
            prices.append(price)

        vol_total = rng.exponential(4000, window_days) + 200
        vol_yes   = vol_total * rng.beta(2, 2, window_days)
        vol_no    = vol_total - vol_yes

        sub = pd.DataFrame({
            "price":     prices,
            "open":      np.array(prices) * (1 + rng.normal(0, 0.005, window_days)),
            "high":      np.array(prices) * (1 + np.abs(rng.normal(0, 0.01, window_days))),
            "low":       np.array(prices) * (1 - np.abs(rng.normal(0, 0.01, window_days))),
            "close":     np.array(prices) * (1 + rng.normal(0, 0.005, window_days)),
            "mean":      np.array(prices) / 100,   # dollars (price = mean * 100)
            "volume":    vol_total,
            "vol_yes":   vol_yes,
            "vol_no":    vol_no,
            "vol_total": vol_total,
        }, index=dates)

        sub.index = pd.MultiIndex.from_arrays(
            [dates, [ticker] * window_days], names=["ts", "ticker"]
        )
        frames.append(sub)

    panel = pd.concat(frames).sort_index()
    print(f"[MOCK] Synthetic panel: {len(tickers)} tickers × {window_days} days = {len(panel)} rows")
    return panel


# monkey-patch
data_series.fetch_series_tickers = _mock_fetch_tickers
data_series.fetch_series_data    = _mock_fetch_data

# propagate into main_series (it imported these names directly)
import importlib
importlib.reload(main_series)


# ─────────────────────────────────────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json

    report = main_series.run_series_pipeline(
        series_ticker   = "DEMO",
        status          = "closed",
        window_days     = WINDOW_DAYS,
        n_lags          = 10,
        min_obs         = 30,
        n_optuna_trials = 20,
        val_ratio       = 0.2,
        min_train       = 40,
        refit_freq      = 5,
        output_dir      = "outputs_series",
    )

    print("\nFull report:")
    printable = {k: v for k, v in report.items() if k != "best_params"}
    print(json.dumps(printable, indent=2))
    print(f"\nBest params: {json.dumps(report['best_params'], indent=2)}")

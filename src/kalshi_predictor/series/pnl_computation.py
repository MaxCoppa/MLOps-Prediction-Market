import numpy as np
import pandas as pd
from ..utils import get_logger
from .data import (
    fetch_tickers,
    fetch_data,
    build_features,
)

log = get_logger()


def compute_yesterday_pnl(
    series_ticker: str, model, n_lags: int = 10, window_days: int = 40
) -> pd.DataFrame:
    """
    Compute PnL for yesterday's predictions across all tickers in a series, 
    using the provided model and features built from recent data.
    """

    tickers = fetch_tickers(series_ticker)
    df = fetch_data(series_ticker, tickers, window_days=window_days)
    X, y = build_features(df, n_lags=n_lags, min_obs=0)

    dates = X.index.get_level_values("ts").unique().sort_values()
    if len(dates) < 2:
        raise ValueError("Insufficient data.")

    target_date = dates[-2]
    X_target = X.loc[[target_date]]
    y_target = y.loc[[target_date]].values

    y_hat = model.predict(X_target)

    abs_sum = np.sum(np.abs(y_hat))
    signals = y_hat / (abs_sum + 1e-9)

    pnls = signals * y_target

    results = pd.DataFrame(
        {
            "ticker": X_target.index.get_level_values("ticker"),
            "y_true": y_target,
            "y_hat": y_hat,
            "signal": signals,
            "pnl": pnls,
        }
    ).set_index("ticker")

    win_rate = float((results["pnl"] > 0).mean())
    log.info(f"PnL evaluation for {target_date.date()} | Win Rate: {win_rate:.2f}")

    return results

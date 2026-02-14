from .markets import KalshiAnalyzer
from typing import Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


class KalshiFeatureEngineer(KalshiAnalyzer):
    def __init__(
        self,
        series_ticker: str,
        market_ticker: str | None = None,
    ):
        super().__init__(series_ticker, market_ticker)

    def build_features(
        self,
        start_ts: str = "2025-09-23",
        end_ts: str = "2026-02-15",
    ) -> Tuple[pd.DataFrame, pd.Series]:

        prices = self.get_price_data(start_ts=start_ts, end_ts=end_ts, plot=False)
        volumes = self.get_trades_data(start_ts=start_ts, end_ts=end_ts, plot=False)
        shift = 10
        ret = prices.pct_change()
        volumes["RET"] = ret.reindex(volumes.index)
        volumes["VOLUME"] = volumes["yes"]

        X = volumes[["RET", "VOLUME"]].copy()
        X.reset_index(
            inplace=True,
            drop=True,
        )
        X.reset_index(inplace=True, names="TS")
        X["TS"] = X["TS"] + np.random.randint(1e5)

        for i in range(1, shift + 1):
            X[f"RET_{i}"] = X["RET"].shift(i)
            X[f"VOLUME_{i}"] = X["VOLUME"].shift(i)

        X = X[shift + 1 :].reset_index(drop=True)
        y = X["RET"] > 0
        X = X.drop(columns=["RET", "VOLUME"])

        self.X = X
        self.y = y

        return X, y

    def split_data(self, train_size: float = 0.8):
        X_train, X_val, y_train, y_val = train_test_split(
            self.X, self.y, train_size=train_size, shuffle=False
        )

        return X_train, X_val, y_train, y_val

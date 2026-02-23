from .markets import KalshiAnalyzer
from typing import Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from datetime import datetime, timedelta


class KalshiFeatureEngineer(KalshiAnalyzer):
    def __init__(
        self,
        series_ticker: str,
        market_ticker: str | None = None,
    ):
        super().__init__(series_ticker, market_ticker)

    def build_features(
        self,
        window_size: int = 100,
        start_ts: str = None,
        end_ts: str = None,
    ) -> Tuple[pd.DataFrame, pd.Series]:
        if end_ts is None:
            end_ts = datetime.now().strftime("%Y-%m-%d")
        if start_ts is None:
            start_ts = (datetime.now() - timedelta(days=window_size)).strftime("%Y-%m-%d")

        # 2. Get synchronized data
        # Assumes get_synchronized_data is updated to accept start/end strings
        df = self.get_price_volume_data(days=window_size)
        
        if df.empty:
            return pd.DataFrame(), pd.Series()
        
        try:
            shift = 10
            df["RET"] = (df["price"]-df["price"].shift())/df["price"].shift().replace(0,np.nan)
            df["VOLUME_TOTAL"] = df["vol_total"]
            df["VOLUME_YES"] = df["vol_yes"]
            df["VOLUME_NO"] = df["vol_no"]

            X = df[["RET", "VOLUME_TOTAL", "VOLUME_YES", "VOLUME_NO"]].copy()
            
            # Lagging
            for i in range(1, shift + 1):
                X[f"RET_{i}"] = X["RET"].shift(i)
                X[f"VOLUME_TOTAL_{i}"] = X["VOLUME_TOTAL"].shift(i)
                X[f"VOLUME_YES_{i}"] = X["VOLUME_YES"].shift(i)
                X[f"VOLUME_NO_{i}"] = X["VOLUME_NO"].shift(i)
            
            y = X["RET"]
            X = X.drop(columns=["RET", "VOLUME_TOTAL"])
        except:
            return pd.DataFrame(), pd.Series()
        return X, y
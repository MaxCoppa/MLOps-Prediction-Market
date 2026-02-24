# %%
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timezone
import time
import pandas as pd
from market_data import KalshiClient, KalshiAnalyzer, KalshiFeatureEngineer

from market_data.utils import convert_ts, request_api

# %% Step 1: Get Series Information


series_ticker = "KXOSCARPIC"  # Oscar for Best Picture ?
market_ticker = "KXOSCARPIC-26-HAM"

path = f"/markets/{market_ticker}/orderbook"
start_ts = "2025-09-23"
end_ts = "2026-02-15"

# %%
analyzer = KalshiAnalyzer(series_ticker=series_ticker, market_ticker=market_ticker)
prices = analyzer.get_price_data(
    start_ts=start_ts,
    end_ts=end_ts,
    plot=False,
)  # Here Yes Price

volumes = analyzer.get_trades_data(
    start_ts=start_ts,
    end_ts=end_ts,
    plot=False,
)
# %%
volumes["prices"] = prices
# %%
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

# %%
for i in range(1, 11):
    X[f"RET_{i}"] = X["RET"].shift(i)
    X[f"VOLUME_{i}"] = X["VOLUME"].shift(i)
# %%
X = X[11:].reset_index(drop=True)
# %%
y = X["RET"] > 0
X.drop(columns=["RET", "VOLUME"])

# %%
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

model = Pipeline(
    [("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=2000))]
)

# %%
X_train, X_val, y_train, y_val = train_test_split(X, y, train_size=0.8, shuffle=False)
# %%

model.fit(X_train, y_train)
# %%
print("Train acc:", model.score(X_train, y_train))
print("Val acc:", model.score(X_val, y_val))

# %%

# %%
import requests
import pandas as pd
from datetime import datetime, timezone
import time
import pandas as pd
from market_data import KalshiClient, KalshiAnalyzer

from market_data.utils import convert_ts, request_api

# %% Step 1: Get Series Information


series_ticker = "KXOSCARPIC"  # Oscar for Best Picture ?
market_ticker = "KXOSCARPIC-26-HAM"

path = f"/markets/{market_ticker}/orderbook"
start_ts = "2025-09-23"
end_ts = "2026-02-15"
# %%
client = KalshiClient(series_ticker=series_ticker)
series_data = client.get_series_information()
markets_data = client.get_markets_data()
# %%
analyzer = KalshiAnalyzer(series_ticker=series_ticker, market_ticker=market_ticker)
orderbook_data = analyzer.get_orderbook_data()
prices = analyzer.get_price_data(
    start_ts=start_ts,
    end_ts=end_ts,
    plot=False,
)

# %%
start = convert_ts(start_ts)
end = convert_ts(end_ts)
path = "/markets/trades"
cursor = ""
trades = []
n = 1

while n != 0:
    params = {
        "ticker": market_ticker,
        "min_ts": start,
        "max_ts": end,
        "limit": 1000,
        "cursor": cursor,
    }

    trades_data = request_api(session=analyzer.session, path=path, params=params)
    cursor = trades_data["cursor"]
    n = len(cursor)
    trades = trades + trades_data["trades"]
# %%
trades_data = pd.DataFrame(trades)
trades_data
# %%
trades_data["trade_day"] = pd.to_datetime(trades_data["created_time"]).dt.normalize()

# %%
trades_data.groupby(["trade_day", "taker_side"])["trade_value"].sum().unstack()[
    "yes"
].plot()
trades_data.groupby(["trade_day", "taker_side"])["trade_value"].sum().unstack()[
    "no"
].plot()

# %%
mask = trades_data["taker_side"] == "yes"
trades_data.loc[mask, "trade_value"] = (
    trades_data.loc[mask, "count"] * trades_data.loc[mask, "yes_price"]
)
trades_data.loc[~mask, "trade_value"] = (
    trades_data.loc[~mask, "count"] * trades_data.loc[~mask, "no_price"]
)
# %%

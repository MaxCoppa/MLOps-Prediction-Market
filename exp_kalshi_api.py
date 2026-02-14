# %%
import requests
import pandas as pd
from datetime import datetime, timezone
import time


# %% Step 1: Get Series Information

series_ticker = "KXHIGHNY"
series_ticker = "KXUCLGAME"
series_ticker = "kxoscarpic".upper()
# Get series information for series_ticker
url = f"https://api.elections.kalshi.com/trade-api/v2/series/{series_ticker}"
response = requests.get(url)
series_data = response.json()

print(f"Series Title: {series_data['series']['title']}")
print(f"Frequency: {series_data['series']['frequency']}")
print(f"Category: {series_data['series']['category']}")
# %%
series_ticker
# %% Step 2: Get Today’s Events and Markets
# Get all open markets for the series_ticker series
markets_url = f"https://api.elections.kalshi.com/trade-api/v2/markets?series_ticker={series_ticker}&status=open"
markets_response = requests.get(markets_url)
markets_data = markets_response.json()

print(f"\nActive markets in {series_ticker} series:")
for market in markets_data["markets"]:
    print(f"- {market['ticker']}: {market['title']}")
    print(f"  Event: {market['event_ticker']}")
    print(
        f"  Yes Price: {market['yes_ask']}¢ | No Price: {market['no_ask']}¢ | Volume: {market['volume']}"
    )
    print()


# %%

# Get details for a specific event if you have its ticker
if markets_data["markets"]:
    # Let's get details for the first market's event
    event_ticker = markets_data["markets"][0]["event_ticker"]
    event_url = f"https://api.elections.kalshi.com/trade-api/v2/events/{event_ticker}"
    event_response = requests.get(event_url)
    event_data = event_response.json()

    print(f"Event Details:")
    print(f"Title: {event_data['event']['title']}")
    print(f"Category: {event_data['event']['category']}")
# %% Step 3: Get Orderbook Data

# Get orderbook for a specific market
# Replace with an actual market ticker from the markets list
if not markets_data["markets"]:
    raise ValueError(
        "No open markets found. Try removing status=open or choose another series."
    )

market_ticker = markets_data["markets"][-1]["ticker"]
orderbook_url = (
    f"https://api.elections.kalshi.com/trade-api/v2/markets/{market_ticker}/orderbook"
)

orderbook_response = requests.get(orderbook_url)
orderbook_data = orderbook_response.json()

print(f"\nOrderbook for {market_ticker}:")
print("YES BIDS:")
for bid in orderbook_data["orderbook"]["yes"][:5]:  # Show top 5
    print(f"  Price: {bid[0]}¢, Quantity: {bid[1]}")

print("\nNO BIDS:")
for bid in orderbook_data["orderbook"]["no"][:5]:  # Show top 5
    print(f"  Price: {bid[0]}¢, Quantity: {bid[1]}")
# %%


ticker = market_ticker
market_ticker, series_ticker

# %%


def convert_ts(datetime_ts):
    dt = datetime.strptime(datetime_ts, "%Y-%m-%d")
    return int(dt.replace(tzinfo=timezone.utc).timestamp())


# %%
start_ts = convert_ts("2025-09-23")
end_ts = start_ts + 86400 * 30
end_ts = convert_ts("2026-02-15")
freq = 1440  # 1, 60, 1440
# %%
candlesticks_url = f"https://api.elections.kalshi.com/trade-api/v2/series/{series_ticker}/markets/{ticker}/candlesticks?start_ts={start_ts}&end_ts={end_ts}&period_interval={freq}"
historic_price = requests.get(candlesticks_url).json()["candlesticks"]


# %%
historic_price_dict = [
    {"ts": data_ts["end_period_ts"]} | data_ts["price"] for data_ts in historic_price
]

historic_price_df = pd.DataFrame.from_dict(historic_price_dict)
historic_price_df["ts"] = pd.to_datetime(
    historic_price_df["ts"],
    unit="s",
    utc=True,
)

historic_price_df.set_index("ts")["mean_dollars"].astype(float).plot()
# %%
historic_price_df


# %%
prices = historic_price_df.set_index("ts")["mean_dollars"].astype(float) * 100
prices.index = prices.index.normalize()
prices.plot()
# %%
ref_prices = pd.read_csv(
    "/Users/maximecoppa/Desktop/Projects/MLOps-Prediction-Market/kalshi-price-history-kxoscarpic-26-day.csv"
)
ref_prices["timestamp"] = pd.to_datetime(ref_prices["timestamp"].str[:10], utc=True)
ref_prices = ref_prices.set_index("timestamp")["Hamnet"]
# %%
diff_prices = prices - ref_prices
# %%

diff_prices.isna().mean(), (diff_prices.abs() > 1e-1).mean()
# %%

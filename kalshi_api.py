# %%
import requests
import pandas as pd
from datetime import datetime, timezone
import time


# %% Step 1: Get Series Information

series_ticker = "KXHIGHNY"
# Get series information for series_ticker
url = f"https://api.elections.kalshi.com/trade-api/v2/series/{series_ticker}"
response = requests.get(url)
series_data = response.json()

print(f"Series Title: {series_data['series']['title']}")
print(f"Frequency: {series_data['series']['frequency']}")
print(f"Category: {series_data['series']['category']}")

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

market_ticker = markets_data["markets"][0]["ticker"]
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
# %%
today_midnight = datetime.now(timezone.utc).replace(
    hour=0, minute=0, second=0, microsecond=0
)

start_ts = int(today_midnight.timestamp())
now_ts = int(time.time())

# %%
candlesticks_url = f"https://api.elections.kalshi.com/trade-api/v2/series/{series_ticker}/markets/{ticker}/candlesticks?start_ts=1770854400&end_ts=1770914728&period_interval=1&include_latest_before_start=true"
price_data = requests.get(candlesticks_url).json()["candlesticks"]

# %%

dict_price = [
    {"ts": data_ts["end_period_ts"]} | data_ts["yes_ask"] for data_ts in price_data
]

df_price = pd.DataFrame.from_dict(dict_price)
df_price["ts"] = pd.to_datetime(
    df_price["ts"],
    unit="s",
)

df_price.set_index("ts")["high_dollars"].astype(float).plot()
# %%

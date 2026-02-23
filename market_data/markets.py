import requests
import pandas as pd
from datetime import datetime, timezone, timedelta

API_URL = "https://api.elections.kalshi.com/trade-api/v2"


class KalshiClient:
    def __init__(
        self,
        series_ticker: str,
    ):
        self.series_ticker = series_ticker
        self.session = requests.Session()

        self.markets_data = None

    def _get(self, path: str, params: dict | None = None) -> dict:
        url = f"{API_URL}{path}"
        r = self.session.get(url, params=params)
        r.raise_for_status()
        return r.json()

    def get_series_information(self, verbose: bool = True) -> dict:
        series_data = self._get(f"/series/{self.series_ticker}")

        if verbose:
            print(f"Series Title: {series_data['series']['title']}")
            print(f"Frequency: {series_data['series']['frequency']}")
            print(f"Category: {series_data['series']['category']}")

        return series_data

    def get_markets_data(self, details: bool = True, status: str = "open") -> dict:
        markets_data = self._get(
            "/markets",
            params={"series_ticker": self.series_ticker, "status": status},
        )
        self.markets_data = markets_data

        if not markets_data.get("markets"):
            raise ValueError(
                "No markets found. Try another series_ticker or a different status."
            )

        if details:
            print(f"\nMarkets in {self.series_ticker} series (status={status}):")
            for market in markets_data["markets"]:
                print(f"- {market['ticker']}: {market['title']}")
                print(f"  Event: {market['event_ticker']}")
                print(
                    f"  Yes Ask: {market.get('yes_ask')}¢ | No Ask: {market.get('no_ask')}¢ | "
                    f"Volume: {market.get('volume')}"
                )
                print()

            # event details for the first market's event
            event_ticker = markets_data["markets"][0]["event_ticker"]
            event_data = self._get(f"/events/{event_ticker}")

            print("Event Details:")
            print(f"Title: {event_data['event']['title']}")
            print(f"Category: {event_data['event']['category']}")

        return markets_data
    


class KalshiAnalyzer(KalshiClient):
    def __init__(self, series_ticker: str, market_ticker: str | None = None):
        super().__init__(series_ticker)

        if market_ticker is None:
            self.get_markets_data(details=False, status="open")
            market_ticker = self.markets_data["markets"][-1]["ticker"]

        self.market_ticker = market_ticker
        self.orderbook_data = None

    def get_orderbook_data(self, top_n: int = 5, verbose: bool = True) -> dict:
        orderbook_data = self._get(f"/markets/{self.market_ticker}/orderbook")
        self.orderbook_data = orderbook_data

        if verbose:
            print(f"\nOrderbook for {self.market_ticker}:")
            print("YES BIDS:")
            for price, qty in orderbook_data["orderbook"]["yes"][:top_n]:
                print(f"  Price: {price}¢, Quantity: {qty}")

            print("\nNO BIDS:")
            for price, qty in orderbook_data["orderbook"]["no"][:top_n]:
                print(f"  Price: {price}¢, Quantity: {qty}")

        return orderbook_data

    @staticmethod
    def _convert_ts(date_str: str) -> int:
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        return int(dt.replace(tzinfo=timezone.utc).timestamp())

    def get_price_data(
        self,
        start_ts: str = None,
        end_ts: str = None,
        window_size: int = 30
    ) -> pd.Series:
        # Default dates handling
        end_ts = end_ts or datetime.now().strftime("%Y-%m-%d")
        start_ts = start_ts or (datetime.now() - timedelta(days=window_size)).strftime("%Y-%m-%d")
        
        start = self._convert_ts(start_ts)
        end = self._convert_ts(end_ts)
        freq = 1440 

        data = self._get(
            f"/series/{self.series_ticker}/markets/{self.market_ticker}/candlesticks",
            params={"start_ts": start, "end_ts": end, "period_interval": freq},
        )

        candles = data.get("candlesticks", [])
        if not candles:
            return pd.Series(name="mean_dollars", dtype=float)

        # Robust extraction: filter out candles without price data
        rows = [{"ts": c["end_period_ts"], **c.get("price", {})} for c in candles if "price" in c]
        
        df = pd.DataFrame(rows)
        if df.empty or "mean_dollars" not in df.columns:
            return pd.Series(name="mean_dollars", dtype=float)

        df["ts"] = pd.to_datetime(df["ts"], unit="s", utc=True)
        prices = df.set_index("ts")["mean_dollars"].astype(float) * 100.0

        return prices.sort_index().ffill()
    

    def get_price_volume_data(
        self,
        start_ts: str = None,
        end_ts: str = None,
        window_size: int = 30
        ) -> pd.DataFrame:
        """
        Fetch and align prices and volumes into a single synchronized DataFrame.
        """
        end_ts = end_ts or datetime.now().strftime("%Y-%m-%d")
        start_ts = start_ts or (datetime.now() - timedelta(days=window_size)).strftime("%Y-%m-%d")
        prices = self.get_price_data(start_ts=start_ts, end_ts=end_ts)
        
        if prices.empty:
            return pd.DataFrame()

        start_unix = int(prices.index.min().timestamp())
        end_unix = int(prices.index.max().timestamp() + 86400) # +1 day buffer
        
        path = "/markets/trades"
        cursor, all_trades = "", []
        
        while True:
            params = {"ticker": self.market_ticker, "min_ts": start_unix, "max_ts": end_unix, "limit": 1000}
            if cursor: params["cursor"] = cursor
            
            data = self._get(path=path, params=params)
            batch = data.get("trades", [])
            if not batch: break
            all_trades.extend(batch)
            cursor = data.get("cursor")
            if not cursor: break

        if not all_trades:
            df_final = prices.to_frame(name="price")
            df_final[["vol_yes", "vol_no"]] = 0.0
            return df_final

        # 3. Process Trades and Synchronize
        df_t = pd.DataFrame(all_trades)
        df_t["ts"] = pd.to_datetime(df_t["created_time"], utc=True)
        
        # Calculate trade values
        for s in ["yes", "no"]:
            if f"{s}_price" not in df_t.columns: df_t[f"{s}_price"] = 0.0
        
        mask_yes = df_t["taker_side"] == "yes"
        df_t["val"] = 0.0
        df_t.loc[mask_yes, "val"] = df_t["count"] * df_t["yes_price"]
        df_t.loc[~mask_yes, "val"] = df_t["count"] * df_t["no_price"]

        # 4. Rigorous Alignment using price index as bins
        # Each trade is assigned to the nearest price timestamp
        pivot = df_t.pivot_table(index="ts", columns="taker_side", values="val", aggfunc="sum")
        pivot = pivot.reindex(columns=["yes", "no"]).fillna(0.0)
        
        # Binning trades into price periods
        bins = pd.cut(pivot.index, bins=prices.index.tolist() + [prices.index.max() + pd.Timedelta(days=1)], 
                    labels=prices.index, right=False)
        volumes = pivot.groupby(bins, observed=False).sum()

        # 5. Final Merge
        dataset = pd.concat([prices.to_frame(name="price"), volumes], axis=1).fillna(0.0)
        dataset.columns = ["price", "vol_yes", "vol_no"]
        dataset["vol_total"] = dataset["vol_yes"] + dataset["vol_no"]
        
        return dataset.sort_index()

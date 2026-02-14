import requests
import pandas as pd
from datetime import datetime, timezone

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
        start_ts: str = "2025-09-23",
        end_ts: str = "2026-02-15",
        plot: bool = True,
    ) -> pd.Series:
        start = self._convert_ts(start_ts)
        end = self._convert_ts(end_ts)
        freq = 1440  # Change to 1, 60, 1440 but .normalize careful

        data = self._get(
            f"/series/{self.series_ticker}/markets/{self.market_ticker}/candlesticks",
            params={"start_ts": start, "end_ts": end, "period_interval": freq},
        )

        candles = data.get("candlesticks", [])
        rows = [{"ts": c["end_period_ts"], **c["price"]} for c in candles]

        df = pd.DataFrame(rows)
        if df.empty:
            return pd.Series(dtype=float)

        df["ts"] = pd.to_datetime(df["ts"], unit="s", utc=True).dt.normalize()
        prices = df.set_index("ts")["mean_dollars"].astype(float) * 100.0

        if plot:
            prices.plot()

        return prices.sort_index()

    def get_trades_data(
        self,
        start_ts: str = "2025-09-23",
        end_ts: str = "2026-02-15",
        plot: bool = True,
    ):

        start = self._convert_ts(start_ts)
        end = self._convert_ts(end_ts)
        path = "/markets/trades"
        cursor = ""
        trades = []
        n = 1

        while n != 0:
            params = {
                "ticker": self.market_ticker,
                "min_ts": start,
                "max_ts": end,
                "limit": 1000,
                "cursor": cursor,
            }

            trades_data = self._get(path=path, params=params)
            cursor = trades_data["cursor"]
            n = len(cursor)
            trades = trades + trades_data["trades"]
        trades_data = pd.DataFrame(trades)
        trades_data["trade_day"] = pd.to_datetime(
            trades_data["created_time"]
        ).dt.normalize()
        volumes = (
            trades_data.groupby(["trade_day", "taker_side"])["count"].sum().unstack()
        )

        self.trades_data = trades_data
        self.volumes = volumes

        if plot:
            volumes["yes"].plot()
            volumes["no"].plot()

        return volumes

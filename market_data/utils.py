import requests
import pandas as pd
from datetime import datetime, timezone


API_URL = "https://api.elections.kalshi.com/trade-api/v2"


def request_api(
    session: requests.Session, path: str, params: dict | None = None
) -> dict:
    url = f"{API_URL}{path}"
    r = session.get(url, params=params)
    r.raise_for_status()
    return r.json()


def convert_ts(date_str: str) -> int:
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    return int(dt.replace(tzinfo=timezone.utc).timestamp())

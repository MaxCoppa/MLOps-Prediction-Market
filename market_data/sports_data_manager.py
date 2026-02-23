from .markets import KalshiAnalyzer, KalshiClient
from .feature_engineering import KalshiFeatureEngineer
import numpy as np
import pandas as pd
from typing import List, Dict
import requests
from tqdm import tqdm
import time
from datetime import datetime, timedelta, timezone

class KalshiSportsManager(KalshiAnalyzer):
    """
    Module spécialisé pour le scan et l'extraction de features 
    sur l'ensemble du catalogue Sport de Kalshi.
    """
    def __init__(self):
        # On passe une chaîne vide ou un placeholder. 
        # On surcharge le comportement pour éviter l'appel automatique à get_markets_data
        KalshiClient.__init__(self, series_ticker="") 
        self.market_ticker = None
        self.markets_data = None

    def get_all_sports_series(self, volume_thresh: float = 1e6, n_ticker: int = 1) -> List[str]:
        """
        Retrieves series tickers in 'Sports' category with volume and ticker count filters.
        """
        params = {
            "category": "Sports", 
            "limit": 1000, 
            "include_volume": True
        }
        
        data = self._get("/series", params=params)
        if not data or 'series' not in data.keys():
            return []
            
        # Filter by volume AND number of markets/tickers within the series
        return [
            s['ticker'] for s in data.get('series', []) 
            if s.get("volume", 0) > volume_thresh 
            and len(data.get('series', [])) >= n_ticker
        ]
    def get_all_active_tickers(
        self, 
        series_list: List[str], 
        min_days: int = 10, 
        max_spread: int = 5,
        min_vol: float = 0
    ) -> List[tuple]:
        """
        Filtre les tickers par âge, liquidité et coût de transaction (spread).
        """
        all_active = []
        now = datetime.now(timezone.utc)
        meta_data = {}
        
        for s_ticker in tqdm(series_list, desc="Filtering Tickers"):
            try:
                params = {"series_ticker": s_ticker, "status": "open", "limit": 1000}
                markets_res = self._get("/markets", params=params)
                
                if not markets_res or "markets" not in markets_res:
                    continue

                for m in markets_res["markets"]:
                    # 1. Extraction de l'âge
                    open_ts = m.get('open_time') or m.get('created_time')
                    if not open_ts: continue
                    dt_open = datetime.fromisoformat(open_ts.replace('Z', '+00:00'))
                    
                    # 2. Calcul du spread bid-ask
                    yes_ask = m.get('yes_ask')
                    yes_bid = m.get('yes_bid')
                    
                    # Gestion des cas où le carnet d'ordres est vide d'un côté
                    if yes_ask is None or yes_bid is None:
                        current_spread = float('inf')
                    else:
                        current_spread = yes_ask - yes_bid

                    # 3. Application des filtres
                    age_days = (now - dt_open).days
                    volume = m.get('volume', 0)

                    meta_data[m["ticker"]] = { k : v for k, v in m.items() if k != "ticker" }

                    if (age_days >= min_days and 
                        current_spread <= max_spread and
                        volume >= min_vol
                        ):
                        
                        all_active.append((s_ticker, m['ticker']))
                
                time.sleep(0.05) 
            except Exception as e:
                print(f"Skipping series {s_ticker}: {e}")
                continue
        self.meta_data = meta_data     
        return all_active
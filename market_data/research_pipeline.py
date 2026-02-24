import requests
import pandas as pd
import numpy as np
import lightgbm as lgb
from tqdm import tqdm
from datetime import datetime, timedelta, timezone
from sklearn.metrics import r2_score, accuracy_score
import warnings
warnings.filterwarnings("ignore")

class KalshiResearch:
    def __init__(self, base_url="https://api.elections.kalshi.com/trade-api/v2"):
        self.base_url = base_url
        self.df = None
        self.X, self.y = None, None

    def fetch_series(self, limit=1000):
        """
        Retrieves top series based on volume and metadata.
        """
        url = f"{self.base_url}/series"
        params = {"limit": limit}
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            return pd.DataFrame(response.json().get("series", []))
        except Exception as e:
            print(f"Error fetching series: {e}")
            return pd.DataFrame()

    def fetch_markets_from_series(self, series_tickers, limit=1000):
        """
        Retrieves all individual markets for a given list of series tickers.
        """
        all_markets = []
        url = f"{self.base_url}/markets/"
        
        for ticker in tqdm(series_tickers, desc="Extracting Markets"):
            params = {"series_ticker": ticker, "limit": limit}
            try:
                res = requests.get(url, params=params)
                res.raise_for_status()
                markets = res.json().get("markets", [])
                all_markets.extend(markets)
            except Exception as e:
                print(f"Error fetching markets for series {ticker}: {e}")
                continue
                
        return pd.DataFrame(all_markets)
    
    def fetch_data(self, tickers, days=1000):
        """Standard batch fetching logic."""
        end_ts = int(datetime.now(timezone.utc).timestamp() - 60)
        start_ts = end_ts - (days * 86400)
        all_candles = []
        batch_size = int(10000/days)
        
        for i in tqdm(range(0, len(tickers), batch_size), desc="Fetching"):
            batch = tickers[i:i+batch_size]
            params = {"market_tickers": ",".join(batch), "start_ts": start_ts, 
                      "end_ts": end_ts, "period_interval": 1440}
            res = requests.get(f"{self.base_url}/markets/candlesticks", params=params).json()
            for m in res.get("markets", []):
                t = m["market_ticker"]
                for c in m.get("candlesticks", []):
                    # all_candles.append({"ticker": t, "ts": pd.to_datetime(c["start_period_ts"], unit="s", utc=True),
                    #                   **c.get("price", {}), "volume": c.get("volume")})
                    all_candles.append({
                        "ticker": t,
                        "ts": pd.to_datetime(c["end_period_ts"], unit="s", utc=True),
                        "volume": c.get("volume"),
                        **c.get("price", {})
                    })
        self.df = pd.DataFrame(all_candles).set_index(["ts", "ticker"]).sort_index().astype(float)
        return self.df

    def prepare_ml_ready(self, feature_func, target_col='close', lags=4):
        """
        Flexibilité sur les features : passe une fonction qui prend le df 
        et retourne un df de features custom.
        """
        print("Executing custom feature engineering...")
        # 1. Calcul des features de base via la fonction fournie
        features = feature_func(self.df)
        
        # 2. Target calculation (Next day return)
        target = self.df.groupby(level=1)[target_col].pct_change().groupby(level=1).shift(-1)
        
        # 3. Automatic Lagging of features
        lagged_list = [features.groupby(level=1).shift(i).add_suffix(f'_L{i}') for i in range(lags)]
        self.X = pd.concat(lagged_list, axis=1)
        self.y = target.reindex(self.X.index).dropna()
        self.X = self.X.loc[self.y.index]
        return self.X, self.y

    def run_backtest(self, model_obj, min_train=10):
        """
        Flexibilité sur le modèle : accepte n'importe quel objet avec .fit() et .predict()
        (LightGBM, XGBoost, Scikit-Learn, LinearRegression, etc.)
        """
        days = self.X.index.get_level_values(0).unique().sort_values()

        results = []
        iter_metrics = []

        for i in tqdm(range(min_train, len(days)), desc="Backtesting"):
            
            X_train, y_train = self.X.loc[days[:i]], self.y.loc[days[:i]]
            X_test, y_test = self.X.loc[[days[i]]], self.y.loc[[days[i]]]
            test_day = days[i]
            
            if len(X_train) < 50 or len(X_test) == 0: 
                continue
        
            model_obj.fit(X_train, y_train)
            
            y_hat_train = model_obj.predict(X_train)
            y_hat_test = model_obj.predict(X_test)

            r2_train = r2_score(y_train, y_hat_train)
            r2_test = r2_score(y_test, y_hat_test)
            
            acc_train = accuracy_score(np.sign(y_train.values), np.sign(y_hat_train))
            acc_test = accuracy_score(np.sign(y_test.values), np.sign(y_hat_test))

            raw_pnl = y_hat_test * y_test.values
            daily_exposure = np.sum(np.abs(y_hat_test)) + 1e-9
            norm_weight = y_hat_test / daily_exposure
            norm_pnl = norm_weight * y_test.values
            
            iter_metrics.append({
                'ts': test_day,
                'r2_train': r2_train,
                'r2_test': r2_test,
                'acc_train': acc_train,
                'acc_test': acc_test,
                'daily_raw_pnl': np.sum(raw_pnl),
                'daily_norm_pnl': np.sum(norm_pnl)
            })
            
            results.append(pd.DataFrame({
                'ticker': X_test.index.get_level_values(1), # Le ticker est au niveau 1
                'y_true': y_test.values,
                'y_hat': y_hat_test,
                'raw_pnl': raw_pnl,
                'norm_pnl': norm_pnl
            }, index=X_test.index))
        
        final_df = pd.concat(results).drop(columns = "ticker")
        metrics_df = pd.DataFrame(iter_metrics).set_index('ts')

        return final_df, metrics_df

# # --- EXEMPLE D'UTILISATION FLEXIBLE ---

# # 1. Définition des features "à la carte"
# def my_custom_features(df):
#     f = pd.DataFrame(index=df.index)
#     g = df.groupby(level=1)
#     f['ret'] = g['close'].pct_change()
#     f['vol'] = (df['high'] - df['low']) / df['open']
#     f['mom_5'] = g['close'].transform(lambda x: x / x.shift(5) - 1) # Momentum 5 jours
#     return f

# # 2. Choix du modèle (ex: Ridge Regression au lieu de LightGBM)
# from sklearn.linear_model import Ridge
# my_model = Ridge(alpha=1.0)

# # 3. Pipeline
# p = KalshiPredictor()
# p.fetch_data(tickers=['TICKER_A', 'TICKER_B'], days=150)
# p.prepare_ml_ready(feature_func=my_custom_features, lags=3)
# results = p.run_backtest(model_obj=my_model)

# print(results.mean())
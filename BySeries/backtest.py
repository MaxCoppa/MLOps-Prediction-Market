import os
import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.metrics import accuracy_score, r2_score
from tqdm import tqdm
from typing import Tuple
from logger import get_logger

log = get_logger()

def run_backtest(
    X: pd.DataFrame,
    y: pd.Series,
    best_params: dict,
    min_train: int = 60,
    refit_freq: int = 1,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    dates = X.index.get_level_values("ts").unique().sort_values()

    if len(dates) < min_train + 1:
        raise ValueError(f"Only {len(dates)} unique dates — need at least {min_train + 1}.")

    model_opt = lgb.LGBMRegressor(**best_params)
    model_def = lgb.LGBMRegressor()
    model_trained = False
    
    results_opt, results_def = [], []

    log.info(f"Panel walk-forward backtest (Opt + Baseline) — {len(dates)} dates, {X.index.get_level_values('ticker').nunique()} tickers")

    for i in tqdm(range(min_train, len(dates)), desc="Backtesting", ncols=80):
        test_date = dates[i]
        train_dates = dates[:i]

        X_train = X.loc[train_dates]
        y_train = y.loc[train_dates]

        if test_date not in X.index.get_level_values("ts"):
            continue
            
        X_test = X.loc[[test_date]]
        y_test = y.loc[[test_date]]

        if len(X_test) == 0:
            continue

        should_refit = (i - min_train) % refit_freq == 0 or not model_trained
        if should_refit and len(X_train) >= 20:
            model_opt.fit(X_train, y_train)
            model_def.fit(X_train, y_train)
            model_trained = True

        if not model_trained:
            continue

        y_true = y_test.values
        y_hat_opt = model_opt.predict(X_test)
        y_hat_def = model_def.predict(X_test)
        
        # L1 Normalization for capital allocation (sum of absolute weights = 1)
        sig_opt = y_hat_opt / (np.sum(np.abs(y_hat_opt)) + 1e-9)
        sig_def = y_hat_def / (np.sum(np.abs(y_hat_def)) + 1e-9)
        
        pnl_opt, pnl_def = sig_opt * y_true, sig_def * y_true

        tickers_today = X_test.index.get_level_values("ticker")
        for j, ticker in enumerate(tickers_today):
            base_dict = {"date": test_date, "ticker": ticker, "y_true": float(y_true[j])}
            
            res_o = base_dict.copy()
            res_o.update({"y_hat": float(y_hat_opt[j]), "signal": float(sig_opt[j]), "pnl": float(pnl_opt[j])})
            results_opt.append(res_o)
            
            res_d = base_dict.copy()
            res_d.update({"y_hat": float(y_hat_def[j]), "signal": float(sig_def[j]), "pnl": float(pnl_def[j])})
            results_def.append(res_d)

    res_df_opt = pd.DataFrame(results_opt).set_index(["date", "ticker"]).sort_index() if results_opt else pd.DataFrame()
    res_df_def = pd.DataFrame(results_def).set_index(["date", "ticker"]).sort_index() if results_def else pd.DataFrame()

    return res_df_opt, res_df_def


def performance_report(
    results: pd.DataFrame,
    series_ticker: str,
    output_dir: str = "outputs",
    val_split_date=None,
    results_default: pd.DataFrame = None,
) -> dict:
    os.makedirs(output_dir, exist_ok=True)

    def _calc_metrics(df: pd.DataFrame):
        if df is None or df.empty:
            return np.nan, np.nan, np.nan
        # Sum used instead of mean due to L1 capital allocation
        dpnl = df["pnl"].groupby(level="date").sum()
        sharpe = (dpnl.mean() / (dpnl.std() + 1e-9)) * np.sqrt(252)
        wr = (dpnl > 0).mean()
        valid = df[df["y_true"] != 0]
        r2 = float(r2_score(valid["y_true"], valid["y_hat"])) if len(valid) > 1 else np.nan
        return sharpe, r2, wr

    daily_pnl = results["pnl"].groupby(level="date").sum()
    cum_pnl = daily_pnl.cumsum()
    max_dd = (cum_pnl - cum_pnl.cummax()).min()
    total_pnl = float(cum_pnl.iloc[-1]) if not cum_pnl.empty else 0.0

    mask_val = results.index.get_level_values("date") >= val_split_date
    sh_tr, r2_tr, wr_tr = _calc_metrics(results[~mask_val])
    sh_va, r2_va, wr_va = _calc_metrics(results[mask_val])

    ticker_pnl = results["pnl"].groupby(level="ticker").sum().sort_values(ascending=False)

    report = {
        "series_ticker": series_ticker,
        "n_tickers": results.index.get_level_values("ticker").nunique(),
        "total_pnl": round(total_pnl, 4),
        "max_drawdown": round(float(max_dd), 4),
        "optuna_train": {"sharpe": round(sh_tr, 3), "r2": round(r2_tr, 4), "win_rate": round(wr_tr, 3)},
        "optuna_val": {"sharpe": round(sh_va, 3), "r2": round(r2_va, 4), "win_rate": round(wr_va, 3)}
    }

    has_base = results_default is not None and not results_default.empty
    if has_base:
        daily_pnl_d = results_default["pnl"].groupby(level="date").sum()
        cum_pnl_d = daily_pnl_d.cumsum()
        max_dd_d = (cum_pnl_d - cum_pnl_d.cummax()).min()
        total_pnl_d = float(cum_pnl_d.iloc[-1]) if not cum_pnl_d.empty else 0.0

        mask_val_d = results_default.index.get_level_values("date") >= val_split_date
        sh_tr_d, r2_tr_d, wr_tr_d = _calc_metrics(results_default[~mask_val_d])
        sh_va_d, r2_va_d, wr_va_d = _calc_metrics(results_default[mask_val_d])

    fig = plt.figure(figsize=(13, 10))
    ax_pnl = plt.subplot2grid((5, 2), (0, 0), rowspan=3, colspan=2)
    ax_ticker = plt.subplot2grid((5, 2), (3, 0), rowspan=2)
    ax_metrics = plt.subplot2grid((5, 2), (3, 1), rowspan=2)

    ax_pnl.plot(cum_pnl.index, cum_pnl.values, color="#1f77b4", linewidth=1.5, label="Optimized PnL")
    if has_base:
        ax_pnl.plot(cum_pnl_d.index, cum_pnl_d.values, color="gray", linewidth=1.2, linestyle="-.", label="Baseline PnL")
        
    if val_split_date is not None and cum_pnl.index[0] <= val_split_date <= cum_pnl.index[-1]:
        ax_pnl.axvline(val_split_date, color="red", linestyle="--", linewidth=1.5, label="Optuna Val Split")
        
    ax_pnl.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax_pnl.set_title(f"Walk-Forward OOS PnL (L1-weighted, {report['n_tickers']} tickers) — {series_ticker}")
    ax_pnl.set_ylabel("Cumulative Return")
    ax_pnl.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax_pnl.grid(True, alpha=0.3)
    ax_pnl.legend(loc="upper left", fontsize=9)

    top_n = min(20, len(ticker_pnl))
    colors = ["#2ca02c" if v >= 0 else "#d62728" for v in ticker_pnl.iloc[:top_n]]
    ax_ticker.barh(range(top_n), ticker_pnl.iloc[:top_n].values, color=colors)
    ax_ticker.set_yticks(range(top_n))
    ax_ticker.set_yticklabels([str(t)[-12:] for t in ticker_pnl.index[:top_n]], fontsize=7)
    ax_ticker.axvline(0, color="black", linewidth=0.8)
    ax_ticker.set_title("Total PnL per Ticker (Optimized)", fontsize=9)
    ax_ticker.set_xlabel("Cumulative PnL")
    ax_ticker.invert_yaxis()

    ax_metrics.axis("off")
    def _fmt(val, dec=3): return str(round(val, dec)) if not np.isnan(val) else "N/A"
    
    if has_base:
        table_data = [
            ["Metric", "Optimized", "Baseline"],
            ["Total PnL", _fmt(total_pnl, 4), _fmt(total_pnl_d, 4)],
            ["Max DD", _fmt(max_dd, 4), _fmt(max_dd_d, 4)],
            ["Sharpe (Optuna Train)", _fmt(sh_tr), _fmt(sh_tr_d)],
            ["Sharpe (Optuna Val)", _fmt(sh_va), _fmt(sh_va_d)],
            ["R² (Optuna Train)", _fmt(r2_tr, 4), _fmt(r2_tr_d, 4)],
            ["R² (Optuna Val)", _fmt(r2_va, 4), _fmt(r2_va_d, 4)],
            ["Win Rate (Optuna Train)", _fmt(wr_tr), _fmt(wr_tr_d)],
            ["Win Rate (Optuna Val)", _fmt(wr_va), _fmt(wr_va_d)]
        ]
        col_widths = [0.4, 0.3, 0.3]
    else:
        table_data = [
            ["Metric", "Optimized"],
            ["Total PnL", _fmt(total_pnl, 4)],
            ["Max DD", _fmt(max_dd, 4)],
            ["Sharpe (Optuna Train)", _fmt(sh_tr)],
            ["Sharpe (Optuna Val)", _fmt(sh_va)],
            ["R² (Optuna Train)", _fmt(r2_tr, 4)],
            ["R² (Optuna Val)", _fmt(r2_va, 4)],
            ["Win Rate (Optuna Train)", _fmt(wr_tr)],
            ["Win Rate (Optuna Val)", _fmt(wr_va)]
        ]
        col_widths = [0.6, 0.4]

    tbl = ax_metrics.table(cellText=table_data, colWidths=col_widths, cellLoc="left", loc="center", edges="open")
    tbl.set_fontsize(10)
    tbl.scale(1, 1.6)
    for i in range(len(table_data)):
        tbl[(i, 0)].get_text().set_fontweight("bold")
        if i == 0:
            for j in range(len(table_data[0])):
                tbl[(i, j)].get_text().set_fontweight("bold")
            
    ax_metrics.set_title("OOS Walk-Forward Metrics (Split by Optuna Timeline)", fontsize=9)

    plt.tight_layout()
    safe = series_ticker.replace("/", "_")
    plot_path = os.path.join(output_dir, f"{safe}_series_pnl.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"Plot saved → {plot_path}")

    ticker_pnl.to_csv(os.path.join(output_dir, f"{safe}_per_ticker_pnl.csv"), header=["total_pnl"])
    return report
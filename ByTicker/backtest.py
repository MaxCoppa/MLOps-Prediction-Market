import os
import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, r2_score
from tqdm import tqdm
from typing import Tuple
from logger import get_logger

log = get_logger()

def run_backtest(X: pd.DataFrame, y: pd.Series, best_params: dict, min_train: int = 60, refit_freq: int = 1) -> Tuple[pd.DataFrame, pd.DataFrame]:
    days = X.index.tolist()
    results, metrics = [], []
    model = lgb.LGBMRegressor(**best_params)
    model_trained = False
    log.info(f"Running walk-forward backtest (min_train={min_train}, refit_freq={refit_freq}) …")

    for i in tqdm(range(min_train, len(days)), desc="Backtesting", ncols=80):
        X_test = X.iloc[[i]]
        y_test = y.iloc[[i]]

        if (i - min_train) % refit_freq == 0 or not model_trained:
            X_train = X.iloc[:i]
            y_train = y.iloc[:i]
            if len(X_train) >= 20:
                model.fit(X_train, y_train)
                model_trained = True
            else:
                continue

        if not model_trained:
            continue

        y_hat = model.predict(X_test)
        y_true = y_test.values

        signal = np.sign(y_hat)
        pnl = signal * y_true

        results.append({
            "date": days[i],
            "y_true": float(y_true[0]),
            "y_hat": float(y_hat[0]),
            "signal": float(signal[0]),
            "pnl": float(pnl[0]),
        })

        if (i - min_train) % refit_freq == 0:
            X_train = X.iloc[:i]
            y_train = y.iloc[:i]
            yhat_tr = model.predict(X_train)
            
            y_tr_arr = y_train.values.flatten()
            yhat_tr_arr = yhat_tr.flatten()
            
            # Filter zero returns
            mask = y_tr_arr != 0
            if mask.sum() > 1:
                r2_tr = float(r2_score(y_tr_arr[mask], yhat_tr_arr[mask]))
                acc_tr = float(accuracy_score(np.sign(y_tr_arr[mask]), np.sign(yhat_tr_arr[mask])))
            else:
                r2_tr, acc_tr = np.nan, np.nan

            metrics.append({
                "date": days[i],
                "r2_train": r2_tr,
                "acc_train": acc_tr,
            })

    results_df = pd.DataFrame(results).set_index("date")
    metrics_df = pd.DataFrame(metrics).set_index("date") if metrics else pd.DataFrame()
    return results_df, metrics_df

def performance_report(results: pd.DataFrame, metrics: pd.DataFrame, ticker: str, output_dir: str = "outputs", val_split_date=None) -> dict:
    pnl = results["pnl"]
    cum_pnl = pnl.cumsum()
    sharpe = (pnl.mean() / (pnl.std() + 1e-9)) * np.sqrt(252)
    
    roll_max = cum_pnl.cummax()
    max_dd = (cum_pnl - roll_max).min()
    
    # Filter flat days for valid metrics
    valid_mask = results["y_true"] != 0
    res_filt = results[valid_mask]
    
    y_true_sgn = np.sign(res_filt["y_true"])
    y_hat_sgn = np.sign(res_filt["y_hat"])
    
    win_rate = (pnl > 0).mean()

    report = {
        "ticker": ticker,
        "n_days": len(results),
        "total_pnl": round(float(cum_pnl.iloc[-1]), 4),
        "mean_daily_pnl": round(float(pnl.mean()), 6),
        "sharpe_ratio": round(float(sharpe), 3),
        "max_drawdown": round(float(max_dd), 4),
        "win_rate": round(float(win_rate), 3),
    }

    if not metrics.empty:
        report["mean_r2_train"] = round(float(metrics["r2_train"].mean()), 4)
        report["mean_acc_train"] = round(float(metrics["acc_train"].mean()), 3)
        
    if not res_filt.empty and len(res_filt) > 1:
        report["global_r2_test"] = round(float(r2_score(res_filt["y_true"], res_filt["y_hat"])), 4)
        report["global_acc_test"] = round(float(accuracy_score(y_true_sgn, y_hat_sgn)), 3)

    os.makedirs(output_dir, exist_ok=True)
    fig = plt.figure(figsize=(10, 8))
    
    ax_plot = plt.subplot2grid((4, 1), (0, 0), rowspan=3)
    ax_table = plt.subplot2grid((4, 1), (3, 0))

    ax_plot.plot(cum_pnl.index, cum_pnl.values, color="#1f77b4", linewidth=1.5, label="Walk-Forward PnL (OOS)")
    if val_split_date is not None and val_split_date >= cum_pnl.index[0] and val_split_date <= cum_pnl.index[-1]:
        ax_plot.axvline(x=val_split_date, color="red", linestyle="--", linewidth=1.5, label="Optuna Val Split")
    
    ax_plot.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax_plot.set_title(f"Cumulative PnL — {ticker}")
    ax_plot.set_ylabel("Cumulative Returns")
    ax_plot.grid(True, alpha=0.3)
    ax_plot.legend(loc="upper left")

    ax_table.axis("off")
    
    table_data = [
        [" Total PnL:", f"{report['total_pnl']} "],
        [" Sharpe Ratio:", f"{report['sharpe_ratio']} "],
        [" Max Drawdown:", f"{report['max_drawdown']} "],
        [" Win Rate:", f"{report['win_rate']} "],
    ]
    
    if "global_r2_test" in report:
        table_data.extend([
            [" Mean R² Train:", f"{report.get('mean_r2_train', 'N/A')} "],
            [" Global R² Test:", f"{report['global_r2_test']} "],
            [" Mean Acc Train:", f"{report.get('mean_acc_train', 'N/A')} "],
            [" Global Acc Test:", f"{report['global_acc_test']} "]
        ])

    the_table = ax_table.table(cellText=table_data, colWidths=[0.3, 0.2], cellLoc='left', loc='center', edges='open')
    the_table.set_fontsize(11)
    the_table.scale(1, 1.5)
    
    for i in range(len(table_data)):
        the_table[(i, 0)].get_text().set_fontweight('bold')

    plt.tight_layout()
    safe_ticker = ticker.replace("/", "_")
    plot_path = os.path.join(output_dir, f"{safe_ticker}_pnl_curve.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    
    log.info(f"PnL plot and metrics table saved to {plot_path}")
    return report
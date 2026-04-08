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


# ─────────────────────────────────────────────────────────────────────────────
# WALK-FORWARD BACKTEST  (panel version)
# ─────────────────────────────────────────────────────────────────────────────

def run_backtest(
    X: pd.DataFrame,
    y: pd.Series,
    best_params: dict,
    min_train: int = 60,
    refit_freq: int = 1,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Walk-forward backtest on a MultiIndex (date, ticker) panel.

    At each date t:
      - Train on all (date, ticker) rows where date < t
      - Predict on all tickers available at date t
      - Refit every `refit_freq` dates

    The train / test split is strictly temporal (no future leakage):
    every ticker's data before date t is used for training regardless
    of which ticker it belongs to.

    Returns
    -------
    results_df : (date, ticker) indexed DataFrame
        y_true, y_hat, signal, pnl
    metrics_df : date-indexed DataFrame
        r2_train, acc_train, r2_test, acc_test  (computed at each refit)
    """
    dates = X.index.get_level_values("ts").unique().sort_values()

    if len(dates) < min_train + 1:
        raise ValueError(
            f"Only {len(dates)} unique dates — need at least {min_train + 1}. "
            "Reduce min_train or increase window_days."
        )

    model         = lgb.LGBMRegressor(**best_params)
    model_trained = False
    results, metrics = [], []

    log.info(
        f"Panel walk-forward backtest — {len(dates)} dates, "
        f"{X.index.get_level_values('ticker').nunique()} tickers, "
        f"min_train={min_train}, refit_freq={refit_freq}"
    )

    for i in tqdm(range(min_train, len(dates)), desc="Backtesting", ncols=80):
        test_date  = dates[i]
        train_dates = dates[:i]

        # ── all rows up to (not including) test_date ──────────────────────────
        X_train = X.loc[train_dates]
        y_train = y.loc[train_dates]

        # ── all tickers available on test_date ───────────────────────────────
        if test_date not in X.index.get_level_values("ts"):
            continue
        X_test = X.loc[[test_date]]
        y_test = y.loc[[test_date]]

        if len(X_test) == 0:
            continue

        # ── refit logic ───────────────────────────────────────────────────────
        should_refit = (i - min_train) % refit_freq == 0 or not model_trained
        if should_refit and len(X_train) >= 20:
            model.fit(X_train, y_train)
            model_trained = True

        if not model_trained:
            continue

        # ── predict ───────────────────────────────────────────────────────────
        y_hat  = model.predict(X_test)
        y_true = y_test.values
        signal = np.sign(y_hat)
        pnl    = signal * y_true

        tickers_today = X_test.index.get_level_values("ticker")
        for j, ticker in enumerate(tickers_today):
            results.append({
                "date":   test_date,
                "ticker": ticker,
                "y_true": float(y_true[j]),
                "y_hat":  float(y_hat[j]),
                "signal": float(signal[j]),
                "pnl":    float(pnl[j]),
            })

        # ── per-refit metrics ─────────────────────────────────────────────────
        if should_refit and model_trained:
            yhat_tr   = model.predict(X_train)
            y_tr_arr  = y_train.values.flatten()
            yhat_tr_a = yhat_tr.flatten()

            mask_tr   = y_tr_arr != 0
            if mask_tr.sum() > 1:
                r2_tr  = float(r2_score(y_tr_arr[mask_tr], yhat_tr_a[mask_tr]))
                acc_tr = float(accuracy_score(np.sign(y_tr_arr[mask_tr]), np.sign(yhat_tr_a[mask_tr])))
            else:
                r2_tr, acc_tr = np.nan, np.nan

            mask_te = y_true != 0
            if mask_te.sum() > 1:
                r2_te  = float(r2_score(y_true[mask_te], y_hat[mask_te]))
                acc_te = float(accuracy_score(np.sign(y_true[mask_te]), np.sign(y_hat[mask_te])))
            else:
                r2_te, acc_te = np.nan, np.nan

            metrics.append({
                "date":      test_date,
                "r2_train":  r2_tr,
                "acc_train": acc_tr,
                "r2_test":   r2_te,
                "acc_test":  acc_te,
                "n_tickers": int(len(X_test)),
            })

    results_df = (
        pd.DataFrame(results)
        .set_index(["date", "ticker"])
        .sort_index()
    )
    metrics_df = (
        pd.DataFrame(metrics).set_index("date")
        if metrics else pd.DataFrame()
    )
    return results_df, metrics_df


# ─────────────────────────────────────────────────────────────────────────────
# PERFORMANCE REPORT  (panel version)
# ─────────────────────────────────────────────────────────────────────────────

def performance_report(
    results: pd.DataFrame,
    metrics: pd.DataFrame,
    series_ticker: str,
    output_dir: str = "outputs",
    val_split_date=None,
) -> dict:
    """
    Compute aggregate + per-ticker performance metrics, save a 3-panel plot.

    Aggregate PnL = average daily PnL across all tickers (equal-weight).
    """
    os.makedirs(output_dir, exist_ok=True)

    # ── aggregate daily PnL (mean across tickers each day) ───────────────────
    daily_pnl = results["pnl"].groupby(level="date").mean()
    cum_pnl   = daily_pnl.cumsum()
    sharpe    = (daily_pnl.mean() / (daily_pnl.std() + 1e-9)) * np.sqrt(252)
    roll_max  = cum_pnl.cummax()
    max_dd    = (cum_pnl - roll_max).min()
    win_rate  = (daily_pnl > 0).mean()

    # ── per-ticker PnL ────────────────────────────────────────────────────────
    ticker_pnl = (
        results["pnl"]
        .groupby(level="ticker")
        .sum()
        .sort_values(ascending=False)
    )

    # ── global OOS accuracy / R² (filter flat days) ──────────────────────────
    valid = results[results["y_true"] != 0]
    global_r2  = float(r2_score(valid["y_true"], valid["y_hat"])) if len(valid) > 1 else np.nan
    global_acc = float(accuracy_score(np.sign(valid["y_true"]), np.sign(valid["y_hat"]))) if len(valid) > 1 else np.nan

    n_tickers = results.index.get_level_values("ticker").nunique()

    report = {
        "series_ticker":   series_ticker,
        "n_tickers":       n_tickers,
        "n_days":          int(len(daily_pnl)),
        "total_pnl":       round(float(cum_pnl.iloc[-1]), 4),
        "mean_daily_pnl":  round(float(daily_pnl.mean()), 6),
        "sharpe_ratio":    round(float(sharpe), 3),
        "max_drawdown":    round(float(max_dd), 4),
        "win_rate":        round(float(win_rate), 3),
        "global_r2_test":  round(global_r2, 4)  if not np.isnan(global_r2)  else None,
        "global_acc_test": round(global_acc, 3) if not np.isnan(global_acc) else None,
    }

    if not metrics.empty:
        report["mean_r2_train"]  = round(float(metrics["r2_train"].mean()),  4)
        report["mean_acc_train"] = round(float(metrics["acc_train"].mean()), 3)

    # ── print ─────────────────────────────────────────────────────────────────
    border = "═" * 54
    print(f"\n{border}")
    print(f"  SERIES BACKTEST REPORT — {series_ticker}")
    print(border)
    for k, v in report.items():
        if k in ("series_ticker", "best_params"):
            continue
        print(f"  {k.replace('_', ' ').title().ljust(24)} : {v}")
    print(border)

    # ── plot : 3 panels ───────────────────────────────────────────────────────
    fig = plt.figure(figsize=(13, 10))
    ax_pnl     = plt.subplot2grid((5, 2), (0, 0), rowspan=3, colspan=2)
    ax_ticker  = plt.subplot2grid((5, 2), (3, 0), rowspan=2)
    ax_metrics = plt.subplot2grid((5, 2), (3, 1), rowspan=2)

    # — cumulative PnL --------------------------------------------------------
    ax_pnl.plot(cum_pnl.index, cum_pnl.values, color="#1f77b4", linewidth=1.5,
                label="Avg Equal-Weight PnL (OOS)")
    if (val_split_date is not None
            and cum_pnl.index[0] <= val_split_date <= cum_pnl.index[-1]):
        ax_pnl.axvline(val_split_date, color="red", linestyle="--",
                       linewidth=1.5, label="Optuna Val Split")
    ax_pnl.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax_pnl.set_title(f"Cumulative PnL (equal-weight, {n_tickers} tickers) — {series_ticker}")
    ax_pnl.set_ylabel("Cumulative Return")
    ax_pnl.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax_pnl.grid(True, alpha=0.3)
    ax_pnl.legend(loc="upper left", fontsize=9)

    # — per-ticker bar chart --------------------------------------------------
    top_n = min(20, len(ticker_pnl))
    colors = ["#2ca02c" if v >= 0 else "#d62728" for v in ticker_pnl.iloc[:top_n]]
    ax_ticker.barh(range(top_n), ticker_pnl.iloc[:top_n].values, color=colors)
    ax_ticker.set_yticks(range(top_n))
    ax_ticker.set_yticklabels(
        [t[-12:] for t in ticker_pnl.index[:top_n]], fontsize=7
    )
    ax_ticker.axvline(0, color="black", linewidth=0.8)
    ax_ticker.set_title("Total PnL per Ticker", fontsize=9)
    ax_ticker.set_xlabel("Cumulative PnL")
    ax_ticker.invert_yaxis()

    # — summary table ---------------------------------------------------------
    ax_metrics.axis("off")
    table_data = [
        [" Tickers:",        f"{report['n_tickers']} "],
        [" Days:",           f"{report['n_days']} "],
        [" Total PnL:",      f"{report['total_pnl']} "],
        [" Sharpe:",         f"{report['sharpe_ratio']} "],
        [" Max Drawdown:",   f"{report['max_drawdown']} "],
        [" Win Rate:",       f"{report['win_rate']} "],
        [" R² OOS:",         f"{report.get('global_r2_test', 'N/A')} "],
        [" Acc OOS:",        f"{report.get('global_acc_test', 'N/A')} "],
        [" R² Train (avg):", f"{report.get('mean_r2_train', 'N/A')} "],
        [" Acc Train (avg):",f"{report.get('mean_acc_train', 'N/A')} "],
    ]
    tbl = ax_metrics.table(
        cellText=table_data, colWidths=[0.55, 0.35],
        cellLoc="left", loc="center", edges="open"
    )
    tbl.set_fontsize(10)
    tbl.scale(1, 1.6)
    for i in range(len(table_data)):
        tbl[(i, 0)].get_text().set_fontweight("bold")
    ax_metrics.set_title("Summary Metrics", fontsize=9)

    plt.tight_layout()
    safe = series_ticker.replace("/", "_")
    plot_path = os.path.join(output_dir, f"{safe}_series_pnl.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"Plot saved → {plot_path}")

    # ── per-ticker CSV ────────────────────────────────────────────────────────
    ticker_pnl.to_csv(os.path.join(output_dir, f"{safe}_per_ticker_pnl.csv"), header=["total_pnl"])

    return report

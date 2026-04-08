import numpy as np
import pandas as pd
import lightgbm as lgb
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.metrics import accuracy_score, r2_score
from tqdm import tqdm
from typing import Tuple
from ..utils import get_logger

log = get_logger()


def run_backtest(
    X: pd.DataFrame,
    y: pd.Series,
    best_params: dict,
    min_train: int = 60,
    refit_freq: int = 1,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Walk-forward backtest evaluating BOTH optimized and default models.
    Returns: (results_opt, metrics_opt, results_def, metrics_def)
    """
    dates = X.index.get_level_values("ts").unique().sort_values()

    if len(dates) < min_train + 1:
        raise ValueError(
            f"Only {len(dates)} unique dates — need at least {min_train + 1}."
        )

    model_opt = lgb.LGBMRegressor(**best_params)
    model_def = lgb.LGBMRegressor()  # Baseline model
    model_trained = False

    results_opt, metrics_opt = [], []
    results_def, metrics_def = [], []

    log.info(
        f"Panel walk-forward backtest (Opt + Baseline) — {len(dates)} dates, {X.index.get_level_values('ticker').nunique()} tickers"
    )

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

        # Predictions for both models
        y_true = y_test.values
        y_hat_opt = model_opt.predict(X_test)
        y_hat_def = model_def.predict(X_test)

        sig_opt, sig_def = np.sign(y_hat_opt), np.sign(y_hat_def)
        pnl_opt, pnl_def = sig_opt * y_true, sig_def * y_true

        tickers_today = X_test.index.get_level_values("ticker")
        for j, ticker in enumerate(tickers_today):
            base_dict = {
                "date": test_date,
                "ticker": ticker,
                "y_true": float(y_true[j]),
            }

            res_o = base_dict.copy()
            res_o.update(
                {
                    "y_hat": float(y_hat_opt[j]),
                    "signal": float(sig_opt[j]),
                    "pnl": float(pnl_opt[j]),
                }
            )
            results_opt.append(res_o)

            res_d = base_dict.copy()
            res_d.update(
                {
                    "y_hat": float(y_hat_def[j]),
                    "signal": float(sig_def[j]),
                    "pnl": float(pnl_def[j]),
                }
            )
            results_def.append(res_d)

        # Refit metrics for both models
        if should_refit:
            y_tr_arr = y_train.values.flatten()
            mask_tr = y_tr_arr != 0
            mask_te = y_true != 0

            # Sub-function to calculate train/test metrics cleanly
            def calc_metrics(model, y_hat_te):
                yhat_tr_a = model.predict(X_train).flatten()

                r2_tr = (
                    float(r2_score(y_tr_arr[mask_tr], yhat_tr_a[mask_tr]))
                    if mask_tr.sum() > 1
                    else np.nan
                )
                acc_tr = (
                    float(
                        accuracy_score(
                            np.sign(y_tr_arr[mask_tr]), np.sign(yhat_tr_a[mask_tr])
                        )
                    )
                    if mask_tr.sum() > 1
                    else np.nan
                )

                r2_te = (
                    float(r2_score(y_true[mask_te], y_hat_te[mask_te]))
                    if mask_te.sum() > 1
                    else np.nan
                )
                acc_te = (
                    float(
                        accuracy_score(
                            np.sign(y_true[mask_te]), np.sign(y_hat_te[mask_te])
                        )
                    )
                    if mask_te.sum() > 1
                    else np.nan
                )

                return r2_tr, acc_tr, r2_te, acc_te

            r2_tr_o, acc_tr_o, r2_te_o, acc_te_o = calc_metrics(model_opt, y_hat_opt)
            r2_tr_d, acc_tr_d, r2_te_d, acc_te_d = calc_metrics(model_def, y_hat_def)

            metrics_opt.append(
                {
                    "date": test_date,
                    "r2_train": r2_tr_o,
                    "acc_train": acc_tr_o,
                    "r2_test": r2_te_o,
                    "acc_test": acc_te_o,
                }
            )
            metrics_def.append(
                {
                    "date": test_date,
                    "r2_train": r2_tr_d,
                    "acc_train": acc_tr_d,
                    "r2_test": r2_te_d,
                    "acc_test": acc_te_d,
                }
            )

    # Convert to DataFrames
    res_df_opt = (
        pd.DataFrame(results_opt).set_index(["date", "ticker"]).sort_index()
        if results_opt
        else pd.DataFrame()
    )
    met_df_opt = (
        pd.DataFrame(metrics_opt).set_index("date") if metrics_opt else pd.DataFrame()
    )

    res_df_def = (
        pd.DataFrame(results_def).set_index(["date", "ticker"]).sort_index()
        if results_def
        else pd.DataFrame()
    )
    met_df_def = (
        pd.DataFrame(metrics_def).set_index("date") if metrics_def else pd.DataFrame()
    )

    return res_df_opt, met_df_opt, res_df_def, met_df_def


def performance_report(
    results: pd.DataFrame,
    metrics: pd.DataFrame,
    series_ticker: str,
    output_dir: str = "outputs",
    val_split_date=None,
    results_default: pd.DataFrame = None,
    metrics_default: pd.DataFrame = None,
) -> dict:
    os.makedirs(output_dir, exist_ok=True)

    daily_pnl = results["pnl"].groupby(level="date").mean()
    cum_pnl = daily_pnl.cumsum()
    sharpe = (daily_pnl.mean() / (daily_pnl.std() + 1e-9)) * np.sqrt(252)
    max_dd = (cum_pnl - cum_pnl.cummax()).min()
    win_rate = (daily_pnl > 0).mean()

    ticker_pnl = (
        results["pnl"].groupby(level="ticker").sum().sort_values(ascending=False)
    )

    valid = results[results["y_true"] != 0]
    global_r2 = (
        float(r2_score(valid["y_true"], valid["y_hat"])) if len(valid) > 1 else np.nan
    )
    global_acc = (
        float(accuracy_score(np.sign(valid["y_true"]), np.sign(valid["y_hat"])))
        if len(valid) > 1
        else np.nan
    )

    report = {
        "series_ticker": series_ticker,
        "n_tickers": results.index.get_level_values("ticker").nunique(),
        "n_days": int(len(daily_pnl)),
        "total_pnl": round(float(cum_pnl.iloc[-1]), 4),
        "mean_daily_pnl": round(float(daily_pnl.mean()), 6),
        "sharpe_ratio": round(float(sharpe), 3),
        "max_drawdown": round(float(max_dd), 4),
        "win_rate": round(float(win_rate), 3),
        "global_r2_test": round(global_r2, 4) if not np.isnan(global_r2) else None,
        "global_acc_test": round(global_acc, 3) if not np.isnan(global_acc) else None,
    }

    if not metrics.empty:
        report["mean_r2_train"] = round(float(metrics["r2_train"].mean()), 4)
        report["mean_acc_train"] = round(float(metrics["acc_train"].mean()), 3)

    has_base = results_default is not None and not results_default.empty
    if has_base:
        daily_pnl_d = results_default["pnl"].groupby(level="date").mean()
        cum_pnl_d = daily_pnl_d.cumsum()
        sharpe_d = (daily_pnl_d.mean() / (daily_pnl_d.std() + 1e-9)) * np.sqrt(252)
        max_dd_d = (cum_pnl_d - cum_pnl_d.cummax()).min()
        win_rate_d = (daily_pnl_d > 0).mean()

        valid_d = results_default[results_default["y_true"] != 0]
        r2_d = (
            float(r2_score(valid_d["y_true"], valid_d["y_hat"]))
            if len(valid_d) > 1
            else np.nan
        )
        acc_d = (
            float(accuracy_score(np.sign(valid_d["y_true"]), np.sign(valid_d["y_hat"])))
            if len(valid_d) > 1
            else np.nan
        )

    fig = plt.figure(figsize=(13, 10))
    ax_pnl = plt.subplot2grid((5, 2), (0, 0), rowspan=3, colspan=2)
    ax_ticker = plt.subplot2grid((5, 2), (3, 0), rowspan=2)
    ax_metrics = plt.subplot2grid((5, 2), (3, 1), rowspan=2)

    ax_pnl.plot(
        cum_pnl.index,
        cum_pnl.values,
        color="#1f77b4",
        linewidth=1.5,
        label="Optimized PnL (OOS)",
    )
    if has_base:
        ax_pnl.plot(
            cum_pnl_d.index,
            cum_pnl_d.values,
            color="gray",
            linewidth=1.2,
            linestyle="-.",
            label="Baseline PnL",
        )

    if (
        val_split_date is not None
        and cum_pnl.index[0] <= val_split_date <= cum_pnl.index[-1]
    ):
        ax_pnl.axvline(
            val_split_date,
            color="red",
            linestyle="--",
            linewidth=1.5,
            label="Optuna Val Split",
        )

    ax_pnl.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax_pnl.set_title(
        f"Cumulative PnL (equal-weight, {report['n_tickers']} tickers) — {series_ticker}"
    )
    ax_pnl.set_ylabel("Cumulative Return")
    ax_pnl.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax_pnl.grid(True, alpha=0.3)
    ax_pnl.legend(loc="upper left", fontsize=9)

    top_n = min(20, len(ticker_pnl))
    colors = ["#2ca02c" if v >= 0 else "#d62728" for v in ticker_pnl.iloc[:top_n]]
    ax_ticker.barh(range(top_n), ticker_pnl.iloc[:top_n].values, color=colors)
    ax_ticker.set_yticks(range(top_n))
    ax_ticker.set_yticklabels(
        [str(t)[-12:] for t in ticker_pnl.index[:top_n]], fontsize=7
    )
    ax_ticker.axvline(0, color="black", linewidth=0.8)
    ax_ticker.set_title("Total PnL per Ticker (Optimized)", fontsize=9)
    ax_ticker.set_xlabel("Cumulative PnL")
    ax_ticker.invert_yaxis()

    ax_metrics.axis("off")
    if has_base:
        table_data = [
            ["Metric", "Optimized", "Baseline"],
            [
                "Total PnL",
                str(report["total_pnl"]),
                str(round(float(cum_pnl_d.iloc[-1]), 4)),
            ],
            ["Sharpe", str(report["sharpe_ratio"]), str(round(float(sharpe_d), 3))],
            ["Max DD", str(report["max_drawdown"]), str(round(float(max_dd_d), 4))],
            ["Win Rate", str(report["win_rate"]), str(round(float(win_rate_d), 3))],
            [
                "R² OOS",
                str(report.get("global_r2_test", "N/A")),
                str(round(r2_d, 4)) if not np.isnan(r2_d) else "N/A",
            ],
            [
                "Acc OOS",
                str(report.get("global_acc_test", "N/A")),
                str(round(acc_d, 3)) if not np.isnan(acc_d) else "N/A",
            ],
        ]

        if metrics_default is not None and not metrics_default.empty:
            r2_tr_d = round(float(metrics_default["r2_train"].mean()), 4)
            acc_tr_d = round(float(metrics_default["acc_train"].mean()), 3)
            table_data.extend(
                [
                    ["R² Train", str(report.get("mean_r2_train", "N/A")), str(r2_tr_d)],
                    [
                        "Acc Train",
                        str(report.get("mean_acc_train", "N/A")),
                        str(acc_tr_d),
                    ],
                ]
            )

        col_widths = [0.35, 0.3, 0.3]
    else:
        table_data = [
            [" Tickers:", f"{report['n_tickers']} "],
            [" Total PnL:", f"{report['total_pnl']} "],
            [" Sharpe:", f"{report['sharpe_ratio']} "],
            [" Max Drawdown:", f"{report['max_drawdown']} "],
            [" Win Rate:", f"{report['win_rate']} "],
            [" R² OOS:", f"{report.get('global_r2_test', 'N/A')} "],
            [" Acc OOS:", f"{report.get('global_acc_test', 'N/A')} "],
            [" R² Train:", f"{report.get('mean_r2_train', 'N/A')} "],
            [" Acc Train:", f"{report.get('mean_acc_train', 'N/A')} "],
        ]
        col_widths = [0.55, 0.35]

    tbl = ax_metrics.table(
        cellText=table_data,
        colWidths=col_widths,
        cellLoc="left",
        loc="center",
        edges="open",
    )
    tbl.set_fontsize(10)
    tbl.scale(1, 1.6)
    for i in range(len(table_data)):
        tbl[(i, 0)].get_text().set_fontweight("bold")
        if has_base and i == 0:
            tbl[(i, 1)].get_text().set_fontweight("bold")
            tbl[(i, 2)].get_text().set_fontweight("bold")

    ax_metrics.set_title("Summary Metrics", fontsize=9)

    plt.tight_layout()
    safe = series_ticker.replace("/", "_")
    plot_path = os.path.join(output_dir, f"{safe}_series_pnl.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"Plot saved → {plot_path}")

    ticker_pnl.to_csv(
        os.path.join(output_dir, f"{safe}_per_ticker_pnl.csv"), header=["total_pnl"]
    )
    return report

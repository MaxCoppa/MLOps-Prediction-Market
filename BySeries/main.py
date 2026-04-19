import argparse
import json
from pathlib import Path
import lightgbm as lgb
import os

from logger import get_logger
from data import fetch_series_tickers, fetch_series_data, build_panel_features
from model import bayesian_optimisation
from backtest import run_backtest, performance_report
from pnl_computation import compute_yesterday_pnl

log = get_logger()

def run_series_pipeline(
    series_ticker: str, window_days: int = 365, n_lags: int = 10,
    min_obs: int = 30, n_optuna_trials: int = 20, val_ratio: float = 0.2,
    n_splits: int = 3, min_train: int = 60, refit_freq: int = 1, 
    output_dir: str = "outputs",
) -> dict:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    log.info(f"{'='*60}\n  Kalshi Series Pipeline — series: {series_ticker}\n{'='*60}")

    tickers = fetch_series_tickers(series_ticker)
    if not tickers:
        raise ValueError(f"No tickers found for series '{series_ticker}'.")

    df = fetch_series_data(series_ticker, tickers, window_days=window_days)
    X, y = build_panel_features(df, n_lags=n_lags, min_obs=min_obs)

    dates = X.index.get_level_values("ts").unique().sort_values()

    best_params, best_val_mse = bayesian_optimisation(
        X, y, n_trials=n_optuna_trials, val_ratio=val_ratio, n_splits=n_splits
    )

    res_opt, res_def = run_backtest(
        X, y, best_params=best_params, min_train=min_train, refit_freq=refit_freq
    )

    split_idx = int(len(dates) * (1 - val_ratio))
    val_split_date = dates[split_idx]

    report = performance_report(
        results=res_opt,
        series_ticker=series_ticker,
        output_dir=output_dir,
        val_split_date=val_split_date,
        results_default=res_def
    )
    
    report["best_val_mse"] = round(best_val_mse, 6)
    report["best_params"] = {
        k: v for k, v in best_params.items() 
        if k not in ("objective", "metric", "verbosity", "n_jobs", "random_state")
    }

    safe = series_ticker.replace("/", "_")

    res_opt.to_csv(f"{output_dir}/{safe}_backtest_results_opt.csv")
    res_def.to_csv(f"{output_dir}/{safe}_backtest_results_def.csv")
    
    with open(f"{output_dir}/{safe}_report.json", "w") as fh:
        json.dump(report, fh, indent=2)

    final_model = lgb.LGBMRegressor(**best_params)
    final_model.fit(X, y)
    final_model.booster_.save_model(f"{output_dir}/{safe}_model.txt")

    log.info(f"All artefacts saved to '{output_dir}/'")
    return report

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kalshi Series Prediction Pipeline")
    parser.add_argument("series_ticker", type=str, help="Series ticker (e.g. KXBTCD)")
    parser.add_argument("--eval-yesterday", action="store_true", help="Calculate yesterday's PnL")
    parser.add_argument("--days", type=int, default=365*5, help="History window in days")
    parser.add_argument("--lags", type=int, default=10, help="Number of lag features")
    parser.add_argument("--min-obs", type=int, default=0, help="Min observations per ticker")
    parser.add_argument("--trials", type=int, default=20, help="Optuna trials")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation ratio for Optuna holdout")
    parser.add_argument("--splits", type=int, default=5, help="Walk-forward CV folds for Optuna")
    parser.add_argument("--min-train", type=int, default=0, help="Min training dates for backtest")
    parser.add_argument("--refit-freq", type=int, default=1, help="Walk-forward refit frequency")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Output directory")
    args = parser.parse_args()

    if args.eval_yesterday:
        safe_ticker = args.series_ticker.replace("/", "_")
        model_file = os.path.join(args.output_dir, f"{safe_ticker}_model.txt")
        
        results_df = compute_yesterday_pnl(
            series_ticker=args.series_ticker,
            model_path=model_file,
            n_lags=args.lags
        )
        print(results_df.head())

    else:
        run_series_pipeline(
            series_ticker=args.series_ticker,
            window_days=args.days,
            n_lags=args.lags,
            min_obs=args.min_obs,
            n_optuna_trials=args.trials,
            val_ratio=args.val_ratio,
            n_splits=args.splits,
            min_train=args.min_train,
            refit_freq=args.refit_freq,
            output_dir=args.output_dir,
        )
import argparse
import json
from pathlib import Path
import mlflow
import lightgbm as lgb

from kalshi_predictor.utils import get_logger, setup_mlflow
from kalshi_predictor.series.data import (
    fetch_tickers,
    fetch_data,
    build_features,
)

from kalshi_predictor.series.model import bayesian_optimisation
from kalshi_predictor.series.backtest import run_backtest, performance_report

log = get_logger()


def run_series_pipeline(
    series_ticker: str,
    window_days: int = 365,
    n_lags: int = 10,
    n_optuna_trials: int = 50,
    val_ratio: float = 0.2,
    min_train: int = 60,
    refit_freq: int = 1,
    output_dir: str = "outputs",
) -> dict:
    
    """
    Main pipeline to run the full series prediction workflow: 
    data fetching, feature engineering, hyperparameter optimization,
    backtesting, and performance reporting.
    Results are logged to MLflow and saved as CSV/JSON artefacts.
    """
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    log.info(f"{'='*60}\n  Kalshi Series Pipeline — {series_ticker}\n{'='*60}")

    with mlflow.start_run(run_name=series_ticker):
        mlflow.log_params(
            {
                "series_ticker": series_ticker,
                "window_days": window_days,
                "n_lags": n_lags,
                "n_optuna_trials": n_optuna_trials,
                "val_ratio": val_ratio,
                "min_train": min_train,
                "refit_freq": refit_freq,
            }
        )

        tickers = fetch_tickers(series_ticker)
        if not tickers:
            raise ValueError(f"No tickers found for {series_ticker}")

        mlflow.log_param("n_tickers", len(tickers))

        df_raw = fetch_data(series_ticker, tickers, window_days=window_days)
        raw_dataset = mlflow.data.from_pandas(
            df_raw.reset_index(),
            name=series_ticker,
        )
        mlflow.log_input(raw_dataset, context="training")
        mlflow.log_params(
            {
                "dataset_start": str(df_raw.index.get_level_values("ts").min().date()),
                "dataset_end": str(df_raw.index.get_level_values("ts").max().date()),
            }
        )

        X, y = build_features(df_raw, n_lags=n_lags, min_obs=min_train)

        dates = X.index.get_level_values("ts").unique().sort_values()
        if len(dates) < min_train + 10:
            raise ValueError(
                f"Not enough dates after feature engineering ({len(dates)}). Increase window_days."
            )

        best_params, best_val_mae = bayesian_optimisation(
            X, y, n_trials=n_optuna_trials, val_ratio=val_ratio
        )

        split_idx = int(len(dates) * (1 - val_ratio))
        val_split_date = dates[split_idx]

        res_opt, met_opt, res_def, met_def = run_backtest(
            X, y, best_params=best_params, min_train=min_train, refit_freq=refit_freq
        )

        report = performance_report(
            results=res_opt,
            metrics=met_opt,
            series_ticker=series_ticker,
            output_dir=output_dir,
            val_split_date=val_split_date,
            results_default=res_def,
            metrics_default=met_def,
        )

        report["best_val_mae"] = round(best_val_mae, 6)
        report["best_params"] = {
            k: v
            for k, v in best_params.items()
            if k not in ("objective", "metric", "verbosity")
        }

        safe_series = series_ticker.replace("/", "_")

        csv_opt_path = f"{output_dir}/{safe_series}_backtest_results_opt.csv"
        csv_def_path = f"{output_dir}/{safe_series}_backtest_results_def.csv"
        res_opt.to_csv(csv_opt_path)
        res_def.to_csv(csv_def_path)

        json_path = f"{output_dir}/{safe_series}_report.json"
        with open(json_path, "w") as fh:
            json.dump(report, fh, indent=2)

        mlflow.log_metric("best_val_mse", best_val_mae)
        mlflow.log_params({k: v for k, v in report.get("best_params", {}).items()})
        mlflow.log_artifact(csv_opt_path)
        mlflow.log_artifact(csv_def_path)
        mlflow.log_artifact(json_path)

        log.info(f"Artefacts saved to '{output_dir}/'")

        final_model = lgb.LGBMRegressor(**best_params)
        final_model.fit(X, y)
        mlflow.lightgbm.log_model(final_model, "model")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kalshi Series Prediction Pipeline")
    parser.add_argument(
        "series_ticker", type=str, help="Series ticker (e.g. KXNASDAQ100U)"
    )
    parser.add_argument("--days", type=int, default=365, help="History window in days")
    parser.add_argument("--lags", type=int, default=10, help="Number of lag features")
    parser.add_argument("--trials", type=int, default=50, help="Optuna trials")
    parser.add_argument(
        "--val-ratio", type=float, default=0.2, help="Validation ratio for Optuna"
    )
    parser.add_argument(
        "--min-train", type=int, default=0, help="Min training samples for backtest"
    )
    parser.add_argument(
        "--refit-freq", type=int, default=1, help="Walk-forward refit frequency"
    )
    parser.add_argument(
        "--output-dir", type=str, default="outputs", help="Output directory"
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default="kalshi-series",
        help="MLFlow experiment name",
    )

    args = parser.parse_args()

    setup_mlflow(args.experiment_name)

    run_series_pipeline(
        series_ticker=args.series_ticker,
        window_days=args.days,
        n_lags=args.lags,
        n_optuna_trials=args.trials,
        val_ratio=args.val_ratio,
        min_train=args.min_train,
        refit_freq=args.refit_freq,
        output_dir=args.output_dir,
    )

import argparse
import json
from pathlib import Path
import joblib
import lightgbm as lgb
import numpy as np

from kalshi_predictor.utils import get_logger
from kalshi_predictor.ticker.data import fetch_data, build_features
from kalshi_predictor.ticker.model import bayesian_optimisation
from kalshi_predictor.ticker.backtest import run_backtest, performance_report

log = get_logger()


def run_pipeline(
    ticker: str,
    window_days: int = 365,
    n_lags: int = 10,
    n_optuna_trials: int = 20,
    val_ratio: float = 0.2,
    min_train: int = 60,
    refit_freq: int = 1,
    output_dir: str = "outputs",
) -> dict:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    log.info(f"{'='*60}\n  Kalshi Pipeline — ticker: {ticker}\n{'='*60}")

    df = fetch_data(ticker, window_days=window_days)
    X, y = build_features(df, n_lags=n_lags)

    if len(X) < min_train + 10:
        raise ValueError(
            f"Not enough data after feature engineering ({len(X)} rows). Increase window_days."
        )

    best_params, best_val_mae = bayesian_optimisation(
        X, y, n_trials=n_optuna_trials, val_ratio=val_ratio
    )
    results, metrics = run_backtest(
        X, y, best_params=best_params, min_train=min_train, refit_freq=refit_freq
    )

    split_idx = int(len(X) * (1 - val_ratio))
    val_split_date = X.index[split_idx]

    report = performance_report(
        results, metrics, ticker, output_dir=output_dir, val_split_date=val_split_date
    )
    report["best_val_mae"] = round(best_val_mae, 6)
    report["best_params"] = {
        k: v
        for k, v in best_params.items()
        if k not in ("objective", "metric", "verbosity", "n_jobs", "random_state")
    }

    safe_ticker = ticker.replace("/", "_")
    results.to_csv(f"{output_dir}/{safe_ticker}_backtest_results.csv")

    with open(f"{output_dir}/{safe_ticker}_report.json", "w") as fh:
        json.dump(report, fh, indent=2)

    final_model = lgb.LGBMRegressor(**best_params)
    final_model.fit(X, y)
    model_path = f"{output_dir}/{safe_ticker}_model.joblib"
    joblib.dump(final_model, model_path)

    saved_model = joblib.load(model_path)
    if not ((np.abs(saved_model.predict(X) - final_model.predict(X)) < 1e-6).all()):
        raise ValueError("The model has been corrupted")
    log.info(f"Artefacts saved to '{output_dir}/'")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kalshi Prediction Market Pipeline")
    parser.add_argument(
        "ticker", type=str, help="Contract ticker (e.g. KXBTCD-25MAR14-B94999)"
    )
    parser.add_argument("--days", type=int, default=365, help="History window in days")
    parser.add_argument("--lags", type=int, default=10, help="Number of lag features")
    parser.add_argument("--trials", type=int, default=50, help="Optuna trials")
    parser.add_argument(
        "--val-ratio", type=float, default=0.2, help="Validation ratio for Optuna"
    )
    parser.add_argument(
        "--min-train", type=int, default=60, help="Min training samples for backtest"
    )
    parser.add_argument(
        "--refit-freq", type=int, default=1, help="Walk-forward refit frequency"
    )
    parser.add_argument(
        "--output-dir", type=str, default="outputs", help="Output directory"
    )
    args = parser.parse_args()

    run_pipeline(
        ticker=args.ticker,
        window_days=args.days,
        n_lags=args.lags,
        n_optuna_trials=args.trials,
        val_ratio=args.val_ratio,
        min_train=args.min_train,
        refit_freq=args.refit_freq,
        output_dir=args.output_dir,
    )

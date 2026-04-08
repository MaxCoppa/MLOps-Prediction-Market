import argparse
import json
from pathlib import Path
import lightgbm as lgb

from logger import get_logger
from data_series import fetch_series_tickers, fetch_series_data, build_panel_features
from model_series import bayesian_optimisation
from backtest_series import run_backtest, performance_report

log = get_logger()


def run_series_pipeline(
    series_ticker: str,
    window_days: int   = 365,
    n_lags: int        = 10,
    min_obs: int       = 30,
    n_optuna_trials: int = 20,
    val_ratio: float   = 0.2,
    min_train: int     = 60,
    refit_freq: int    = 1,
    output_dir: str    = "outputs",
) -> dict:
    """
    Full series-level pipeline.

    Parameters
    ----------
    series_ticker    : Kalshi series ticker  (e.g. "KXBTCD")
    window_days      : History window in days
    n_lags           : Number of lag features
    min_obs          : Minimum observations per ticker to keep it
    n_optuna_trials  : Bayesian optimisation trials
    val_ratio        : Fraction kept for Optuna validation
    min_train        : Minimum dates before first backtest prediction
    refit_freq       : Walk-forward refit frequency (in dates)
    output_dir       : Directory for output artefacts

    Returns
    -------
    dict  performance report
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    log.info(f"{'='*60}\n  Kalshi Series Pipeline — series: {series_ticker}\n{'='*60}")

    # ── 1. discover tickers ───────────────────────────────────────────────────
    tickers = fetch_series_tickers(series_ticker)
    if not tickers:
        raise ValueError(f"No tickers found for series '{series_ticker}'.")

    # ── 2. fetch raw panel data ───────────────────────────────────────────────
    df = fetch_series_data(series_ticker, tickers, window_days=window_days)

    # ── 3. feature engineering ────────────────────────────────────────────────
    X, y = build_panel_features(df, n_lags=n_lags, min_obs=min_obs)

    if len(X.index.get_level_values("ts").unique()) < min_train + 10:
        raise ValueError(
            f"Not enough dates after feature engineering. "
            "Increase window_days or reduce min_train."
        )

    # ── 4. bayesian optimisation (model trained on full panel) ────────────────
    best_params, best_val_mae = bayesian_optimisation(
        X, y, n_trials=n_optuna_trials, val_ratio=val_ratio
    )

    # ── 5. walk-forward backtest ──────────────────────────────────────────────
    results, metrics = run_backtest(
        X, y,
        best_params=best_params,
        min_train=min_train,
        refit_freq=refit_freq,
    )

    # ── 6. performance report + plots ─────────────────────────────────────────
    split_idx      = int(len(X.index.get_level_values("ts").unique()) * (1 - val_ratio))
    val_split_date = X.index.get_level_values("ts").unique().sort_values()[split_idx]

    report = performance_report(
        results, metrics,
        series_ticker=series_ticker,
        output_dir=output_dir,
        val_split_date=val_split_date,
    )
    report["best_val_mae"] = round(best_val_mae, 6)
    report["best_params"]  = {
        k: v for k, v in best_params.items()
        if k not in ("objective", "metric", "verbosity", "n_jobs", "random_state")
    }

    # ── 7. save artefacts ─────────────────────────────────────────────────────
    safe = series_ticker.replace("/", "_")

    results.to_csv(f"{output_dir}/{safe}_backtest_results.csv")
    if not metrics.empty:
        metrics.to_csv(f"{output_dir}/{safe}_metrics.csv")

    with open(f"{output_dir}/{safe}_report.json", "w") as fh:
        json.dump(report, fh, indent=2)

    final_model = lgb.LGBMRegressor(**best_params)
    final_model.fit(X, y)
    final_model.booster_.save_model(f"{output_dir}/{safe}_model.txt")

    log.info(f"All artefacts saved to '{output_dir}/'")
    return report


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kalshi Series Prediction Pipeline")
    parser.add_argument("series_ticker",       type=str,   help="Series ticker  (e.g. KXBTCD)")
    parser.add_argument("--days",              type=int,   default=365,        help="History window in days")
    parser.add_argument("--lags",              type=int,   default=10,         help="Number of lag features")
    parser.add_argument("--min-obs",           type=int,   default=0,         help="Min observations per ticker")
    parser.add_argument("--trials",            type=int,   default=50,         help="Optuna trials")
    parser.add_argument("--val-ratio",         type=float, default=0.2,        help="Validation ratio for Optuna")
    parser.add_argument("--min-train",         type=int,   default=0,         help="Min training dates for backtest")
    parser.add_argument("--refit-freq",        type=int,   default=1,          help="Walk-forward refit frequency")
    parser.add_argument("--output-dir",        type=str,   default="outputs",  help="Output directory")
    args = parser.parse_args()

    run_series_pipeline(
        series_ticker    = args.series_ticker,
        window_days      = args.days,
        n_lags           = args.lags,
        min_obs          = args.min_obs,
        n_optuna_trials  = args.trials,
        val_ratio        = args.val_ratio,
        min_train        = args.min_train,
        refit_freq       = args.refit_freq,
        output_dir       = args.output_dir,
    )

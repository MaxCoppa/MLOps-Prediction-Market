import numpy as np
import pandas as pd
import optuna
import lightgbm as lgb
from typing import Tuple
import json
from sklearn.model_selection import TimeSeriesSplit
from logger import get_logger

log = get_logger()

def bayesian_optimisation(
    X: pd.DataFrame, 
    y: pd.Series, 
    n_trials: int = 50, 
    val_ratio: float = 0.2,
    n_splits: int = 5
) -> Tuple[dict, float]:
    
    dates = X.index.get_level_values("ts").unique().sort_values()
    split_idx = int(len(dates) * (1 - val_ratio))
    optuna_dates = dates[:split_idx]
    
    log.info(f"Bayesian optimisation — Train dates: {len(optuna_dates)} (Holdout: {len(dates)-split_idx}), CV folds: {n_splits}")

    tscv = TimeSeriesSplit(n_splits=n_splits)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "objective": "regression",
            "metric": "mse",
            "verbosity": -1,
            "boosting_type": trial.suggest_categorical("boosting_type", ["gbdt", "dart"]),
            "num_leaves": trial.suggest_int("num_leaves", 16, 256),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 50, 500),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "subsample": trial.suggest_float("subsample", 0.4, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "n_jobs": -1,
            "random_state": 42,
        }

        fold_mses = []
        
        for train_idx, val_idx in tscv.split(optuna_dates):
            train_dates = optuna_dates[train_idx]
            val_dates = optuna_dates[val_idx]

            X_tr, y_tr = X.loc[train_dates], y.loc[train_dates]
            X_val, y_val = X.loc[val_dates], y.loc[val_dates]

            model = lgb.LGBMRegressor(**params)
            
            callbacks = [lgb.early_stopping(stopping_rounds=20, verbose=False)] if params["boosting_type"] != "dart" else []
            
            model.fit(
                X_tr, y_tr, 
                eval_set=[(X_val, y_val)],
                callbacks=callbacks
            )
            
            preds = model.predict(X_val)
            fold_mses.append(np.mean((preds - y_val.values)**2))

        return float(np.mean(fold_mses))

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best_params = study.best_params
    best_params.update({"objective": "regression", "metric": "mse", "verbosity": -1, "n_jobs": -1, "random_state": 42})
    
    log.info(f"Best Avg MSE (CV): {study.best_value:.6f}")
    clean_params = {k: v for k, v in best_params.items() if k not in ('objective','metric','verbosity','n_jobs','random_state')}
    log.info(f"Best params: {json.dumps(clean_params, indent=2)}")
    
    return best_params, study.best_value
import numpy as np
import pandas as pd
import optuna
import lightgbm as lgb
from typing import Tuple
import json
from logger import get_logger

log = get_logger()

def bayesian_optimisation(X: pd.DataFrame, y: pd.Series, n_trials: int = 50, val_ratio: float = 0.2) -> Tuple[dict, float]:
    split = int(len(X) * (1 - val_ratio))
    X_tr, y_tr = X.iloc[:split], y.iloc[:split]
    X_val, y_val = X.iloc[split:], y.iloc[split:]

    log.info(f"Bayesian optimisation — train={len(X_tr)}, val={len(X_val)}, n_trials={n_trials}")

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
        model = lgb.LGBMRegressor(**params)
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)])
        preds = model.predict(X_val)
        mse = np.mean((preds - y_val.values)**2)
        return mse

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best_params = study.best_params
    best_params.update({"objective": "regression", "metric": "mse", "verbosity": -1, "n_jobs": -1, "random_state": 42})
    
    log.info(f"Best MSE (val): {study.best_value:.6f}")
    clean_params = {k: v for k, v in best_params.items() if k not in ('objective','metric','verbosity','n_jobs','random_state')}
    log.info(f"Best params: {json.dumps(clean_params, indent=2)}")
    
    return best_params, study.best_value
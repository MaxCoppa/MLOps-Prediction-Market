import logging
import os
import warnings
import optuna
import mlflow


warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logging.getLogger("mlflow").setLevel(logging.WARNING)


def get_logger(name="KalshiPipeline"):
    return logging.getLogger(name)


def setup_mlflow(experiment_name: str = "kalshi-predictor") -> None:

    """
    Configure MLflow tracking URI and experiment name 
    for logging hyperparameter optimization results.
    """

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    mlflow.set_experiment(experiment_name)

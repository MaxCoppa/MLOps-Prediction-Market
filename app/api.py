from fastapi import FastAPI
import pandas as pd
import mlflow
import logging
from kalshi_predictor.series import compute_yesterday_pnl

logging.basicConfig(
    format="{asctime} - {levelname} - {message}",
    style="{",
    datefmt="%Y-%m-%d %H:%M",
    level=logging.DEBUG,
    handlers=[logging.FileHandler("api.log"), logging.StreamHandler()],
)

# Preload model -------------------
logging.info("Getting model from MLFlow")

# Load the model from the Model Registry
model_name = "model_KXCPI"

model_uri = f"models:/{model_name}@production"
model_uri = f"models:/{model_name}/2"
model = mlflow.lightgbm.load_model(model_uri)

# Default input from CSV (first row)
DEFAULT_X = pd.read_csv("app/X_example.csv").iloc[-1].to_dict()

app = FastAPI(
    title="Kalshi Prediction API",
    description="API for model Kalshi serie prediction",
)


@app.get("/", tags=["Welcome"])
def show_welcome_page():
    return {
        "Message": "Kalshi prediction API",
        "Model_name": "KXGDP ML",
        "Model_version": "0.3",
    }


@app.post("/predict", tags=["Predict"])
async def predict(params: dict) -> dict:
    """
    Predict
    """

    X = pd.DataFrame([params])

    if "ts" in X.columns:
        X = X.drop(columns=["ts"])

    X = X.apply(pd.to_numeric, errors="coerce")

    prediction = float(model.predict(X)[0])
    return {"prediction": prediction}


@app.get("/predict/example", tags=["Predict"])
async def predict_example():
    X = pd.DataFrame([DEFAULT_X])

    if "ts" in X.columns:
        X = X.drop(columns=["ts"])

    X = X.apply(pd.to_numeric, errors="coerce")

    prediction = float(model.predict(X)[0])

    return {
        "prediction": prediction,
        "note": "This uses a default example from X_example.csv",
    }


@app.get("/predict/pnl", tags=["Predict"])
async def predict_pnl():

    prediction = compute_yesterday_pnl("KXCPI")

    return {
        "prediction": prediction,
        "note": "Hello!",
    }

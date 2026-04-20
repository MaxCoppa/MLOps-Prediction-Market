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
model = mlflow.lightgbm.load_model(model_uri)

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


@app.get("/predict", tags=["Predict"])
async def predict(
    RET_1: float = None,
    VOL_YES_1: float = None,
    VOL_NO_1: float = None,
    VOL_TOTAL_1: float = None,
    RET_2: float = None,
    VOL_YES_2: float = None,
    VOL_NO_2: float = None,
    VOL_TOTAL_2: float = None,
    RET_3: float = None,
    VOL_YES_3: float = None,
    VOL_NO_3: float = None,
    VOL_TOTAL_3: float = None,
    RET_4: float = None,
    VOL_YES_4: float = None,
    VOL_NO_4: float = None,
    VOL_TOTAL_4: float = None,
    RET_5: float = None,
    VOL_YES_5: float = None,
    VOL_NO_5: float = None,
    VOL_TOTAL_5: float = None,
    RET_6: float = None,
    VOL_YES_6: float = None,
    VOL_NO_6: float = None,
    VOL_TOTAL_6: float = None,
    RET_7: float = None,
    VOL_YES_7: float = None,
    VOL_NO_7: float = None,
    VOL_TOTAL_7: float = None,
    RET_8: float = None,
    VOL_YES_8: float = None,
    VOL_NO_8: float = None,
    VOL_TOTAL_8: float = None,
    RET_9: float = None,
    VOL_YES_9: float = None,
    VOL_NO_9: float = None,
    VOL_TOTAL_9: float = None,
    RET_10: float = None,
    VOL_YES_10: float = None,
    VOL_NO_10: float = None,
    VOL_TOTAL_10: float = None,
    RET_MEAN_5: float = None,
    RET_STD_5: float = None,
    VOL_MEAN_5: float = None,
    RET_MEAN_10: float = None,
    RET_STD_10: float = None,
    VOL_MEAN_10: float = None,
    DIST_50: float = None,
    DIST_MIN_20: float = None,
    DIST_MAX_20: float = None,
    VOL_IMBALANCE: float = None,
) -> dict:
    params = {
        "RET_1": RET_1,
        "VOL_YES_1": VOL_YES_1,
        "VOL_NO_1": VOL_NO_1,
        "VOL_TOTAL_1": VOL_TOTAL_1,
        "RET_2": RET_2,
        "VOL_YES_2": VOL_YES_2,
        "VOL_NO_2": VOL_NO_2,
        "VOL_TOTAL_2": VOL_TOTAL_2,
        "RET_3": RET_3,
        "VOL_YES_3": VOL_YES_3,
        "VOL_NO_3": VOL_NO_3,
        "VOL_TOTAL_3": VOL_TOTAL_3,
        "RET_4": RET_4,
        "VOL_YES_4": VOL_YES_4,
        "VOL_NO_4": VOL_NO_4,
        "VOL_TOTAL_4": VOL_TOTAL_4,
        "RET_5": RET_5,
        "VOL_YES_5": VOL_YES_5,
        "VOL_NO_5": VOL_NO_5,
        "VOL_TOTAL_5": VOL_TOTAL_5,
        "RET_6": RET_6,
        "VOL_YES_6": VOL_YES_6,
        "VOL_NO_6": VOL_NO_6,
        "VOL_TOTAL_6": VOL_TOTAL_6,
        "RET_7": RET_7,
        "VOL_YES_7": VOL_YES_7,
        "VOL_NO_7": VOL_NO_7,
        "VOL_TOTAL_7": VOL_TOTAL_7,
        "RET_8": RET_8,
        "VOL_YES_8": VOL_YES_8,
        "VOL_NO_8": VOL_NO_8,
        "VOL_TOTAL_8": VOL_TOTAL_8,
        "RET_9": RET_9,
        "VOL_YES_9": VOL_YES_9,
        "VOL_NO_9": VOL_NO_9,
        "VOL_TOTAL_9": VOL_TOTAL_9,
        "RET_10": RET_10,
        "VOL_YES_10": VOL_YES_10,
        "VOL_NO_10": VOL_NO_10,
        "VOL_TOTAL_10": VOL_TOTAL_10,
        "RET_MEAN_5": RET_MEAN_5,
        "RET_STD_5": RET_STD_5,
        "VOL_MEAN_5": VOL_MEAN_5,
        "RET_MEAN_10": RET_MEAN_10,
        "RET_STD_10": RET_STD_10,
        "VOL_MEAN_10": VOL_MEAN_10,
        "DIST_50": DIST_50,
        "DIST_MIN_20": DIST_MIN_20,
        "DIST_MAX_20": DIST_MAX_20,
        "VOL_IMBALANCE": VOL_IMBALANCE,
    }

    X = pd.DataFrame([params])
    X = X.apply(pd.to_numeric, errors="coerce")

    prediction = float(model.predict(X)[0])
    return {"prediction": prediction}


@app.get("/predict/pnl", tags=["Predict"])
async def predict_pnl():

    prediction = compute_yesterday_pnl("KXCPI", model)

    return {
        "prediction": prediction,
        "note": "Hello!",
    }

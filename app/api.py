from fastapi import FastAPI
import joblib
import pandas as pd

model = joblib.load("outputs/KXGDP_model.joblib")

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
        "Model_version": "0.1",
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

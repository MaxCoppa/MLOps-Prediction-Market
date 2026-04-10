import requests
import pandas as pd

url = "http://127.0.0.1:8000/predict"
X_example = pd.read_csv("app/X_example.csv")
params = X_example.iloc[-1].to_dict()

params = {k: (None if pd.isna(v) else v) for k, v in params.items()}

response = requests.post(url, json=params)
print("POST /predict")
print(response.json())


response = requests.get("http://127.0.0.1:8000/predict/example")
print("\nGET /predict/example")
print(response.json())

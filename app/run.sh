#/bin/bash

uv run predict_series.py
uv run uvicorn app.api:app --host "0.0.0.0"
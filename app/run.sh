#/bin/bash

uv run predict_series.py KXGDP
uv run uvicorn app.api:app --host "0.0.0.0"
FROM python:3.13-slim

WORKDIR /app

# Copy project
COPY pyproject.toml uv.lock* README.md ./
COPY src ./src
COPY predict_series.py predict_ticker.py ./

# Install uv + project
RUN pip install --no-cache-dir uv \
    && uv sync

ENV PATH="/app/.venv/bin:$PATH"

# Default command
CMD ["python", "predict_ticker.py", "--help"]

FROM ubuntu:22.04

# Install Python and system tools
RUN apt-get update && \
    apt-get install -y python3 python3-pip curl && \
    rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin/:$PATH"

# Set working directory
WORKDIR /app

# Copy dependency files first
COPY pyproject.toml ./
COPY uv.lock* ./

# Copy source code
COPY src ./src
COPY predict_series.py ./
COPY predict_ticker.py ./
COPY README.md ./
COPY app ./app

# Install project dependencies
RUN uv sync

# Default command
CMD ["bash", "-c", "./app/run.sh"]
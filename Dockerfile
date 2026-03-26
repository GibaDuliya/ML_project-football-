# CPU training image (smaller; PyTorch CPU wheels).
# Build:  docker build -t ml-football-train .
# Run:    docker run --rm -v "$(pwd)/dataset:/app/dataset:ro" -v "$(pwd)/outputs:/app/outputs" ml-football-train \
#            --data /app/dataset/your_data.csv --output /app/outputs/mpp_run --epochs 10

FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

# Default: show CLI help (override with docker run ... --data ... --output ...)
CMD ["python", "run/run_mpp.py", "--help"]

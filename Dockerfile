# Finstreet Trading System
# Multi-stage build for optimized image size

FROM python:3.11-slim as builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

FROM python:3.11-slim

WORKDIR /app

COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

COPY . .

RUN mkdir -p data/raw data/processed models reports/figures

ENV PYTHONUNBUFFERED=1
ENV LOG_LEVEL=INFO

CMD ["python", "run.py", "all"]

LABEL maintainer="finstreet"
LABEL description="Finstreet Trading System"
LABEL version="2.0"

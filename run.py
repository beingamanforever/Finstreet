#!/usr/bin/env python
"""
Finstreet Trading System

Usage:
    python run.py fetch     - Fetch data from FYERS (Nov-Dec 2025)
    python run.py train     - Train XGBoost model
    python run.py backtest  - Run backtest simulation
    python run.py all       - Full pipeline
"""

import sys
import os
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

START_DATE = "2025-11-01"
END_DATE = "2025-12-31"


def fetch_data() -> bool:
    from src.data.fetch_data import fetch_and_save
    logger.info(f"Fetching data: {START_DATE} to {END_DATE}")
    df = fetch_and_save(START_DATE, END_DATE)
    if df is not None:
        logger.info(f"Data fetched: {len(df)} rows")
        return True
    return False


def train_model() -> bool:
    from src.model.train import train_model as _train
    logger.info("Training model")
    model, features = _train()
    if model is not None:
        logger.info(f"Model trained with {len(features)} features")
        return True
    return False


def run_backtest() -> bool:
    from src.backtest.backtest import Backtester
    os.makedirs("data/processed", exist_ok=True)

    logger.info("Running backtest")
    bt = Backtester("data/raw/NSE_SONATSOFTW-EQ.csv")
    trades, equity = bt.run()
    metrics = bt.calculate_metrics(trades, equity)

    print("\n" + "=" * 50)
    print("BACKTEST RESULTS")
    print("=" * 50)
    for k, v in metrics.items():
        print(f"  {k:20s}: {v}")
    print("=" * 50)

    trades.to_csv("data/processed/trades.csv", index=False)
    equity.to_csv("data/processed/equity.csv", index=False)
    return True


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    cmd = sys.argv[1].lower()
    commands = {
        "fetch": fetch_data,
        "train": train_model,
        "backtest": run_backtest,
        "all": lambda: fetch_data() and train_model() and run_backtest()
    }

    if cmd not in commands:
        print(f"Unknown command: {cmd}")
        print(__doc__)
        sys.exit(1)

    success = commands[cmd]()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""
Finstreet Trading System

Usage:
    python run.py fetch      - Fetch data from FYERS
    python run.py train      - Train XGBoost model
    python run.py ensemble   - Train ensemble model (XGBoost + LightGBM)
    python run.py backtest   - Run backtest simulation
    python run.py predict    - Generate Jan 1-8, 2026 predictions
    python run.py visualize  - Generate performance visualizations
    python run.py all        - Full pipeline
"""

import sys
import os
import logging
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config.settings import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def fetch_data() -> bool:
    from src.data.fetch_data import fetch_and_save
    logger.info("Fetching data")
    df = fetch_and_save()
    if df is not None:
        logger.info(f"Data fetched: {len(df)} rows")
        return True
    return False


def train_model() -> bool:
    from src.model.train import train_model as _train
    logger.info("Training XGBoost model")
    model, features = _train()
    if model is not None:
        logger.info(f"Model trained with {len(features)} features")
        return True
    return False


def train_ensemble() -> bool:
    """Train ensemble model with walk-forward validation."""
    try:
        from src.model.ensemble import train_ensemble_model
        logger.info("Training ensemble model (XGBoost + LightGBM)")
        model, features = train_ensemble_model()
        if model is not None:
            logger.info(f"Ensemble trained with {len(features)} features")
            return True
    except ImportError as e:
        logger.warning(f"Ensemble training requires additional dependencies: {e}")
        logger.info("Falling back to standard XGBoost training")
        return train_model()
    return False


def run_backtest() -> bool:
    from src.backtest.backtest import Backtester, _export_trade_log
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("reports", exist_ok=True)

    logger.info("Running backtest")
    data_path = str(settings.data.data_path)
    bt = Backtester(data_path)
    trades, equity = bt.run()
    metrics = bt.calculate_metrics(trades, equity)

    print("\n" + "=" * 60)
    print("BACKTEST RESULTS")
    print("=" * 60)
    for k, v in metrics.items():
        print(f"  {k:20s}: {v}")
    print("=" * 60)

    trades.to_csv("data/processed/trades.csv", index=False)
    equity.to_csv("data/processed/equity.csv", index=False)
    _export_trade_log(trades)
    return True


def generate_predictions() -> bool:
    from src.forecast.daily_predictions import DailyPredictionGenerator
    os.makedirs("reports", exist_ok=True)
    
    logger.info("Generating daily predictions for Jan 1-8, 2026")
    generator = DailyPredictionGenerator()
    predictions = generator.generate_all_predictions()
    
    if not predictions:
        logger.error("Failed to generate predictions")
        return False
    
    generator.export_csv(predictions)
    
    print("\n" + "=" * 80)
    print("DAILY PREDICTIONS (Jan 1-8, 2026)")
    print("=" * 80)
    print(generator.format_table(predictions))
    print("=" * 80)
    
    return True


def generate_visualizations() -> bool:
    from src.visualization.performance import PerformanceVisualizer
    os.makedirs("reports/figures", exist_ok=True)
    
    logger.info("Generating visualizations")
    
    equity_path = "data/processed/equity.csv"
    trades_path = "data/processed/trades.csv"
    price_path = str(settings.data.data_path)
    
    if not os.path.exists(equity_path):
        logger.error("Run backtest first to generate equity data")
        return False
    
    equity_df = pd.read_csv(equity_path)
    equity_df["date"] = pd.to_datetime(equity_df["date"], format="mixed", errors="coerce")
    equity_df = equity_df.dropna(subset=["date"])
    equity_df = equity_df.set_index("date")
    equity = equity_df["equity"]
    
    trades_df = pd.read_csv(trades_path) if os.path.exists(trades_path) else pd.DataFrame()
    price_df = pd.read_csv(price_path) if os.path.exists(price_path) else None
    
    if price_df is not None:
        from src.features.indicators import add_indicators
        price_df = add_indicators(price_df)
    
    viz = PerformanceVisualizer()
    viz.generate_report(equity, trades_df, price_df=price_df)
    
    logger.info("Visualizations saved to reports/figures/")
    return True


def run_full_pipeline() -> bool:
    """Run complete pipeline: fetch -> train -> backtest -> predict -> visualize."""
    # Check if data exists, skip fetch if so
    data_path = str(settings.data.data_path)
    if os.path.exists(data_path):
        logger.info(f"Data already exists at {data_path}, skipping fetch")
        steps = [
            ("Training ensemble model", train_ensemble),
            ("Running backtest", run_backtest),
            ("Generating predictions", generate_predictions),
            ("Creating visualizations", generate_visualizations)
        ]
    else:
        steps = [
            ("Fetching data", fetch_data),
            ("Training ensemble model", train_ensemble),
            ("Running backtest", run_backtest),
            ("Generating predictions", generate_predictions),
            ("Creating visualizations", generate_visualizations)
        ]
    
    for step_name, step_func in steps:
        logger.info(f"Step: {step_name}")
        if not step_func():
            logger.error(f"Failed at: {step_name}")
            return False
    
    logger.info("Full pipeline completed successfully")
    return True


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    cmd = sys.argv[1].lower()
    commands = {
        "fetch": fetch_data,
        "train": train_model,
        "ensemble": train_ensemble,
        "backtest": run_backtest,
        "predict": generate_predictions,
        "visualize": generate_visualizations,
        "all": run_full_pipeline
    }

    if cmd not in commands:
        print(f"Unknown command: {cmd}")
        print(__doc__)
        sys.exit(1)

    success = commands[cmd]()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
 
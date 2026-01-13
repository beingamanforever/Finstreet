import os
import logging
from datetime import datetime

import pandas as pd

from src.data.fyers_client import FyersClient
from src.strategy.strategy import Strategy

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

SYMBOL = "NSE:SONATSOFTW-EQ"
DATA_PATH = "data/raw/NSE_SONATSOFTW-EQ.csv"


def execute():
    client = FyersClient()
    strategy = Strategy(capital=100000)

    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - pd.Timedelta(days=60)).strftime("%Y-%m-%d")

    df = client.fetch_historical_data(SYMBOL, start_date, end_date)

    if df is None or df.empty:
        if os.path.exists(DATA_PATH):
            df = pd.read_csv(DATA_PATH)
        else:
            logger.error("No data available")
            return

    signal = strategy.generate_signal(df)

    if signal is None:
        logger.info("No signal")
        return

    logger.info(f"Signal: {signal['signal']}")
    logger.info(f"Entry: {signal['entry_price']:.2f}")
    logger.info(f"Qty: {signal['quantity']}")
    logger.info(f"SL: {signal['stop_loss']:.2f}")
    logger.info(f"TP: {signal['take_profit']:.2f}")
    logger.info(f"Confidence: {signal['confidence']:.3f}")


if __name__ == "__main__":
    execute()

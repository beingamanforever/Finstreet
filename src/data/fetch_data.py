import os
import sys
import logging
from typing import Optional

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.data.fyers_client import FyersClient

logger = logging.getLogger(__name__)

SYMBOL = "NSE:SONATSOFTW-EQ"
START_DATE = "2025-11-01"
END_DATE = "2025-12-31"
OUTPUT_FILE = "data/raw/NSE_SONATSOFTW-EQ.csv"


def fetch_and_save(start_date: str = START_DATE, end_date: str = END_DATE) -> Optional[pd.DataFrame]:
    os.makedirs("data/raw", exist_ok=True)
    client = FyersClient()

    if not client.is_connected:
        logger.error("FYERS client not connected")
        return None

    df = client.fetch_historical_data(SYMBOL, start_date, end_date)
    if df is None or df.empty:
        logger.error("Failed to fetch data")
        return None

    df['date'] = pd.to_datetime(df['date'])
    df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
    df = df[['date', 'open', 'high', 'low', 'close', 'volume']]
    df = df.sort_values('date').reset_index(drop=True)

    df.to_csv(OUTPUT_FILE, index=False)
    logger.info(f"Saved {len(df)} rows to {OUTPUT_FILE}")
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    fetch_and_save()

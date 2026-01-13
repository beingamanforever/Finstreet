import os
import logging
from typing import Optional

import pandas as pd
from fyers_apiv3 import fyersModel
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


class FyersClient:
    def __init__(self):
        self._client_id = os.getenv("FYERS_CLIENT_ID")
        self._access_token = os.getenv("FYERS_ACCESS_TOKEN")
        self._fyers: Optional[fyersModel.FyersModel] = None
        self._connected = False
        self._initialize()

    @property
    def is_connected(self) -> bool:
        return self._connected and self._fyers is not None

    def _initialize(self) -> None:
        if not self._client_id or not self._access_token:
            logger.warning("FYERS credentials not configured")
            return

        try:
            self._fyers = fyersModel.FyersModel(
                client_id=self._client_id,
                token=self._access_token,
                is_async=False,
                log_path=""
            )

            profile = self._fyers.get_profile()
            if profile.get("code") == 200:
                self._connected = True
            else:
                self._fyers = None
        except Exception as e:
            logger.error(f"FYERS init failed: {e}")
            self._fyers = None

    def fetch_historical_data(self, symbol: str, start_date: str, end_date: str, resolution: str = "D") -> Optional[pd.DataFrame]:
        if not self._fyers:
            return None

        request = {
            "symbol": symbol,
            "resolution": resolution,
            "date_format": "1",
            "range_from": start_date,
            "range_to": end_date,
            "cont_flag": "1"
        }

        try:
            response = self._fyers.history(data=request)
            if response.get("code") != 200:
                return None

            candles = response.get("candles", [])
            if not candles:
                return None

            df = pd.DataFrame(candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
            df["date"] = pd.to_datetime(df["timestamp"], unit="s").dt.date
            return df[["date", "open", "high", "low", "close", "volume"]]
        except Exception as e:
            logger.error(f"Request failed: {e}")
            return None

    def get_quote(self, symbol: str) -> Optional[dict]:
        if not self._fyers:
            return None
        try:
            response = self._fyers.quotes(data={"symbols": symbol})
            if response.get("code") == 200:
                return response.get("d", [{}])[0]
            return None
        except Exception:
            return None

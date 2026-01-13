import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import MACD, SMAIndicator
from ta.volatility import BollingerBands, AverageTrueRange


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    close, high, low = df["close"], df["high"], df["low"]

    df["RSI_14"] = RSIIndicator(close=close, window=14).rsi()
    df["SMA_10"] = SMAIndicator(close=close, window=10).sma_indicator()
    df["SMA_20"] = SMAIndicator(close=close, window=20).sma_indicator()

    bb = BollingerBands(close=close, window=20, window_dev=2)
    df["BBU_20_2.0"] = bb.bollinger_hband()
    df["BBL_20_2.0"] = bb.bollinger_lband()
    df["BBM_20_2.0"] = bb.bollinger_mavg()
    df["BBP_20_2.0"] = bb.bollinger_pband()

    df["ATR_14"] = AverageTrueRange(high=high, low=low, close=close, window=14).average_true_range()

    macd = MACD(close=close)
    df["MACD_12_26_9"] = macd.macd()
    df["MACD_SIGNAL"] = macd.macd_signal()
    df["MACD_DIFF"] = macd.macd_diff()

    return df

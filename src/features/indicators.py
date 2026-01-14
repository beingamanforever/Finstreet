import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator
from ta.trend import MACD, SMAIndicator, ADXIndicator, EMAIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator


def compute_adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.DataFrame:
    """
    Compute ADX (Average Directional Index) and directional indicators.
    ADX quantifies trend strength: <20 weak/sideways, 20-25 developing, >25 strong trend.
    """
    adx_indicator = ADXIndicator(high=high, low=low, close=close, window=period)
    return pd.DataFrame({
        "ADX": adx_indicator.adx(),
        "ADX_POS": adx_indicator.adx_pos(),  # +DI
        "ADX_NEG": adx_indicator.adx_neg(),  # -DI
    })


def compute_zscore(series: pd.Series, window: int = 20) -> pd.Series:
    """
    Compute rolling Z-score: measures how stretched price is from its mean.
    Values > 2 indicate overbought, < -2 indicate oversold.
    """
    rolling_mean = series.rolling(window=window).mean()
    rolling_std = series.rolling(window=window).std()
    return (series - rolling_mean) / (rolling_std + 1e-10)


def compute_relative_volume(volume: pd.Series, window: int = 20) -> pd.Series:
    """
    Compute relative volume: today's volume / average volume.
    Values > 1.5 indicate significant institutional activity.
    """
    avg_volume = volume.rolling(window=window).mean()
    return volume / (avg_volume + 1e-10)


def compute_kaufman_efficiency_ratio(close: pd.Series, window: int = 10) -> pd.Series:
    """Kaufman Efficiency Ratio: |Price Change| / Sum(|Daily Changes|). Range 0-1."""
    price_change = abs(close - close.shift(window))
    volatility = abs(close.diff()).rolling(window=window).sum()
    ker = price_change / (volatility + 1e-10)
    return ker.clip(0, 1)


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add technical indicators for ML features."""
    df = df.copy()
    close, high, low, volume = df["close"], df["high"], df["low"], df["volume"]

    df["RSI_14"] = RSIIndicator(close=close, window=14).rsi()
    
    df["EMA_10"] = EMAIndicator(close=close, window=10).ema_indicator()
    df["SMA_10"] = SMAIndicator(close=close, window=10).sma_indicator()
    df["SMA_20"] = SMAIndicator(close=close, window=20).sma_indicator()
    df["SMA_50"] = SMAIndicator(close=close, window=50).sma_indicator()
    df["SMA_100"] = SMAIndicator(close=close, window=100).sma_indicator()
    
    df["EMA_SMA_spread"] = (df["EMA_10"] - df["SMA_20"]) / df["SMA_20"] * 100
    df["price_to_sma20"] = (close - df["SMA_20"]) / df["SMA_20"] * 100
    df["price_to_sma50"] = (close - df["SMA_50"]) / df["SMA_50"] * 100

    bb = BollingerBands(close=close, window=20, window_dev=2)
    df["BBU_20_2.0"] = bb.bollinger_hband()
    df["BBL_20_2.0"] = bb.bollinger_lband()
    df["BBM_20_2.0"] = bb.bollinger_mavg()
    df["BBP_20_2.0"] = bb.bollinger_pband()
    df["BB_width"] = (df["BBU_20_2.0"] - df["BBL_20_2.0"]) / df["BBM_20_2.0"]

    df["ATR_14"] = AverageTrueRange(high=high, low=low, close=close, window=14).average_true_range()
    df["ATR_pct"] = df["ATR_14"] / close * 100
    df["ATR_percentile"] = df["ATR_14"].expanding(min_periods=20).apply(
        lambda x: (x.iloc[-1] <= x).sum() / len(x) if len(x) > 0 else 0.5
    )

    # === MACD ===
    macd = MACD(close=close)
    df["MACD_12_26_9"] = macd.macd()
    df["MACD_SIGNAL"] = macd.macd_signal()
    df["MACD_DIFF"] = macd.macd_diff()
    df["MACD_normalized"] = df["MACD_12_26_9"] / close * 100

    # === ADX - Trend Strength ===
    adx_df = compute_adx(high, low, close, period=14)
    df["ADX"] = adx_df["ADX"]
    df["ADX_POS"] = adx_df["ADX_POS"]
    df["ADX_NEG"] = adx_df["ADX_NEG"]
    df["DI_spread"] = df["ADX_POS"] - df["ADX_NEG"]  # Positive = bullish dominance
    
    # ADX regime classification
    df["trend_strength"] = pd.cut(
        df["ADX"], 
        bins=[0, 20, 25, 40, 100], 
        labels=[0, 1, 2, 3]
    ).astype(float)

    df["close_zscore"] = compute_zscore(close, window=20)
    df["volume_zscore"] = compute_zscore(volume, window=20)
    df["atr_zscore"] = compute_zscore(df["ATR_14"], window=20)

    df["relative_volume"] = compute_relative_volume(volume, window=20)
    df["volume_breakout"] = (df["relative_volume"] > 1.5).astype(int)

    df["OBV"] = OnBalanceVolumeIndicator(close=close, volume=volume).on_balance_volume()
    df["OBV_slope"] = df["OBV"].diff(5) / df["OBV"].shift(5) * 100

    df["daily_range"] = (high - low) / close * 100
    df["body_size"] = abs(close - df["open"]) / (high - low + 1e-10)
    df["upper_wick"] = (high - pd.concat([close, df["open"]], axis=1).max(axis=1)) / (high - low + 1e-10)
    df["lower_wick"] = (pd.concat([close, df["open"]], axis=1).min(axis=1) - low) / (high - low + 1e-10)

    df["weekly_trend"] = np.where(
        (close > df["SMA_100"]) & (df["SMA_20"] > df["SMA_50"]), 1,
        np.where((close < df["SMA_100"]) & (df["SMA_20"] < df["SMA_50"]), -1, 0)
    )

    df["KER_10"] = compute_kaufman_efficiency_ratio(close, window=10)
    df["KER_20"] = compute_kaufman_efficiency_ratio(close, window=20)
    df["trend_efficient"] = (df["KER_10"] > 0.5).astype(int)

    df["market_regime"] = np.select(
        [
            (df["ADX"] < 20) | (df["KER_10"] < 0.3),
            (df["ADX"] >= 20) & (df["ADX"] < 25) & (df["KER_10"] >= 0.3),
            (df["ADX"] >= 25) & (df["ATR_percentile"] < 0.7),
            (df["ADX"] >= 25) & (df["ATR_percentile"] >= 0.7),
        ],
        [0, 1, 2, 3],
        default=1
    )

    return df

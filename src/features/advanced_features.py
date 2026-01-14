import pandas as pd
import numpy as np


def add_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["typical_price"] = (df["high"] + df["low"] + df["close"]) / 3
    df["vwap"] = (df["typical_price"] * df["volume"]).rolling(20).sum() / df["volume"].rolling(20).sum()
    df["vwap_deviation"] = (df["close"] - df["vwap"]) / df["vwap"]
    df["volume_ratio"] = df["volume"] / df["volume"].rolling(20).mean()

    df["pvt"] = ((df["close"] - df["close"].shift(1)) / df["close"].shift(1) * df["volume"]).cumsum()
    df["pvt_normalized"] = (df["pvt"] - df["pvt"].rolling(20).mean()) / df["pvt"].rolling(20).std()
    return df


def add_statistical_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    returns = df["close"].pct_change()

    df["returns_skew_20"] = returns.rolling(20).skew()
    df["returns_kurt_20"] = returns.rolling(20).kurt()
    df["returns_std_10"] = returns.rolling(10).std()
    df["returns_std_20"] = returns.rolling(20).std()

    mean_ret = returns.rolling(20).mean()
    std_ret = returns.rolling(20).std()
    df["cv_20"] = std_ret / (mean_ret.abs() + 1e-8)
    df["returns_zscore"] = (returns - mean_ret) / (std_ret + 1e-8)
    return df


def add_volatility_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    returns = df["close"].pct_change()

    df["realized_vol_20"] = returns.rolling(20).std() * np.sqrt(252)
    df["vol_percentile"] = df["realized_vol_20"].expanding(min_periods=20).apply(
        lambda x: (x.iloc[-1] <= x).sum() / len(x) if len(x) > 0 else 0.5
    )

    df["hl_range"] = (df["high"] - df["low"]) / df["close"]
    df["hl_range_ma"] = df["hl_range"].rolling(20).mean()

    df["parkinson_vol"] = np.sqrt(
        1 / (4 * np.log(2)) * ((np.log(df["high"] / df["low"])) ** 2)
    ).rolling(20).mean() * np.sqrt(252)

    df["vol_regime"] = (df["vol_percentile"] > 0.7).astype(int)
    return df


def add_momentum_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["roc_5"] = df["close"].pct_change(5)
    df["roc_10"] = df["close"].pct_change(10)
    df["momentum_10"] = df["close"] - df["close"].shift(10)

    returns = df["close"].pct_change()
    up_days = returns.clip(lower=0).rolling(14).sum()
    down_days = (-returns.clip(upper=0)).rolling(14).sum()
    df["rel_strength"] = up_days / (down_days + 1e-8)
    return df


def add_all_features(df: pd.DataFrame) -> pd.DataFrame:
    df = add_volume_features(df)
    df = add_statistical_features(df)
    df = add_volatility_features(df)
    df = add_momentum_features(df)
    return df

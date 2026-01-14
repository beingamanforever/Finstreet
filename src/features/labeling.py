"""
Triple Barrier Labeling with ATR-Based Significance Filtering.

Implements:
1. Rolling volatility estimation for dynamic thresholds
2. Triple barrier labeling (upper/lower/timeout)
3. ATR-based move significance filtering to reduce noise
4. Sample weight calculation for overlapping labels
"""

import logging
from typing import Tuple

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Compute Average True Range for volatility measurement.
    
    ATR = EMA of True Range over specified period.
    True Range = max(high-low, abs(high-prev_close), abs(low-prev_close))
    """
    high = df["high"]
    low = df["low"]
    close = df["close"]
    prev_close = close.shift(1)
    
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = true_range.ewm(span=period, adjust=False).mean()
    
    atr = atr.fillna(high - low)
    return atr


def get_rolling_volatility(prices: pd.Series, span: int = 60) -> pd.Series:
    """Rolling volatility using only past data. Expanding window for early periods."""
    returns = prices.pct_change()
    rolling_vol = returns.rolling(window=span).std()
    expanding_vol = returns.expanding(min_periods=2).std()
    rolling_vol = rolling_vol.fillna(expanding_vol)
    if rolling_vol.iloc[0] != rolling_vol.iloc[0]:
        rolling_vol.iloc[0] = 0.02
    return rolling_vol


def triple_barrier_labeling(
    df: pd.DataFrame,
    min_hold: int = 2,
    max_hold: int = 5,
    vol_mult: float = 2.0,
    use_atr_filter: bool = True,
    atr_significance: float = 0.3
) -> pd.DataFrame:
    """
    Triple-barrier labeling with ATR-based significance filtering.
    
    The triple barrier method sets:
    - Upper barrier: profit target based on volatility
    - Lower barrier: stop loss based on volatility
    - Vertical barrier: maximum holding period (timeout)
    
    With ATR filtering enabled, labels are only assigned to moves that
    exceed a significance threshold, reducing noise in the training data.
    
    Args:
        df: DataFrame with OHLCV data
        min_hold: Minimum holding period before checking barriers
        max_hold: Maximum holding period (vertical barrier)
        vol_mult: Volatility multiplier for barrier distance
        use_atr_filter: Whether to apply ATR significance filtering
        atr_significance: ATR multiplier for significance threshold
    """
    closes = df["close"].values
    highs = df["high"].values
    lows = df["low"].values
    n = len(closes)
    
    volatility = get_rolling_volatility(df["close"], span=20)
    threshold = vol_mult * volatility
    
    atr = compute_atr(df) if use_atr_filter else None
    significance_threshold = atr_significance * atr if atr is not None else None

    labels, t1_indices, returns, touched, raw_labels = [], [], [], [], []

    for i in range(n):
        if i + min_hold >= n:
            labels.append(0)
            t1_indices.append(np.nan)
            returns.append(0)
            touched.append("none")
            raw_labels.append(0)
            continue

        start = closes[i]
        upper = start * (1 + threshold.iloc[i])
        lower = start * (1 - threshold.iloc[i])

        label, t1, barrier = 0, None, "timeout"

        for j in range(i + min_hold, min(i + max_hold + 1, n)):
            if highs[j] >= upper:
                label, t1, barrier = 1, j, "upper"
                break
            elif lows[j] <= lower:
                label, t1, barrier = -1, j, "lower"
                break

        if t1 is None:
            final_idx = min(i + max_hold, n - 1)
            label = 1 if closes[final_idx] > start else -1
            t1 = final_idx

        raw_labels.append(label)
        
        # Apply ATR significance filter
        if use_atr_filter and significance_threshold is not None:
            price_move = abs(closes[t1] - start) if t1 is not None else 0
            sig_thresh = significance_threshold.iloc[i]
            if price_move < sig_thresh:
                label = 0  # Not significant enough, mark as HOLD

        labels.append(label)
        t1_indices.append(t1)
        ret = (closes[t1] / start - 1) if t1 is not None else 0
        returns.append(ret)
        touched.append(barrier)

    result = df.copy()
    result["label"] = (np.array(labels) > 0).astype(int)
    result["label_3class"] = np.array(labels)
    result["raw_label"] = np.array(raw_labels)
    result["t1"] = t1_indices
    result["future_return"] = returns
    result["barrier_touched"] = touched
    
    if use_atr_filter:
        result["atr"] = atr
        result["significance_threshold"] = significance_threshold

    return result


def calculate_sample_weights(df: pd.DataFrame) -> np.ndarray:
    """Calculate sample weights based on label concurrency."""
    n = len(df)
    t1_array = df["t1"].values

    concurrency = np.zeros(n)
    for i in range(n):
        if pd.notna(t1_array[i]):
            end_idx = min(int(t1_array[i]), n - 1)
            concurrency[i:end_idx + 1] += 1

    concurrency[concurrency == 0] = 1

    weights = np.zeros(n)
    for i in range(n):
        if pd.notna(t1_array[i]):
            end_idx = min(int(t1_array[i]), n - 1)
            event_conc = concurrency[i:end_idx + 1]
            if len(event_conc) > 0:
                weights[i] = np.mean(1.0 / event_conc)

    if weights.sum() > 0:
        weights = weights / weights.sum() * n

    return weights

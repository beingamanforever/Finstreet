import logging
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


def get_rolling_volatility(prices: pd.Series, span: int = 60) -> pd.Series:
    """Rolling volatility using only past data. Expanding window for early periods."""
    returns = prices.pct_change()
    rolling_vol = returns.rolling(window=span).std()
    expanding_vol = returns.expanding(min_periods=2).std()
    rolling_vol = rolling_vol.fillna(expanding_vol)
    if rolling_vol.iloc[0] != rolling_vol.iloc[0]:
        rolling_vol.iloc[0] = 0.02
    return rolling_vol


def triple_barrier_labeling(df: pd.DataFrame, min_hold: int = 2, max_hold: int = 5, vol_mult: float = 2.0) -> pd.DataFrame:
    """Triple-barrier labeling using intraday high/low for realistic barrier detection."""
    closes = df["close"].values
    highs = df["high"].values
    lows = df["low"].values
    n = len(closes)
    volatility = get_rolling_volatility(df["close"], span=20)
    threshold = vol_mult * volatility

    labels, t1_indices, returns, touched = [], [], [], []

    for i in range(n):
        if i + min_hold >= n:
            labels.append(0)
            t1_indices.append(np.nan)
            returns.append(0)
            touched.append("none")
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

        labels.append(label)
        t1_indices.append(t1)
        ret = (closes[t1] / start - 1) if t1 is not None else 0
        returns.append(ret)
        touched.append(barrier)

    result = df.copy()
    result["label"] = (np.array(labels) > 0).astype(int)
    result["t1"] = t1_indices
    result["future_return"] = returns
    result["barrier_touched"] = touched

    return result


def calculate_sample_weights(df: pd.DataFrame) -> np.ndarray:
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

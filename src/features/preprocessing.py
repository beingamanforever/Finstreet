import logging
from typing import List

import pandas as pd
import numpy as np

from src.features.indicators import add_indicators
from src.features.advanced_features import add_all_features
from src.features.labeling import triple_barrier_labeling, calculate_sample_weights

logger = logging.getLogger(__name__)


def preprocess_pipeline(df: pd.DataFrame, is_training: bool = True, min_hold: int = 2, max_hold: int = 5) -> pd.DataFrame:
    df = add_indicators(df)
    df = add_all_features(df)

    initial_len = len(df)
    df = df.dropna(subset=["close", "volume"])

    if is_training:
        df = triple_barrier_labeling(df, min_hold=min_hold, max_hold=max_hold)
        df["sample_weight"] = calculate_sample_weights(df)
        df = df.dropna(subset=["label", "t1", "sample_weight"])
    else:
        feature_cols = get_feature_columns(df)
        df[feature_cols] = df[feature_cols].ffill().fillna(0)

    return df


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    exclude = [
        "date", "open", "high", "low", "close", "volume",
        "label", "label_3class", "t1", "future_return",
        "barrier_touched", "sample_weight", "typical_price",
        "raw_label", "atr", "significance_threshold"
    ]
    return [col for col in df.columns if col not in exclude]

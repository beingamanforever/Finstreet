import os
import sys
import logging
from typing import Optional, Dict

import pandas as pd
import joblib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.features.preprocessing import preprocess_pipeline

logger = logging.getLogger(__name__)

MODEL_PATH = "models/xgb_model.pkl"
FEATURES_PATH = "models/features.pkl"


def predict_probabilities(df: pd.DataFrame) -> Optional[Dict]:
    if not os.path.exists(MODEL_PATH) or not os.path.exists(FEATURES_PATH):
        return None

    model = joblib.load(MODEL_PATH)
    features = joblib.load(FEATURES_PATH)

    df_processed = preprocess_pipeline(df, is_training=False)
    if df_processed.empty:
        return None

    feature_values = df_processed.iloc[[-1]][features].fillna(0)
    proba = model.predict_proba(feature_values)[0]
    predicted_class = model.predict(feature_values)[0]

    p_down, p_up = float(proba[0]), float(proba[1])

    return {
        "date": df.iloc[-1]["date"],
        "p_down": p_down,
        "p_up": p_up,
        "predicted_class": int(predicted_class),
        "confidence": max(p_down, p_up),
        "directional_edge": abs(p_up - p_down)
    }

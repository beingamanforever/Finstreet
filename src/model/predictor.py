import os
import sys
import logging
from typing import Optional, Dict

import pandas as pd
import numpy as np
import joblib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.features.preprocessing import preprocess_pipeline

logger = logging.getLogger(__name__)

MODEL_PATH = "models/xgb_model.pkl"
ENSEMBLE_PATH = "models/ensemble_model.pkl"
FEATURES_PATH = "models/features.pkl"

# Ensemble weights
XGB_WEIGHT = 0.6
LGBM_WEIGHT = 0.4


def predict_probabilities(df: pd.DataFrame) -> Optional[Dict]:
    """
    Generate predictions using XGBoost + LightGBM ensemble.
    Falls back to XGBoost-only if LightGBM model not available.
    """
    if not os.path.exists(MODEL_PATH) or not os.path.exists(FEATURES_PATH):
        return None

    xgb_model = joblib.load(MODEL_PATH)
    features = joblib.load(FEATURES_PATH)
    
    # Load LightGBM model if available
    lgbm_model = None
    if os.path.exists(ENSEMBLE_PATH):
        lgbm_model = joblib.load(ENSEMBLE_PATH)

    df_processed = preprocess_pipeline(df, is_training=False)
    if df_processed.empty:
        return None

    feature_values = df_processed.iloc[[-1]][features].fillna(0)
    
    # XGBoost prediction
    xgb_proba = xgb_model.predict_proba(feature_values)[0]
    
    # Ensemble prediction if LightGBM available
    if lgbm_model is not None:
        lgbm_proba = lgbm_model.predict_proba(feature_values)[0]
        # Weighted average
        proba = XGB_WEIGHT * xgb_proba + LGBM_WEIGHT * lgbm_proba
        ensemble_used = True
    else:
        proba = xgb_proba
        ensemble_used = False
    
    predicted_class = 1 if proba[1] > proba[0] else 0
    p_down, p_up = float(proba[0]), float(proba[1])

    return {
        "date": df.iloc[-1]["date"],
        "p_down": p_down,
        "p_up": p_up,
        "predicted_class": int(predicted_class),
        "confidence": max(p_down, p_up),
        "directional_edge": abs(p_up - p_down),
        "ensemble_used": ensemble_used
    }

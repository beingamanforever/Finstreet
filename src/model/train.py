import os
import sys
import logging
from typing import Tuple, Optional, List

import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, log_loss, f1_score
import joblib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.features.preprocessing import preprocess_pipeline, get_feature_columns
from config.settings import settings

logger = logging.getLogger(__name__)

MODEL_PATH = "models/xgb_model.pkl"
FEATURES_PATH = "models/features.pkl"


def train_model() -> Tuple[Optional[XGBClassifier], Optional[List[str]]]:
    os.makedirs("models", exist_ok=True)
    data_path = str(settings.data.data_path)

    if not os.path.exists(data_path):
        logger.error(f"Data file not found: {data_path}")
        return None, None

    df = pd.read_csv(data_path)
    
    # Competition constraint: Use only Nov 1 - Dec 31, 2025 for training
    # Earlier data is used only for indicator warmup (required for technical features)
    df["date"] = pd.to_datetime(df["date"])
    train_mask = (df["date"] >= "2025-11-01") & (df["date"] <= "2025-12-31")
    
    # Compute features on full data (indicators need history), then filter
    df_processed = preprocess_pipeline(df, is_training=True, min_hold=2, max_hold=5)
    df_processed["date"] = pd.to_datetime(df_processed["date"])
    df_processed = df_processed[(df_processed["date"] >= "2025-11-01") & (df_processed["date"] <= "2025-12-31")]

    if df_processed.empty:
        logger.error("No samples after preprocessing")
        return None, None

    feature_cols = get_feature_columns(df_processed)
    X = df_processed[feature_cols].copy()
    y = (df_processed['label'].values == 1).astype(int)
    weights = df_processed['sample_weight'].values

    n_splits = min(3, len(X) // 10)
    tscv = TimeSeriesSplit(n_splits=n_splits)

    model = XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=1.0,
        reg_lambda=1.0,
        random_state=42,
        eval_metric='logloss',
        n_jobs=-1
    )

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        w_train = weights[train_idx]

        model.fit(X_train, y_train, sample_weight=w_train, verbose=False)

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        ll = log_loss(y_test, y_proba)
        logger.info(f"Fold {fold}: Acc={acc:.3f}, F1={f1:.3f}, LogLoss={ll:.3f}")

    model.fit(X, y, sample_weight=weights, verbose=False)

    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    joblib.dump(model, MODEL_PATH)
    joblib.dump(feature_cols, FEATURES_PATH)
    importance_df.to_csv("models/feature_importance.csv", index=False)

    logger.info(f"Model saved to {MODEL_PATH}")
    return model, feature_cols


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    train_model()

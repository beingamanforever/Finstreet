"""
Ensemble Model with Rolling Walk-Forward Training

Implements:
1. XGBoost + LightGBM ensemble with weighted voting
2. Rolling walk-forward retraining for robustness
3. Kelly Criterion position sizing estimation
4. Model performance tracking (confusion matrix metrics)
"""

import os
import sys
import logging
from typing import Tuple, Optional, List, Dict
from dataclasses import dataclass

import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    accuracy_score, log_loss, precision_score, recall_score, 
    f1_score, confusion_matrix, classification_report
)
import joblib

# Import LightGBM with fallback
try:
    from lightgbm import LGBMClassifier
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    LGBMClassifier = None

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.features.preprocessing import preprocess_pipeline, get_feature_columns

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DATA_PATH = "data/raw/NSE_SONATSOFTW-EQ.csv"
MODEL_PATH = "models/xgb_model.pkl"
ENSEMBLE_PATH = "models/ensemble_model.pkl"
FEATURES_PATH = "models/features.pkl"
METRICS_PATH = "models/model_metrics.csv"


@dataclass
class ModelMetrics:
    """Container for model performance metrics."""
    accuracy: float
    precision: float
    recall: float
    f1: float
    log_loss_val: float
    confusion: np.ndarray
    kelly_fraction: float
    
    def to_dict(self) -> Dict:
        tn, fp, fn, tp = self.confusion.ravel()
        return {
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "log_loss": self.log_loss_val,
            "true_positives": tp,
            "true_negatives": tn,
            "false_positives": fp,
            "false_negatives": fn,
            "kelly_fraction": self.kelly_fraction
        }


def calculate_kelly_fraction(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> float:
    """
    Calculate Kelly Criterion optimal fraction.
    Kelly = (bp - q) / b
    where: b = odds, p = win probability, q = loss probability
    
    For trading: f* = (win_rate * avg_win/avg_loss - loss_rate) / (avg_win/avg_loss)
    """
    if len(y_true) == 0:
        return 0.0
    
    # Use probability-based estimation
    correct = y_true == y_pred
    win_rate = correct.mean()
    loss_rate = 1 - win_rate
    
    if loss_rate == 0:
        return 0.5  # Cap at 50% if no losses
    
    # Estimate win/loss ratio from probability confidence
    avg_win_conf = y_proba[correct].max(axis=1).mean() if correct.any() else 0.5
    avg_loss_conf = y_proba[~correct].max(axis=1).mean() if (~correct).any() else 0.5
    
    # Win/loss ratio proxy (higher confidence on wins = better edge)
    win_loss_ratio = avg_win_conf / (avg_loss_conf + 1e-10)
    
    kelly = (win_rate * win_loss_ratio - loss_rate) / (win_loss_ratio + 1e-10)
    
    # Half-Kelly for safety (standard practice)
    return max(0, min(0.25, kelly / 2))


class EnsembleTrainer:
    """
    Ensemble model trainer with rolling walk-forward validation.
    """
    
    def __init__(
        self, 
        xgb_weight: float = 0.6, 
        lgbm_weight: float = 0.4,
        rolling_window_days: int = 30,
        retrain_frequency_days: int = 7
    ):
        self.xgb_weight = xgb_weight
        self.lgbm_weight = lgbm_weight
        self.rolling_window = rolling_window_days
        self.retrain_freq = retrain_frequency_days
        
        self.xgb_model = None
        self.lgbm_model = None
        self.features = None
        self.metrics_history: List[Dict] = []
        
    def _create_xgb_model(self) -> XGBClassifier:
        """Create XGBoost classifier with optimized hyperparameters."""
        return XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=1.0,
            reg_lambda=1.0,
            min_child_weight=3,
            gamma=0.1,
            random_state=42,
            eval_metric='logloss',
            n_jobs=-1
        )
    
    def _create_lgbm_model(self) -> Optional['LGBMClassifier']:
        """Create LightGBM classifier if available."""
        if not LIGHTGBM_AVAILABLE:
            logger.warning("LightGBM not installed. Using XGBoost only.")
            return None
            
        return LGBMClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=1.0,
            reg_lambda=1.0,
            min_child_samples=20,
            random_state=42,
            verbose=-1,
            n_jobs=-1
        )
    
    def _evaluate_model(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray, 
        y_proba: np.ndarray
    ) -> ModelMetrics:
        """Calculate comprehensive model metrics."""
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        ll = log_loss(y_true, y_proba)
        cm = confusion_matrix(y_true, y_pred)
        kelly = calculate_kelly_fraction(y_true, y_pred, y_proba)
        
        return ModelMetrics(
            accuracy=acc,
            precision=prec,
            recall=rec,
            f1=f1,
            log_loss_val=ll,
            confusion=cm,
            kelly_fraction=kelly
        )
    
    def train_static(self, df: pd.DataFrame) -> Tuple[Optional[XGBClassifier], Optional[List[str]]]:
        """
        Train models on full dataset (static approach).
        Used for final model training after walk-forward validation.
        """
        df["date"] = pd.to_datetime(df["date"])
        train_mask = (df["date"] >= "2025-11-01") & (df["date"] <= "2025-12-31")
        
        df_processed = preprocess_pipeline(df.copy(), is_training=True, min_hold=2, max_hold=5)
        df_processed["date"] = pd.to_datetime(df_processed["date"])
        df_processed = df_processed[
            (df_processed["date"] >= "2025-11-01") & 
            (df_processed["date"] <= "2025-12-31")
        ]
        
        if df_processed.empty:
            logger.error("No samples after preprocessing")
            return None, None
        
        self.features = get_feature_columns(df_processed)
        X = df_processed[self.features].copy()
        y = (df_processed['label'].values == 1).astype(int)
        weights = df_processed['sample_weight'].values
        
        # Time series cross-validation
        n_splits = min(3, len(X) // 10)
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        self.xgb_model = self._create_xgb_model()
        self.lgbm_model = self._create_lgbm_model()
        
        all_metrics = []
        
        for fold, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            w_train = weights[train_idx]
            
            # Train XGBoost
            self.xgb_model.fit(X_train, y_train, sample_weight=w_train, verbose=False)
            xgb_proba = self.xgb_model.predict_proba(X_test)
            
            # Train LightGBM if available
            if self.lgbm_model is not None:
                self.lgbm_model.fit(X_train, y_train, sample_weight=w_train)
                lgbm_proba = self.lgbm_model.predict_proba(X_test)
                
                # Ensemble prediction
                ensemble_proba = (
                    self.xgb_weight * xgb_proba + 
                    self.lgbm_weight * lgbm_proba
                )
            else:
                ensemble_proba = xgb_proba
            
            y_pred = (ensemble_proba[:, 1] > 0.5).astype(int)
            metrics = self._evaluate_model(y_test, y_pred, ensemble_proba)
            all_metrics.append(metrics.to_dict())
            
            logger.info(
                f"Fold {fold}: Acc={metrics.accuracy:.3f}, F1={metrics.f1:.3f}, "
                f"Kelly={metrics.kelly_fraction:.3f}"
            )
        
        # Final training on all data
        self.xgb_model.fit(X, y, sample_weight=weights, verbose=False)
        if self.lgbm_model is not None:
            self.lgbm_model.fit(X, y, sample_weight=weights)
        
        # Save metrics history
        self.metrics_history = all_metrics
        
        return self.xgb_model, self.features
    
    def train_walk_forward(
        self, 
        df: pd.DataFrame,
        train_window: int = 30,
        test_window: int = 5
    ) -> Dict[str, float]:
        """
        Rolling walk-forward training for robust out-of-sample evaluation.
        
        Process:
        1. Train on days [0, train_window]
        2. Test on days [train_window, train_window + test_window]
        3. Roll forward by test_window days
        4. Repeat
        
        Returns aggregate metrics across all folds.
        """
        df["date"] = pd.to_datetime(df["date"])
        df_processed = preprocess_pipeline(df.copy(), is_training=True, min_hold=2, max_hold=5)
        df_processed["date"] = pd.to_datetime(df_processed["date"])
        df_processed = df_processed[
            (df_processed["date"] >= "2025-11-01") & 
            (df_processed["date"] <= "2025-12-31")
        ]
        
        if len(df_processed) < train_window + test_window:
            logger.warning("Insufficient data for walk-forward validation")
            return {}
        
        self.features = get_feature_columns(df_processed)
        X = df_processed[self.features].copy()
        y = (df_processed['label'].values == 1).astype(int)
        weights = df_processed['sample_weight'].values
        
        all_y_true, all_y_pred, all_y_proba = [], [], []
        
        start_idx = 0
        fold = 0
        
        while start_idx + train_window + test_window <= len(X):
            train_end = start_idx + train_window
            test_end = train_end + test_window
            
            X_train = X.iloc[start_idx:train_end]
            X_test = X.iloc[train_end:test_end]
            y_train = y[start_idx:train_end]
            y_test = y[train_end:test_end]
            w_train = weights[start_idx:train_end]
            
            # Train models
            xgb = self._create_xgb_model()
            xgb.fit(X_train, y_train, sample_weight=w_train, verbose=False)
            xgb_proba = xgb.predict_proba(X_test)
            
            if LIGHTGBM_AVAILABLE:
                lgbm = self._create_lgbm_model()
                lgbm.fit(X_train, y_train, sample_weight=w_train)
                lgbm_proba = lgbm.predict_proba(X_test)
                ensemble_proba = self.xgb_weight * xgb_proba + self.lgbm_weight * lgbm_proba
            else:
                ensemble_proba = xgb_proba
            
            y_pred = (ensemble_proba[:, 1] > 0.5).astype(int)
            
            all_y_true.extend(y_test)
            all_y_pred.extend(y_pred)
            all_y_proba.extend(ensemble_proba)
            
            fold += 1
            start_idx += test_window
        
        if not all_y_true:
            return {}
        
        # Aggregate metrics
        y_true_arr = np.array(all_y_true)
        y_pred_arr = np.array(all_y_pred)
        y_proba_arr = np.array(all_y_proba)
        
        final_metrics = self._evaluate_model(y_true_arr, y_pred_arr, y_proba_arr)
        
        logger.info(f"Walk-Forward Results ({fold} folds):")
        logger.info(f"  Accuracy: {final_metrics.accuracy:.3f}")
        logger.info(f"  Precision: {final_metrics.precision:.3f}")
        logger.info(f"  Recall: {final_metrics.recall:.3f}")
        logger.info(f"  F1 Score: {final_metrics.f1:.3f}")
        logger.info(f"  Kelly Fraction: {final_metrics.kelly_fraction:.3f}")
        
        return final_metrics.to_dict()
    
    def save_models(self) -> None:
        """Save trained models and features."""
        os.makedirs("models", exist_ok=True)
        
        if self.xgb_model is not None:
            joblib.dump(self.xgb_model, MODEL_PATH)
            logger.info(f"XGBoost model saved to {MODEL_PATH}")
        
        if self.lgbm_model is not None:
            joblib.dump(self.lgbm_model, ENSEMBLE_PATH)
            logger.info(f"LightGBM model saved to {ENSEMBLE_PATH}")
        
        if self.features is not None:
            joblib.dump(self.features, FEATURES_PATH)
            
        # Save feature importance
        if self.xgb_model is not None:
            importance_df = pd.DataFrame({
                'feature': self.features,
                'importance': self.xgb_model.feature_importances_
            }).sort_values('importance', ascending=False)
            importance_df.to_csv("models/feature_importance.csv", index=False)
        
        # Save metrics history
        if self.metrics_history:
            pd.DataFrame(self.metrics_history).to_csv(METRICS_PATH, index=False)
    
    def get_confusion_matrix_summary(self) -> str:
        """Generate confusion matrix summary for visualization/reporting."""
        if not self.metrics_history:
            return "No metrics available"
        
        # Average across folds
        avg_metrics = {}
        for key in self.metrics_history[0].keys():
            if key not in ['confusion']:
                values = [m[key] for m in self.metrics_history]
                avg_metrics[key] = np.mean(values)
        
        lines = [
            "Model Performance Summary (Cross-Validation Average)",
            "=" * 50,
            f"Accuracy:     {avg_metrics.get('accuracy', 0):.2%}",
            f"Precision:    {avg_metrics.get('precision', 0):.2%}",
            f"Recall:       {avg_metrics.get('recall', 0):.2%}",
            f"F1 Score:     {avg_metrics.get('f1', 0):.2%}",
            "",
            "Confusion Matrix (Last Fold):",
            f"  True Positives:  {int(avg_metrics.get('true_positives', 0))}",
            f"  True Negatives:  {int(avg_metrics.get('true_negatives', 0))}",
            f"  False Positives: {int(avg_metrics.get('false_positives', 0))}",
            f"  False Negatives: {int(avg_metrics.get('false_negatives', 0))}",
            "",
            f"Kelly Criterion:  {avg_metrics.get('kelly_fraction', 0):.1%} of capital per trade",
        ]
        
        return "\n".join(lines)


def train_ensemble_model() -> Tuple[Optional[XGBClassifier], Optional[List[str]]]:
    """Main training function with ensemble and walk-forward validation."""
    os.makedirs("models", exist_ok=True)
    
    if not os.path.exists(DATA_PATH):
        logger.error(f"Data file not found: {DATA_PATH}")
        return None, None
    
    df = pd.read_csv(DATA_PATH)
    
    trainer = EnsembleTrainer(xgb_weight=0.6, lgbm_weight=0.4)
    
    # Run walk-forward validation first
    logger.info("Running walk-forward validation...")
    wf_metrics = trainer.train_walk_forward(df, train_window=25, test_window=5)
    
    # Train final models on all data
    logger.info("Training final models...")
    model, features = trainer.train_static(df)
    
    if model is not None:
        trainer.save_models()
        logger.info("\n" + trainer.get_confusion_matrix_summary())
    
    return model, features


if __name__ == "__main__":
    train_ensemble_model()

"""
Daily Predictions Generator for Competition Compliance

Generates specific Buy/Sell/Hold predictions for Jan 1-8, 2026 trading days
as required by competition rules. Outputs both CSV and formatted table.
"""

import os
import sys
import logging
from dataclasses import dataclass
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import joblib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.features.preprocessing import preprocess_pipeline, get_feature_columns
from src.features.indicators import add_indicators

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

MODEL_PATH = "models/xgb_model.pkl"
FEATURES_PATH = "models/features.pkl"
ENSEMBLE_PATH = "models/ensemble_model.pkl"


@dataclass
class DailyPrediction:
    """Container for a single day's prediction."""
    date: str
    signal: str  # BUY / SELL / HOLD
    direction: str  # UP / DOWN / NEUTRAL
    confidence: float
    probability_up: float
    probability_down: float
    expected_return: str  # Categorical: POSITIVE / NEGATIVE / FLAT
    regime: str
    adx_strength: float
    rationale: str


class DailyPredictionGenerator:
    """
    Generates daily predictions for competition compliance.
    Combines ML model output with technical regime analysis.
    """
    
    # Jan 1-8, 2026 trading days (excluding weekends)
    FORECAST_DATES = [
        "2026-01-01",  # Thursday
        "2026-01-02",  # Friday
        "2026-01-05",  # Monday
        "2026-01-06",  # Tuesday
        "2026-01-07",  # Wednesday
        "2026-01-08",  # Thursday
    ]
    
    def __init__(self, data_path: str = "data/raw/NSE_SONATSOFTW-EQ.csv"):
        self.data_path = data_path
        self.model = None
        self.features = None
        self.ensemble = None
        self._load_models()
        
    def _load_models(self) -> bool:
        """Load trained models."""
        if os.path.exists(MODEL_PATH) and os.path.exists(FEATURES_PATH):
            self.model = joblib.load(MODEL_PATH)
            self.features = joblib.load(FEATURES_PATH)
            
        if os.path.exists(ENSEMBLE_PATH):
            self.ensemble = joblib.load(ENSEMBLE_PATH)
            
        return self.model is not None
    
    def _compute_adx(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Compute Average Directional Index for trend strength."""
        high, low, close = df["high"], df["low"], df["close"]
        
        plus_dm = high.diff()
        minus_dm = low.diff().abs()
        
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
        
        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs()
        ], axis=1).max(axis=1)
        
        atr = tr.ewm(span=period, adjust=False).mean()
        plus_di = 100 * (plus_dm.ewm(span=period, adjust=False).mean() / atr)
        minus_di = 100 * (minus_dm.ewm(span=period, adjust=False).mean() / atr)
        
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10)
        adx = dx.ewm(span=period, adjust=False).mean()
        
        return adx
    
    def _get_regime(self, df: pd.DataFrame) -> str:
        """Determine market regime from price structure."""
        if len(df) < 30:
            return "INSUFFICIENT_DATA"
            
        close = df["close"].values
        ema_10 = df["close"].ewm(span=10).mean().values
        sma_20 = df["close"].rolling(20).mean().values
        
        recent_close = close[-1]
        recent_ema = ema_10[-1]
        recent_sma = sma_20[-1]
        
        adx = self._compute_adx(df).iloc[-1]
        
        if adx < 20:
            return "SIDEWAYS_WEAK"
        
        if recent_close > recent_ema > recent_sma:
            return "UPTREND_STRONG" if adx > 25 else "UPTREND_WEAK"
        elif recent_close < recent_ema < recent_sma:
            return "DOWNTREND_STRONG" if adx > 25 else "DOWNTREND_WEAK"
        else:
            return "TRANSITIONAL"
    
    def _generate_rationale(self, prob_up: float, prob_down: float, regime: str, adx: float) -> str:
        """Generate human-readable rationale for prediction."""
        rationale_parts = []
        
        edge = abs(prob_up - prob_down)
        if edge > 0.3:
            rationale_parts.append("Strong directional edge")
        elif edge > 0.15:
            rationale_parts.append("Moderate directional edge")
        else:
            rationale_parts.append("Weak directional signal")
        
        if "STRONG" in regime:
            rationale_parts.append("trending conditions favor continuation")
        elif "WEAK" in regime:
            rationale_parts.append("trend weakening suggests caution")
        elif "SIDEWAYS" in regime:
            rationale_parts.append("range-bound market limits conviction")
        else:
            rationale_parts.append("regime transition in progress")
        
        if adx > 25:
            rationale_parts.append(f"ADX={adx:.1f} confirms trend")
        elif adx < 20:
            rationale_parts.append(f"ADX={adx:.1f} suggests choppy conditions")
            
        return "; ".join(rationale_parts)
    
    def _determine_signal(self, prob_up: float, prob_down: float, regime: str, adx: float) -> str:
        """
        Determine Buy/Sell/Hold signal based on model output and regime.
        Conservative approach: require confluence of factors.
        """
        edge = abs(prob_up - prob_down)
        confidence = max(prob_up, prob_down)
        
        # HOLD conditions
        if edge < 0.10:  # Weak directional signal
            return "HOLD"
        if adx < 20 and edge < 0.20:  # Choppy market without strong edge
            return "HOLD"
        if confidence < 0.55:  # Low confidence
            return "HOLD"
        
        # Directional signals require confluence
        if prob_up > prob_down:
            if "DOWNTREND_STRONG" in regime:
                return "HOLD"  # Don't buy against strong downtrend
            return "BUY" if edge > 0.15 or (confidence > 0.60 and adx > 20) else "HOLD"
        else:
            if "UPTREND_STRONG" in regime:
                return "HOLD"  # Don't sell against strong uptrend
            return "SELL" if edge > 0.15 or (confidence > 0.60 and adx > 20) else "HOLD"
    
    def generate_prediction(self, forecast_date: str) -> Optional[DailyPrediction]:
        """Generate prediction for a specific date."""
        if not os.path.exists(self.data_path):
            logger.error(f"Data file not found: {self.data_path}")
            return None
            
        df = pd.read_csv(self.data_path)
        df["date"] = pd.to_datetime(df["date"])
        
        if self.model is None:
            logger.error("Model not loaded")
            return None
        
        df_processed = preprocess_pipeline(df, is_training=False)
        if df_processed.empty:
            return None
        
        feature_values = df_processed.iloc[[-1]][self.features].fillna(0)
        
        # Get primary model predictions
        proba = self.model.predict_proba(feature_values)[0]
        p_down, p_up = float(proba[0]), float(proba[1])
        
        # If ensemble exists, blend predictions
        if self.ensemble is not None:
            try:
                ensemble_proba = self.ensemble.predict_proba(feature_values)[0]
                # Weighted average: 60% primary, 40% ensemble
                p_down = 0.6 * p_down + 0.4 * float(ensemble_proba[0])
                p_up = 0.6 * p_up + 0.4 * float(ensemble_proba[1])
            except Exception:
                pass  # Fall back to primary model only
        
        regime = self._get_regime(df)
        adx = self._compute_adx(df).iloc[-1]
        signal = self._determine_signal(p_up, p_down, regime, adx)
        rationale = self._generate_rationale(p_up, p_down, regime, adx)
        
        direction = "UP" if p_up > p_down else "DOWN" if p_down > p_up else "NEUTRAL"
        expected_return = "POSITIVE" if p_up > 0.55 else "NEGATIVE" if p_down > 0.55 else "FLAT"
        
        return DailyPrediction(
            date=forecast_date,
            signal=signal,
            direction=direction,
            confidence=max(p_up, p_down),
            probability_up=p_up,
            probability_down=p_down,
            expected_return=expected_return,
            regime=regime,
            adx_strength=adx,
            rationale=rationale
        )
    
    def generate_all_predictions(self) -> List[DailyPrediction]:
        """Generate predictions for all forecast dates (Jan 1-8, 2026)."""
        predictions = []
        
        for date in self.FORECAST_DATES:
            pred = self.generate_prediction(date)
            if pred:
                predictions.append(pred)
                
        return predictions
    
    def to_dataframe(self, predictions: List[DailyPrediction]) -> pd.DataFrame:
        """Convert predictions to DataFrame."""
        return pd.DataFrame([
            {
                "Date": p.date,
                "Signal": p.signal,
                "Direction": p.direction,
                "Confidence": f"{p.confidence:.2%}",
                "P(Up)": f"{p.probability_up:.2%}",
                "P(Down)": f"{p.probability_down:.2%}",
                "Expected_Return": p.expected_return,
                "Regime": p.regime,
                "ADX": f"{p.adx_strength:.1f}",
                "Rationale": p.rationale
            }
            for p in predictions
        ])
    
    def export_csv(self, predictions: List[DailyPrediction], path: str = "reports/daily_predictions.csv") -> None:
        """Export predictions to CSV."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df = self.to_dataframe(predictions)
        df.to_csv(path, index=False)
        logger.info(f"Predictions exported to {path}")
    
    def format_table(self, predictions: List[DailyPrediction]) -> str:
        """Format predictions as markdown table for README."""
        lines = []
        lines.append("| Date | Signal | Direction | Confidence | Expected Return | Regime |")
        lines.append("|------|--------|-----------|------------|-----------------|--------|")
        
        for p in predictions:
            lines.append(
                f"| {p.date} | **{p.signal}** | {p.direction} | "
                f"{p.confidence:.1%} | {p.expected_return} | {p.regime.replace('_', ' ').title()} |"
            )
            
        return "\n".join(lines)


def generate_predictions() -> str:
    """Main function to generate and export predictions."""
    generator = DailyPredictionGenerator()
    predictions = generator.generate_all_predictions()
    
    if not predictions:
        return "Predictions unavailable - model not trained or data missing"
    
    generator.export_csv(predictions)
    
    return generator.format_table(predictions)


if __name__ == "__main__":
    output = generate_predictions()
    print("\nDaily Predictions (Jan 1-8, 2026):")
    print("=" * 80)
    print(output)

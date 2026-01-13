"""
Forecast Generation Module

Generates forward-looking predictions based on the trained model and current
market regime. Forecasts are probabilistic and include confidence intervals.
"""

import os
import sys
import logging
from dataclasses import dataclass
from typing import Optional, List, Dict
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
import joblib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.features.preprocessing import preprocess_pipeline, get_feature_columns
from src.strategy.trend_momentum import TrendMomentum, Config

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

MODEL_PATH = "models/xgb_model.pkl"
FEATURES_PATH = "models/features.pkl"


@dataclass
class ForecastResult:
    """Container for a single day's forecast."""
    date: str
    direction: str
    probability: float
    confidence_level: str
    regime: str
    key_factors: List[str]


class ForecastGenerator:
    """
    Generates probabilistic forecasts combining ML predictions with
    technical regime analysis.
    """
    
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.model = None
        self.features = None
        self.trend_analyzer = TrendMomentum(Config(
            ema=10, sma=20, slope_period=5,
            pullback_pct=0.20, confirm_bars=1,
            stop_atr=2.0, target_atr=4.0
        ))
        self._load_model()
        
    def _load_model(self) -> bool:
        """Load trained model and feature list."""
        if not os.path.exists(MODEL_PATH) or not os.path.exists(FEATURES_PATH):
            logger.warning("Model not found. Train the model first.")
            return False
        self.model = joblib.load(MODEL_PATH)
        self.features = joblib.load(FEATURES_PATH)
        return True
    
    def _get_regime(self, df: pd.DataFrame) -> str:
        """Determine current market regime from price structure."""
        if len(df) < 30:
            return "INSUFFICIENT_DATA"
            
        close = df["close"].values
        ema_10 = df["close"].ewm(span=10).mean().values
        sma_20 = df["close"].rolling(20).mean().values
        
        recent_close = close[-1]
        recent_ema = ema_10[-1]
        recent_sma = sma_20[-1]
        
        slope_window = min(5, len(close) - 1)
        ema_slope = (ema_10[-1] - ema_10[-slope_window]) / slope_window
        
        if recent_close > recent_ema > recent_sma and ema_slope > 0:
            return "STRONG_UP" if ema_slope > 0.5 else "WEAK_UP"
        elif recent_close < recent_ema < recent_sma and ema_slope < 0:
            return "STRONG_DOWN" if ema_slope < -0.5 else "WEAK_DOWN"
        else:
            return "SIDEWAYS"
    
    def _identify_key_factors(self, df: pd.DataFrame) -> List[str]:
        """Identify the primary technical factors driving the forecast."""
        factors = []
        
        if len(df) < 20:
            return ["Insufficient data for factor analysis"]
            
        close = df["close"].iloc[-1]
        sma_20 = df["close"].rolling(20).mean().iloc[-1]
        
        if close > sma_20 * 1.02:
            factors.append("Price above 20-day mean")
        elif close < sma_20 * 0.98:
            factors.append("Price below 20-day mean")
            
        recent_returns = df["close"].pct_change().iloc[-5:]
        if recent_returns.mean() > 0.005:
            factors.append("Positive short-term momentum")
        elif recent_returns.mean() < -0.005:
            factors.append("Negative short-term momentum")
            
        vol_current = df["close"].pct_change().iloc[-10:].std()
        vol_historical = df["close"].pct_change().iloc[-60:-10].std() if len(df) > 60 else vol_current
        
        if vol_current > vol_historical * 1.5:
            factors.append("Elevated volatility regime")
        elif vol_current < vol_historical * 0.7:
            factors.append("Compressed volatility")
            
        if not factors:
            factors.append("Neutral technical conditions")
            
        return factors[:3]
    
    def _confidence_level(self, probability: float) -> str:
        """Map probability to human-readable confidence level."""
        if probability >= 0.75:
            return "HIGH"
        elif probability >= 0.60:
            return "MODERATE"
        else:
            return "LOW"
    
    def generate_forecast(self, forecast_date: str) -> Optional[ForecastResult]:
        """
        Generate forecast for a specific date.
        
        Args:
            forecast_date: Target date for forecast (YYYY-MM-DD)
            
        Returns:
            ForecastResult containing direction, probability, and supporting factors
        """
        df = pd.read_csv(self.data_path)
        
        if self.model is None:
            return None
            
        df_processed = preprocess_pipeline(df, is_training=False)
        if df_processed.empty:
            return None
            
        feature_values = df_processed.iloc[[-1]][self.features].fillna(0)
        
        proba = self.model.predict_proba(feature_values)[0]
        p_down, p_up = float(proba[0]), float(proba[1])
        
        direction = "BULLISH" if p_up > p_down else "BEARISH"
        probability = max(p_up, p_down)
        
        regime = self._get_regime(df)
        factors = self._identify_key_factors(df)
        
        return ForecastResult(
            date=forecast_date,
            direction=direction,
            probability=probability,
            confidence_level=self._confidence_level(probability),
            regime=regime,
            key_factors=factors
        )
    
    def generate_weekly_forecast(self, start_date: str, num_days: int = 5) -> List[ForecastResult]:
        """
        Generate forecasts for multiple consecutive trading days.
        
        Note: Multi-day forecasts carry increasing uncertainty. The probability
        values represent single-day directional bias, not cumulative returns.
        """
        forecasts = []
        current_date = datetime.strptime(start_date, "%Y-%m-%d")
        
        for i in range(num_days):
            forecast_date = current_date + timedelta(days=i)
            if forecast_date.weekday() < 5:
                result = self.generate_forecast(forecast_date.strftime("%Y-%m-%d"))
                if result:
                    forecasts.append(result)
                    
        return forecasts


def generate_readme_forecast() -> str:
    """Generate forecast section for README."""
    generator = ForecastGenerator("data/raw/NSE_SONATSOFTW-EQ.csv")
    
    forecast = generator.generate_forecast("2026-01-08")
    
    if forecast is None:
        return "Forecast unavailable - model not trained"
    
    output = []
    output.append(f"**Date:** Jan 1-8, 2026")
    output.append(f"**Bias:** {forecast.direction}")
    output.append(f"**Confidence:** {forecast.confidence_level} ({forecast.probability:.1%})")
    output.append(f"**Regime:** {forecast.regime.replace('_', ' ').title()}")
    output.append(f"**Factors:** {', '.join(forecast.key_factors)}")
    
    return "\n".join(output)


if __name__ == "__main__":
    output = generate_readme_forecast()
    logger.info("\n" + output)

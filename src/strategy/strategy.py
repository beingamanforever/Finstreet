import logging
from dataclasses import dataclass
from typing import Optional, Dict

import pandas as pd
import numpy as np

from src.model.predictor import predict_probabilities
from src.strategy.trend_momentum import TrendMomentum, Config

logger = logging.getLogger(__name__)


@dataclass
class Params:
    """Regime-specific trading parameters."""
    stop_mult: float = 1.5
    target_mult: float = 3.0
    min_edge: float = 0.08
    min_conf: float = 0.55
    base_risk_pct: float = 0.02
    min_risk_pct: float = 0.015
    max_risk_pct: float = 0.025


class Strategy:
    """
    Enhanced trading strategy with:
    - Dynamic position sizing based on confidence
    - ADX-based regime filtering
    - Multi-timeframe trend alignment
    - Volume confirmation
    - ML signal filtering
    """
    
    def __init__(self, capital: float = 100_000.0):
        self.capital = capital
        self.trend = TrendMomentum(Config(
            ema=10,
            sma=20,
            slope_period=5,
            pullback_pct=0.06,
            pullback_max=0.35,
            confirm_bars=1,
            stop_atr=1.5,
            target_atr=3.0,
            base_risk_pct=0.04,
            min_risk_pct=0.03,
            max_risk_pct=0.05,
            adx_min_trade=15.0,
            adx_strong=25.0,
            volume_confirm=False,
            volume_threshold=0.8,
            use_weekly_filter=False,
            use_trailing_stop=True,
            trailing_atr_mult=2.5
        ))
        
        # Regime-specific parameters
        self.params = {
            "STRONG_UP": Params(
                stop_mult=1.5, target_mult=3.0, min_edge=0.08, 
                min_conf=0.55, base_risk_pct=0.025, min_risk_pct=0.02, max_risk_pct=0.03
            ),
            "WEAK_UP": Params(
                stop_mult=1.5, target_mult=2.5, min_edge=0.10, 
                min_conf=0.58, base_risk_pct=0.02, min_risk_pct=0.015, max_risk_pct=0.025
            ),
            "SIDEWAYS": Params(
                stop_mult=1.2, target_mult=2.0, min_edge=0.12, 
                min_conf=0.62, base_risk_pct=0.015, min_risk_pct=0.01, max_risk_pct=0.02
            ),
            "WEAK_DOWN": Params(
                stop_mult=1.5, target_mult=2.5, min_edge=0.10, 
                min_conf=0.58, base_risk_pct=0.02, min_risk_pct=0.015, max_risk_pct=0.025
            ),
            "STRONG_DOWN": Params(
                stop_mult=1.5, target_mult=3.0, min_edge=0.08, 
                min_conf=0.55, base_risk_pct=0.025, min_risk_pct=0.02, max_risk_pct=0.03
            )
        }

    def _calculate_dynamic_risk(self, conf: float, params: Params) -> float:
        """
        Dynamic position sizing based on signal confidence.
        Formula: risk = base_risk * (0.5 + confidence)
        Scales from min_risk to max_risk based on confidence.
        """
        risk = params.base_risk_pct * (0.5 + conf)
        return max(params.min_risk_pct, min(params.max_risk_pct, risk))

    def generate_signal(self, df: pd.DataFrame) -> Optional[Dict]:
        if len(df) < 50:  # Increased for new indicators
            return None

        sig = self.trend.signal(df, self.capital)
        if sig is None:
            return None

        ml = predict_probabilities(df)
        if ml is None:
            return sig

        regime = sig["regime"]
        p = self.params.get(regime, Params())

        tech_dir = 1 if sig["signal"] == "BUY" else -1
        ml_dir = 1 if ml["p_up"] > ml["p_down"] else -1

        # Require ML and technical signal alignment
        if tech_dir != ml_dir:
            return None

        edge = ml.get("directional_edge", 0.0)
        conf = ml["confidence"]

        # Filter by minimum edge and confidence
        if edge < p.min_edge or conf < p.min_conf:
            return None

        # Blend technical and ML confidence
        combined_conf = (sig["confidence"] + conf) / 2
        
        # Calculate dynamic risk based on combined confidence
        risk_pct = self._calculate_dynamic_risk(combined_conf, p)

        # Update signal with ML and risk information
        sig["ml_confidence"] = conf
        sig["ml_edge"] = edge
        sig["confidence"] = combined_conf
        sig["edge"] = max(sig["edge"], edge)
        sig["risk_pct"] = risk_pct
        
        # Recalculate position size with dynamic risk
        atr = sig.get("stop_loss", sig["entry_price"] * 0.02)
        if sig["signal"] == "BUY":
            stop_dist = sig["entry_price"] - sig["stop_loss"]
        else:
            stop_dist = sig["stop_loss"] - sig["entry_price"]
        
        if stop_dist > 0:
            risk_amount = self.capital * risk_pct
            new_qty = max(1, int(risk_amount / stop_dist))
            new_qty = min(new_qty, int(self.capital * 0.15 / sig["entry_price"]))
            sig["quantity"] = new_qty

        return sig

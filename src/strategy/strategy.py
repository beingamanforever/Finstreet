import logging
from dataclasses import dataclass
from typing import Optional, Dict

import pandas as pd

from src.model.predictor import predict_probabilities
from src.strategy.trend_momentum import TrendMomentum, Config

logger = logging.getLogger(__name__)


@dataclass
class Params:
    stop_mult: float = 1.5
    target_mult: float = 3.0
    min_edge: float = 0.08
    min_conf: float = 0.55
    risk_pct: float = 0.02


class Strategy:
    def __init__(self, capital: float = 100_000.0):
        self.capital = capital
        self.trend = TrendMomentum(Config(
            ema=10,
            sma=20,
            slope_period=5,
            pullback_pct=0.20,
            confirm_bars=1,
            stop_atr=2.0,
            target_atr=4.0
        ))
        
        self.params = {
            "STRONG_UP": Params(stop_mult=1.5, target_mult=3.0, min_edge=0.08, min_conf=0.55, risk_pct=0.025),
            "WEAK_UP": Params(stop_mult=1.5, target_mult=2.5, min_edge=0.10, min_conf=0.58, risk_pct=0.02),
            "SIDEWAYS": Params(stop_mult=1.2, target_mult=2.0, min_edge=0.12, min_conf=0.62, risk_pct=0.015),
            "WEAK_DOWN": Params(stop_mult=1.5, target_mult=2.5, min_edge=0.10, min_conf=0.58, risk_pct=0.02),
            "STRONG_DOWN": Params(stop_mult=1.5, target_mult=3.0, min_edge=0.08, min_conf=0.55, risk_pct=0.025)
        }

    def generate_signal(self, df: pd.DataFrame) -> Optional[Dict]:
        if len(df) < 30:
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

        if tech_dir != ml_dir:
            return None

        edge = ml.get("directional_edge", 0.0)
        conf = ml["confidence"]

        if edge < p.min_edge or conf < p.min_conf:
            return None

        sig["ml_confidence"] = conf
        sig["ml_edge"] = edge
        sig["confidence"] = (sig["confidence"] + conf) / 2
        sig["edge"] = max(sig["edge"], edge)

        return sig

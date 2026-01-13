import numpy as np
import pandas as pd
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Dict, Tuple


class Trend(Enum):
    STRONG_UP = "STRONG_UP"
    WEAK_UP = "WEAK_UP"
    SIDEWAYS = "SIDEWAYS"
    WEAK_DOWN = "WEAK_DOWN"
    STRONG_DOWN = "STRONG_DOWN"


class Pullback(Enum):
    NONE = "NONE"
    FORMING = "FORMING"
    READY = "READY"
    BROKEN = "BROKEN"


@dataclass
class Config:
    ema: int = 8
    sma: int = 21
    slope_period: int = 5
    pullback_pct: float = 0.18
    confirm_bars: int = 1
    stop_atr: float = 1.8
    target_atr: float = 3.5
    risk_pct: float = 0.03


class TrendMomentum:
    def __init__(self, cfg: Config = None):
        self.cfg = cfg or Config()
        self._pending = False
        self._direction = 0
        self._confirms = 0

    def _indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        c = df["close"]
        h, l = df["high"], df["low"]
        
        df["ema"] = c.ewm(span=self.cfg.ema, adjust=False).mean()
        df["sma"] = c.rolling(self.cfg.sma).mean()
        
        tr = pd.concat([h - l, abs(h - c.shift(1)), abs(l - c.shift(1))], axis=1).max(axis=1)
        df["atr"] = tr.rolling(14).mean()
        
        delta = c.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        df["rsi"] = 100 - 100 / (1 + gain / (loss + 1e-10))
        
        df["ema_slope"] = (df["ema"] - df["ema"].shift(self.cfg.slope_period)) / c * 100
        df["sma_slope"] = (df["sma"] - df["sma"].shift(self.cfg.slope_period)) / c * 100
        df["roc"] = c.pct_change(5)
        
        return df

    def _trend(self, df: pd.DataFrame) -> Tuple[Trend, float]:
        r = df.iloc[-1]
        gap = (r["ema"] - r["sma"]) / r["sma"] * 100
        
        if abs(gap) < 0.4:
            return Trend.SIDEWAYS, gap
        
        if gap > 0:
            if r["ema_slope"] > 0.08 and r["sma_slope"] > 0:
                return Trend.STRONG_UP, gap
            return Trend.WEAK_UP, gap
        else:
            if r["ema_slope"] < -0.08 and r["sma_slope"] < 0:
                return Trend.STRONG_DOWN, abs(gap)
            return Trend.WEAK_DOWN, abs(gap)

    def _pullback(self, df: pd.DataFrame, trend: Trend) -> Tuple[Pullback, float]:
        if len(df) < 20:
            return Pullback.NONE, 0.0
        
        r = df.iloc[-1]
        window = df.iloc[-20:]
        hi, lo = window["high"].max(), window["low"].min()
        rng = hi - lo
        
        if rng < r["atr"] * 0.5:
            return Pullback.NONE, 0.0
        
        price, ema, sma, atr = r["close"], r["ema"], r["sma"], r["atr"]
        
        if trend in [Trend.STRONG_UP, Trend.WEAK_UP]:
            depth = (hi - price) / rng
            if price < sma:
                return Pullback.BROKEN, depth
            dist = abs(price - ema) / atr
            if depth >= self.cfg.pullback_pct:
                return (Pullback.READY, depth) if dist < 0.6 else (Pullback.FORMING, depth)
        
        elif trend in [Trend.STRONG_DOWN, Trend.WEAK_DOWN]:
            depth = (price - lo) / rng
            if price > sma:
                return Pullback.BROKEN, depth
            dist = abs(price - ema) / atr
            if depth >= self.cfg.pullback_pct:
                return (Pullback.READY, depth) if dist < 0.6 else (Pullback.FORMING, depth)
        
        return Pullback.NONE, 0.0

    def _momentum(self, df: pd.DataFrame, trend: Trend) -> Tuple[bool, float]:
        r, p = df.iloc[-1], df.iloc[-2]
        rsi, prev_rsi, roc = r["rsi"], p["rsi"], r["roc"]
        
        if trend in [Trend.STRONG_UP, Trend.WEAK_UP]:
            accel = (rsi > prev_rsi or roc > -0.005) and rsi > 35
            return accel, rsi
        elif trend in [Trend.STRONG_DOWN, Trend.WEAK_DOWN]:
            accel = (rsi < prev_rsi or roc < 0.005) and rsi < 65
            return accel, rsi
        return False, rsi

    def _check_setup(self, trend: Trend, pb: Pullback, accel: bool) -> int:
        if trend in [Trend.STRONG_UP, Trend.WEAK_UP]:
            if pb == Pullback.READY:
                return 1
            if pb == Pullback.FORMING and accel:
                return 1
        elif trend in [Trend.STRONG_DOWN, Trend.WEAK_DOWN]:
            if pb == Pullback.READY:
                return -1
            if pb == Pullback.FORMING and accel:
                return -1
        return 0

    def signal(self, df: pd.DataFrame, capital: float = 100000) -> Optional[Dict]:
        if len(df) < self.cfg.sma + self.cfg.slope_period + 5:
            return None
        
        df = self._indicators(df)
        trend, strength = self._trend(df)
        pb, depth = self._pullback(df, trend)
        accel, rsi = self._momentum(df, trend)
        
        direction = self._check_setup(trend, pb, accel)
        
        if direction == 0:
            self._pending, self._direction, self._confirms = False, 0, 0
            return None
        
        if self._pending and self._direction == direction:
            self._confirms += 1
        else:
            self._pending, self._direction, self._confirms = True, direction, 1
        
        if self._confirms < self.cfg.confirm_bars:
            return None
        
        self._pending, self._direction, self._confirms = False, 0, 0
        
        r = df.iloc[-1]
        price, atr = r["close"], r["atr"]
        
        mult = 1.15 if trend in [Trend.STRONG_UP, Trend.STRONG_DOWN] else 1.0
        stop_dist = self.cfg.stop_atr * atr * mult
        target_dist = self.cfg.target_atr * atr * mult
        
        if direction == 1:
            sl, tp = price - stop_dist, price + target_dist
        else:
            sl, tp = price + stop_dist, price - target_dist
        
        risk = capital * self.cfg.risk_pct
        qty = max(1, int(risk / stop_dist))
        qty = min(qty, int(capital * 0.15 / price))
        
        conf = 0.72 if trend in [Trend.STRONG_UP, Trend.STRONG_DOWN] else 0.62
        if 0.25 <= depth <= 0.45:
            conf += 0.05
        
        return {
            "signal": "BUY" if direction == 1 else "SELL",
            "entry_price": price,
            "quantity": qty,
            "stop_loss": sl,
            "take_profit": tp,
            "confidence": conf,
            "edge": min(strength / 40, 1.0),
            "regime": trend.value,
            "depth": depth,
            "rsi": rsi,
            "rr": target_dist / stop_dist
        }

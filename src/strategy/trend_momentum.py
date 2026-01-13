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


class RegimeState(Enum):
    """Market regime based on ADX and volatility."""
    HIGH_TREND = "HIGH_TREND"      # ADX > 25, strong directional move
    LOW_TREND = "LOW_TREND"        # ADX 20-25, developing trend
    CHOPPY = "CHOPPY"              # ADX < 20, avoid trading
    HIGH_VOL = "HIGH_VOL"          # ATR percentile > 70%
    LOW_VOL = "LOW_VOL"            # ATR percentile < 30%


@dataclass
class Config:
    ema: int = 10
    sma: int = 20
    slope_period: int = 5
    pullback_pct: float = 0.06
    pullback_max: float = 0.35
    confirm_bars: int = 1
    stop_atr: float = 1.5
    target_atr: float = 3.0
    base_risk_pct: float = 0.04
    min_risk_pct: float = 0.03
    max_risk_pct: float = 0.05
    adx_min_trade: float = 15.0
    adx_strong: float = 25.0
    volume_confirm: bool = False
    volume_threshold: float = 0.8
    use_weekly_filter: bool = False
    use_trailing_stop: bool = True
    trailing_atr_mult: float = 2.5


class TrendMomentum:
    def __init__(self, cfg: Config = None):
        self.cfg = cfg or Config()
        self._pending = False
        self._direction = 0
        self._confirms = 0

    def _indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        c = df["close"]
        h, l, v = df["high"], df["low"], df["volume"]
        
        df["ema"] = c.ewm(span=self.cfg.ema, adjust=False).mean()
        df["sma"] = c.rolling(self.cfg.sma).mean()
        df["sma_50"] = c.rolling(50).mean()
        df["sma_100"] = c.rolling(100).mean()
        
        tr = pd.concat([h - l, abs(h - c.shift(1)), abs(l - c.shift(1))], axis=1).max(axis=1)
        df["atr"] = tr.rolling(14).mean()
        df["atr_pct"] = df["atr"] / c * 100
        
        # ATR percentile for regime detection
        df["atr_percentile"] = df["atr"].rolling(60).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else 0.5
        )
        
        delta = c.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        df["rsi"] = 100 - 100 / (1 + gain / (loss + 1e-10))
        
        df["ema_slope"] = (df["ema"] - df["ema"].shift(self.cfg.slope_period)) / c * 100
        df["sma_slope"] = (df["sma"] - df["sma"].shift(self.cfg.slope_period)) / c * 100
        df["roc"] = c.pct_change(5)
        
        # ADX calculation for trend strength
        df = self._compute_adx(df)
        
        # Volume features
        df["vol_ma"] = v.rolling(20).mean()
        df["relative_vol"] = v / (df["vol_ma"] + 1e-10)
        
        # Z-score for mean reversion detection
        df["close_zscore"] = (c - c.rolling(20).mean()) / (c.rolling(20).std() + 1e-10)
        
        # Weekly trend proxy
        df["weekly_trend"] = np.where(
            (c > df["sma_100"]) & (df["sma"] > df["sma_50"]), 1,
            np.where((c < df["sma_100"]) & (df["sma"] < df["sma_50"]), -1, 0)
        )
        
        return df
    
    def _compute_adx(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Compute ADX for trend strength quantification."""
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
        
        atr_adx = tr.ewm(span=period, adjust=False).mean()
        plus_di = 100 * (plus_dm.ewm(span=period, adjust=False).mean() / (atr_adx + 1e-10))
        minus_di = 100 * (minus_dm.ewm(span=period, adjust=False).mean() / (atr_adx + 1e-10))
        
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10)
        df["adx"] = dx.ewm(span=period, adjust=False).mean()
        df["plus_di"] = plus_di
        df["minus_di"] = minus_di
        
        return df

    def _check_regime(self, df: pd.DataFrame) -> Tuple[bool, str]:
        """Check if market regime is suitable for trading."""
        r = df.iloc[-1]
        adx = r.get("adx", 25)
        atr_pct = r.get("atr_percentile", 0.5)
        
        if adx < self.cfg.adx_min_trade and atr_pct < 0.15:
            return False, f"ADX={adx:.1f} and low vol - avoid"
        
        regime = "HIGH_TREND" if adx > self.cfg.adx_strong else "DEVELOPING_TREND"
        return True, regime

    def _check_volume(self, df: pd.DataFrame) -> bool:
        """Check if volume confirms the move."""
        if not self.cfg.volume_confirm:
            return True
        
        r = df.iloc[-1]
        relative_vol = r.get("relative_vol", 1.0)
        return relative_vol >= self.cfg.volume_threshold
    
    def _check_weekly_filter(self, df: pd.DataFrame, direction: int) -> bool:
        """
        Multi-timeframe filter: only trade in direction of weekly trend.
        direction: 1 for long, -1 for short
        """
        if not self.cfg.use_weekly_filter:
            return True
        
        r = df.iloc[-1]
        weekly_trend = r.get("weekly_trend", 0)
        
        # Allow trades aligned with weekly trend, or when weekly is neutral
        if weekly_trend == 0:
            return True
        return weekly_trend == direction

    def _trend(self, df: pd.DataFrame) -> Tuple[Trend, float]:
        r = df.iloc[-1]
        gap = (r["ema"] - r["sma"]) / r["sma"] * 100
        adx = r.get("adx", 25)
        
        if abs(gap) < 0.2 and adx < self.cfg.adx_min_trade:
            return Trend.SIDEWAYS, gap
        
        is_strong = adx > self.cfg.adx_strong
        
        if gap > 0 or (r["close"] > r["sma"] and r["rsi"] > 50):
            if is_strong and r["ema_slope"] > 0.05:
                return Trend.STRONG_UP, gap
            return Trend.WEAK_UP, gap
        else:
            if is_strong and r["ema_slope"] < -0.05:
                return Trend.STRONG_DOWN, abs(gap)
            return Trend.WEAK_DOWN, abs(gap)

    def _pullback(self, df: pd.DataFrame, trend: Trend) -> Tuple[Pullback, float]:
        if len(df) < 20:
            return Pullback.NONE, 0.0
        
        r = df.iloc[-1]
        window = df.iloc[-20:]
        hi, lo = window["high"].max(), window["low"].min()
        rng = hi - lo
        
        if rng < r["atr"] * 0.3:
            return Pullback.NONE, 0.0
        
        price, ema, sma, atr = r["close"], r["ema"], r["sma"], r["atr"]
        
        if trend in [Trend.STRONG_UP, Trend.WEAK_UP]:
            depth = (hi - price) / rng if rng > 0 else 0
            
            if depth > self.cfg.pullback_max:
                return Pullback.BROKEN, depth
            
            if depth >= self.cfg.pullback_pct:
                dist = abs(price - ema) / (atr + 1e-10)
                return (Pullback.READY, depth) if dist < 1.0 else (Pullback.FORMING, depth)
            
            if price < ema and price > sma * 0.98:
                return Pullback.FORMING, depth
        
        elif trend in [Trend.STRONG_DOWN, Trend.WEAK_DOWN]:
            depth = (price - lo) / rng if rng > 0 else 0
            
            if depth > self.cfg.pullback_max:
                return Pullback.BROKEN, depth
            
            if depth >= self.cfg.pullback_pct:
                dist = abs(price - ema) / (atr + 1e-10)
                return (Pullback.READY, depth) if dist < 1.0 else (Pullback.FORMING, depth)
            
            if price > ema and price < sma * 1.02:
                return Pullback.FORMING, depth
        
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
    
    def _calculate_dynamic_risk(self, confidence: float) -> float:
        """
        Dynamic position sizing based on signal confidence.
        Formula: risk = base_risk * (0.5 + confidence)
        Range: 1.5% to 2.5% for confidence 0.5 to 1.0
        """
        risk = self.cfg.base_risk_pct * (0.5 + confidence)
        return max(self.cfg.min_risk_pct, min(self.cfg.max_risk_pct, risk))

    def signal(self, df: pd.DataFrame, capital: float = 100000) -> Optional[Dict]:
        if len(df) < self.cfg.sma + self.cfg.slope_period + 5:
            return None
        
        df = self._indicators(df)
        
        # Check regime suitability
        can_trade, regime_desc = self._check_regime(df)
        if not can_trade:
            return None
        
        trend, strength = self._trend(df)
        pb, depth = self._pullback(df, trend)
        accel, rsi = self._momentum(df, trend)
        
        direction = self._check_setup(trend, pb, accel)
        
        if direction == 0:
            self._pending, self._direction, self._confirms = False, 0, 0
            return None
        
        # Volume confirmation check
        if not self._check_volume(df):
            return None
        
        # Multi-timeframe filter
        if not self._check_weekly_filter(df, direction):
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
        adx = r.get("adx", 25)
        
        # Confidence based on trend strength and ADX
        base_conf = 0.72 if trend in [Trend.STRONG_UP, Trend.STRONG_DOWN] else 0.62
        if 0.18 <= depth <= 0.35:
            base_conf += 0.05
        if adx > self.cfg.adx_strong:
            base_conf += 0.03
        
        # Dynamic risk sizing based on confidence
        risk_pct = self._calculate_dynamic_risk(base_conf)
        
        # ATR multipliers adjusted for ADX regime
        mult = 1.15 if adx > self.cfg.adx_strong else 1.0
        stop_dist = self.cfg.stop_atr * atr * mult
        target_dist = self.cfg.target_atr * atr * mult
        
        if direction == 1:
            sl, tp = price - stop_dist, price + target_dist
        else:
            sl, tp = price + stop_dist, price - target_dist
        
        # Position sizing with dynamic risk
        risk = capital * risk_pct
        qty = max(1, int(risk / stop_dist))
        qty = min(qty, int(capital * 0.15 / price))
        
        # Trailing stop initial level (for reference)
        trailing_stop = None
        if self.cfg.use_trailing_stop:
            trailing_atr = self.cfg.trailing_atr_mult * atr
            trailing_stop = price - trailing_atr if direction == 1 else price + trailing_atr
        
        return {
            "signal": "BUY" if direction == 1 else "SELL",
            "entry_price": price,
            "quantity": qty,
            "stop_loss": sl,
            "take_profit": tp,
            "trailing_stop": trailing_stop,
            "confidence": base_conf,
            "edge": min(strength / 40, 1.0),
            "regime": trend.value,
            "depth": depth,
            "rsi": rsi,
            "adx": adx,
            "relative_volume": r.get("relative_vol", 1.0),
            "risk_pct": risk_pct,
            "rr": target_dist / stop_dist
        }

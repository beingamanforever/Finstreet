import os
import sys
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.strategy.strategy import Strategy
from config.settings import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class Config:
    capital: float = 100_000.0
    commission: float = 0.0003
    slippage: float = 0.0005
    max_pos_pct: float = 0.10
    use_trailing_stop: bool = True
    trailing_atr_mult: float = 2.0


@dataclass
class TradeRecord:
    """Detailed trade record for analysis."""
    date: str
    signal_type: str
    entry: float
    exit: float
    qty: int
    pnl: float
    reason: str
    confidence: float
    edge: float
    regime: str
    adx: float = 0.0
    risk_pct: float = 0.02
    holding_bars: int = 1
    max_favorable: float = 0.0  # Maximum favorable excursion
    max_adverse: float = 0.0   # Maximum adverse excursion


class Backtester:
    def __init__(self, data_path: str, capital: float = 100_000.0, config: Optional[Config] = None):
        self.data_path = data_path
        self.capital = capital
        self.config = config or Config(capital=capital)
        self.strategy = Strategy(capital=capital)
        self.trades: List[TradeRecord] = []
        self.equity_curve: List[Dict] = []
        self.signal_log: List[Dict] = []  # Track all signals for confusion matrix

    def run(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        df = pd.read_csv(self.data_path)
        df["date"] = pd.to_datetime(df["date"])
        
        # Competition constraint: Trade only within Nov 1 - Dec 31, 2025
        trade_start = df[df["date"] >= "2025-11-01"].index[0]
        warmup = max(50, trade_start)  # Increased for new indicators

        current_capital = self.capital
        self.equity_curve.append({"date": "Start", "equity": current_capital})

        for i in range(warmup, len(df) - 1):
            df_slice = df.iloc[:i + 1].copy()
            current_date = df_slice.iloc[-1]["date"]
            
            # Only generate signals for Nov-Dec 2025 period
            if current_date < pd.Timestamp("2025-11-01") or current_date > pd.Timestamp("2025-12-31"):
                self.equity_curve.append({"date": str(current_date.date()), "equity": current_capital})
                continue
                
            next_bar = df.iloc[i + 1]

            try:
                signal = self.strategy.generate_signal(df_slice)
            except Exception as e:
                logger.debug(f"Signal generation error: {e}")
                signal = None

            if signal:
                pnl, trade = self._execute_with_trailing(
                    signal, df, i, str(current_date.date())
                )
                current_capital += pnl
                if trade:
                    self.trades.append(trade)

            self.equity_curve.append({"date": str(current_date.date()), "equity": current_capital})

        trades_df = pd.DataFrame([vars(t) for t in self.trades]) if self.trades else pd.DataFrame()
        return trades_df, pd.DataFrame(self.equity_curve)

    def _execute_with_trailing(
        self, 
        signal: Dict, 
        df: pd.DataFrame, 
        bar_idx: int, 
        date: str
    ) -> Tuple[float, Optional[TradeRecord]]:
        """Execute trade with optional trailing stop."""
        next_bar = df.iloc[bar_idx + 1]
        entry = next_bar["open"]
        qty = signal["quantity"]
        sl, tp = signal["stop_loss"], signal["take_profit"]
        trailing_stop = signal.get("trailing_stop")
        
        # Track execution over multiple bars if trailing stop is used
        exit_price, reason = next_bar["close"], "EOD"
        high, low = next_bar["high"], next_bar["low"]
        holding_bars = 1
        max_favorable, max_adverse = 0.0, 0.0
        
        is_long = signal["signal"] == "BUY"
        
        if is_long:
            max_favorable = max(0, high - entry)
            max_adverse = max(0, entry - low)
            
            if low <= sl:
                exit_price, reason = sl, "SL"
            elif high >= tp:
                exit_price, reason = tp, "TP"
            elif self.config.use_trailing_stop and trailing_stop:
                # Update trailing stop if price moved favorably
                if high > entry + (entry - trailing_stop):
                    trailing_stop = high - (entry - sl)  # Trail by original stop distance
                if low <= trailing_stop:
                    exit_price, reason = trailing_stop, "TSL"
            
            pnl = (exit_price - entry) * qty
        else:
            max_favorable = max(0, entry - low)
            max_adverse = max(0, high - entry)
            
            if high >= sl:
                exit_price, reason = sl, "SL"
            elif low <= tp:
                exit_price, reason = tp, "TP"
            elif self.config.use_trailing_stop and trailing_stop:
                if low < entry - (trailing_stop - entry):
                    trailing_stop = low + (sl - entry)
                if high >= trailing_stop:
                    exit_price, reason = trailing_stop, "TSL"
            
            pnl = (entry - exit_price) * qty
        
        cost = entry * qty * (self.config.commission + self.config.slippage) * 2
        pnl -= cost
        
        trade = TradeRecord(
            date=date,
            signal_type=signal["signal"],
            entry=entry,
            exit=exit_price,
            qty=qty,
            pnl=round(pnl, 2),
            reason=reason,
            confidence=signal["confidence"],
            edge=signal["edge"],
            regime=signal["regime"],
            adx=signal.get("adx", 0),
            risk_pct=signal.get("risk_pct", 0.02),
            holding_bars=holding_bars,
            max_favorable=round(max_favorable, 2),
            max_adverse=round(max_adverse, 2)
        )
        
        return pnl, trade

    def _execute(self, signal: Dict, entry: float, high: float, low: float, close: float, date: str) -> float:
        """Legacy execution method for compatibility."""
        qty = signal["quantity"]
        sl, tp = signal["stop_loss"], signal["take_profit"]
        exit_price, reason = close, "EOD"

        if signal["signal"] == "BUY":
            if low <= sl:
                exit_price, reason = sl, "SL"
            elif high >= tp:
                exit_price, reason = tp, "TP"
            pnl = (exit_price - entry) * qty
        else:
            if high >= sl:
                exit_price, reason = sl, "SL"
            elif low <= tp:
                exit_price, reason = tp, "TP"
            pnl = (entry - exit_price) * qty

        cost = entry * qty * (self.config.commission + self.config.slippage) * 2
        pnl -= cost

        return pnl

    def calculate_metrics(self, trades_df: pd.DataFrame, equity_df: pd.DataFrame) -> Dict[str, str]:
        """Calculate comprehensive backtest metrics."""
        if trades_df.empty:
            return {"Status": "No trades"}

        total = len(trades_df)
        wins = (trades_df["pnl"] > 0).sum()
        losses = total - wins
        win_rate = wins / total

        equity_df["returns"] = equity_df["equity"].pct_change()
        total_return = equity_df["equity"].iloc[-1] / self.capital - 1
        days = len(equity_df)
        ann_return = (1 + total_return) ** (252 / days) - 1

        std = equity_df["returns"].std()
        sharpe = (equity_df["returns"].mean() / std * np.sqrt(252)) if std > 0 else 0

        running_max = equity_df["equity"].cummax()
        drawdown = equity_df["equity"] / running_max - 1
        max_dd = abs(drawdown.min())
        calmar = ann_return / max_dd if max_dd > 0 else 0

        # Enhanced metrics
        gross_profit = trades_df[trades_df["pnl"] > 0]["pnl"].sum()
        gross_loss = abs(trades_df[trades_df["pnl"] < 0]["pnl"].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        avg_win = trades_df[trades_df["pnl"] > 0]["pnl"].mean() if wins > 0 else 0
        avg_loss = abs(trades_df[trades_df["pnl"] < 0]["pnl"].mean()) if losses > 0 else 0
        win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else float('inf')
        
        # Expectancy
        expectancy = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)
        
        # Kelly Criterion
        if avg_loss > 0:
            kelly = (win_rate * win_loss_ratio - (1 - win_rate)) / win_loss_ratio
            kelly = max(0, min(0.25, kelly / 2))  # Half-Kelly, capped at 25%
        else:
            kelly = 0.25
        
        # Holding period stats
        if "holding_bars" in trades_df.columns:
            avg_hold = trades_df["holding_bars"].mean()
        else:
            avg_hold = 1.0

        return {
            "Total Trades": str(total),
            "Win Rate": f"{win_rate * 100:.2f}%",
            "Total Return": f"{total_return * 100:.2f}%",
            "Annualized Return": f"{ann_return * 100:.2f}%",
            "Sharpe Ratio": f"{sharpe:.2f}",
            "Max Drawdown": f"{max_dd * 100:.2f}%",
            "Calmar Ratio": f"{calmar:.2f}",
            "Profit Factor": f"{profit_factor:.2f}",
            "Avg Win": f"INR {avg_win:,.2f}",
            "Avg Loss": f"INR {avg_loss:,.2f}",
            "Win/Loss Ratio": f"{win_loss_ratio:.2f}",
            "Expectancy": f"INR {expectancy:,.2f}",
            "Kelly Fraction": f"{kelly * 100:.1f}%",
            "Avg Holding": f"{avg_hold:.1f} bars",
            "Final Equity": f"INR {equity_df['equity'].iloc[-1]:,.2f}"
        }
    
    def sensitivity_analysis(
        self, 
        param_ranges: Optional[Dict] = None
    ) -> pd.DataFrame:
        """
        Run sensitivity analysis on key parameters.
        Tests strategy robustness across parameter ranges.
        """
        if param_ranges is None:
            param_ranges = {
                "pullback_pct": [0.12, 0.15, 0.18, 0.20],
                "stop_atr": [1.2, 1.5, 1.8, 2.0],
                "target_atr": [2.5, 3.0, 3.5, 4.0]
            }
        
        results = []
        
        # Test pullback depth variations
        for pb in param_ranges.get("pullback_pct", [0.15]):
            self.strategy.trend.cfg.pullback_pct = pb
            trades_df, equity_df = self.run()
            metrics = self.calculate_metrics(trades_df, equity_df)
            results.append({
                "param": "pullback_pct",
                "value": pb,
                "trades": int(metrics.get("Total Trades", 0)),
                "win_rate": metrics.get("Win Rate", "0%"),
                "return": metrics.get("Total Return", "0%"),
                "sharpe": metrics.get("Sharpe Ratio", "0")
            })
        
        return pd.DataFrame(results)


def _export_trade_log(trades_df: pd.DataFrame) -> None:
    """Export trades to a standardized format for transparency."""
    if trades_df.empty:
        return
    
    # Map column names if needed
    pnl_col = "pnl" if "pnl" in trades_df.columns else "pnl"
    type_col = "signal_type" if "signal_type" in trades_df.columns else "type"
    entry_col = "entry" if "entry" in trades_df.columns else "entry"
    exit_col = "exit" if "exit" in trades_df.columns else "exit"
    
    trade_log = pd.DataFrame({
        "entry_date": trades_df["date"],
        "exit_date": trades_df["date"],
        "side": trades_df[type_col].map({"BUY": "LONG", "SELL": "SHORT"}),
        "entry_price": trades_df[entry_col].round(2),
        "exit_price": trades_df[exit_col].round(2),
        "pnl": trades_df[pnl_col].round(2),
        "holding_days": trades_df.get("holding_bars", 1),
        "reason_exit": trades_df["reason"],
        "confidence": trades_df.get("confidence", 0.5).round(3),
        "regime": trades_df.get("regime", "UNKNOWN")
    })
    
    os.makedirs("reports", exist_ok=True)
    trade_log.to_csv("reports/trades.csv", index=False)


def main():
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("reports", exist_ok=True)
    
    data_path = str(settings.data.data_path)
    bt = Backtester(data_path)
    trades, equity = bt.run()
    metrics = bt.calculate_metrics(trades, equity)

    print("\n" + "=" * 60)
    print("BACKTEST RESULTS")
    print("=" * 60)
    for k, v in metrics.items():
        print(f"  {k:20s}: {v}")
    print("=" * 60)

    trades.to_csv("data/processed/trades.csv", index=False)
    equity.to_csv("data/processed/equity.csv", index=False)
    
    _export_trade_log(trades)


if __name__ == "__main__":
    main()

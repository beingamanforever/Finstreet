import os
import sys
import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.strategy.strategy import Strategy

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class Config:
    capital: float = 100_000.0
    commission: float = 0.0003
    slippage: float = 0.0005
    max_pos_pct: float = 0.10


class Backtester:
    def __init__(self, data_path: str, capital: float = 100_000.0, config: Optional[Config] = None):
        self.data_path = data_path
        self.capital = capital
        self.config = config or Config(capital=capital)
        self.strategy = Strategy(capital=capital)
        self.trades: List[Dict] = []
        self.equity_curve: List[Dict] = []

    def run(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        df = pd.read_csv(self.data_path)
        warmup = max(30, int(len(df) * 0.5))

        current_capital = self.capital
        self.equity_curve.append({"date": "Start", "equity": current_capital})

        for i in range(warmup, len(df) - 1):
            df_slice = df.iloc[:i + 1].copy()
            current_date = df_slice.iloc[-1]["date"]
            next_bar = df.iloc[i + 1]

            try:
                signal = self.strategy.generate_signal(df_slice)
            except Exception:
                signal = None

            if signal:
                pnl = self._execute(
                    signal, next_bar["open"], next_bar["high"],
                    next_bar["low"], next_bar["close"], current_date
                )
                current_capital += pnl

            self.equity_curve.append({"date": current_date, "equity": current_capital})

        return pd.DataFrame(self.trades), pd.DataFrame(self.equity_curve)

    def _execute(self, signal: Dict, entry: float, high: float, low: float, close: float, date: str) -> float:
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

        self.trades.append({
            "date": date, "type": signal["signal"], "entry": entry, "exit": exit_price,
            "qty": qty, "pnl": round(pnl, 2), "reason": reason,
            "confidence": signal["confidence"], "edge": signal["edge"], "regime": signal["regime"]
        })

        return pnl

    def calculate_metrics(self, trades_df: pd.DataFrame, equity_df: pd.DataFrame) -> Dict[str, str]:
        if trades_df.empty:
            return {"Status": "No trades"}

        total = len(trades_df)
        wins = (trades_df["pnl"] > 0).sum()
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

        return {
            "Total Trades": str(total),
            "Win Rate": f"{win_rate * 100:.2f}%",
            "Total Return": f"{total_return * 100:.2f}%",
            "Annualized Return": f"{ann_return * 100:.2f}%",
            "Sharpe Ratio": f"{sharpe:.2f}",
            "Max Drawdown": f"{max_dd * 100:.2f}%",
            "Calmar Ratio": f"{calmar:.2f}",
            "Final Equity": f"INR {equity_df['equity'].iloc[-1]:,.2f}"
        }


def _export_trade_log(trades_df: pd.DataFrame) -> None:
    """
    Export trades to a standardized format for transparency.
    
    Output columns:
        entry_date, exit_date, side, entry_price, exit_price,
        pnl, holding_days, reason_exit
    """
    if trades_df.empty:
        return
        
    trade_log = pd.DataFrame({
        "entry_date": trades_df["date"],
        "exit_date": trades_df["date"],
        "side": trades_df["type"].map({"BUY": "LONG", "SELL": "SHORT"}),
        "entry_price": trades_df["entry"].round(2),
        "exit_price": trades_df["exit"].round(2),
        "pnl": trades_df["pnl"].round(2),
        "holding_days": 1,
        "reason_exit": trades_df["reason"]
    })
    
    trade_log.to_csv("reports/trades.csv", index=False)


def main():
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("reports", exist_ok=True)
    bt = Backtester("data/raw/NSE_SONATSOFTW-EQ.csv")
    trades, equity = bt.run()
    metrics = bt.calculate_metrics(trades, equity)

    print("\n" + "=" * 50)
    print("BACKTEST RESULTS")
    print("=" * 50)
    for k, v in metrics.items():
        print(f"  {k:20s}: {v}")
    print("=" * 50)

    trades.to_csv("data/processed/trades.csv", index=False)
    equity.to_csv("data/processed/equity.csv", index=False)
    
    _export_trade_log(trades)


if __name__ == "__main__":
    main()

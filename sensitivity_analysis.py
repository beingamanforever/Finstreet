"""Parameter Sensitivity Analysis."""

import os
import sys
import logging
from typing import Dict, List
from dataclasses import dataclass

import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.strategy.trend_momentum import TrendMomentum, Config as TMConfig
from src.strategy.strategy import Strategy
from src.backtest.backtest import Backtester, Config as BTConfig
from config.settings import settings

logging.basicConfig(level=logging.WARNING, format='%(message)s')


@dataclass
class SensitivityResult:
    param_name: str
    param_value: float
    trades: int
    win_rate: float
    total_return: float
    sharpe: float
    drawdown: float
    profit_factor: float


class ParameterizedBacktester(Backtester):
    """Backtester with parameterized strategy for sensitivity analysis."""
    
    def __init__(self, data_path: str, capital: float, tm_config: TMConfig):
        super().__init__(data_path, capital)
        # Override the strategy's TrendMomentum config
        self.strategy.trend.cfg = tm_config


def run_sensitivity_analysis() -> pd.DataFrame:
    """Run parameter sensitivity analysis on key strategy parameters."""
    
    results: List[SensitivityResult] = []
    
    # Base configuration (optimal values)
    base_config = {
        'pullback_pct': 0.06,
        'stop_atr': 1.5,
        'target_atr': 3.0,
        'min_risk_pct': 0.03,
        'max_risk_pct': 0.05,
        'adx_min_trade': 15.0,
    }
    
    # Parameter ranges to test
    sensitivity_params = {
        'pullback_pct': [0.04, 0.05, 0.06, 0.07, 0.08, 0.10],
        'stop_atr': [1.2, 1.5, 1.8, 2.0],
        'target_atr': [2.5, 3.0, 3.5, 4.0, 5.0],
        'adx_min_trade': [12.0, 15.0, 18.0, 20.0],
        'max_risk_pct': [0.03, 0.04, 0.05, 0.06]
    }
    
    print("=" * 70)
    print("PARAMETER SENSITIVITY ANALYSIS")
    print("=" * 70)
    print()
    
    for param_name, values in sensitivity_params.items():
        print(f"Testing: {param_name}")
        print("-" * 50)
        
        for value in values:
            # Create config with modified parameter
            test_config = base_config.copy()
            test_config[param_name] = value
            
            tm_config = TMConfig(
                ema=10,
                sma=20,
                pullback_pct=test_config['pullback_pct'],
                pullback_max=0.35,
                stop_atr=test_config['stop_atr'],
                target_atr=test_config['target_atr'],
                base_risk_pct=0.04,
                min_risk_pct=test_config['min_risk_pct'],
                max_risk_pct=test_config['max_risk_pct'],
                adx_min_trade=test_config['adx_min_trade'],
                adx_strong=25.0,
                volume_confirm=False,
                use_weekly_filter=False,
                use_trailing_stop=True,
                trailing_atr_mult=2.5
            )
            
            try:
                data_path = str(settings.data.data_path)
                backtester = ParameterizedBacktester(data_path, 100000, tm_config)
                
                # Debug: Verify config was applied
                actual_pullback = backtester.strategy.trend.cfg.pullback_pct
                
                trades_df, equity_df = backtester.run()
                
                metrics = backtester.calculate_metrics(trades_df, equity_df)
                
                # Parse string metrics into numeric values
                def parse_pct(s: str) -> float:
                    return float(s.replace('%', '').replace(',', '')) / 100 if '%' in s else 0.0
                
                def parse_num(s: str) -> float:
                    s = s.replace(',', '').replace('INR ', '')
                    try:
                        return float(s)
                    except:
                        return 0.0
                
                result = SensitivityResult(
                    param_name=param_name,
                    param_value=value,
                    trades=int(metrics.get('Total Trades', '0')),
                    win_rate=parse_pct(metrics.get('Win Rate', '0%')),
                    total_return=parse_pct(metrics.get('Total Return', '0%')),
                    sharpe=parse_num(metrics.get('Sharpe Ratio', '0')),
                    drawdown=parse_pct(metrics.get('Max Drawdown', '0%')),
                    profit_factor=parse_num(metrics.get('Profit Factor', '0'))
                )
                results.append(result)
                
                marker = " [SELECTED]" if value == base_config[param_name] else ""
                print(f"  {value:>6} | Trades: {result.trades:>2} | Win: {result.win_rate:>5.1%} | "
                      f"Return: {result.total_return:>5.2%} | Sharpe: {result.sharpe:>5.2f}{marker}")
                
            except Exception as e:
                print(f"  {value:>6} | ERROR: {str(e)[:40]}")
        
        print()
    
    # Convert to DataFrame
    df = pd.DataFrame([vars(r) for r in results])
    
    # Save results
    os.makedirs("reports", exist_ok=True)
    df.to_csv("reports/sensitivity_analysis.csv", index=False)
    
    # Print summary
    print("=" * 70)
    print("ROBUSTNESS SUMMARY")
    print("=" * 70)
    
    for param in sensitivity_params.keys():
        param_results = df[df['param_name'] == param]
        if len(param_results) > 0:
            avg_sharpe = param_results['sharpe'].mean()
            std_sharpe = param_results['sharpe'].std()
            print(f"{param:>15}: Avg Sharpe = {avg_sharpe:.2f} (std = {std_sharpe:.2f})")
    
    print()
    print(f"Results saved to: reports/sensitivity_analysis.csv")
    
    return df


if __name__ == "__main__":
    run_sensitivity_analysis()

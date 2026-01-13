"""
Centralized configuration management for the trading system.
"""

from dataclasses import dataclass, field
from typing import List, Optional
from pathlib import Path
import os


@dataclass
class DataConfig:
    """Configuration for data acquisition and storage."""
    raw_data_dir: Path = Path("data/raw")
    processed_data_dir: Path = Path("data/processed")
    default_symbol: str = "NSE:SONATSOFTW-EQ"
    default_resolution: str = "D"
    lookback_days: int = 365


@dataclass
class FeatureConfig:
    """Configuration for feature engineering."""
    rsi_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    bollinger_period: int = 20
    bollinger_std: float = 2.0
    atr_period: int = 14
    volume_ma_period: int = 20
    momentum_periods: List[int] = field(default_factory=lambda: [5, 10, 20])


@dataclass
class ModelConfig:
    """Configuration for model training and inference."""
    model_dir: Path = Path("models")
    test_size: float = 0.2
    random_state: int = 42
    n_estimators: int = 100
    max_depth: Optional[int] = None
    min_samples_split: int = 2
    feature_importance_threshold: float = 0.01


@dataclass
class BacktestConfig:
    """Configuration for backtesting."""
    initial_capital: float = 100000.0
    commission_pct: float = 0.001
    slippage_pct: float = 0.0005
    position_size_pct: float = 0.1
    max_positions: int = 5
    stop_loss_pct: float = 0.02
    take_profit_pct: float = 0.04


@dataclass
class ExecutionConfig:
    """Configuration for trade execution."""
    paper_trading: bool = True
    max_order_size: int = 100
    order_timeout_seconds: int = 30
    retry_attempts: int = 3
    retry_delay_seconds: int = 5


@dataclass
class VisualizationConfig:
    """Configuration for visualization and reporting."""
    output_dir: Path = Path("reports/figures")
    figure_dpi: int = 100
    figure_format: str = "png"
    color_scheme: str = "seaborn-v0_8-whitegrid"


@dataclass
class Settings:
    """Main settings container aggregating all configurations."""
    data: DataConfig = field(default_factory=DataConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)

    def __post_init__(self):
        """Create necessary directories after initialization."""
        self._ensure_directories()

    def _ensure_directories(self) -> None:
        """Ensure all required directories exist."""
        directories = [
            self.data.raw_data_dir,
            self.data.processed_data_dir,
            self.model.model_dir,
            self.visualization.output_dir,
        ]
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)


class ConfigLoader:
    """Utility class for loading and managing configurations."""

    _instance: Optional[Settings] = None

    @classmethod
    def get_settings(cls) -> Settings:
        """Get singleton settings instance."""
        if cls._instance is None:
            cls._instance = Settings()
        return cls._instance

    @classmethod
    def reload(cls) -> Settings:
        """Force reload of settings."""
        cls._instance = Settings()
        return cls._instance

    @classmethod
    def get_env(cls, key: str, default: str = "") -> str:
        """Get environment variable with default."""
        return os.environ.get(key, default)


settings = ConfigLoader.get_settings()

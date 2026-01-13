"""
Logging configuration for the trading system.
Provides structured logging with file and console handlers.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional


class LoggerFactory:
    """Factory for creating configured logger instances."""

    _loggers: dict = {}
    _initialized: bool = False
    _log_dir: Path = Path("logs")

    @classmethod
    def _ensure_log_dir(cls) -> None:
        """Ensure log directory exists."""
        cls._log_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def _get_formatter(cls) -> logging.Formatter:
        """Get standard log formatter."""
        return logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

    @classmethod
    def _setup_root_logger(cls, level: int = logging.INFO) -> None:
        """Configure the root logger with handlers."""
        if cls._initialized:
            return

        cls._ensure_log_dir()

        root_logger = logging.getLogger()
        root_logger.setLevel(level)

        if root_logger.handlers:
            root_logger.handlers.clear()

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(cls._get_formatter())
        root_logger.addHandler(console_handler)

        log_filename = cls._log_dir / f"trading_{datetime.now().strftime('%Y%m%d')}.log"
        file_handler = logging.FileHandler(log_filename, mode="a", encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(cls._get_formatter())
        root_logger.addHandler(file_handler)

        cls._initialized = True

    @classmethod
    def get_logger(
        cls,
        name: str,
        level: Optional[int] = None
    ) -> logging.Logger:
        """
        Get or create a logger instance.

        Args:
            name: Logger name, typically __name__
            level: Optional logging level override

        Returns:
            Configured logger instance
        """
        if not cls._initialized:
            cls._setup_root_logger()

        if name in cls._loggers:
            return cls._loggers[name]

        logger = logging.getLogger(name)

        if level is not None:
            logger.setLevel(level)

        cls._loggers[name] = logger
        return logger

    @classmethod
    def set_level(cls, level: int) -> None:
        """Set logging level for all loggers."""
        logging.getLogger().setLevel(level)
        for logger in cls._loggers.values():
            logger.setLevel(level)


def get_logger(name: str) -> logging.Logger:
    """Convenience function to get a logger."""
    return LoggerFactory.get_logger(name)


class TradeLogger:
    """Specialized logger for trade events."""

    def __init__(self):
        self._logger = get_logger("trades")
        self._trade_log_path = Path("logs") / "trades.log"
        self._trade_log_path.parent.mkdir(parents=True, exist_ok=True)

    def log_order(
        self,
        symbol: str,
        side: str,
        quantity: int,
        price: float,
        order_id: Optional[str] = None
    ) -> None:
        """Log order placement."""
        self._logger.info(
            f"ORDER | {symbol} | {side} | qty={quantity} | price={price:.2f} | id={order_id}"
        )

    def log_fill(
        self,
        symbol: str,
        side: str,
        quantity: int,
        fill_price: float,
        order_id: Optional[str] = None
    ) -> None:
        """Log order fill."""
        self._logger.info(
            f"FILL | {symbol} | {side} | qty={quantity} | fill_price={fill_price:.2f} | id={order_id}"
        )

    def log_position(
        self,
        symbol: str,
        quantity: int,
        avg_price: float,
        unrealized_pnl: float
    ) -> None:
        """Log position update."""
        self._logger.info(
            f"POSITION | {symbol} | qty={quantity} | avg_price={avg_price:.2f} | unrealized_pnl={unrealized_pnl:.2f}"
        )

    def log_signal(
        self,
        symbol: str,
        direction: int,
        confidence: float,
        strategy: str
    ) -> None:
        """Log trading signal."""
        direction_str = {1: "LONG", -1: "SHORT", 0: "NEUTRAL"}.get(direction, "UNKNOWN")
        self._logger.info(
            f"SIGNAL | {symbol} | {direction_str} | confidence={confidence:.2f} | strategy={strategy}"
        )


trade_logger = TradeLogger()

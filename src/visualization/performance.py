import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from typing import Optional, Tuple
from pathlib import Path


class PerformanceVisualizer:

    def __init__(self, output_dir: str = "reports/figures"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._setup_style()

    def _setup_style(self) -> None:
        plt.style.use("seaborn-v0_8-whitegrid")
        plt.rcParams.update({
            "figure.figsize": (12, 6),
            "figure.dpi": 100,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "lines.linewidth": 1.5,
        })

    def plot_equity_curve(
        self,
        equity: pd.Series,
        benchmark: Optional[pd.Series] = None,
        title: str = "Portfolio Equity Curve",
        save: bool = True
    ) -> plt.Figure:
        fig, ax = plt.subplots(figsize=(14, 7))

        ax.plot(equity.index, equity.values, label="Strategy", color="#2E86AB")

        if benchmark is not None:
            normalized_benchmark = benchmark / benchmark.iloc[0] * equity.iloc[0]
            ax.plot(
                normalized_benchmark.index,
                normalized_benchmark.values,
                label="Benchmark",
                color="#A23B72",
                linestyle="--",
                alpha=0.7
            )
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Date")
        ax.set_ylabel("Portfolio Value")
        ax.legend(loc="upper left")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        plt.xticks(rotation=45)
        plt.tight_layout()

        if save:
            fig.savefig(self.output_dir / "equity_curve.png", bbox_inches="tight")

        return fig

    def plot_drawdown(
        self,
        equity: pd.Series,
        title: str = "Portfolio Drawdown",
        save: bool = True
    ) -> plt.Figure:
        rolling_max = equity.expanding().max()
        drawdown = (equity - rolling_max) / rolling_max * 100

        fig, ax = plt.subplots(figsize=(14, 5))

        ax.fill_between(
            drawdown.index,
            drawdown.values,
            0,
            color="#E63946",
            alpha=0.4,
            label="Drawdown"
        )
        ax.plot(drawdown.index, drawdown.values, color="#E63946", linewidth=0.8)

        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Date")
        ax.set_ylabel("Drawdown (%)")
        ax.set_ylim(drawdown.min() * 1.1, 5)
        ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        plt.xticks(rotation=45)
        plt.tight_layout()

        if save:
            fig.savefig(self.output_dir / "drawdown.png", bbox_inches="tight")

        return fig

    def plot_monthly_returns(
        self,
        returns: pd.Series,
        title: str = "Monthly Returns Heatmap",
        save: bool = True
    ) -> plt.Figure:
        monthly_returns = returns.resample("M").apply(
            lambda x: (1 + x).prod() - 1
        ) * 100

        monthly_df = pd.DataFrame({
            "year": monthly_returns.index.year,
            "month": monthly_returns.index.month,
            "return": monthly_returns.values
        })

        pivot_table = monthly_df.pivot(index="year", columns="month", values="return")

        fig, ax = plt.subplots(figsize=(12, 6))

        cmap = plt.cm.RdYlGn
        im = ax.imshow(pivot_table.values, cmap=cmap, aspect="auto", vmin=-10, vmax=10)

        ax.set_xticks(np.arange(12))
        ax.set_xticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                           "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
        ax.set_yticks(np.arange(len(pivot_table.index)))
        ax.set_yticklabels(pivot_table.index)

        for i in range(len(pivot_table.index)):
            for j in range(12):
                if j < len(pivot_table.columns):
                    val = pivot_table.iloc[i, j]
                    if not np.isnan(val):
                        text_color = "white" if abs(val) > 5 else "black"
                        ax.text(j, i, f"{val:.1f}%", ha="center", va="center",
                               color=text_color, fontsize=9)

        ax.set_title(title, fontweight="bold")
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label("Return (%)")
        plt.tight_layout()

        if save:
            fig.savefig(self.output_dir / "monthly_returns.png", bbox_inches="tight")

        return fig

    def plot_trade_distribution(
        self,
        trades: pd.DataFrame,
        save: bool = True
    ) -> plt.Figure:
        if "pnl" not in trades.columns:
            raise ValueError("trades DataFrame must contain 'pnl' column")

        pnl = trades["pnl"].dropna()

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        colors = ["#2E86AB" if x >= 0 else "#E63946" for x in pnl]
        axes[0].hist(pnl, bins=50, color="#2E86AB", alpha=0.7, edgecolor="white")
        axes[0].axvline(x=0, color="black", linestyle="--", linewidth=1)
        axes[0].axvline(x=pnl.mean(), color="#F77F00", linestyle="-", linewidth=2,
                       label=f"Mean: {pnl.mean():.2f}")
        axes[0].set_title("Trade P&L Distribution", fontweight="bold")
        axes[0].set_xlabel("P&L")
        axes[0].set_ylabel("Frequency")
        axes[0].legend()

        win_rate = (pnl > 0).sum() / len(pnl) * 100
        loss_rate = 100 - win_rate
        axes[1].pie(
            [win_rate, loss_rate],
            labels=["Winners", "Losers"],
            colors=["#2E86AB", "#E63946"],
            autopct="%1.1f%%",
            startangle=90,
            explode=(0.02, 0.02)
        )
        axes[1].set_title("Win/Loss Ratio", fontweight="bold")

        plt.tight_layout()

        if save:
            fig.savefig(self.output_dir / "trade_distribution.png", bbox_inches="tight")

        return fig

    def generate_report(
        self,
        equity: pd.Series,
        trades: pd.DataFrame,
        benchmark: Optional[pd.Series] = None
    ) -> None:
        returns = equity.pct_change().dropna()
        self.plot_equity_curve(equity, benchmark)
        self.plot_drawdown(equity)
        self.plot_monthly_returns(returns)
        if "pnl" in trades.columns:
            self.plot_trade_distribution(trades)

    def calculate_metrics(self, equity: pd.Series) -> dict:
        returns = equity.pct_change().dropna()
        total_return = (equity.iloc[-1] / equity.iloc[0] - 1) * 100
        annual_return = (1 + total_return / 100) ** (252 / len(returns)) - 1
        volatility = returns.std() * np.sqrt(252) * 100
        sharpe_ratio = (annual_return * 100) / volatility if volatility != 0 else 0
        rolling_max = equity.expanding().max()
        drawdown = (equity - rolling_max) / rolling_max
        max_drawdown = drawdown.min() * 100
        return {
            "total_return": round(total_return, 2),
            "annual_return": round(annual_return * 100, 2),
            "volatility": round(volatility, 2),
            "sharpe_ratio": round(sharpe_ratio, 2),
            "max_drawdown": round(max_drawdown, 2),
            "calmar_ratio": round(annual_return * 100 / abs(max_drawdown), 2) if max_drawdown != 0 else 0
        }

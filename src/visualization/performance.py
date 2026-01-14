import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
from typing import Optional, Tuple, List
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

        ax.plot(equity.index, equity.values, label="Strategy", color="#2E86AB", linewidth=2)

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
        ax.set_ylabel("Portfolio Value (INR)")
        ax.legend(loc="upper left")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        plt.xticks(rotation=45)
        plt.tight_layout()

        if save:
            fig.savefig(self.output_dir / "equity_curve.png", bbox_inches="tight")

        return fig

    def plot_equity_and_drawdown(
        self,
        equity: pd.Series,
        title: str = "Equity Curve with Drawdown Analysis",
        save: bool = True
    ) -> plt.Figure:
        """
        Dual-panel visualization: Equity curve on top, drawdown below.
        Highlights the low drawdown as a key strength.
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), 
                                        gridspec_kw={'height_ratios': [2, 1]},
                                        sharex=True)
        
        # Panel 1: Equity Curve
        ax1.plot(equity.index, equity.values, color="#2E86AB", linewidth=2, label="Strategy Equity")
        ax1.fill_between(equity.index, equity.iloc[0], equity.values, 
                         alpha=0.3, color="#2E86AB")
        ax1.axhline(y=equity.iloc[0], color='gray', linestyle='--', alpha=0.5, label='Initial Capital')
        ax1.set_title(title, fontweight="bold", fontsize=14)
        ax1.set_ylabel("Portfolio Value (INR)")
        ax1.legend(loc="upper left")
        ax1.grid(True, alpha=0.3)
        
        # Panel 2: Drawdown
        rolling_max = equity.expanding().max()
        drawdown = (equity - rolling_max) / rolling_max * 100
        
        ax2.fill_between(drawdown.index, drawdown.values, 0, 
                         color="#E63946", alpha=0.5, label="Drawdown")
        ax2.plot(drawdown.index, drawdown.values, color="#E63946", linewidth=1)
        ax2.axhline(y=0, color='black', linewidth=0.5)
        ax2.set_ylabel("Drawdown (%)")
        ax2.set_xlabel("Date")
        ax2.set_ylim(min(drawdown.min() * 1.2, -0.5), 1)
        
        # Annotate max drawdown
        max_dd_idx = drawdown.idxmin()
        max_dd_val = drawdown.min()
        ax2.annotate(
            f'Max DD: {max_dd_val:.2f}%',
            xy=(max_dd_idx, max_dd_val),
            xytext=(max_dd_idx, max_dd_val - 0.3),
            fontsize=10,
            color='#E63946',
            fontweight='bold'
        )
        
        ax2.legend(loc="lower left")
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            fig.savefig(self.output_dir / "equity_drawdown_dual.png", bbox_inches="tight", dpi=120)
        
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

    def plot_trades_on_price(
        self,
        price_df: pd.DataFrame,
        trades_df: pd.DataFrame,
        title: str = "Price Chart with Trade Entries",
        save: bool = True,
        annotate_count: int = 3
    ) -> plt.Figure:
        """
        Plot price chart with trade entry/exit markers.
        Green triangles for buys, red triangles for sells.
        Annotates top performing trades.
        """
        fig, ax = plt.subplots(figsize=(16, 8))
        
        # Price line
        dates = pd.to_datetime(price_df["date"])
        ax.plot(dates, price_df["close"], color="#333333", linewidth=1.5, label="Close Price")
        
        # Moving averages for context
        if "SMA_20" in price_df.columns:
            ax.plot(dates, price_df["SMA_20"], color="#2E86AB", linewidth=1, 
                    alpha=0.7, linestyle="--", label="SMA(20)")
        if "EMA_10" in price_df.columns:
            ax.plot(dates, price_df["EMA_10"], color="#F77F00", linewidth=1, 
                    alpha=0.7, linestyle="--", label="EMA(10)")
        
        if not trades_df.empty:
            # Map trades to dates
            trade_dates = pd.to_datetime(trades_df["date"])
            
            # Determine trade type column
            type_col = "signal_type" if "signal_type" in trades_df.columns else "type"
            entry_col = "entry" if "entry" in trades_df.columns else "entry_price"
            
            # Plot buy signals
            buys = trades_df[trades_df[type_col] == "BUY"]
            if not buys.empty:
                buy_dates = pd.to_datetime(buys["date"])
                ax.scatter(buy_dates, buys[entry_col], marker="^", s=150, 
                          color="#2E86AB", edgecolors="black", linewidths=1,
                          label="BUY", zorder=5)
            
            # Plot sell signals
            sells = trades_df[trades_df[type_col] == "SELL"]
            if not sells.empty:
                sell_dates = pd.to_datetime(sells["date"])
                ax.scatter(sell_dates, sells[entry_col], marker="v", s=150,
                          color="#E63946", edgecolors="black", linewidths=1,
                          label="SELL", zorder=5)
            
            # Annotate top trades
            if "pnl" in trades_df.columns:
                top_trades = trades_df.nlargest(annotate_count, "pnl")
                for _, trade in top_trades.iterrows():
                    trade_date = pd.to_datetime(trade["date"])
                    ax.annotate(
                        f'+{trade["pnl"]:.0f}',
                        xy=(trade_date, trade[entry_col]),
                        xytext=(10, 20),
                        textcoords="offset points",
                        fontsize=9,
                        color="#2E86AB",
                        fontweight="bold",
                        arrowprops=dict(arrowstyle="->", color="#2E86AB", alpha=0.7)
                    )
        
        ax.set_title(title, fontweight="bold", fontsize=14)
        ax.set_xlabel("Date")
        ax.set_ylabel("Price (INR)")
        ax.legend(loc="upper left")
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            fig.savefig(self.output_dir / "price_with_trades.png", bbox_inches="tight", dpi=120)
        
        return fig

    def plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        title: str = "ML Filter Confusion Matrix",
        save: bool = True
    ) -> plt.Figure:
        """
        Plot confusion matrix for the ML filter.
        Shows TP, TN, FP, FN rates.
        """
        from sklearn.metrics import confusion_matrix
        
        cm = confusion_matrix(y_true, y_pred)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Create heatmap
        im = ax.imshow(cm, cmap='Blues')
        
        # Labels
        labels = ['Predicted Negative', 'Predicted Positive']
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(labels)
        ax.set_yticklabels(['Actual Negative', 'Actual Positive'])
        
        # Add values to cells
        for i in range(2):
            for j in range(2):
                total = cm.sum()
                value = cm[i, j]
                pct = value / total * 100
                text = f'{value}\n({pct:.1f}%)'
                color = 'white' if value > total / 4 else 'black'
                ax.text(j, i, text, ha='center', va='center', 
                       fontsize=14, color=color, fontweight='bold')
        
        # Labels for metrics
        tn, fp, fn, tp = cm.ravel()
        total = tn + fp + fn + tp
        
        metrics_text = (
            f"Accuracy: {(tp+tn)/total:.1%}\n"
            f"Precision: {tp/(tp+fp):.1%}\n"
            f"Recall: {tp/(tp+fn):.1%}\n"
            f"F1 Score: {2*tp/(2*tp+fp+fn):.1%}"
        )
        
        ax.text(1.35, 0.5, metrics_text, transform=ax.transAxes, 
               fontsize=11, verticalalignment='center',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        ax.set_title(title, fontweight="bold", fontsize=14)
        plt.colorbar(im, ax=ax, shrink=0.8)
        plt.tight_layout()
        
        if save:
            fig.savefig(self.output_dir / "confusion_matrix.png", bbox_inches="tight", dpi=120)
        
        return fig

    def plot_monthly_returns(
        self,
        returns: pd.Series,
        title: str = "Monthly Returns Heatmap",
        save: bool = True
    ) -> plt.Figure:
        monthly_returns = returns.resample("ME").apply(
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
        pnl_col = "pnl" if "pnl" in trades.columns else "pnl"
        if pnl_col not in trades.columns:
            raise ValueError("trades DataFrame must contain 'pnl' column")

        pnl = trades[pnl_col].dropna()

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        colors = ["#2E86AB" if x >= 0 else "#E63946" for x in pnl]
        axes[0].hist(pnl, bins=50, color="#2E86AB", alpha=0.7, edgecolor="white")
        axes[0].axvline(x=0, color="black", linestyle="--", linewidth=1)
        axes[0].axvline(x=pnl.mean(), color="#F77F00", linestyle="-", linewidth=2,
                       label=f"Mean: {pnl.mean():.2f}")
        axes[0].set_title("Trade P&L Distribution", fontweight="bold")
        axes[0].set_xlabel("P&L (INR)")
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

    def plot_regime_performance(
        self,
        trades: pd.DataFrame,
        title: str = "Performance by Market Regime",
        save: bool = True
    ) -> plt.Figure:
        """Analyze and visualize performance across different market regimes."""
        if "regime" not in trades.columns or trades.empty:
            return None
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Trades by regime
        regime_counts = trades["regime"].value_counts()
        colors = plt.cm.Set2(np.linspace(0, 1, len(regime_counts)))
        axes[0].bar(regime_counts.index, regime_counts.values, color=colors)
        axes[0].set_title("Trade Count by Regime", fontweight="bold")
        axes[0].set_xlabel("Market Regime")
        axes[0].set_ylabel("Number of Trades")
        axes[0].tick_params(axis='x', rotation=45)
        
        # Win rate by regime
        pnl_col = "pnl" if "pnl" in trades.columns else "pnl"
        regime_winrate = trades.groupby("regime")[pnl_col].apply(
            lambda x: (x > 0).mean() * 100
        )
        axes[1].bar(regime_winrate.index, regime_winrate.values, color=colors)
        axes[1].axhline(y=50, color='gray', linestyle='--', alpha=0.7, label='50% baseline')
        axes[1].set_title("Win Rate by Regime", fontweight="bold")
        axes[1].set_xlabel("Market Regime")
        axes[1].set_ylabel("Win Rate (%)")
        axes[1].tick_params(axis='x', rotation=45)
        axes[1].legend()
        
        plt.tight_layout()
        
        if save:
            fig.savefig(self.output_dir / "regime_performance.png", bbox_inches="tight")
        
        return fig

    def plot_confusion_matrix_from_metrics(self, save: bool = True) -> Optional[plt.Figure]:
        """Generate confusion matrix from saved model metrics."""
        metrics_path = Path("models/model_metrics.csv")
        if not metrics_path.exists():
            return None
            
        metrics_df = pd.read_csv(metrics_path)
        
        tp = int(metrics_df["true_positives"].iloc[-1])
        tn = int(metrics_df["true_negatives"].iloc[-1])
        fp = int(metrics_df["false_positives"].iloc[-1])
        fn = int(metrics_df["false_negatives"].iloc[-1])
        
        cm = np.array([[tn, fp], [fn, tp]])
        
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(cm, cmap="Blues")
        
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Predicted DOWN", "Predicted UP"], fontsize=11)
        ax.set_yticklabels(["Actual DOWN", "Actual UP"], fontsize=11)
        
        for i in range(2):
            for j in range(2):
                total = max(cm.sum(), 1)
                value = cm[i, j]
                pct = value / total * 100
                text = f"{value}\n({pct:.1f}%)"
                color = "white" if value > total / 4 else "black"
                ax.text(j, i, text, ha="center", va="center", fontsize=14, color=color, fontweight="bold")
        
        total = max(tp + tn + fp + fn, 1)
        acc = (tp + tn) / total
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * tp / max(2 * tp + fp + fn, 1)
        
        metrics_text = f"Accuracy: {acc:.1%}\nPrecision: {prec:.1%}\nRecall: {rec:.1%}\nF1 Score: {f1:.1%}"
        ax.text(1.35, 0.5, metrics_text, transform=ax.transAxes, fontsize=11, verticalalignment="center",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
        
        ax.set_title("Model Confusion Matrix", fontweight="bold", fontsize=14)
        plt.colorbar(im, ax=ax, shrink=0.8)
        plt.tight_layout()
        
        if save:
            fig.savefig(self.output_dir / "confusion_matrix.png", bbox_inches="tight", dpi=120)
        
        plt.close()
        return fig

    def plot_feature_importance(self, save: bool = True) -> Optional[plt.Figure]:
        """Generate feature importance chart from saved metrics."""
        importance_path = Path("models/feature_importance.csv")
        if not importance_path.exists():
            return None
            
        importance_df = pd.read_csv(importance_path)
        top_features = importance_df.nlargest(15, "importance")
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(top_features)))[::-1]
        ax.barh(range(len(top_features)), top_features["importance"].values, color=colors)
        
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features["feature"].values, fontsize=10)
        ax.set_xlabel("Importance Score", fontsize=11)
        ax.set_title("Top 15 Feature Importance", fontweight="bold", fontsize=14)
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            fig.savefig(self.output_dir / "feature_importance.png", bbox_inches="tight", dpi=120)
        
        plt.close()
        return fig

    def generate_report(
        self,
        equity: pd.Series,
        trades: pd.DataFrame,
        benchmark: Optional[pd.Series] = None,
        price_df: Optional[pd.DataFrame] = None
    ) -> None:
        """Generate comprehensive performance report with all visualizations."""
        returns = equity.pct_change().dropna()
        
        # Core plots
        self.plot_equity_curve(equity, benchmark)
        self.plot_drawdown(equity)
        self.plot_equity_and_drawdown(equity)
        self.plot_monthly_returns(returns)
        
        if "pnl" in trades.columns:
            self.plot_trade_distribution(trades)
        
        if "regime" in trades.columns:
            self.plot_regime_performance(trades)
        
        # Price chart with trades if data available
        if price_df is not None and not trades.empty:
            self.plot_trades_on_price(price_df, trades)
        
        # Model performance visualizations
        self.plot_confusion_matrix_from_metrics()
        self.plot_feature_importance()

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

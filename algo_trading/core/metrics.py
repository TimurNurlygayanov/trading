"""
Comprehensive metrics calculator for backtesting results.

Includes:
- Risk-adjusted returns (Sharpe, Sortino, Calmar, Omega)
- Drawdown metrics (Max DD, Avg DD, Duration)
- Trade metrics (Win rate, Profit factor, Expectancy)
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class BacktestResults:
    """Container for backtest results."""

    equity_curve: pd.Series
    returns: pd.Series
    trades: pd.DataFrame
    metrics: Dict[str, float]

    def summary(self) -> str:
        """Generate a text summary of results."""
        lines = [
            f"{'='*50}",
            "BACKTEST RESULTS SUMMARY",
            f"{'='*50}",
            "",
            "Returns:",
            f"  Total Return:      {self.metrics.get('total_return', 0):.2%}",
            f"  Annualized Return: {self.metrics.get('annualized_return', 0):.2%}",
            f"  Volatility:        {self.metrics.get('volatility', 0):.2%}",
            "",
            "Risk-Adjusted:",
            f"  Sharpe Ratio:      {self.metrics.get('sharpe_ratio', 0):.2f}",
            f"  Sortino Ratio:     {self.metrics.get('sortino_ratio', 0):.2f}",
            f"  Calmar Ratio:      {self.metrics.get('calmar_ratio', 0):.2f}",
            "",
            "Drawdown:",
            f"  Max Drawdown:      {self.metrics.get('max_drawdown', 0):.2%}",
            f"  Avg Drawdown:      {self.metrics.get('avg_drawdown', 0):.2%}",
            f"  Max DD Duration:   {self.metrics.get('max_drawdown_duration', 0):.0f} bars",
            "",
            "Trades:",
            f"  Number of Trades:  {self.metrics.get('num_trades', 0):.0f}",
            f"  Win Rate:          {self.metrics.get('win_rate', 0):.2%}",
            f"  Profit Factor:     {self.metrics.get('profit_factor', 0):.2f}",
            f"  Expectancy:        {self.metrics.get('expectancy', 0):.4f}",
            f"{'='*50}",
        ]
        return "\n".join(lines)


class MetricsCalculator:
    """
    Calculator for all trading performance metrics.

    Metrics calculated:
    - Returns: total, annualized, volatility
    - Risk-adjusted: Sharpe, Sortino, Calmar, Information, Omega
    - Drawdown: max, average, duration
    - Trade: win rate, profit factor, avg win/loss, expectancy
    """

    RISK_FREE_RATE = 0.02  # 2% annual
    TRADING_DAYS = 252

    @staticmethod
    def calculate_all(
        equity_curve: pd.Series,
        returns: pd.Series,
        trades: pd.DataFrame,
        commission_rate: float = 0.0001
    ) -> Dict[str, float]:
        """
        Calculate all performance metrics.

        Args:
            equity_curve: Portfolio value over time
            returns: Period returns
            trades: DataFrame with trade information (must have 'pnl' column)
            commission_rate: Commission rate (for reference)

        Returns:
            Dictionary of metric names to values
        """
        metrics = {}

        # Returns metrics
        metrics['total_return'] = MetricsCalculator._total_return(equity_curve)
        metrics['annualized_return'] = MetricsCalculator._annualize_return(
            metrics['total_return'], len(returns)
        )
        metrics['volatility'] = MetricsCalculator._volatility(returns)
        metrics['downside_volatility'] = MetricsCalculator._downside_volatility(returns)

        # Risk-adjusted metrics
        metrics['sharpe_ratio'] = MetricsCalculator._sharpe_ratio(returns)
        metrics['sortino_ratio'] = MetricsCalculator._sortino_ratio(returns)
        metrics['calmar_ratio'] = MetricsCalculator._calmar_ratio(
            metrics['annualized_return'], equity_curve
        )
        metrics['omega_ratio'] = MetricsCalculator._omega_ratio(returns)

        # Drawdown metrics
        dd_metrics = MetricsCalculator._calculate_drawdowns(equity_curve)
        metrics['max_drawdown'] = dd_metrics['max_drawdown']
        metrics['avg_drawdown'] = dd_metrics['avg_drawdown']
        metrics['max_drawdown_duration'] = dd_metrics['max_duration_bars']

        # Trade metrics
        if len(trades) > 0 and 'pnl' in trades.columns:
            trade_metrics = MetricsCalculator._calculate_trade_metrics(trades)
            metrics.update(trade_metrics)
        else:
            metrics.update({
                'win_rate': 0.0,
                'profit_factor': 0.0,
                'avg_win': 0.0,
                'avg_loss': 0.0,
                'expectancy': 0.0,
                'num_trades': 0,
                'avg_trade_duration': 0,
            })

        return metrics

    @staticmethod
    def _total_return(equity_curve: pd.Series) -> float:
        """Calculate total return."""
        if len(equity_curve) < 2 or equity_curve.iloc[0] == 0:
            return 0.0
        return (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1

    @staticmethod
    def _annualize_return(total_return: float, periods: int) -> float:
        """Annualize total return."""
        if periods == 0:
            return 0.0
        years = periods / MetricsCalculator.TRADING_DAYS
        if years <= 0:
            return 0.0
        return (1 + total_return) ** (1 / years) - 1

    @staticmethod
    def _volatility(returns: pd.Series) -> float:
        """Calculate annualized volatility."""
        if len(returns) < 2:
            return 0.0
        return returns.std() * np.sqrt(MetricsCalculator.TRADING_DAYS)

    @staticmethod
    def _downside_volatility(returns: pd.Series) -> float:
        """Calculate annualized downside volatility."""
        negative_returns = returns[returns < 0]
        if len(negative_returns) < 2:
            return 0.0
        return negative_returns.std() * np.sqrt(MetricsCalculator.TRADING_DAYS)

    @staticmethod
    def _sharpe_ratio(returns: pd.Series) -> float:
        """
        Calculate Sharpe Ratio.

        Sharpe = (R - Rf) / sigma
        Where:
            R = mean return
            Rf = risk-free rate
            sigma = standard deviation of returns
        """
        if len(returns) < 2 or returns.std() == 0:
            return 0.0

        daily_rf = MetricsCalculator.RISK_FREE_RATE / MetricsCalculator.TRADING_DAYS
        excess_return = returns.mean() - daily_rf

        return (excess_return / returns.std()) * np.sqrt(MetricsCalculator.TRADING_DAYS)

    @staticmethod
    def _sortino_ratio(returns: pd.Series) -> float:
        """
        Calculate Sortino Ratio.

        Sortino = (R - Rf) / sigma_downside
        Only considers downside volatility.
        """
        if len(returns) < 2:
            return 0.0

        negative_returns = returns[returns < 0]
        if len(negative_returns) < 2 or negative_returns.std() == 0:
            return 0.0

        daily_rf = MetricsCalculator.RISK_FREE_RATE / MetricsCalculator.TRADING_DAYS
        excess_return = returns.mean() - daily_rf

        return (excess_return / negative_returns.std()) * np.sqrt(MetricsCalculator.TRADING_DAYS)

    @staticmethod
    def _calmar_ratio(annualized_return: float, equity_curve: pd.Series) -> float:
        """
        Calculate Calmar Ratio.

        Calmar = Annualized Return / |Max Drawdown|
        """
        dd = MetricsCalculator._calculate_drawdowns(equity_curve)
        max_dd = abs(dd['max_drawdown'])

        if max_dd == 0:
            return 0.0

        return annualized_return / max_dd

    @staticmethod
    def _omega_ratio(returns: pd.Series, threshold: float = 0.0) -> float:
        """
        Calculate Omega Ratio.

        Omega = Probability-weighted gains / Probability-weighted losses
        """
        if len(returns) < 2:
            return 0.0

        gains = returns[returns > threshold] - threshold
        losses = threshold - returns[returns <= threshold]

        if losses.sum() == 0:
            return float('inf') if gains.sum() > 0 else 0.0

        return gains.sum() / losses.sum()

    @staticmethod
    def _calculate_drawdowns(equity_curve: pd.Series) -> Dict[str, float]:
        """
        Calculate drawdown metrics.

        Returns:
            Dictionary with max_drawdown, avg_drawdown, max_duration_bars
        """
        if len(equity_curve) < 2:
            return {'max_drawdown': 0.0, 'avg_drawdown': 0.0, 'max_duration_bars': 0}

        # Calculate running maximum (peak)
        peak = equity_curve.expanding().max()

        # Calculate drawdown as percentage from peak
        drawdown = (equity_curve - peak) / peak

        # Max drawdown
        max_dd = drawdown.min()

        # Average drawdown (only when in drawdown)
        in_drawdown = drawdown[drawdown < 0]
        avg_dd = in_drawdown.mean() if len(in_drawdown) > 0 else 0.0

        # Max drawdown duration
        max_duration = MetricsCalculator._max_drawdown_duration(drawdown)

        return {
            'max_drawdown': max_dd,
            'avg_drawdown': avg_dd,
            'max_duration_bars': max_duration
        }

    @staticmethod
    def _max_drawdown_duration(drawdown: pd.Series) -> int:
        """Calculate maximum drawdown duration in bars."""
        if len(drawdown) == 0:
            return 0

        is_in_dd = drawdown < 0

        if not is_in_dd.any():
            return 0

        # Group consecutive drawdown periods
        groups = (~is_in_dd).cumsum()
        durations = is_in_dd.groupby(groups).cumsum()

        return int(durations.max()) if len(durations) > 0 else 0

    @staticmethod
    def _calculate_trade_metrics(trades: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate trade-based metrics.

        Args:
            trades: DataFrame with 'pnl' column at minimum

        Returns:
            Dictionary of trade metrics
        """
        if len(trades) == 0 or 'pnl' not in trades.columns:
            return {}

        pnl = trades['pnl']
        winning = pnl[pnl > 0]
        losing = pnl[pnl < 0]

        # Win rate
        win_rate = len(winning) / len(pnl) if len(pnl) > 0 else 0.0

        # Profit factor
        gross_profit = winning.sum() if len(winning) > 0 else 0.0
        gross_loss = abs(losing.sum()) if len(losing) > 0 else 0.0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        # Average win/loss
        avg_win = winning.mean() if len(winning) > 0 else 0.0
        avg_loss = losing.mean() if len(losing) > 0 else 0.0

        # Expectancy (average PnL per trade)
        expectancy = pnl.mean() if len(pnl) > 0 else 0.0

        # Average trade duration (if entry_time and exit_time columns exist)
        avg_duration = 0
        if 'entry_time' in trades.columns and 'exit_time' in trades.columns:
            durations = (
                pd.to_datetime(trades['exit_time']) -
                pd.to_datetime(trades['entry_time'])
            )
            avg_duration = durations.mean().total_seconds() / 3600  # in hours

        return {
            'win_rate': win_rate,
            'profit_factor': profit_factor if profit_factor != float('inf') else 999.99,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'expectancy': expectancy,
            'num_trades': len(trades),
            'avg_trade_duration': avg_duration,
        }

    @staticmethod
    def calculate_rolling_metrics(
        returns: pd.Series,
        window: int = 252
    ) -> pd.DataFrame:
        """
        Calculate rolling performance metrics.

        Args:
            returns: Period returns
            window: Rolling window size (default: 252 for 1 year)

        Returns:
            DataFrame with rolling Sharpe, volatility, etc.
        """
        if len(returns) < window:
            return pd.DataFrame()

        daily_rf = MetricsCalculator.RISK_FREE_RATE / MetricsCalculator.TRADING_DAYS

        rolling_df = pd.DataFrame(index=returns.index)

        # Rolling return
        rolling_df['return'] = returns.rolling(window).mean() * MetricsCalculator.TRADING_DAYS

        # Rolling volatility
        rolling_df['volatility'] = returns.rolling(window).std() * np.sqrt(MetricsCalculator.TRADING_DAYS)

        # Rolling Sharpe
        rolling_excess = returns.rolling(window).mean() - daily_rf
        rolling_std = returns.rolling(window).std()
        rolling_df['sharpe'] = (rolling_excess / rolling_std) * np.sqrt(MetricsCalculator.TRADING_DAYS)

        return rolling_df.dropna()

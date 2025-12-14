"""
Tests for the metrics calculator.
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.metrics import MetricsCalculator, BacktestResults


class TestMetricsCalculator:
    """Test suite for MetricsCalculator."""

    @pytest.fixture
    def sample_equity_curve(self):
        """Create a sample equity curve."""
        dates = pd.date_range('2023-01-01', periods=252, freq='D')
        # Simulated equity curve with growth and drawdown
        np.random.seed(42)
        returns = np.random.normal(0.0005, 0.01, 252)
        equity = 10000 * np.cumprod(1 + returns)
        return pd.Series(equity, index=dates)

    @pytest.fixture
    def sample_returns(self, sample_equity_curve):
        """Create sample returns from equity curve."""
        return sample_equity_curve.pct_change().fillna(0)

    @pytest.fixture
    def sample_trades(self):
        """Create sample trades DataFrame."""
        return pd.DataFrame({
            'entry_time': pd.date_range('2023-01-01', periods=50, freq='5D'),
            'exit_time': pd.date_range('2023-01-02', periods=50, freq='5D'),
            'entry_price': np.random.uniform(1.08, 1.12, 50),
            'exit_price': np.random.uniform(1.08, 1.12, 50),
            'position': np.random.choice([1, -1], 50),
            'pnl': np.random.normal(50, 100, 50),
        })

    def test_total_return(self, sample_equity_curve):
        """Test total return calculation."""
        total_return = MetricsCalculator._total_return(sample_equity_curve)
        expected = (sample_equity_curve.iloc[-1] / sample_equity_curve.iloc[0]) - 1
        assert abs(total_return - expected) < 1e-10

    def test_sharpe_ratio(self, sample_returns):
        """Test Sharpe ratio calculation."""
        sharpe = MetricsCalculator._sharpe_ratio(sample_returns)
        assert isinstance(sharpe, float)
        # Sharpe should be reasonable for random returns
        assert -5 < sharpe < 5

    def test_sortino_ratio(self, sample_returns):
        """Test Sortino ratio calculation."""
        sortino = MetricsCalculator._sortino_ratio(sample_returns)
        assert isinstance(sortino, float)

    def test_calmar_ratio(self, sample_equity_curve):
        """Test Calmar ratio calculation."""
        ann_return = 0.10  # 10% annual return
        calmar = MetricsCalculator._calmar_ratio(ann_return, sample_equity_curve)
        assert isinstance(calmar, float)

    def test_max_drawdown(self, sample_equity_curve):
        """Test max drawdown calculation."""
        dd = MetricsCalculator._calculate_drawdowns(sample_equity_curve)
        assert 'max_drawdown' in dd
        assert dd['max_drawdown'] <= 0  # Drawdown is negative
        assert dd['max_drawdown'] >= -1  # Can't lose more than 100%

    def test_trade_metrics(self, sample_trades):
        """Test trade metrics calculation."""
        metrics = MetricsCalculator._calculate_trade_metrics(sample_trades)

        assert 'win_rate' in metrics
        assert 0 <= metrics['win_rate'] <= 1

        assert 'profit_factor' in metrics
        assert metrics['profit_factor'] >= 0

        assert 'num_trades' in metrics
        assert metrics['num_trades'] == 50

    def test_calculate_all(self, sample_equity_curve, sample_returns, sample_trades):
        """Test full metrics calculation."""
        metrics = MetricsCalculator.calculate_all(
            sample_equity_curve,
            sample_returns,
            sample_trades
        )

        # Check all expected metrics are present
        expected_keys = [
            'total_return', 'annualized_return', 'volatility',
            'sharpe_ratio', 'sortino_ratio', 'calmar_ratio',
            'max_drawdown', 'win_rate', 'profit_factor', 'num_trades'
        ]

        for key in expected_keys:
            assert key in metrics, f"Missing metric: {key}"

    def test_empty_trades(self, sample_equity_curve, sample_returns):
        """Test metrics with no trades."""
        empty_trades = pd.DataFrame(columns=['pnl'])
        metrics = MetricsCalculator.calculate_all(
            sample_equity_curve,
            sample_returns,
            empty_trades
        )

        assert metrics['num_trades'] == 0
        assert metrics['win_rate'] == 0

    def test_rolling_metrics(self, sample_returns):
        """Test rolling metrics calculation."""
        rolling = MetricsCalculator.calculate_rolling_metrics(
            sample_returns,
            window=60
        )

        assert 'return' in rolling.columns
        assert 'volatility' in rolling.columns
        assert 'sharpe' in rolling.columns


class TestBacktestResults:
    """Test BacktestResults dataclass."""

    def test_summary(self):
        """Test results summary generation."""
        equity = pd.Series([10000, 10100, 10200], index=pd.date_range('2023-01-01', periods=3))
        returns = equity.pct_change().fillna(0)
        trades = pd.DataFrame({'pnl': [100, 100]})
        metrics = {'total_return': 0.02, 'sharpe_ratio': 1.5, 'max_drawdown': -0.05}

        result = BacktestResults(
            equity_curve=equity,
            returns=returns,
            trades=trades,
            metrics=metrics
        )

        summary = result.summary()
        assert 'BACKTEST RESULTS' in summary
        assert 'Total Return' in summary


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

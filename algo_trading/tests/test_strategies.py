"""
Tests for trading strategies.
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from strategies.base_strategy import Strategy
from strategies.mean_reversion.strategy import MeanReversionStrategy
from strategies.momentum.strategy import MomentumStrategy
from data.downloaders.forex_downloader import ForexDownloader


class TestBaseStrategy:
    """Test suite for base Strategy class."""

    def test_cannot_instantiate_abstract(self):
        """Test that Strategy cannot be instantiated directly."""
        with pytest.raises(TypeError):
            Strategy({})

    def test_strategy_interface(self):
        """Test that strategies implement required interface."""
        strategy = MeanReversionStrategy()

        # Check required methods exist
        assert hasattr(strategy, 'generate_signals')
        assert hasattr(strategy, 'get_position_size')
        assert hasattr(strategy, 'required_history')
        assert hasattr(strategy, 'preprocess_data')


class TestMeanReversionStrategy:
    """Test suite for MeanReversionStrategy."""

    @pytest.fixture
    def strategy(self):
        """Create strategy instance."""
        return MeanReversionStrategy()

    @pytest.fixture
    def sample_data(self):
        """Create sample OHLCV data."""
        downloader = ForexDownloader()
        return downloader.generate_sample_data(
            symbol='EURUSD',
            timeframe='1h',
            start_date='2023-01-01',
            end_date='2023-06-01'
        )

    def test_initialization(self, strategy):
        """Test strategy initialization."""
        assert strategy.name == 'MeanReversionStrategy'
        assert 'bb_period' in strategy.params
        assert 'rsi_period' in strategy.params

    def test_required_history(self, strategy):
        """Test required history property."""
        assert strategy.required_history > 0
        assert isinstance(strategy.required_history, int)

    def test_preprocess_data(self, strategy, sample_data):
        """Test data preprocessing."""
        processed = strategy.preprocess_data(sample_data)

        # Check indicators were added
        assert 'bb_upper' in processed.columns
        assert 'bb_lower' in processed.columns
        assert 'bb_middle' in processed.columns
        assert 'rsi' in processed.columns

    def test_generate_signals(self, strategy, sample_data):
        """Test signal generation."""
        processed = strategy.preprocess_data(sample_data)
        signals = strategy.generate_signals(processed)

        # Check signals are valid
        assert isinstance(signals, pd.Series)
        assert len(signals) == len(sample_data)
        assert signals.isin([-1, 0, 1]).all()

    def test_position_size(self, strategy):
        """Test risk-based position sizing."""
        # With $100k account and 2% risk ($2k), 15 pip stop:
        # lots = $2000 / (15 pips * $10/pip) = 13.33 lots
        # units = 13.33 * 100,000 = 1,333,333
        # notional = 1,333,333 * 1.10 = $1,466,666
        portfolio = 100_000
        price = 1.10
        size = strategy.get_position_size(1, portfolio, price)
        assert size > 0

        # Check size is within leverage limits (30x = $3M max notional)
        max_notional = portfolio * 30
        assert size <= max_notional

        # No position for hold signal
        size_hold = strategy.get_position_size(0, portfolio, price)
        assert size_hold == 0

    def test_custom_params(self):
        """Test strategy with custom parameters."""
        custom_params = {
            'bb_period': 30,
            'rsi_period': 21,
            'rsi_oversold': 25,
        }
        strategy = MeanReversionStrategy(params=custom_params)

        assert strategy.params['bb_period'] == 30
        assert strategy.params['rsi_period'] == 21
        assert strategy.params['rsi_oversold'] == 25


class TestMomentumStrategy:
    """Test suite for MomentumStrategy."""

    @pytest.fixture
    def strategy(self):
        """Create strategy instance."""
        return MomentumStrategy()

    @pytest.fixture
    def sample_data(self):
        """Create sample OHLCV data."""
        downloader = ForexDownloader()
        return downloader.generate_sample_data(
            symbol='EURUSD',
            timeframe='1h',
            start_date='2023-01-01',
            end_date='2023-06-01'
        )

    def test_initialization(self, strategy):
        """Test strategy initialization."""
        assert strategy.name == 'MomentumStrategy'
        assert 'fast_ema' in strategy.params
        assert 'slow_ema' in strategy.params
        assert 'adx_threshold' in strategy.params

    def test_preprocess_data(self, strategy, sample_data):
        """Test data preprocessing."""
        processed = strategy.preprocess_data(sample_data)

        # Check indicators
        assert 'ema_fast' in processed.columns
        assert 'ema_slow' in processed.columns
        assert 'macd' in processed.columns
        assert 'adx' in processed.columns
        assert 'atr' in processed.columns

    def test_generate_signals(self, strategy, sample_data):
        """Test signal generation."""
        processed = strategy.preprocess_data(sample_data)
        signals = strategy.generate_signals(processed)

        assert isinstance(signals, pd.Series)
        assert len(signals) == len(sample_data)
        assert signals.isin([-1, 0, 1]).all()

    def test_stop_loss_calculation(self, strategy):
        """Test stop loss calculation."""
        entry = 1.1000
        atr = 0.0050

        # Long position stop loss
        stop_long = strategy.get_stop_loss(entry, atr, 1)
        assert stop_long < entry

        # Short position stop loss
        stop_short = strategy.get_stop_loss(entry, atr, -1)
        assert stop_short > entry


class TestDataDownloader:
    """Test suite for ForexDownloader."""

    @pytest.fixture
    def downloader(self):
        """Create downloader instance."""
        return ForexDownloader()

    def test_generate_sample_data(self, downloader):
        """Test synthetic data generation."""
        data = downloader.generate_sample_data(
            symbol='EURUSD',
            timeframe='1h',
            start_date='2023-01-01',
            end_date='2023-02-01'
        )

        # Check structure
        assert isinstance(data, pd.DataFrame)
        assert 'open' in data.columns
        assert 'high' in data.columns
        assert 'low' in data.columns
        assert 'close' in data.columns
        assert 'volume' in data.columns

        # Check OHLC consistency
        assert (data['high'] >= data['low']).all()
        assert (data['high'] >= data['open']).all()
        assert (data['high'] >= data['close']).all()
        assert (data['low'] <= data['open']).all()
        assert (data['low'] <= data['close']).all()

    def test_available_symbols(self, downloader):
        """Test available symbols list."""
        symbols = downloader.get_available_symbols()

        assert isinstance(symbols, list)
        assert len(symbols) > 0
        assert 'EURUSD' in symbols


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

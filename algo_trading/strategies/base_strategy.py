"""
Base strategy class that all trading strategies must inherit from.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import pandas as pd


class Strategy(ABC):
    """
    Abstract base class for all trading strategies.

    All strategies must implement:
    - generate_signals: Generate trading signals from data
    - get_position_size: Determine position sizing
    - required_history: Minimum bars needed for signal generation
    """

    def __init__(self, params: Dict[str, Any]):
        """
        Initialize strategy with parameters.

        Args:
            params: Dictionary of strategy-specific parameters
        """
        self.params = params
        self.name = self.__class__.__name__

    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate trading signals from market data.

        Args:
            data: OHLCV DataFrame with columns ['open', 'high', 'low', 'close', 'volume']
                  Index should be DatetimeIndex

        Returns:
            pd.Series with signal values:
                1: Buy signal
                -1: Sell signal
                0: Hold/No position
        """
        pass

    @abstractmethod
    def get_position_size(
        self,
        signal: int,
        portfolio_value: float,
        current_price: float
    ) -> float:
        """
        Determine the position size for a given signal.

        Args:
            signal: Trading signal (1, -1, or 0)
            portfolio_value: Current portfolio value
            current_price: Current market price

        Returns:
            Position size in base currency units
        """
        pass

    @property
    @abstractmethod
    def required_history(self) -> int:
        """
        Minimum number of bars required to generate a signal.

        Returns:
            Number of historical bars needed
        """
        pass

    def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess data before signal generation.

        Override this method to add indicators, normalize data, etc.

        Args:
            data: Raw OHLCV data

        Returns:
            Preprocessed DataFrame
        """
        return data

    def validate_data(self, data: pd.DataFrame) -> bool:
        """
        Validate input data has required columns and format.

        Args:
            data: Input DataFrame

        Returns:
            True if data is valid

        Raises:
            ValueError: If data is invalid
        """
        required_columns = ['open', 'high', 'low', 'close']
        missing = [col for col in required_columns if col not in data.columns]

        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("DataFrame index must be DatetimeIndex")

        if len(data) < self.required_history:
            raise ValueError(
                f"Insufficient data: need {self.required_history} bars, "
                f"got {len(data)}"
            )

        return True

    def get_params(self) -> Dict[str, Any]:
        """Get strategy parameters."""
        return self.params.copy()

    def set_params(self, **params) -> None:
        """Update strategy parameters."""
        self.params.update(params)

    def __repr__(self) -> str:
        return f"{self.name}(params={self.params})"

"""
Abstract base class for data downloaders.
"""
from abc import ABC, abstractmethod
from typing import Optional, List
import pandas as pd
from pathlib import Path


class DataDownloader(ABC):
    """
    Abstract base class for all data downloaders.

    All downloaders must implement:
    - download: Fetch data from source
    - get_available_symbols: List available symbols
    """

    def __init__(self, cache_dir: Optional[Path] = None):
        """
        Initialize downloader.

        Args:
            cache_dir: Directory for caching downloaded data
        """
        self.cache_dir = cache_dir
        if cache_dir:
            cache_dir.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def download(
        self,
        symbol: str,
        timeframe: str,
        start_date: str,
        end_date: str,
        **kwargs
    ) -> pd.DataFrame:
        """
        Download historical data.

        Args:
            symbol: Trading symbol (e.g., 'EURUSD')
            timeframe: Data timeframe (e.g., '5min', '1h', '1d')
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            **kwargs: Additional downloader-specific arguments

        Returns:
            DataFrame with columns: open, high, low, close, volume
            Index: DatetimeIndex
        """
        pass

    @abstractmethod
    def get_available_symbols(self) -> List[str]:
        """
        Get list of available symbols.

        Returns:
            List of symbol strings
        """
        pass

    def _get_cache_path(
        self,
        symbol: str,
        timeframe: str,
        start_date: str,
        end_date: str
    ) -> Optional[Path]:
        """Get cache file path for given parameters."""
        if not self.cache_dir:
            return None

        filename = f"{symbol}_{timeframe}_{start_date}_{end_date}.parquet"
        return self.cache_dir / filename

    def _load_from_cache(
        self,
        symbol: str,
        timeframe: str,
        start_date: str,
        end_date: str
    ) -> Optional[pd.DataFrame]:
        """Load data from cache if available."""
        cache_path = self._get_cache_path(symbol, timeframe, start_date, end_date)

        if cache_path and cache_path.exists():
            return pd.read_parquet(cache_path)

        return None

    def _save_to_cache(
        self,
        data: pd.DataFrame,
        symbol: str,
        timeframe: str,
        start_date: str,
        end_date: str
    ) -> None:
        """Save data to cache."""
        cache_path = self._get_cache_path(symbol, timeframe, start_date, end_date)

        if cache_path:
            data.to_parquet(cache_path)

    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names to lowercase."""
        df.columns = df.columns.str.lower()

        # Ensure required columns exist
        required = ['open', 'high', 'low', 'close']
        missing = [col for col in required if col not in df.columns]

        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        # Add volume if missing
        if 'volume' not in df.columns:
            df['volume'] = 0

        return df[['open', 'high', 'low', 'close', 'volume']]

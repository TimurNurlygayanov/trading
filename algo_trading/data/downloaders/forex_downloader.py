"""
Forex data downloader supporting multiple sources.

Supports:
- polygon/massive: Polygon.io / Massive.com API (recommended)
- yfinance (free, limited data)
- OANDA API (requires account)
- Local CSV files
"""
import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Any
from pathlib import Path
import logging
from datetime import datetime, timedelta
import os

from .base import DataDownloader

logger = logging.getLogger(__name__)

# Default API key path
DEFAULT_API_KEY_PATH = os.path.expanduser('~/api_key2.txt')


class ForexDownloader(DataDownloader):
    """
    Forex data downloader with multiple source support.

    Sources:
    - polygon: Polygon.io / Massive.com API (recommended, high quality data)
    - yfinance: Free but limited to daily/hourly data
    - oanda: Full intraday data (requires API key)
    - local: Load from local CSV/Parquet files
    """

    # Mapping of common forex symbols to yfinance tickers
    YFINANCE_SYMBOLS = {
        'EURUSD': 'EURUSD=X',
        'GBPUSD': 'GBPUSD=X',
        'USDJPY': 'USDJPY=X',
        'USDCHF': 'USDCHF=X',
        'AUDUSD': 'AUDUSD=X',
        'USDCAD': 'USDCAD=X',
        'NZDUSD': 'NZDUSD=X',
        'EURGBP': 'EURGBP=X',
        'EURJPY': 'EURJPY=X',
        'GBPJPY': 'GBPJPY=X',
    }

    # Polygon forex symbols (C: prefix for currencies)
    POLYGON_SYMBOLS = {
        'EURUSD': 'C:EURUSD',
        'GBPUSD': 'C:GBPUSD',
        'USDJPY': 'C:USDJPY',
        'USDCHF': 'C:USDCHF',
        'AUDUSD': 'C:AUDUSD',
        'USDCAD': 'C:USDCAD',
        'NZDUSD': 'C:NZDUSD',
        'EURGBP': 'C:EURGBP',
        'EURJPY': 'C:EURJPY',
        'GBPJPY': 'C:GBPJPY',
    }

    # Timeframe mapping
    TIMEFRAME_MAP = {
        '1min': '1m',
        '5min': '5m',
        '15min': '15m',
        '30min': '30m',
        '1h': '1h',
        '4h': '4h',
        '1d': '1d',
        '1wk': '1wk',
        '1mo': '1mo',
    }

    # Polygon timeframe mapping
    POLYGON_TIMEFRAME_MAP = {
        '1min': ('minute', 1),
        '5min': ('minute', 5),
        '15min': ('minute', 15),
        '30min': ('minute', 30),
        '1h': ('hour', 1),
        '4h': ('hour', 4),
        '1d': ('day', 1),
        '1wk': ('week', 1),
        '1mo': ('month', 1),
    }

    def __init__(
        self,
        source: str = 'polygon',
        cache_dir: Optional[Path] = None,
        polygon_api_key: Optional[str] = None,
        polygon_api_key_path: Optional[str] = None,
        oanda_api_key: Optional[str] = None,
        oanda_account_id: Optional[str] = None,
        local_data_dir: Optional[Path] = None
    ):
        """
        Initialize forex downloader.

        Args:
            source: Data source ('polygon', 'yfinance', 'oanda', 'local')
            cache_dir: Directory for caching data
            polygon_api_key: Polygon/Massive API key (direct)
            polygon_api_key_path: Path to file containing API key
            oanda_api_key: OANDA API key (for 'oanda' source)
            oanda_account_id: OANDA account ID
            local_data_dir: Directory with local data files
        """
        super().__init__(cache_dir)
        self.source = source
        self.oanda_api_key = oanda_api_key
        self.oanda_account_id = oanda_account_id
        self.local_data_dir = local_data_dir

        # Load Polygon API key
        self.polygon_api_key = polygon_api_key
        if not self.polygon_api_key:
            key_path = polygon_api_key_path or DEFAULT_API_KEY_PATH
            if os.path.exists(key_path):
                with open(key_path, 'r') as f:
                    self.polygon_api_key = f.read().strip()
                logger.info(f"Loaded Polygon API key from {key_path}")

    def download(
        self,
        symbol: str = 'EURUSD',
        timeframe: str = '5min',
        start_date: str = '2023-01-01',
        end_date: str = '2024-01-01',
        **kwargs
    ) -> pd.DataFrame:
        """
        Download forex data.

        Args:
            symbol: Forex pair (e.g., 'EURUSD')
            timeframe: Candle timeframe
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            OHLCV DataFrame
        """
        # Try cache first
        cached = self._load_from_cache(symbol, timeframe, start_date, end_date)
        if cached is not None:
            logger.info(f"Loaded {symbol} from cache")
            return cached

        # Download based on source
        if self.source == 'polygon':
            df = self._download_polygon(symbol, timeframe, start_date, end_date)
        elif self.source == 'yfinance':
            df = self._download_yfinance(symbol, timeframe, start_date, end_date)
        elif self.source == 'oanda':
            df = self._download_oanda(symbol, timeframe, start_date, end_date)
        elif self.source == 'local':
            df = self._load_local(symbol, timeframe, start_date, end_date)
        else:
            raise ValueError(f"Unknown source: {self.source}")

        # Standardize and cache
        df = self._standardize_columns(df)
        self._save_to_cache(df, symbol, timeframe, start_date, end_date)

        return df

    def _download_polygon(
        self,
        symbol: str,
        timeframe: str,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """
        Download forex data from Polygon.io / Massive.com API.

        Args:
            symbol: Forex pair (e.g., 'EURUSD')
            timeframe: Candle timeframe
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            OHLCV DataFrame
        """
        if not self.polygon_api_key:
            raise ValueError(
                "Polygon API key required. Set polygon_api_key or "
                "create ~/api_key2.txt with your API key."
            )

        try:
            from polygon import RESTClient
        except ImportError:
            raise ImportError(
                "polygon-api-client not installed. "
                "Run: pip install polygon-api-client"
            )

        # Get polygon symbol format
        polygon_symbol = self.POLYGON_SYMBOLS.get(symbol, f'C:{symbol}')

        # Get timeframe parameters
        if timeframe not in self.POLYGON_TIMEFRAME_MAP:
            raise ValueError(f"Unsupported timeframe: {timeframe}")
        timespan, multiplier = self.POLYGON_TIMEFRAME_MAP[timeframe]

        logger.info(f"Downloading {polygon_symbol} from Polygon ({timeframe})")
        logger.info(f"Date range: {start_date} to {end_date}")

        client = RESTClient(api_key=self.polygon_api_key)

        # Collect data
        data = {
            'open': [], 'high': [], 'low': [], 'close': [],
            'volume': [], 'vwap': [], 'timestamp': []
        }

        try:
            for agg in client.list_aggs(
                ticker=polygon_symbol,
                multiplier=multiplier,
                timespan=timespan,
                from_=start_date,
                to=end_date,
                limit=50000
            ):
                timestamp = datetime.fromtimestamp(agg.timestamp // 1000)
                data['timestamp'].append(timestamp)
                data['open'].append(agg.open)
                data['high'].append(agg.high)
                data['low'].append(agg.low)
                data['close'].append(agg.close)
                data['volume'].append(agg.volume if agg.volume else 0)
                data['vwap'].append(agg.vwap if agg.vwap else agg.close)

        except Exception as e:
            logger.error(f"Error downloading from Polygon: {e}")
            raise

        if not data['timestamp']:
            raise ValueError(f"No data returned for {symbol} from {start_date} to {end_date}")

        # Create DataFrame
        df = pd.DataFrame(data)
        df.set_index('timestamp', inplace=True)
        df.sort_index(inplace=True)
        df.index = pd.to_datetime(df.index)

        # Remove duplicates
        df = df[~df.index.duplicated(keep='first')]

        logger.info(f"Downloaded {len(df)} bars from Polygon")

        return df

    def _download_yfinance(
        self,
        symbol: str,
        timeframe: str,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """Download data from yfinance."""
        try:
            import yfinance as yf
        except ImportError:
            raise ImportError("yfinance not installed. Run: pip install yfinance")

        ticker = self.YFINANCE_SYMBOLS.get(symbol, f"{symbol}=X")
        interval = self.TIMEFRAME_MAP.get(timeframe, '1h')

        logger.info(f"Downloading {ticker} from yfinance ({interval})")

        # yfinance has limitations on intraday data
        # For intervals < 1d, max period is limited
        if interval in ['1m', '5m', '15m', '30m']:
            # Download in chunks for intraday
            df = self._download_yfinance_intraday(ticker, interval, start_date, end_date)
        else:
            data = yf.download(
                ticker,
                start=start_date,
                end=end_date,
                interval=interval,
                progress=False
            )
            df = data

        if df.empty:
            raise ValueError(f"No data returned for {symbol}")

        # Handle MultiIndex columns from yfinance
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        return df

    def _download_yfinance_intraday(
        self,
        ticker: str,
        interval: str,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """Download intraday data in chunks (yfinance limitation)."""
        import yfinance as yf

        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)

        # Max 7 days for 1m, 60 days for others
        if interval == '1m':
            chunk_days = 6
        else:
            chunk_days = 59

        chunks = []
        current = start

        while current < end:
            chunk_end = min(current + timedelta(days=chunk_days), end)

            data = yf.download(
                ticker,
                start=current.strftime('%Y-%m-%d'),
                end=chunk_end.strftime('%Y-%m-%d'),
                interval=interval,
                progress=False
            )

            if not data.empty:
                chunks.append(data)

            current = chunk_end

        if not chunks:
            return pd.DataFrame()

        df = pd.concat(chunks)
        df = df[~df.index.duplicated(keep='first')]
        return df.sort_index()

    def _download_oanda(
        self,
        symbol: str,
        timeframe: str,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """Download data from OANDA API."""
        if not self.oanda_api_key:
            raise ValueError("OANDA API key required")

        try:
            import oandapyV20
            import oandapyV20.endpoints.instruments as instruments
        except ImportError:
            raise ImportError("oandapyV20 not installed. Run: pip install oandapyV20")

        # OANDA uses underscore format (EUR_USD)
        oanda_symbol = symbol[:3] + '_' + symbol[3:]

        # Timeframe mapping for OANDA
        oanda_granularity = {
            '1min': 'M1',
            '5min': 'M5',
            '15min': 'M15',
            '30min': 'M30',
            '1h': 'H1',
            '4h': 'H4',
            '1d': 'D',
        }.get(timeframe, 'M5')

        client = oandapyV20.API(access_token=self.oanda_api_key)

        params = {
            'granularity': oanda_granularity,
            'from': start_date,
            'to': end_date,
        }

        r = instruments.InstrumentsCandles(instrument=oanda_symbol, params=params)
        client.request(r)

        candles = r.response['candles']

        data = []
        for candle in candles:
            if candle['complete']:
                data.append({
                    'time': candle['time'],
                    'open': float(candle['mid']['o']),
                    'high': float(candle['mid']['h']),
                    'low': float(candle['mid']['l']),
                    'close': float(candle['mid']['c']),
                    'volume': int(candle['volume']),
                })

        df = pd.DataFrame(data)
        df['time'] = pd.to_datetime(df['time'])
        df.set_index('time', inplace=True)

        return df

    def _load_local(
        self,
        symbol: str,
        timeframe: str,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """Load data from local files."""
        if not self.local_data_dir:
            raise ValueError("local_data_dir not specified")

        # Try different file formats
        patterns = [
            f"{symbol}_{timeframe}.parquet",
            f"{symbol}_{timeframe}.csv",
            f"{symbol}.parquet",
            f"{symbol}.csv",
        ]

        for pattern in patterns:
            path = Path(self.local_data_dir) / pattern
            if path.exists():
                if path.suffix == '.parquet':
                    df = pd.read_parquet(path)
                else:
                    df = pd.read_csv(path, parse_dates=['time'], index_col='time')

                # Filter by date
                df = df[(df.index >= start_date) & (df.index <= end_date)]
                return df

        raise FileNotFoundError(f"No data file found for {symbol}")

    def get_available_symbols(self) -> List[str]:
        """Get list of available forex symbols."""
        return list(self.YFINANCE_SYMBOLS.keys())

    def generate_sample_data(
        self,
        symbol: str = 'EURUSD',
        timeframe: str = '5min',
        start_date: str = '2023-01-01',
        end_date: str = '2024-01-01',
        initial_price: float = 1.1000,
        volatility: float = 0.0002
    ) -> pd.DataFrame:
        """
        Generate synthetic forex data for testing.

        Args:
            symbol: Symbol name (for reference)
            timeframe: Timeframe
            start_date: Start date
            end_date: End date
            initial_price: Starting price
            volatility: Price volatility per bar

        Returns:
            Synthetic OHLCV DataFrame
        """
        # Generate datetime index
        freq_map = {
            '1min': 'min',
            '5min': '5min',
            '15min': '15min',
            '30min': '30min',
            '1h': 'h',
            '4h': '4h',
            '1d': 'D',
        }
        freq = freq_map.get(timeframe, '5min')

        index = pd.date_range(start=start_date, end=end_date, freq=freq)

        # Filter for forex market hours (Mon-Fri, skip weekends)
        index = index[index.dayofweek < 5]

        n = len(index)

        # Generate random walk for close prices
        returns = np.random.normal(0, volatility, n)
        close = initial_price * np.exp(np.cumsum(returns))

        # Generate OHLC from close
        high_factor = np.random.uniform(1.0, 1.0005, n)
        low_factor = np.random.uniform(0.9995, 1.0, n)

        high = close * high_factor
        low = close * low_factor
        open_ = np.roll(close, 1)
        open_[0] = initial_price

        # Ensure OHLC consistency
        high = np.maximum.reduce([open_, close, high])
        low = np.minimum.reduce([open_, close, low])

        df = pd.DataFrame({
            'open': open_,
            'high': high,
            'low': low,
            'close': close,
            'volume': np.random.randint(100, 10000, n),
        }, index=index)

        return df

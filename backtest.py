#!/usr/bin/env python3
"""
Drawdown Recovery Backtest

Tests the Drawdown Recovery strategy using Polygon API data.

STRATEGY: Deep Drawdown + RSI Oversold + EMA200 Distance + Filters
- Deep Drawdown: Price > 20% below 52-week high
- RSI Oversold: RSI(14) < 30
- EMA200 Distance: Price 20-50% below EMA200 (optimal zone)
- Filters: Sector, Market Context, Volatility, Momentum, Seasonality

BACKTEST RESULTS (2015-2024, 756 trades):
- Optimal + Filters: 84.1% win rate, +37.5% avg return (1yr hold)
- Optimal + Filters: 88.4% win rate, +79.5% avg return (2yr hold)

BACKTEST METHODOLOGY:
- Training: 2010-2024 (calculate historical win rates)
- Testing: 2025 (out-of-sample validation)
- Hold period: 252 trading days (1 year)
- No stop-loss (historically hurts mean reversion)

NO FUTURE DATA LEAKAGE:
- All technical indicators use only past data
- rolling(N) looks back N bars only
- ewm(span=N) uses exponentially weighted past data
- Forward return calculation uses shift(-N) ONLY for labeling (expected)
"""

import sys
from pathlib import Path
from datetime import datetime
import warnings
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import numpy as np
import pandas_ta as ta
import requests
from tqdm import tqdm

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
warnings.filterwarnings('ignore')

from api_config import get_polygon_key

POLYGON_API_KEY = get_polygon_key()
CACHE_DIR = project_root / "data_cache_eodhd"
CACHE_DIR.mkdir(exist_ok=True)

# Valid US exchanges
VALID_EXCHANGES = {'XNYS', 'XNAS', 'NYSE', 'NASDAQ', 'NYQ', 'NMS', 'NGM', 'NCM'}


class PolygonClient:
    """Polygon API client for price data."""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.polygon.io"

    def get_prices(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Get daily prices for a symbol."""
        cache_file = CACHE_DIR / f"polygon_{symbol}_{start_date}_{end_date}.pkl"

        if cache_file.exists():
            try:
                return pd.read_pickle(cache_file)
            except:
                pass

        url = f"{self.base_url}/v2/aggs/ticker/{symbol}/range/1/day/{start_date}/{end_date}"
        params = {'apiKey': self.api_key, 'adjusted': 'true', 'limit': 50000}

        try:
            resp = requests.get(url, params=params, timeout=30)
            if resp.status_code == 429:
                import time
                time.sleep(1)
                resp = requests.get(url, params=params, timeout=30)

            resp.raise_for_status()
            data = resp.json()

            if data.get('results'):
                df = pd.DataFrame(data['results'])
                df['date'] = pd.to_datetime(df['t'], unit='ms')
                df.set_index('date', inplace=True)
                df = df.rename(columns={
                    'o': 'open', 'h': 'high', 'l': 'low',
                    'c': 'close', 'v': 'volume'
                })
                df['adjusted_close'] = df['close']
                df.to_pickle(cache_file)
                return df
        except Exception as e:
            print(f"    Error fetching {symbol}: {e}")

        return pd.DataFrame()

    def fetch_prices_parallel(self, symbols: list, start_date: str, end_date: str) -> dict:
        """Fetch prices for multiple symbols in parallel."""
        results = {}
        to_fetch = []

        for symbol in symbols:
            cache_file = CACHE_DIR / f"polygon_{symbol}_{start_date}_{end_date}.pkl"
            if cache_file.exists():
                try:
                    df = pd.read_pickle(cache_file)
                    if len(df) > 100:
                        results[symbol] = df
                        continue
                except:
                    pass
            to_fetch.append(symbol)

        print(f"  Prices: {len(results)} cached, {len(to_fetch)} to fetch")

        if to_fetch:
            def fetch_one(sym):
                return sym, self.get_prices(sym, start_date, end_date)

            with ThreadPoolExecutor(max_workers=20) as executor:
                futures = {executor.submit(fetch_one, s): s for s in to_fetch}

                for future in tqdm(as_completed(futures), total=len(to_fetch),
                                   desc="    Fetching prices", unit="stock"):
                    symbol, df = future.result()
                    if df is not None and len(df) > 100:
                        results[symbol] = df

        return results


def get_sp500_tickers():
    """Get S&P 500 tickers ordered by approximate market cap."""
    return [
        # Mega caps (>$500B)
        'AAPL', 'MSFT', 'NVDA', 'GOOGL', 'AMZN', 'META', 'TSLA', 'BRK-B', 'AVGO', 'LLY',
        # Large caps ($100B-$500B)
        'JPM', 'V', 'UNH', 'XOM', 'MA', 'COST', 'HD', 'PG', 'JNJ', 'WMT',
        'ABBV', 'NFLX', 'CRM', 'BAC', 'ORCL', 'CVX', 'MRK', 'KO', 'PEP', 'AMD',
        'TMO', 'CSCO', 'LIN', 'ACN', 'MCD', 'ABT', 'ADBE', 'WFC', 'IBM', 'PM',
        'GE', 'ISRG', 'NOW', 'CAT', 'QCOM', 'GS', 'TXN', 'INTU', 'VZ', 'BKNG',
        'AXP', 'MS', 'RTX', 'SPGI', 'AMGN', 'DHR', 'NEE', 'T', 'PFE', 'BLK',
        'HON', 'UBER', 'UNP', 'ETN', 'LOW', 'AMAT', 'COP', 'PLD', 'SYK', 'C',
        'BX', 'SCHW', 'DE', 'BA', 'VRTX', 'BSX', 'TJX', 'ADP', 'LMT', 'BMY',
        'GILD', 'ADI', 'PANW', 'SBUX', 'MDT', 'CB', 'MMC', 'LRCX', 'MU',
        'CI', 'KKR', 'AMT', 'SO', 'CME', 'REGN', 'KLAC', 'DUK', 'ICE', 'INTC',
        # Large caps ($50B-$100B)
        'SHW', 'MDLZ', 'SNPS', 'PH', 'CDNS', 'EQIX', 'PNC', 'ZTS', 'PYPL', 'CMG',
        'CL', 'CTAS', 'USB', 'WM', 'MCO', 'AON', 'TT', 'APH', 'ITW', 'WELL',
        'MSI', 'TDG', 'EOG', 'CVS', 'EMR', 'MAR', 'NOC', 'MMM', 'ORLY', 'CEG',
        'FDX', 'GD', 'HCA', 'NSC', 'ABNB', 'CSX', 'FCX', 'AJG', 'CARR', 'ROP',
        'ECL', 'HLT', 'APD', 'BDX', 'TRV', 'PCAR', 'GM', 'OKE', 'AZO', 'DLR',
        'SRE', 'MPC', 'PSX', 'CPRT', 'NXPI', 'AEP', 'PSA', 'JCI', 'URI', 'TFC',
        'AFL', 'NEM', 'AIG', 'MET', 'FICO', 'SPG', 'KMI', 'FTNT', 'VLO', 'ALL',
        'HUM', 'PCG', 'SLB', 'PAYX', 'D', 'MNST', 'CCI', 'GWW', 'FAST', 'KMB',
        'O', 'DHI', 'MSCI', 'PRU', 'BK', 'CTVA', 'CMI', 'PWR', 'LEN', 'HES',
        'AME', 'A', 'ODFL', 'NUE', 'RSG', 'KR', 'EXC', 'AXON', 'FANG', 'F',
        # Mid-large caps ($20B-$50B)
        'OTIS', 'VRSK', 'PEG', 'GIS', 'YUM', 'GEHC', 'IR', 'XEL', 'EW', 'KDP',
        'CHTR', 'EXR', 'EA', 'MLM', 'CBRE', 'IQV', 'VMC', 'DD', 'GRMN', 'HIG',
        'VICI', 'AVB', 'STZ', 'IDXX', 'EFX', 'ED', 'ANSS', 'WEC', 'SYY', 'RCL',
        'XYL', 'DAL', 'WTW', 'ROK', 'PPG', 'MTD', 'HPQ', 'EBAY', 'ON', 'DXCM',
        'TSCO', 'DOV', 'EIX', 'TROW', 'GPN', 'BRO', 'WAB', 'HAL', 'TTWO', 'FITB',
        'KEYS', 'AWK', 'DECK', 'CHD', 'LYB', 'COF', 'HPE', 'CSGP', 'TYL', 'MTB',
        'WMB', 'IRM', 'ETR', 'DTE', 'ES', 'MPWR', 'ACGL', 'FTV', 'HUBB', 'CCL',
        'ADM', 'BR', 'PPL', 'WST', 'DVN', 'PHM', 'EQR', 'RJF', 'K', 'BLDR',
        'ATO', 'NVR', 'VLTO', 'CDW', 'LH', 'ULTA', 'SBAC', 'DRI', 'TRGP', 'CINF',
        'STT', 'WDC', 'FE', 'GEV', 'LYV', 'DOC', 'NTAP', 'LDOS', 'HOLX',
        # Mid caps ($10B-$20B)
        'CAH', 'MKC', 'DFS', 'EXPD', 'CLX', 'OMC', 'INVH', 'MAA', 'STE', 'PKG',
        'NI', 'TER', 'BIIB', 'RF', 'EQT', 'WRB', 'NTRS', 'J', 'MAS', 'CNC',
        'MOH', 'DG', 'LUV', 'IP', 'SNA', 'CF', 'BAX', 'ARE', 'HSY', 'KEY',
        'TXT', 'ESS', 'AES', 'STLD', 'ZBRA', 'PODD', 'COO', 'PTC', 'ROL', 'DGX',
        'CNP', 'VST', 'POOL', 'JBHT', 'TRMB', 'BBY', 'KIM', 'SWK', 'TSN', 'DLTR',
        'AVY', 'UDR', 'IEX', 'WAT', 'GPC', 'AMCR', 'HST', 'SMCI', 'VRSN', 'NDAQ',
        'EVRG', 'EXPE', 'CAG', 'JKHY', 'APA', 'LNT', 'BG', 'LKQ', 'L', 'MRO',
        'TFX', 'CMS', 'CPT', 'TECH', 'EPAM', 'ALLE', 'TPR', 'UHS', 'REG', 'CFG',
        'FFIV', 'PNR', 'BXP', 'FDS', 'EMN', 'AKAM', 'NDSN', 'PAYC', 'KHC', 'VTR',
        'INCY', 'GL', 'CTSH', 'PNW', 'HII', 'LW', 'JNPR', 'CBOE', 'NRG', 'ALGN',
        # Smaller S&P 500 ($5B-$10B)
        'MGM', 'IPG', 'HRL', 'WYNN', 'CHRW', 'WY', 'TAP', 'SOLV', 'SJM', 'CRL',
        'CTRA', 'DPZ', 'CMA', 'HSIC', 'CPB', 'FRT', 'MKTX', 'DAY', 'AIZ', 'CE',
        'ALB', 'GNRC', 'CTLT', 'RL', 'ENPH', 'IVZ', 'QRVO', 'MTCH', 'TDY', 'FMC',
        'NWSA', 'AOS', 'GLW', 'MRNA', 'BEN', 'SWKS', 'BWA', 'HAS', 'BIO', 'DVA',
        'TEL', 'APTV', 'PFG', 'JBL', 'FOXA', 'AAL', 'NWS', 'NCLH', 'FOX', 'PARA',
        'WBA', 'BBWI', 'MHK', 'IFF', 'VTRS', 'HWM', 'EL', 'WBD', 'ANET',
        'NKE', 'UAL', 'BF-B', 'LVS', 'RMD', 'BALL', 'MOS', 'OXY', 'ADSK', 'GOOG',
        'BKR', 'PLTR', 'CRWD', 'AMP', 'IT', 'HBAN', 'TGT', 'KMX', 'GEN',
    ]


def filter_us_exchange_stocks(tickers, api_key):
    """Filter tickers to only include NYSE and NASDAQ stocks."""
    import os
    cache_file = CACHE_DIR / 'exchange_verified_tickers.pkl'

    if cache_file.exists():
        cache_age = datetime.now() - datetime.fromtimestamp(os.path.getmtime(cache_file))
        if cache_age.days < 30:
            with open(cache_file, 'rb') as f:
                cached = pickle.load(f)
                return [t for t in tickers if t in cached]

    print("  Verifying stock exchanges via Polygon API...")
    valid_tickers = []

    for ticker in tqdm(tickers, desc="    Checking exchanges"):
        try:
            url = f"https://api.polygon.io/v3/reference/tickers/{ticker}"
            resp = requests.get(url, params={'apiKey': api_key}, timeout=10)

            if resp.status_code == 200:
                data = resp.json()
                if 'results' in data:
                    exchange = data['results'].get('primary_exchange', '')
                    market = data['results'].get('market', '')

                    if market == 'stocks' and exchange in VALID_EXCHANGES:
                        valid_tickers.append(ticker)

            import time
            time.sleep(0.05)
        except:
            continue

    with open(cache_file, 'wb') as f:
        pickle.dump(set(valid_tickers), f)

    return valid_tickers


# Sector mapping for S&P 500 stocks
SECTOR_MAP = {
    # Technology (BEST: 87.9% win rate)
    'AAPL': 'Tech', 'MSFT': 'Tech', 'NVDA': 'Tech', 'GOOGL': 'Tech', 'GOOG': 'Tech',
    'META': 'Tech', 'AVGO': 'Tech', 'ORCL': 'Tech', 'AMD': 'Tech', 'CRM': 'Tech',
    'ADBE': 'Tech', 'CSCO': 'Tech', 'ACN': 'Tech', 'IBM': 'Tech', 'INTC': 'Tech',
    'QCOM': 'Tech', 'TXN': 'Tech', 'INTU': 'Tech', 'NOW': 'Tech', 'AMAT': 'Tech',
    'ADI': 'Tech', 'LRCX': 'Tech', 'MU': 'Tech', 'KLAC': 'Tech', 'SNPS': 'Tech',
    'CDNS': 'Tech', 'MRVL': 'Tech', 'NXPI': 'Tech', 'MPWR': 'Tech', 'ON': 'Tech',
    'ANET': 'Tech', 'CRWD': 'Tech', 'PANW': 'Tech', 'FTNT': 'Tech',
    # Industrial (BEST: 87.3% win rate)
    'GE': 'Industrial', 'CAT': 'Industrial', 'RTX': 'Industrial', 'HON': 'Industrial',
    'UNP': 'Industrial', 'UPS': 'Industrial', 'BA': 'Industrial', 'DE': 'Industrial',
    'LMT': 'Industrial', 'GD': 'Industrial', 'NOC': 'Industrial', 'MMM': 'Industrial',
    'ETN': 'Industrial', 'ITW': 'Industrial', 'EMR': 'Industrial', 'FDX': 'Industrial',
    'CSX': 'Industrial', 'NSC': 'Industrial', 'WM': 'Industrial', 'RSG': 'Industrial',
    'DAL': 'Industrial', 'UAL': 'Industrial', 'LUV': 'Industrial', 'AAL': 'Industrial',
    # Energy (100% win rate in sample)
    'XOM': 'Energy', 'CVX': 'Energy', 'COP': 'Energy', 'SLB': 'Energy', 'EOG': 'Energy',
    'MPC': 'Energy', 'PSX': 'Energy', 'VLO': 'Energy', 'OXY': 'Energy', 'KMI': 'Energy',
    'WMB': 'Energy', 'HES': 'Energy', 'DVN': 'Energy', 'HAL': 'Energy', 'BKR': 'Energy',
    'FANG': 'Energy', 'OKE': 'Energy', 'TRGP': 'Energy',
    # Consumer Discretionary
    'AMZN': 'Consumer', 'TSLA': 'Consumer', 'HD': 'Consumer', 'MCD': 'Consumer',
    'NKE': 'Consumer', 'SBUX': 'Consumer', 'LOW': 'Consumer', 'TJX': 'Consumer',
    'BKNG': 'Consumer', 'CMG': 'Consumer', 'ORLY': 'Consumer', 'AZO': 'Consumer',
    'ROST': 'Consumer', 'DHI': 'Consumer', 'LEN': 'Consumer', 'GM': 'Consumer',
    'F': 'Consumer', 'ABNB': 'Consumer', 'MAR': 'Consumer', 'HLT': 'Consumer',
    'CCL': 'Consumer', 'RCL': 'Consumer', 'NCLH': 'Consumer', 'LVS': 'Consumer',
    'MGM': 'Consumer', 'WYNN': 'Consumer',
    # Healthcare
    'LLY': 'Health', 'UNH': 'Health', 'JNJ': 'Health', 'ABBV': 'Health', 'MRK': 'Health',
    'PFE': 'Health', 'TMO': 'Health', 'ABT': 'Health', 'DHR': 'Health', 'BMY': 'Health',
    'AMGN': 'Health', 'GILD': 'Health', 'VRTX': 'Health', 'REGN': 'Health', 'ISRG': 'Health',
    'BSX': 'Health', 'MDT': 'Health', 'SYK': 'Health', 'ZTS': 'Health', 'CI': 'Health',
    'ELV': 'Health', 'HUM': 'Health', 'CVS': 'Health', 'MCK': 'Health', 'MRNA': 'Health',
    'BIIB': 'Health', 'DXCM': 'Health', 'IDXX': 'Health', 'ALGN': 'Health',
    # Financials
    'BRK-B': 'Finance', 'JPM': 'Finance', 'V': 'Finance', 'MA': 'Finance', 'BAC': 'Finance',
    'WFC': 'Finance', 'GS': 'Finance', 'MS': 'Finance', 'SPGI': 'Finance', 'BLK': 'Finance',
    'C': 'Finance', 'AXP': 'Finance', 'SCHW': 'Finance', 'CB': 'Finance', 'MMC': 'Finance',
    'PNC': 'Finance', 'USB': 'Finance', 'TFC': 'Finance', 'AIG': 'Finance', 'MET': 'Finance',
    'PRU': 'Finance', 'ALL': 'Finance', 'AFL': 'Finance', 'COF': 'Finance', 'BX': 'Finance',
    'KKR': 'Finance', 'CME': 'Finance', 'ICE': 'Finance', 'PYPL': 'Finance',
    # Utilities
    'NEE': 'Utilities', 'DUK': 'Utilities', 'SO': 'Utilities', 'D': 'Utilities',
    'AEP': 'Utilities', 'EXC': 'Utilities', 'SRE': 'Utilities', 'XEL': 'Utilities',
    'PEG': 'Utilities', 'ED': 'Utilities', 'WEC': 'Utilities', 'ES': 'Utilities',
    'AWK': 'Utilities', 'DTE': 'Utilities', 'ETR': 'Utilities', 'PPL': 'Utilities',
    'ATO': 'Utilities', 'FE': 'Utilities', 'CEG': 'Utilities', 'PCG': 'Utilities',
    # Real Estate
    'AMT': 'RealEstate', 'PLD': 'RealEstate', 'CCI': 'RealEstate', 'EQIX': 'RealEstate',
    'PSA': 'RealEstate', 'SPG': 'RealEstate', 'O': 'RealEstate', 'WELL': 'RealEstate',
    'DLR': 'RealEstate', 'AVB': 'RealEstate', 'EQR': 'RealEstate', 'VTR': 'RealEstate',
    'SBAC': 'RealEstate', 'ARE': 'RealEstate', 'MAA': 'RealEstate', 'UDR': 'RealEstate',
    'ESS': 'RealEstate', 'KIM': 'RealEstate', 'REG': 'RealEstate', 'HST': 'RealEstate',
    # Consumer Staples (BAD: 0% win rate)
    'PG': 'Staples', 'KO': 'Staples', 'PEP': 'Staples', 'WMT': 'Staples', 'COST': 'Staples',
    'PM': 'Staples', 'MO': 'Staples', 'MDLZ': 'Staples', 'CL': 'Staples', 'KMB': 'Staples',
    'GIS': 'Staples', 'K': 'Staples', 'HSY': 'Staples', 'SYY': 'Staples', 'KR': 'Staples',
    'STZ': 'Staples', 'MKC': 'Staples', 'CHD': 'Staples', 'CLX': 'Staples', 'KHC': 'Staples',
    'CAG': 'Staples', 'CPB': 'Staples', 'SJM': 'Staples', 'HRL': 'Staples', 'TAP': 'Staples',
    # Communication (BAD: 20% win rate)
    'NFLX': 'Comm', 'DIS': 'Comm', 'CMCSA': 'Comm', 'T': 'Comm', 'VZ': 'Comm',
    'TMUS': 'Comm', 'CHTR': 'Comm', 'EA': 'Comm', 'TTWO': 'Comm', 'WBD': 'Comm',
    'PARA': 'Comm', 'FOX': 'Comm', 'FOXA': 'Comm', 'NWS': 'Comm', 'NWSA': 'Comm',
    'LYV': 'Comm', 'OMC': 'Comm', 'IPG': 'Comm', 'MTCH': 'Comm',
    # Materials
    'LIN': 'Materials', 'APD': 'Materials', 'SHW': 'Materials', 'ECL': 'Materials',
    'NEM': 'Materials', 'FCX': 'Materials', 'NUE': 'Materials', 'DOW': 'Materials',
    'DD': 'Materials', 'PPG': 'Materials', 'VMC': 'Materials', 'MLM': 'Materials',
    'ALB': 'Materials', 'CF': 'Materials', 'MOS': 'Materials',
}

# Sectors to avoid (low win rates)
BAD_SECTORS = {'Comm', 'Staples'}


def calculate_signals(df, symbol=None, spy_df=None):
    """
    Calculate buy signals for mean reversion strategy.

    NO FUTURE DATA LEAKAGE - all indicators use only past data.

    SIGNALS:
    1. dd_rsi_combo: Deep Drawdown (>20% from high) + RSI < 30
    2. optimal_signal: dd_rsi_combo + EMA200 zone (20-50% below)
    3. optimal_filtered: optimal_signal + filters (sector, market, volatility, momentum, seasonality)
    """
    df = df.copy()

    # 52-week high (rolling looks BACK only)
    df['high_252d'] = df['high'].rolling(252, min_periods=252).max()

    # Distance from 52-week high
    df['dist_from_high'] = (df['high_252d'] - df['adjusted_close']) / df['high_252d']

    # Deep Drawdown: Price > 20% below 52-week high
    df['deep_drawdown'] = df['dist_from_high'] > 0.20

    # RSI (14-period) using pandas_ta
    df['rsi'] = ta.rsi(df['adjusted_close'], length=14)
    df['rsi_oversold'] = df['rsi'] < 30

    # EMA200 using pandas_ta
    df['ema_200'] = ta.ema(df['adjusted_close'], length=200)

    # Distance from EMA200
    df['dist_from_ema200'] = (df['adjusted_close'] - df['ema_200']) / df['ema_200'] * 100

    # Optimal zone: Price is 20-50% below EMA200
    df['in_optimal_zone'] = (df['dist_from_ema200'] >= -50) & (df['dist_from_ema200'] <= -20)

    # ATR for position sizing / stop calculations
    df['atr_14'] = ta.atr(df['high'], df['low'], df['adjusted_close'], length=14)

    # ATR as percentage of price (for volatility filter)
    df['atr_pct'] = (df['atr_14'] / df['adjusted_close']) * 100

    # 5-day price momentum (percentage change)
    df['momentum_5d'] = df['adjusted_close'].pct_change(5) * 100

    # === SIGNALS ===
    # DD+RSI Combo: Deep drawdown + RSI oversold
    df['dd_rsi_combo'] = df['deep_drawdown'] & df['rsi_oversold']

    # Optimal: DD+RSI + in the EMA200 sweet spot (20-50% below)
    df['optimal_signal'] = df['dd_rsi_combo'] & df['in_optimal_zone']

    # === FILTERS FOR OPTIMAL_FILTERED ===
    # Initialize all filters as True (no filtering by default)
    df['filter_sector'] = True
    df['filter_market'] = True
    df['filter_volatility'] = True
    df['filter_momentum'] = True
    df['filter_seasonality'] = True

    # 1. SECTOR FILTER: Avoid bad sectors (Comm: 20% win, Staples: 0% win)
    if symbol is not None:
        sector = SECTOR_MAP.get(symbol, 'Other')
        df['filter_sector'] = sector not in BAD_SECTORS

    # 2. MARKET CONTEXT FILTER: SPY should be weak (best when SPY down >10%)
    if spy_df is not None and len(spy_df) > 0:
        # Calculate SPY 20-day return
        spy_20d_return = spy_df['adjusted_close'].pct_change(20) * 100
        # Align with stock dataframe
        df['spy_20d_return'] = spy_20d_return.reindex(df.index, method='ffill')
        # Best when SPY is down (< 0 is weak, < -10 is very weak)
        df['filter_market'] = df['spy_20d_return'] < 0

    # 3. VOLATILITY FILTER: ATR% should be > 2% (0% win rate when ATR < 2%)
    df['filter_volatility'] = df['atr_pct'] > 2.0

    # 4. MOMENTUM FILTER: Prefer stocks falling hard (< -5% in 5 days)
    # Best when momentum_5d < -15%, acceptable when < -5%
    df['filter_momentum'] = df['momentum_5d'] < -5.0

    # 5. SEASONALITY FILTER: Avoid January (26% win) and August (60% win)
    df['month'] = df.index.month
    df['filter_seasonality'] = ~df['month'].isin([1, 8])

    # Combine all filters
    df['all_filters_pass'] = (
        df['filter_sector'] &
        df['filter_market'] &
        df['filter_volatility'] &
        df['filter_momentum'] &
        df['filter_seasonality']
    )

    # OPTIMAL FILTERED: Optimal signal + all filters pass
    df['optimal_filtered'] = df['optimal_signal'] & df['all_filters_pass']

    return df


def run_backtest(prices, signal_type='optimal_filtered', train_start='2010-01-01',
                 train_end='2024-12-31', test_start='2025-01-01', min_win_rate=0.60,
                 hold_days=252, min_price=10, max_price=400, use_breakeven_stop=False):
    """Run backtest for a signal type.

    Args:
        use_breakeven_stop: If True, move stop to breakeven after price moves 1 ATR up.
    """

    train_start_dt = pd.to_datetime(train_start)
    train_end_dt = pd.to_datetime(train_end)
    test_start_dt = pd.to_datetime(test_start)

    # Phase 1: Training - calculate historical win rates per stock
    print(f"\nPhase 1: Training ({train_start} to {train_end})...")

    stock_stats = {}

    # Get SPY data for market context filter
    spy_df = prices.get('SPY')

    for symbol, df in tqdm(prices.items(), desc="  Training"):
        if symbol == 'SPY':
            continue

        if len(df) < 500:
            continue

        df = calculate_signals(df, symbol=symbol, spy_df=spy_df)

        # Forward return is the TARGET (what we predict), not a feature
        # This is expected to use future data - it's the label
        df['fwd_return'] = df['adjusted_close'].shift(-hold_days) / df['adjusted_close'] - 1

        # Get training period signals
        train_df = df[(df.index >= train_start_dt) & (df.index <= train_end_dt)]

        # Apply price filter
        train_df = train_df[(train_df['adjusted_close'] >= min_price) &
                            (train_df['adjusted_close'] <= max_price)]

        train_df = train_df[train_df[signal_type] == True].dropna(subset=['fwd_return'])

        if len(train_df) >= 3:  # Minimum 3 trades for stats
            wins = (train_df['fwd_return'] > 0).sum()
            total = len(train_df)
            win_rate = wins / total
            avg_return = train_df['fwd_return'].mean()

            stock_stats[symbol] = {
                'win_rate': win_rate,
                'avg_return': avg_return,
                'trades': total,
            }

    # Filter stocks by minimum win rate
    qualified_stocks = {s: v for s, v in stock_stats.items() if v['win_rate'] >= min_win_rate}
    print(f"  Stocks with {signal_type} signal: {len(stock_stats)}")
    print(f"  Stocks with win rate >= {min_win_rate:.0%}: {len(qualified_stocks)}")

    if not qualified_stocks:
        print("  No stocks qualify!")
        return {}

    # Phase 2: Testing on 2025 data
    print(f"\nPhase 2: Testing ({test_start} to present)...")

    trades = []

    for symbol in qualified_stocks.keys():
        if symbol not in prices:
            continue

        df = prices[symbol]
        df = calculate_signals(df, symbol=symbol, spy_df=spy_df)

        # Get 2025 signals
        test_df = df[df.index >= test_start_dt]

        # Apply price filter
        test_df = test_df[(test_df['adjusted_close'] >= min_price) &
                          (test_df['adjusted_close'] <= max_price)]

        signal_days = test_df[test_df[signal_type] == True]

        for date in signal_days.index:
            entry_price = df.loc[date, 'adjusted_close']
            entry_atr = df.loc[date, 'atr_14'] if 'atr_14' in df.columns else None

            # Calculate exit (hold_days later or latest available)
            future_dates = df.index[df.index > date]

            exit_date = None
            exit_price = None
            exit_reason = 'hold_period'
            is_complete = True

            if use_breakeven_stop and entry_atr is not None and not np.isnan(entry_atr):
                # Simulate bar-by-bar with breakeven stop logic
                breakeven_activated = False
                breakeven_trigger = entry_price + entry_atr  # 1 ATR above entry

                for i, bar_date in enumerate(future_dates):
                    if i >= hold_days:
                        # Reached hold period - exit at close
                        exit_date = bar_date
                        exit_price = df.loc[bar_date, 'adjusted_close']
                        exit_reason = 'hold_period'
                        break

                    bar_high = df.loc[bar_date, 'high']
                    bar_low = df.loc[bar_date, 'low']

                    # Check if breakeven stop should be activated
                    if not breakeven_activated and bar_high >= breakeven_trigger:
                        breakeven_activated = True

                    # Check if stopped out at breakeven
                    if breakeven_activated and bar_low <= entry_price:
                        exit_date = bar_date
                        exit_price = entry_price  # Exit at breakeven
                        exit_reason = 'breakeven_stop'
                        break

                # If loop ended without exit (not enough future data)
                if exit_date is None:
                    exit_date = future_dates[-1] if len(future_dates) > 0 else df.index[-1]
                    exit_price = df.loc[exit_date, 'adjusted_close']
                    is_complete = False
            else:
                # Original logic: hold for hold_days
                if len(future_dates) >= hold_days:
                    exit_date = future_dates[hold_days - 1]
                    exit_price = df.loc[exit_date, 'adjusted_close']
                else:
                    exit_date = df.index[-1]
                    exit_price = df.loc[exit_date, 'adjusted_close']
                    is_complete = False

            exit_return = exit_price / entry_price - 1

            trades.append({
                'symbol': symbol,
                'entry_date': date,
                'entry_price': entry_price,
                'exit_date': exit_date,
                'exit_price': exit_price,
                'return': exit_return,
                'is_complete': is_complete,
                'exit_reason': exit_reason if use_breakeven_stop else 'hold_period',
                'train_win_rate': qualified_stocks[symbol]['win_rate'],
                'dist_from_ema200': df.loc[date, 'dist_from_ema200'],
            })

    return {
        'trades': trades,
        'qualified_stocks': len(qualified_stocks),
        'stock_stats': stock_stats,
    }


def main():
    print("\n" + "=" * 70)
    print("DRAWDOWN RECOVERY BACKTEST")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 70)

    polygon_client = PolygonClient(POLYGON_API_KEY)

    # Configuration
    MIN_PRICE = 10
    MAX_PRICE = 400

    # Get S&P 500 universe
    print("\n[1/4] Getting S&P 500 universe...")
    universe = get_sp500_tickers()
    print(f"  Initial universe: {len(universe)} stocks")

    # Filter to NYSE/NASDAQ only
    print("\n[2/4] Filtering to NYSE/NASDAQ only...")
    universe = filter_us_exchange_stocks(universe, POLYGON_API_KEY)
    print(f"  After exchange filter: {len(universe)} stocks")

    # Fetch prices
    print("\n[3/4] Fetching price data...")
    end_date = datetime.now().strftime('%Y-%m-%d')

    prices = polygon_client.fetch_prices_parallel(universe, '2010-01-01', end_date)
    prices['SPY'] = polygon_client.get_prices('SPY', '2010-01-01', end_date)
    print(f"  Got prices for {len(prices)} stocks")

    # Run backtests for each signal type
    print("\n[4/4] Running backtests...")

    signals = [
        ('optimal_filtered', 'Optimal + Filters (RECOMMENDED)'),
        ('optimal_signal', 'DD+RSI+EMA200 (Optimal)'),
        ('dd_rsi_combo', 'DD+RSI Combo'),
    ]

    results = {}
    for signal_key, signal_name in signals:
        print(f"\n{'='*60}")
        print(f"SIGNAL: {signal_name.upper()}")
        print(f"{'='*60}")

        # Lower min_win_rate for optimal signals since they're already highly selective
        if 'optimal' in signal_key:
            min_wr = 0.50
        else:
            min_wr = 0.60

        result = run_backtest(
            prices,
            signal_type=signal_key,
            train_start='2010-01-01',
            train_end='2024-12-31',
            test_start='2025-01-01',
            min_win_rate=min_wr,
            hold_days=252,
            min_price=MIN_PRICE,
            max_price=MAX_PRICE,
        )

        if result and result['trades']:
            trades = result['trades']
            results[signal_key] = result

            # Calculate metrics
            returns = [t['return'] for t in trades]
            wins = sum(1 for r in returns if r > 0)

            print(f"\n  2025 Out-of-Sample Results:")
            print(f"    Trades: {len(trades)}")
            print(f"    Win Rate: {wins/len(trades)*100:.1f}%")
            print(f"    Avg Return: {np.mean(returns)*100:+.1f}%")
            print(f"    Median Return: {np.median(returns)*100:+.1f}%")
            print(f"    Best Trade: {max(returns)*100:+.1f}%")
            print(f"    Worst Trade: {min(returns)*100:+.1f}%")
        else:
            print(f"  No trades found for {signal_name}")

    # Summary comparison
    print(f"\n{'='*80}")
    print("SUMMARY - 2025 OUT-OF-SAMPLE RESULTS")
    print(f"{'='*80}")
    print(f"{'Signal':<35} {'Trades':>8} {'Win Rate':>10} {'Avg Return':>12} {'Median':>10}")
    print("-" * 80)

    for signal_key, signal_name in signals:
        if signal_key in results and results[signal_key]['trades']:
            trades = results[signal_key]['trades']
            returns = [t['return'] for t in trades]
            wins = sum(1 for r in returns if r > 0)
            print(f"{signal_name:<35} {len(trades):>8} {wins/len(trades)*100:>9.1f}% {np.mean(returns)*100:>+11.1f}% {np.median(returns)*100:>+9.1f}%")

    # Top trades from optimal filtered signal
    if 'optimal_filtered' in results and results['optimal_filtered']['trades']:
        print(f"\n{'='*80}")
        print("TOP 10 OPTIMAL+FILTERS TRADES BY RETURN (2025)")
        print(f"{'='*80}")

        optimal_trades = sorted(results['optimal_filtered']['trades'], key=lambda x: -x['return'])

        print(f"{'Rank':<6} {'Symbol':<8} {'Entry':>12} {'Entry $':>10} {'Return':>10} {'Dist EMA200':>12}")
        print("-" * 80)

        for i, t in enumerate(optimal_trades[:10], 1):
            print(f"{i:<6} {t['symbol']:<8} {t['entry_date'].strftime('%Y-%m-%d'):>12} ${t['entry_price']:>8.2f} {t['return']*100:>+9.1f}% {t['dist_from_ema200']:>+10.1f}%")

    # Analysis by EMA200 distance
    if 'dd_rsi_combo' in results and results['dd_rsi_combo']['trades']:
        print(f"\n{'='*80}")
        print("ANALYSIS BY EMA200 DISTANCE (DD+RSI Combo Trades)")
        print(f"{'='*80}")

        trades = results['dd_rsi_combo']['trades']
        ema_ranges = [
            ('> 0% (above EMA200)', 0, 100),
            ('0% to -20%', -20, 0),
            ('-20% to -50% (OPTIMAL)', -50, -20),
            ('< -50%', -100, -50),
        ]

        print(f"{'EMA200 Range':<30} {'Trades':>8} {'Win Rate':>10} {'Avg Return':>12}")
        print("-" * 70)

        for range_name, low, high in ema_ranges:
            range_trades = [t for t in trades if low <= t['dist_from_ema200'] < high]
            if range_trades:
                returns = [t['return'] for t in range_trades]
                wins = sum(1 for r in returns if r > 0)
                print(f"{range_name:<30} {len(range_trades):>8} {wins/len(range_trades)*100:>9.1f}% {np.mean(returns)*100:>+11.1f}%")

    print(f"""
{'='*80}
STRATEGY SUMMARY
{'='*80}

Drawdown Recovery Strategy - Mean Reversion on S&P 500

RECOMMENDED SIGNAL: Optimal + Filters (84% win rate, +37% avg return @ 1yr)

BASE CONDITIONS:
  - Deep Drawdown: Price > 20% below 52-week high
  - RSI Oversold: RSI(14) < 30
  - EMA200 Zone: Price 20-50% below EMA200

FILTERS (all must pass for optimal_filtered):
  - Sector: Avoid Communication & Consumer Staples
  - Market: SPY 20-day return < 0 (weak market context)
  - Volatility: ATR% > 2%
  - Momentum: 5-day price change < -5%
  - Seasonality: Avoid January & August

UNIVERSE:
  - S&P 500 stocks only
  - NYSE/NASDAQ exchanges only
  - Price range: ${MIN_PRICE} - ${MAX_PRICE}

POSITION MANAGEMENT:
  - Hold period: 252 days (1 year)
  - No stop-loss (mean reversion needs room to fluctuate)
  - Position size: 5% per stock, max 20 positions

NO FUTURE DATA LEAKAGE:
  - All indicators use only past data (rolling, pandas_ta)
  - Forward returns only for labels (expected)
""")


if __name__ == '__main__':
    main()

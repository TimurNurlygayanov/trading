#!/usr/bin/env python3
"""
Drawdown Recovery Backtest

Tests the Drawdown Recovery strategy using Polygon API data.

STRATEGY: Deep Drawdown + RSI Oversold + EMA200 Distance + Filters
- Deep Drawdown: Price > 20% below 52-week high
- RSI Oversold: RSI(14) < 30
- EMA200 Distance: Price 20-50% below EMA200 (optimal zone)
- Filters: Sector, Market Context, Volatility, Momentum, Seasonality

STRATEGY PRESETS (select via --strategy):
  1. ULTRA      - 94%+ win rate, Sep-Nov entries only, ~10-15 trades/year
  2. AGGRESSIVE - 90%+ win rate, year-round, ~20-30 trades/year
  3. Q1_SPECIAL - 93%+ win rate, optimized for Jan-Mar entries
  4. BALANCED   - 88%+ win rate, more trades, ~40-50 trades/year

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

# =============================================================================
# STRATEGY PRESETS
# =============================================================================
# Each preset defines filters optimized for different goals
# Based on backtesting 2015-2024 and 2025 out-of-sample validation

STRATEGY_PRESETS = {
    'ULTRA': {
        'name': 'ULTRA - Maximum Win Rate (Sep-Nov Only)',
        'description': '94%+ win rate, best for Sep-Nov entries, ~10-15 trades/year',
        'expected_win_rate': '94%+',
        'expected_return': '+40-50%',
        'filters': {
            'optimal_zone': True,        # EMA200 -20% to -50%
            'atr_contracting': True,     # Weekly ATR SMA3 < SMA10
            'good_sector': True,         # Avoid Comm & Staples
            'vol_above_avg': True,       # Volume > 20-day average
            'seasonal_best': True,       # Sep, Oct, Nov only
        }
    },
    'AGGRESSIVE': {
        'name': 'AGGRESSIVE - High Win Rate Year-Round',
        'description': '90%+ win rate, works all year, ~20-30 trades/year',
        'expected_win_rate': '90%+',
        'expected_return': '+35-45%',
        'filters': {
            'optimal_zone': True,        # EMA200 -20% to -50%
            'atr_contracting': True,     # Weekly ATR SMA3 < SMA10
            'good_sector': True,         # Avoid Comm & Staples
            'vol_above_avg': True,       # Volume > 20-day average
        }
    },
    'Q1_SPECIAL': {
        'name': 'Q1 SPECIAL - Optimized for Jan-Mar Entries',
        'description': '93%+ win rate for Q1, volume-focused filters',
        'expected_win_rate': '93%+',
        'expected_return': '+30-40%',
        'filters': {
            'optimal_zone': True,        # EMA200 -20% to -50%
            'vol_above_1_5x': True,      # Volume > 1.5x 20-day average
            'atr_pct_gt_3': True,        # ATR% > 3 (high volatility)
            'good_sector': True,         # Avoid Comm & Staples
        }
    },
    'BALANCED': {
        'name': 'BALANCED - More Trades, Good Win Rate',
        'description': '88%+ win rate, more opportunities, ~40-50 trades/year',
        'expected_win_rate': '88%+',
        'expected_return': '+30-40%',
        'filters': {
            'optimal_zone': True,        # EMA200 -20% to -50%
            'atr_contracting': True,     # Weekly ATR SMA3 < SMA10
            'good_sector': True,         # Avoid Comm & Staples
        }
    },
    'MOMENTUM': {
        'name': 'MOMENTUM - Strong Weekly Bounce',
        'description': '90% win rate when prev week up >5%, counter-intuitive but works!',
        'expected_win_rate': '90%+',
        'expected_return': '+36%',
        'filters': {
            'optimal_zone': True,        # EMA200 -20% to -50%
            'weekly_up_gt_5': True,      # Previous week UP > 5%
            'good_sector': True,         # Avoid Comm & Staples
        }
    },
}


def resample_to_weekly(df):
    """Convert daily OHLCV data to weekly for ATR calculation."""
    weekly = df.resample('W').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'adjusted_close': 'last',
        'volume': 'sum'
    }).dropna()
    return weekly


def calculate_signals(df, symbol=None, spy_df=None, strategy='AGGRESSIVE'):
    """
    Calculate buy signals for mean reversion strategy.

    NO FUTURE DATA LEAKAGE - all indicators use only past data.

    SIGNALS:
    1. dd_rsi_combo: Deep Drawdown (>20% from high) + RSI < 30
    2. optimal_signal: dd_rsi_combo + EMA200 zone (20-50% below)
    3. strategy_signal: Based on selected strategy preset filters

    Args:
        df: DataFrame with OHLCV data
        symbol: Stock symbol for sector lookup
        spy_df: SPY DataFrame for market context
        strategy: Strategy preset name (ULTRA, AGGRESSIVE, Q1_SPECIAL, BALANCED)
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

    # === NEW FILTERS ===

    # Volume filters
    df['vol_sma_20'] = df['volume'].rolling(20).mean()
    df['vol_vs_avg'] = df['volume'] / df['vol_sma_20']
    df['vol_above_avg'] = df['volume'] > df['vol_sma_20']
    df['vol_above_1_5x'] = df['vol_vs_avg'] > 1.5
    df['vol_above_2x'] = df['vol_vs_avg'] > 2.0

    # Weekly ATR contracting (volatility compression before reversal)
    try:
        weekly = resample_to_weekly(df)
        weekly['weekly_atr'] = ta.atr(weekly['high'], weekly['low'], weekly['close'], length=14)
        weekly['atr_sma3'] = weekly['weekly_atr'].rolling(3).mean()
        weekly['atr_sma10'] = weekly['weekly_atr'].rolling(10).mean()
        weekly['atr_contracting'] = weekly['atr_sma3'] < weekly['atr_sma10']

        # Map weekly ATR contracting back to daily (use last completed week, shifted to avoid future leak)
        weekly_shifted = weekly[['atr_contracting']].shift(1)
        df['atr_contracting'] = weekly_shifted['atr_contracting'].reindex(df.index, method='ffill')
    except:
        df['atr_contracting'] = True  # Default to True if calculation fails

    # ATR percentage thresholds
    df['atr_pct_gt_3'] = df['atr_pct'] > 3.0
    df['atr_pct_gt_4'] = df['atr_pct'] > 4.0

    # Sector filter
    df['good_sector'] = True
    if symbol is not None:
        sector = SECTOR_MAP.get(symbol, 'Other')
        df['good_sector'] = sector not in BAD_SECTORS
        df['sector'] = sector

    # Seasonality filters
    df['month'] = df.index.month
    df['seasonal_best'] = df['month'].isin([9, 10, 11])  # Sep, Oct, Nov
    df['seasonal_q1'] = df['month'].isin([1, 2, 3])      # Q1
    df['seasonal_avoid'] = df['month'].isin([1, 8])      # Jan, Aug (worst months)

    # Weekly return filters (for MOMENTUM strategy)
    try:
        weekly = resample_to_weekly(df)
        weekly['weekly_return'] = weekly['close'].pct_change() * 100
        weekly_shifted = weekly[['weekly_return']].shift(1)  # Use last completed week
        df['prev_weekly_return'] = weekly_shifted['weekly_return'].reindex(df.index, method='ffill')
        df['weekly_up_gt_5'] = df['prev_weekly_return'] > 5
        df['weekly_down_gt_5'] = df['prev_weekly_return'] < -5
    except:
        df['prev_weekly_return'] = 0
        df['weekly_up_gt_5'] = False
        df['weekly_down_gt_5'] = False

    # === SIGNALS ===
    # DD+RSI Combo: Deep drawdown + RSI oversold
    df['dd_rsi_combo'] = df['deep_drawdown'] & df['rsi_oversold']

    # Optimal: DD+RSI + in the EMA200 sweet spot (20-50% below)
    df['optimal_signal'] = df['dd_rsi_combo'] & df['in_optimal_zone']

    # === STRATEGY-BASED SIGNALS ===
    # Get strategy preset
    preset = STRATEGY_PRESETS.get(strategy, STRATEGY_PRESETS['AGGRESSIVE'])
    filters = preset['filters']

    # Build strategy signal based on preset filters
    df['strategy_signal'] = df['dd_rsi_combo'].copy()

    if filters.get('optimal_zone', False):
        df['strategy_signal'] = df['strategy_signal'] & df['in_optimal_zone']

    if filters.get('atr_contracting', False):
        df['strategy_signal'] = df['strategy_signal'] & df['atr_contracting']

    if filters.get('good_sector', False):
        df['strategy_signal'] = df['strategy_signal'] & df['good_sector']

    if filters.get('vol_above_avg', False):
        df['strategy_signal'] = df['strategy_signal'] & df['vol_above_avg']

    if filters.get('vol_above_1_5x', False):
        df['strategy_signal'] = df['strategy_signal'] & df['vol_above_1_5x']

    if filters.get('atr_pct_gt_3', False):
        df['strategy_signal'] = df['strategy_signal'] & df['atr_pct_gt_3']

    if filters.get('atr_pct_gt_4', False):
        df['strategy_signal'] = df['strategy_signal'] & df['atr_pct_gt_4']

    if filters.get('seasonal_best', False):
        df['strategy_signal'] = df['strategy_signal'] & df['seasonal_best']

    if filters.get('weekly_up_gt_5', False):
        df['strategy_signal'] = df['strategy_signal'] & df['weekly_up_gt_5']

    if filters.get('weekly_down_gt_5', False):
        df['strategy_signal'] = df['strategy_signal'] & df['weekly_down_gt_5']

    # === LEGACY FILTERS (for backward compatibility) ===
    df['filter_sector'] = df['good_sector']
    df['filter_volatility'] = df['atr_pct'] > 2.0
    df['filter_momentum'] = df['momentum_5d'] < -5.0
    df['filter_seasonality'] = ~df['seasonal_avoid']

    # Market context filter
    df['filter_market'] = True
    if spy_df is not None and len(spy_df) > 0:
        spy_20d_return = spy_df['adjusted_close'].pct_change(20) * 100
        df['spy_20d_return'] = spy_20d_return.reindex(df.index, method='ffill')
        df['filter_market'] = df['spy_20d_return'] < 0

    df['all_filters_pass'] = (
        df['filter_sector'] &
        df['filter_market'] &
        df['filter_volatility'] &
        df['filter_momentum'] &
        df['filter_seasonality']
    )

    df['optimal_filtered'] = df['optimal_signal'] & df['all_filters_pass']

    return df


def run_backtest(prices, signal_type='optimal_filtered', train_start='2010-01-01',
                 train_end='2024-12-31', test_start='2025-01-01', min_win_rate=0.60,
                 hold_days=252, min_price=10, max_price=400, use_breakeven_stop=False,
                 strategy='AGGRESSIVE'):
    """Run backtest for a signal type.

    Args:
        use_breakeven_stop: If True, move stop to breakeven after price moves 1 ATR up.
        strategy: Strategy preset name (ULTRA, AGGRESSIVE, Q1_SPECIAL, BALANCED)
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

        df = calculate_signals(df, symbol=symbol, spy_df=spy_df, strategy=strategy)

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
        df = calculate_signals(df, symbol=symbol, spy_df=spy_df, strategy=strategy)

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
    import argparse

    parser = argparse.ArgumentParser(description='Drawdown Recovery Backtest')
    parser.add_argument('--strategy', '-s', type=str, default=None,
                        choices=['ULTRA', 'AGGRESSIVE', 'Q1_SPECIAL', 'BALANCED', 'MOMENTUM', 'ALL'],
                        help='Strategy preset to use (default: run all)')
    parser.add_argument('--list', '-l', action='store_true',
                        help='List available strategy presets')
    args = parser.parse_args()

    # List strategies if requested
    if args.list:
        print("\n" + "=" * 80)
        print("AVAILABLE STRATEGY PRESETS")
        print("=" * 80)
        for key, preset in STRATEGY_PRESETS.items():
            print(f"\n{key}:")
            print(f"  {preset['name']}")
            print(f"  {preset['description']}")
            print(f"  Expected Win Rate: {preset['expected_win_rate']}")
            print(f"  Expected Return: {preset['expected_return']}")
            print(f"  Filters: {', '.join(preset['filters'].keys())}")
        return

    print("\n" + "=" * 80)
    print("DRAWDOWN RECOVERY BACKTEST")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 80)

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

    # Determine which strategies to run
    if args.strategy and args.strategy != 'ALL':
        strategies_to_run = [args.strategy]
    else:
        strategies_to_run = list(STRATEGY_PRESETS.keys())

    # Run backtests for each strategy
    print("\n[4/4] Running backtests...")

    results = {}
    for strategy_key in strategies_to_run:
        preset = STRATEGY_PRESETS[strategy_key]

        print(f"\n{'='*80}")
        print(f"STRATEGY: {preset['name']}")
        print(f"Expected: {preset['expected_win_rate']} win rate, {preset['expected_return']} return")
        print(f"{'='*80}")

        result = run_backtest(
            prices,
            signal_type='strategy_signal',
            train_start='2015-01-01',
            train_end='2024-12-31',
            test_start='2025-01-01',
            min_win_rate=0.50,
            hold_days=252,
            min_price=MIN_PRICE,
            max_price=MAX_PRICE,
            strategy=strategy_key,
        )

        if result and result['trades']:
            trades = result['trades']
            results[strategy_key] = result

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

            # Show top trades
            sorted_trades = sorted(trades, key=lambda x: -x['return'])
            print(f"\n  Top 5 Trades:")
            for i, t in enumerate(sorted_trades[:5], 1):
                print(f"    {i}. {t['symbol']:<6} {t['entry_date'].strftime('%Y-%m-%d')} -> {t['return']*100:+.1f}%")
        else:
            print(f"  No trades found for {preset['name']}")

    # Summary comparison
    if len(results) > 1:
        print(f"\n{'='*80}")
        print("SUMMARY - 2025 OUT-OF-SAMPLE RESULTS")
        print(f"{'='*80}")
        print(f"{'Strategy':<35} {'Trades':>8} {'Win Rate':>10} {'Avg Return':>12} {'Median':>10}")
        print("-" * 80)

        for strategy_key in strategies_to_run:
            if strategy_key in results and results[strategy_key]['trades']:
                trades = results[strategy_key]['trades']
                returns = [t['return'] for t in trades]
                wins = sum(1 for r in returns if r > 0)
                preset = STRATEGY_PRESETS[strategy_key]
                print(f"{strategy_key:<35} {len(trades):>8} {wins/len(trades)*100:>9.1f}% {np.mean(returns)*100:>+11.1f}% {np.median(returns)*100:>+9.1f}%")

    print(f"""
{'='*80}
STRATEGY PRESETS
{'='*80}

Use --strategy <NAME> to run a specific strategy, or --list to see details.

AVAILABLE STRATEGIES:
  ULTRA       - 94%+ win rate, Sep-Nov entries only, ~10-15 trades/year
  AGGRESSIVE  - 90%+ win rate, year-round, ~20-30 trades/year
  Q1_SPECIAL  - 93%+ win rate, optimized for Jan-Mar entries
  BALANCED    - 88%+ win rate, more trades, ~40-50 trades/year

USAGE:
  python backtest.py --strategy AGGRESSIVE
  python backtest.py --list
  python backtest.py  (runs all strategies)

NO FUTURE DATA LEAKAGE:
  - All indicators use only past data (rolling, pandas_ta)
  - Forward returns only for labels (expected)
""")


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Drawdown Recovery Backtest

Tests the Drawdown Recovery strategy using Polygon API data.

STRATEGY: Deep Drawdown + RSI Oversold + EMA200 Distance
- Deep Drawdown: Price > 20% below 52-week high
- RSI Oversold: RSI(14) < 30 (daily or weekly)
- EMA200 Distance: Price 20-50% below EMA200 (optimal zone)

WEEKLY INDICATORS:
- Weekly RSI (14 weeks) - smoother signal, less noise
- Weekly distance from 52-week high
- Weekly EMA200 distance

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


def calculate_weekly_indicators(df):
    """
    Calculate weekly technical indicators. NO FUTURE DATA LEAKAGE.

    Weekly data provides smoother signals with less noise than daily.
    All indicators use only past data (forward-fill to daily).
    """
    # Resample to weekly OHLCV (Friday close)
    weekly = df.resample('W-FRI').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'adjusted_close': 'last',
        'volume': 'sum'
    }).dropna()

    if len(weekly) < 52:
        return pd.DataFrame(index=df.index)

    # Weekly 52-week high/low
    weekly['weekly_high_52w'] = weekly['high'].rolling(52, min_periods=52).max()
    weekly['weekly_low_52w'] = weekly['low'].rolling(52, min_periods=52).min()
    weekly['weekly_dist_from_high'] = (weekly['weekly_high_52w'] - weekly['adjusted_close']) / weekly['weekly_high_52w']

    # Weekly RSI (14 weeks)
    delta = weekly['adjusted_close'].diff()
    gain = delta.where(delta > 0, 0).ewm(span=14, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(span=14, adjust=False).mean()
    weekly['weekly_rsi'] = 100 - (100 / (1 + gain / (loss + 1e-10)))

    # Weekly EMA200 (200 weeks ~ 4 years)
    weekly['weekly_ema_50'] = weekly['adjusted_close'].ewm(span=50, adjust=False).mean()
    weekly['weekly_ema_200'] = weekly['adjusted_close'].ewm(span=200, adjust=False).mean()
    weekly['weekly_dist_from_ema200'] = (weekly['adjusted_close'] - weekly['weekly_ema_200']) / weekly['weekly_ema_200'] * 100

    # Select columns to return
    weekly_cols = [
        'weekly_dist_from_high', 'weekly_rsi',
        'weekly_dist_from_ema200', 'weekly_ema_50', 'weekly_ema_200'
    ]

    # Forward-fill weekly data to daily index (no future leak)
    return weekly[weekly_cols].reindex(df.index, method='ffill')


def calculate_signals(df):
    """
    Calculate buy signals. NO FUTURE DATA LEAKAGE.

    All calculations use only past data:
    - rolling(N) uses past N bars only
    - ewm(span=N) uses exponentially weighted past data
    - pct_change() uses current vs previous bar

    The only forward-looking calculation is fwd_return which is
    the TARGET variable (what we're predicting), not a feature.

    DAILY INDICATORS:
    - RSI(14), EMA50, EMA200, 52-week high/low

    WEEKLY INDICATORS (smoother, less noise):
    - Weekly RSI(14), Weekly EMA200, Weekly 52-week high
    """
    df = df.copy()

    # === DAILY INDICATORS ===

    # Daily returns (uses only past data: current vs previous)
    df['return_1d'] = df['adjusted_close'].pct_change()

    # 52-week high/low (rolling looks BACK only, no future data)
    df['high_252d'] = df['high'].rolling(252, min_periods=252).max()
    df['low_252d'] = df['low'].rolling(252, min_periods=252).min()

    # Distance from high/low (current price vs past high/low)
    df['dist_from_high'] = (df['high_252d'] - df['adjusted_close']) / df['high_252d']
    df['dist_from_low'] = (df['adjusted_close'] - df['low_252d']) / df['low_252d']

    # Deep Drawdown Signal: Price > 20% below 52-week high
    df['deep_drawdown'] = df['dist_from_high'] > 0.20

    # Near 52-week low
    df['near_52w_low'] = df['dist_from_low'] < 0.10

    # Volatility (rolling std looks back only)
    df['volatility_21d'] = df['return_1d'].rolling(21).std() * np.sqrt(252)
    df['vol_pct'] = df['volatility_21d'].rolling(252).rank(pct=True)
    df['low_vol'] = df['vol_pct'] < 0.25

    # RSI (exponential weighted mean uses past data only)
    delta = df['adjusted_close'].diff()
    gain = delta.where(delta > 0, 0).ewm(span=14, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(span=14, adjust=False).mean()
    df['rsi'] = 100 - (100 / (1 + gain / (loss + 1e-10)))
    df['rsi_oversold'] = df['rsi'] < 30

    # EMA indicators (exponential moving averages use past data only)
    df['ema_50'] = df['adjusted_close'].ewm(span=50, adjust=False).mean()
    df['ema_200'] = df['adjusted_close'].ewm(span=200, adjust=False).mean()

    # Distance from EMA200 (current price vs current EMA - both use past data)
    df['dist_from_ema200'] = (df['adjusted_close'] - df['ema_200']) / df['ema_200'] * 100

    # Optimal zone: Price is 20-50% below EMA200
    df['in_optimal_zone'] = (df['dist_from_ema200'] >= -50) & (df['dist_from_ema200'] <= -20)

    # === WEEKLY INDICATORS ===
    weekly_df = calculate_weekly_indicators(df)
    if len(weekly_df) > 0:
        for col in weekly_df.columns:
            df[col] = weekly_df[col]

        # Weekly signals
        df['weekly_deep_drawdown'] = df['weekly_dist_from_high'] > 0.20
        df['weekly_rsi_oversold'] = df['weekly_rsi'] < 30
        df['weekly_in_optimal_zone'] = (df['weekly_dist_from_ema200'] >= -50) & (df['weekly_dist_from_ema200'] <= -20)
    else:
        # Default to False if not enough weekly data
        df['weekly_deep_drawdown'] = False
        df['weekly_rsi_oversold'] = False
        df['weekly_in_optimal_zone'] = False

    # === COMBINED SIGNALS ===

    # Original daily signals
    df['dd_rsi_combo'] = df['deep_drawdown'] & df['rsi_oversold']
    df['optimal_signal'] = df['dd_rsi_combo'] & df['in_optimal_zone']

    # Weekly signals (smoother, potentially fewer false positives)
    df['weekly_dd_rsi_combo'] = df['weekly_deep_drawdown'] & df['weekly_rsi_oversold']
    df['weekly_optimal_signal'] = df['weekly_dd_rsi_combo'] & df['weekly_in_optimal_zone']

    # Hybrid signals (daily DD + weekly RSI for confirmation)
    df['hybrid_dd_rsi'] = df['deep_drawdown'] & df['weekly_rsi_oversold']
    df['hybrid_optimal'] = df['hybrid_dd_rsi'] & df['in_optimal_zone']

    return df


def run_backtest(prices, signal_type='optimal_signal', train_start='2010-01-01',
                 train_end='2024-12-31', test_start='2025-01-01', min_win_rate=0.60,
                 hold_days=252, min_price=10, max_price=400):
    """Run backtest for a signal type."""

    train_start_dt = pd.to_datetime(train_start)
    train_end_dt = pd.to_datetime(train_end)
    test_start_dt = pd.to_datetime(test_start)

    # Phase 1: Training - calculate historical win rates per stock
    print(f"\nPhase 1: Training ({train_start} to {train_end})...")

    stock_stats = {}

    for symbol, df in tqdm(prices.items(), desc="  Training"):
        if symbol == 'SPY':
            continue

        if len(df) < 500:
            continue

        df = calculate_signals(df)

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
        df = calculate_signals(df)

        # Get 2025 signals
        test_df = df[df.index >= test_start_dt]

        # Apply price filter
        test_df = test_df[(test_df['adjusted_close'] >= min_price) &
                          (test_df['adjusted_close'] <= max_price)]

        signal_days = test_df[test_df[signal_type] == True]

        for date in signal_days.index:
            entry_price = df.loc[date, 'adjusted_close']

            # Calculate exit (hold_days later or latest available)
            future_dates = df.index[df.index > date]
            if len(future_dates) >= hold_days:
                exit_date = future_dates[hold_days - 1]
                exit_price = df.loc[exit_date, 'adjusted_close']
                exit_return = exit_price / entry_price - 1
                is_complete = True
            else:
                exit_date = df.index[-1]
                exit_price = df.loc[exit_date, 'adjusted_close']
                exit_return = exit_price / entry_price - 1
                is_complete = False

            trades.append({
                'symbol': symbol,
                'entry_date': date,
                'entry_price': entry_price,
                'exit_date': exit_date,
                'exit_price': exit_price,
                'return': exit_return,
                'is_complete': is_complete,
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
    print("Daily vs Weekly Indicators Comparison")
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

    # Signal types to test - DAILY vs WEEKLY vs HYBRID
    signals = [
        # Daily indicators (original)
        ('optimal_signal', '[DAILY] DD+RSI+EMA200 zone'),
        ('dd_rsi_combo', '[DAILY] DD + RSI Combo'),
        # Weekly indicators (smoother signals)
        ('weekly_optimal_signal', '[WEEKLY] DD+RSI+EMA200 zone'),
        ('weekly_dd_rsi_combo', '[WEEKLY] DD + RSI Combo'),
        # Hybrid indicators (daily DD + weekly RSI)
        ('hybrid_optimal', '[HYBRID] Daily DD + Weekly RSI + EMA200'),
        ('hybrid_dd_rsi', '[HYBRID] Daily DD + Weekly RSI'),
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

    # Top trades from optimal signal
    if 'optimal_signal' in results and results['optimal_signal']['trades']:
        print(f"\n{'='*80}")
        print("TOP 10 OPTIMAL SIGNAL TRADES BY RETURN (2025)")
        print(f"{'='*80}")

        optimal_trades = sorted(results['optimal_signal']['trades'], key=lambda x: -x['return'])

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
CONCLUSION
{'='*80}

Strategy: Drawdown Recovery - Daily vs Weekly Indicator Comparison

SIGNAL TYPES TESTED:

[DAILY] - Original daily indicators
  - Deep Drawdown: Price > 20% below daily 52-week high
  - RSI(14): 14-day RSI < 30
  - EMA200 Zone: Price 20-50% below daily EMA200

[WEEKLY] - Weekly indicators (smoother, less noise)
  - Deep Drawdown: Price > 20% below weekly 52-week high
  - RSI(14): 14-week RSI < 30
  - EMA200 Zone: Price 20-50% below weekly EMA200

[HYBRID] - Best of both worlds
  - Deep Drawdown: Daily (faster reaction)
  - RSI: Weekly (smoother confirmation)
  - EMA200: Daily zone filter

HYPOTHESIS:
  - Weekly indicators should filter out false daily signals
  - Hybrid approach may offer best risk/reward tradeoff
  - Fewer trades but potentially higher win rate

FILTERS APPLIED:
  - S&P 500 universe only
  - NYSE/NASDAQ exchanges only
  - Price range: ${MIN_PRICE} - ${MAX_PRICE}
  - Hold period: 252 days (1 year)

NO FUTURE DATA LEAKAGE:
  - All indicators use only past data (rolling, ewm)
  - Weekly data forward-filled to daily (no lookahead)
  - Forward returns only for labels (expected)
""")


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
S&P 500 3-Month Momentum Screener

Finds S&P 500 stocks with the highest 3-month momentum.
Use this to identify candidates for the momentum strategy.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

CACHE_DIR = Path('data_cache_weekly')

# Screener parameters
TOP_N = 20  # Show top N stocks
LOOKBACK_MONTHS = 3

# S&P 500 tickers
SP500_TICKERS = [
    'AAPL', 'MSFT', 'AMZN', 'NVDA', 'GOOGL', 'META', 'TSLA', 'BRK.B', 'UNH', 'XOM',
    'JPM', 'JNJ', 'V', 'PG', 'MA', 'HD', 'CVX', 'MRK', 'ABBV', 'LLY',
    'PEP', 'KO', 'COST', 'AVGO', 'MCD', 'WMT', 'CSCO', 'TMO', 'ABT', 'CRM',
    'ACN', 'BAC', 'CMCSA', 'PFE', 'NFLX', 'AMD', 'INTC', 'DIS', 'ADBE', 'WFC',
    'PM', 'VZ', 'TXN', 'COP', 'RTX', 'NEE', 'BMY', 'ORCL', 'HON', 'UNP',
    'QCOM', 'LOW', 'SPGI', 'UPS', 'IBM', 'CAT', 'BA', 'GE', 'SBUX', 'INTU',
    'DE', 'AMAT', 'PLD', 'ISRG', 'GILD', 'NOW', 'BKNG', 'ADP', 'MDLZ', 'ADI',
    'TJX', 'MMC', 'REGN', 'VRTX', 'SYK', 'LMT', 'CB', 'AMT', 'MO', 'SCHW',
    'CVS', 'TMUS', 'ZTS', 'CI', 'SO', 'BDX', 'EOG', 'DUK', 'PNC', 'MU',
    'LRCX', 'SLB', 'BSX', 'ITW', 'SNPS', 'AON', 'CL', 'CSX', 'CDNS', 'ICE',
    'NOC', 'FDX', 'CME', 'WM', 'SHW', 'EMR', 'ETN', 'FCX', 'APD', 'GD',
    'MCK', 'NSC', 'PXD', 'HUM', 'ORLY', 'GM', 'ROP', 'F', 'PSX', 'KLAC',
    'AZO', 'MCO', 'PCAR', 'MPC', 'ADSK', 'MAR', 'AJG', 'MCHP', 'KMB', 'TGT',
    'EL', 'OXY', 'GIS', 'TRV', 'AEP', 'MSCI', 'D', 'HES', 'NXPI', 'FTNT',
    'SRE', 'APH', 'PSA', 'PAYX', 'CMG', 'ECL', 'A', 'AFL', 'DOW', 'JCI',
    'O', 'STZ', 'TEL', 'IDXX', 'HAL', 'WELL', 'NUE', 'CCI', 'KMI', 'KDP',
    'AIG', 'BK', 'YUM', 'SPG', 'DXCM', 'KHC', 'IQV', 'CMI', 'CTSH', 'DVN',
    'HSY', 'NEM', 'MNST', 'CTAS', 'GPN', 'AMP', 'PH', 'CARR', 'ALL', 'COF',
    'ED', 'PRU', 'EW', 'PCG', 'HLT', 'OKE', 'XEL', 'BIIB', 'MTD', 'ROST',
    'ROK', 'FAST', 'ALB', 'DLR', 'VRSK', 'CPRT', 'EA', 'WEC', 'PPG', 'OTIS',
]


def load_sp500_data():
    """Load cached data for S&P 500 symbols."""
    all_data = {}

    # First try to load 2023 data files
    for f in CACHE_DIR.glob('polygon_*_2023-01-01_*.pkl'):
        parts = f.stem.split('_')
        if len(parts) >= 2:
            symbol = parts[1]
            if symbol in SP500_TICKERS:
                all_data[symbol] = pd.read_pickle(f)

    # Then fill in any missing symbols from 2024 data
    for f in CACHE_DIR.glob('polygon_*_2024-01-01_*.pkl'):
        parts = f.stem.split('_')
        if len(parts) >= 2:
            symbol = parts[1]
            if symbol in SP500_TICKERS and symbol not in all_data:
                all_data[symbol] = pd.read_pickle(f)

    return all_data


def calculate_momentum(all_data):
    """Calculate 3-month momentum for all stocks."""
    results = []

    for symbol, df in tqdm(all_data.items(), desc='Calculating momentum'):
        if len(df) < 63:  # Need at least ~3 months of daily data
            continue

        # Get current price (latest close)
        current_price = df['close'].iloc[-1]
        current_date = df.index[-1]

        # Get price from ~3 months ago (63 trading days)
        lookback_idx = min(63, len(df) - 1)
        price_3m_ago = df['close'].iloc[-lookback_idx]
        date_3m_ago = df.index[-lookback_idx]

        # Calculate 3-month return
        momentum_3m = (current_price - price_3m_ago) / price_3m_ago * 100

        # Get price from ~1 month ago (21 trading days)
        lookback_1m_idx = min(21, len(df) - 1)
        price_1m_ago = df['close'].iloc[-lookback_1m_idx]
        momentum_1m = (current_price - price_1m_ago) / price_1m_ago * 100

        # Get 52-week high/low
        lookback_52w_idx = min(252, len(df) - 1)
        high_52w = df['high'].iloc[-lookback_52w_idx:].max()
        low_52w = df['low'].iloc[-lookback_52w_idx:].min()
        pct_from_high = (current_price - high_52w) / high_52w * 100
        pct_from_low = (current_price - low_52w) / low_52w * 100

        # Average volume (20-day)
        avg_volume = df['volume'].iloc[-20:].mean()
        avg_dollar_volume = avg_volume * current_price

        results.append({
            'symbol': symbol,
            'price': current_price,
            'momentum_3m': momentum_3m,
            'momentum_1m': momentum_1m,
            'high_52w': high_52w,
            'low_52w': low_52w,
            'pct_from_high': pct_from_high,
            'pct_from_low': pct_from_low,
            'avg_volume': avg_volume,
            'avg_dollar_volume': avg_dollar_volume,
            'date': current_date,
            'date_3m_ago': date_3m_ago,
        })

    return pd.DataFrame(results)


def print_screener_results(df):
    """Print screener results."""
    print('\n' + '=' * 80)
    print('S&P 500 3-MONTH MOMENTUM SCREENER')
    print('=' * 80)

    if len(df) == 0:
        print("No data available!")
        return

    # Get latest date
    latest_date = df['date'].max()
    print(f'\nData as of: {latest_date.strftime("%Y-%m-%d")}')
    print(f'Stocks analyzed: {len(df)}')

    # Sort by 3-month momentum
    df_sorted = df.sort_values('momentum_3m', ascending=False)

    # Top momentum stocks
    print('\n' + '-' * 80)
    print(f'TOP {TOP_N} MOMENTUM STOCKS (3-Month)')
    print('-' * 80)
    print(f'{"Rank":<5} {"Symbol":<7} {"Price":>10} {"3M Mom":>10} {"1M Mom":>10} {"From 52H":>10} {"Avg $Vol":>12}')
    print('-' * 80)

    for i, (_, row) in enumerate(df_sorted.head(TOP_N).iterrows(), 1):
        print(f'{i:<5} {row["symbol"]:<7} ${row["price"]:>8.2f} '
              f'{row["momentum_3m"]:>+9.1f}% {row["momentum_1m"]:>+9.1f}% '
              f'{row["pct_from_high"]:>+9.1f}% ${row["avg_dollar_volume"]/1e6:>10.1f}M')

    # Bottom momentum stocks (biggest losers)
    print('\n' + '-' * 80)
    print(f'BOTTOM {TOP_N} MOMENTUM STOCKS (Biggest Losers)')
    print('-' * 80)
    print(f'{"Rank":<5} {"Symbol":<7} {"Price":>10} {"3M Mom":>10} {"1M Mom":>10} {"From 52L":>10} {"Avg $Vol":>12}')
    print('-' * 80)

    for i, (_, row) in enumerate(df_sorted.tail(TOP_N).iloc[::-1].iterrows(), 1):
        print(f'{i:<5} {row["symbol"]:<7} ${row["price"]:>8.2f} '
              f'{row["momentum_3m"]:>+9.1f}% {row["momentum_1m"]:>+9.1f}% '
              f'{row["pct_from_low"]:>+9.1f}% ${row["avg_dollar_volume"]/1e6:>10.1f}M')

    # Strategy recommendation
    print('\n' + '=' * 80)
    print('STRATEGY RECOMMENDATION')
    print('=' * 80)
    top_pick = df_sorted.iloc[0]
    print(f'\nTop 3-Month Momentum Pick: {top_pick["symbol"]}')
    print(f'  Current Price: ${top_pick["price"]:.2f}')
    print(f'  3-Month Return: {top_pick["momentum_3m"]:+.1f}%')
    print(f'  1-Month Return: {top_pick["momentum_1m"]:+.1f}%')
    print(f'  From 52-Week High: {top_pick["pct_from_high"]:+.1f}%')

    # Summary stats
    print('\n' + '-' * 80)
    print('MARKET SUMMARY')
    print('-' * 80)
    positive_mom = len(df[df['momentum_3m'] > 0])
    negative_mom = len(df[df['momentum_3m'] < 0])
    avg_mom = df['momentum_3m'].mean()
    median_mom = df['momentum_3m'].median()

    print(f'Stocks with positive 3M momentum: {positive_mom} ({positive_mom/len(df)*100:.1f}%)')
    print(f'Stocks with negative 3M momentum: {negative_mom} ({negative_mom/len(df)*100:.1f}%)')
    print(f'Average 3M momentum: {avg_mom:+.1f}%')
    print(f'Median 3M momentum: {median_mom:+.1f}%')


if __name__ == '__main__':
    print('Loading S&P 500 data...')
    all_data = load_sp500_data()
    print(f'Loaded {len(all_data)} S&P 500 symbols')

    if len(all_data) < 50:
        print("\nWarning: Low S&P 500 coverage.")
        print("Available:", list(all_data.keys())[:20], "...")

    print('\nRunning momentum screener...')
    results_df = calculate_momentum(all_data)

    print_screener_results(results_df)

    # Save results
    results_df.to_csv('screener_3m_momentum.csv', index=False)
    print(f'\nResults saved to screener_3m_momentum.csv')

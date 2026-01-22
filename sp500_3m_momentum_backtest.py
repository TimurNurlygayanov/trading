#!/usr/bin/env python3
"""
S&P 500 3-Month Momentum Winner Strategy

Strategy Rules:
1. Every month, find S&P 500 stocks with the largest gain over the LAST 3 MONTHS
2. Buy at the start of the new month
3. Hold for exactly 1 year
4. Position size: 10% of capital per stock (max 10 concurrent positions)

This is a momentum strategy betting that winning stocks continue to win.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

CACHE_DIR = Path('data_cache_weekly')

# Position sizing parameters
TOTAL_CAPITAL = 20000  # $20k account
POSITION_PCT = 0.10  # 10% per position
POSITION_SIZE = TOTAL_CAPITAL * POSITION_PCT  # $2k per trade
MAX_POSITIONS = 10  # Max concurrent positions

# How many winners to buy each month
WINNERS_PER_MONTH = 1  # Buy the single biggest winner

# Lookback period for momentum calculation
LOOKBACK_MONTHS = 3  # Look at last 3 months performance

# S&P 500 tickers (current constituents)
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
    """Load cached data for S&P 500 symbols, preferring 2023 data."""
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


def resample_to_monthly(df):
    """Resample daily data to monthly OHLCV."""
    if len(df) < 60:  # Need at least ~3 months
        return None

    df = df.copy()

    # Resample to monthly (month end)
    monthly = df.resample('ME').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()

    if len(monthly) < 3:
        return None

    # Calculate monthly return
    monthly['monthly_return'] = (monthly['close'] - monthly['open']) / monthly['open'] * 100

    return monthly


def run_backtest(all_data, start_year=2024):
    """Run backtest for the biggest loser strategy."""
    print("Resampling to monthly data...")

    # Resample all stocks to monthly
    monthly_data = {}
    for symbol, daily_df in tqdm(all_data.items(), desc='Resampling'):
        monthly = resample_to_monthly(daily_df)
        if monthly is not None:
            monthly_data[symbol] = monthly

    print(f"S&P 500 stocks with monthly data: {len(monthly_data)}")

    # Get all months where we can trade
    all_months = set()
    for symbol, monthly_df in monthly_data.items():
        for date in monthly_df.index:
            if date.year >= start_year:
                all_months.add(date)

    all_months = sorted(all_months)
    print(f"Months available from {start_year}: {len(all_months)}")

    # Track active positions and completed trades
    active_positions = []  # List of {symbol, entry_date, entry_price, exit_date, shares}
    completed_trades = []
    portfolio_history = []

    for i, month_end in enumerate(tqdm(all_months, desc='Processing months')):
        if i < LOOKBACK_MONTHS:
            continue  # Need enough months for lookback calculation

        prev_month = all_months[i - 1]
        lookback_start_month = all_months[i - LOOKBACK_MONTHS]

        # First, check if any positions need to be closed (held for 1 year)
        still_active = []
        for pos in active_positions:
            # Check if 1 year has passed
            if month_end >= pos['exit_date']:
                # Close the position
                symbol = pos['symbol']
                if symbol in monthly_data:
                    monthly_df = monthly_data[symbol]
                    # Find the closest month to exit date
                    valid_months = monthly_df.index[monthly_df.index <= pos['exit_date']]
                    if len(valid_months) > 0:
                        actual_exit_month = valid_months[-1]
                        exit_price = monthly_df.loc[actual_exit_month, 'close']

                        pct_return = (exit_price - pos['entry_price']) / pos['entry_price'] * 100
                        dollar_return = pos['shares'] * (exit_price - pos['entry_price'])

                        completed_trades.append({
                            'symbol': symbol,
                            'entry_date': pos['entry_date'],
                            'entry_price': pos['entry_price'],
                            'exit_date': actual_exit_month,
                            'exit_price': exit_price,
                            'shares': pos['shares'],
                            'position_size': pos['position_size'],
                            'momentum_gain': pos['momentum_gain'],
                            'pct_return': pct_return,
                            'dollar_return': dollar_return,
                            'win': pct_return > 0,
                        })
            else:
                still_active.append(pos)

        active_positions = still_active

        # Find biggest winner(s) over last 3 months
        momentum_returns = []
        for symbol, monthly_df in monthly_data.items():
            if lookback_start_month not in monthly_df.index:
                continue
            if prev_month not in monthly_df.index:
                continue
            if month_end not in monthly_df.index:
                continue

            # Calculate 3-month return (from start of lookback to end of prev month)
            start_price = monthly_df.loc[lookback_start_month, 'open']
            end_price = monthly_df.loc[prev_month, 'close']
            curr_row = monthly_df.loc[month_end]

            three_month_return = (end_price - start_price) / start_price * 100
            if pd.notna(three_month_return) and three_month_return > 0:  # Only consider winners
                momentum_returns.append({
                    'symbol': symbol,
                    'momentum_return': three_month_return,
                    'entry_price': curr_row['open'],  # Buy at month open
                })

        if not momentum_returns:
            continue

        # Sort by return (most positive first = biggest winners)
        momentum_returns = sorted(momentum_returns, key=lambda x: x['momentum_return'], reverse=True)

        # Check how many positions we can open
        available_slots = MAX_POSITIONS - len(active_positions)
        if available_slots <= 0:
            continue

        # Buy the biggest winner(s)
        for winner in momentum_returns[:min(WINNERS_PER_MONTH, available_slots)]:
            symbol = winner['symbol']
            entry_price = winner['entry_price']

            # Skip if already holding this stock
            if any(p['symbol'] == symbol for p in active_positions):
                continue

            shares = POSITION_SIZE / entry_price
            exit_date = month_end + relativedelta(years=1)

            active_positions.append({
                'symbol': symbol,
                'entry_date': month_end,
                'entry_price': entry_price,
                'exit_date': exit_date,
                'shares': shares,
                'position_size': POSITION_SIZE,
                'momentum_gain': winner['momentum_return'],
            })

        # Record portfolio value
        portfolio_value = TOTAL_CAPITAL - len(active_positions) * POSITION_SIZE  # Cash
        for pos in active_positions:
            symbol = pos['symbol']
            if symbol in monthly_data and month_end in monthly_data[symbol].index:
                current_price = monthly_data[symbol].loc[month_end, 'close']
                portfolio_value += pos['shares'] * current_price

        portfolio_history.append({
            'date': month_end,
            'portfolio_value': portfolio_value,
            'num_positions': len(active_positions),
            'cash': TOTAL_CAPITAL - len(active_positions) * POSITION_SIZE,
        })

    # Close any remaining positions at latest available price
    last_month = all_months[-1] if all_months else None
    for pos in active_positions:
        symbol = pos['symbol']
        if symbol in monthly_data:
            monthly_df = monthly_data[symbol]
            if len(monthly_df) > 0:
                exit_price = monthly_df['close'].iloc[-1]
                actual_exit_date = monthly_df.index[-1]

                pct_return = (exit_price - pos['entry_price']) / pos['entry_price'] * 100
                dollar_return = pos['shares'] * (exit_price - pos['entry_price'])

                completed_trades.append({
                    'symbol': symbol,
                    'entry_date': pos['entry_date'],
                    'entry_price': pos['entry_price'],
                    'exit_date': actual_exit_date,
                    'exit_price': exit_price,
                    'shares': pos['shares'],
                    'position_size': pos['position_size'],
                    'momentum_gain': pos['momentum_gain'],
                    'pct_return': pct_return,
                    'dollar_return': dollar_return,
                    'win': pct_return > 0,
                    'still_open': True,  # Mark as not fully held
                })

    return {
        'trades': pd.DataFrame(completed_trades),
        'portfolio_history': pd.DataFrame(portfolio_history),
    }


def calculate_metrics(trades_df):
    """Calculate performance metrics."""
    if len(trades_df) == 0:
        return {}

    # Separate completed trades from still-open ones
    if 'still_open' in trades_df.columns:
        closed_trades = trades_df[trades_df['still_open'] != True].copy()
    else:
        closed_trades = trades_df.copy()

    wins = trades_df[trades_df['win']]
    losses = trades_df[~trades_df['win']]

    win_rate = len(wins) / len(trades_df) * 100
    avg_ret = trades_df['pct_return'].mean()

    gross_profit = wins['pct_return'].sum() if len(wins) > 0 else 0
    gross_loss = abs(losses['pct_return'].sum()) if len(losses) > 0 else 0.001
    profit_factor = gross_profit / gross_loss

    total_dollar_return = trades_df['dollar_return'].sum()

    return {
        'trades': len(trades_df),
        'closed_trades': len(closed_trades),
        'wins': len(wins),
        'losses': len(losses),
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'avg_return': avg_ret,
        'max_win': trades_df['pct_return'].max(),
        'max_loss': trades_df['pct_return'].min(),
        'total_dollar_return': total_dollar_return,
        'avg_gain_bought': trades_df['momentum_gain'].mean(),
    }


def print_results(results):
    """Print backtest results."""
    trades_df = results['trades']
    history_df = results['portfolio_history']

    if len(trades_df) == 0:
        print("No trades executed!")
        return

    metrics = calculate_metrics(trades_df)

    print('\n' + '=' * 70)
    print('S&P 500 3-MONTH MOMENTUM WINNER STRATEGY')
    print('=' * 70)
    print(f'\nStrategy: Buy S&P 500 stock with largest {LOOKBACK_MONTHS}-month gain')
    print(f'Hold Period: 1 year')
    print(f'Position Size: ${POSITION_SIZE:,} ({POSITION_PCT*100:.0f}% of ${TOTAL_CAPITAL:,})')
    print(f'Max Positions: {MAX_POSITIONS}')
    print('-' * 70)
    print(f"Total Trades:     {metrics['trades']}")
    print(f"Wins/Losses:      {metrics['wins']} / {metrics['losses']}")
    print(f"Win Rate:         {metrics['win_rate']:.1f}%")
    print(f"Profit Factor:    {metrics['profit_factor']:.2f}")
    print('-' * 70)
    print(f"Avg Return/Trade: {metrics['avg_return']:.1f}%")
    print(f"Max Win:          {metrics['max_win']:.1f}%")
    print(f"Max Loss:         {metrics['max_loss']:.1f}%")
    print(f"Avg {LOOKBACK_MONTHS}M Gain Bought: {metrics['avg_gain_bought']:.1f}%")
    print('-' * 70)
    print(f"Total Dollar P&L: ${metrics['total_dollar_return']:+,.0f}")

    # Show all trades
    print('\n' + '-' * 70)
    print('ALL TRADES')
    print('-' * 70)
    for _, trade in trades_df.iterrows():
        result = "WIN" if trade['win'] else "LOSS"
        still_open = " (OPEN)" if trade.get('still_open', False) else ""
        print(f"{trade['entry_date'].strftime('%Y-%m')} -> {trade['exit_date'].strftime('%Y-%m')} "
              f"{trade['symbol']:5s} (3M: +{trade['momentum_gain']:.1f}%) "
              f"-> {trade['pct_return']:+.1f}% [{result}]{still_open}")

    # Separate 2025 results
    print('\n' + '=' * 70)
    print('2025 OUT-OF-SAMPLE ANALYSIS')
    print('=' * 70)

    # Trades that were ENTERED in 2025
    trades_2025_entry = trades_df[trades_df['entry_date'].dt.year == 2025]
    # Trades that were EXITED in 2025 (completed)
    trades_2025_exit = trades_df[trades_df['exit_date'].dt.year == 2025]

    print(f"\nTrades entered in 2025: {len(trades_2025_entry)}")
    if len(trades_2025_entry) > 0:
        print("  (These are still open or recently closed)")
        for _, trade in trades_2025_entry.iterrows():
            result = "WIN" if trade['win'] else "LOSS"
            print(f"  {trade['entry_date'].strftime('%Y-%m')} {trade['symbol']:5s} "
                  f"(3M: +{trade['momentum_gain']:.1f}%) -> {trade['pct_return']:+.1f}% [{result}]")

    print(f"\nTrades exited in 2025: {len(trades_2025_exit)}")
    if len(trades_2025_exit) > 0:
        metrics_2025 = calculate_metrics(trades_2025_exit)
        print(f"  Win Rate: {metrics_2025['win_rate']:.1f}%")
        print(f"  Avg Return: {metrics_2025['avg_return']:.1f}%")
        print(f"  Profit Factor: {metrics_2025['profit_factor']:.2f}")

    # Portfolio value over time
    if len(history_df) > 0:
        print('\n' + '-' * 70)
        print('PORTFOLIO VALUE OVER TIME')
        print('-' * 70)
        for _, row in history_df.iterrows():
            pct_change = (row['portfolio_value'] / TOTAL_CAPITAL - 1) * 100
            print(f"{row['date'].strftime('%Y-%m')}: ${row['portfolio_value']:,.0f} "
                  f"({pct_change:+.1f}%) - {row['num_positions']} positions")


if __name__ == '__main__':
    print('=' * 70)
    print('S&P 500 3-MONTH MOMENTUM WINNER BACKTEST')
    print('=' * 70)

    # Load S&P 500 data
    print('\nLoading S&P 500 data...')
    all_data = load_sp500_data()
    print(f'Loaded {len(all_data)} S&P 500 symbols')

    if len(all_data) < 50:
        print("\nWarning: Low S&P 500 coverage. Some stocks may not be in cache.")
        print("Available:", list(all_data.keys())[:20], "...")

    # Run backtest starting from 2023
    print('\nRunning backtest (starting 2023)...')
    results = run_backtest(all_data, start_year=2023)

    print_results(results)

    # Save trades
    if len(results['trades']) > 0:
        results['trades'].to_csv('trades_sp500_3m_momentum.csv', index=False)
        print(f'\nTrades saved to trades_sp500_3m_momentum.csv')

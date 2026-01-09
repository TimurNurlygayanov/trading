#!/usr/bin/env python3
"""
Compare actual portfolio returns between different hold periods.

Simulates a portfolio that:
- Starts with $100,000
- Allocates equal weight to each position
- Tracks overlapping positions over time
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
import warnings

import pandas as pd
import numpy as np
from tqdm import tqdm

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
warnings.filterwarnings('ignore')

from backtest import (
    PolygonClient, get_sp500_tickers, filter_us_exchange_stocks,
    calculate_signals, POLYGON_API_KEY
)


def simulate_portfolio(prices, hold_days, test_start, test_end,
                       signal_type='optimal_filtered', initial_capital=100000,
                       max_positions=20, min_price=10, max_price=400):
    """
    Simulate portfolio with realistic position management.

    Args:
        max_positions: Maximum concurrent positions allowed
    """
    test_start_dt = pd.to_datetime(test_start)
    test_end_dt = pd.to_datetime(test_end)

    spy_df = prices.get('SPY')

    # Collect all signals first
    all_signals = []

    for symbol, df in prices.items():
        if symbol == 'SPY':
            continue
        if len(df) < 500:
            continue

        df = calculate_signals(df, symbol=symbol, spy_df=spy_df)

        # Get test period signals
        test_df = df[(df.index >= test_start_dt) & (df.index <= test_end_dt)]
        test_df = test_df[(test_df['adjusted_close'] >= min_price) &
                          (test_df['adjusted_close'] <= max_price)]

        signal_days = test_df[test_df[signal_type] == True]

        for date in signal_days.index:
            all_signals.append({
                'symbol': symbol,
                'entry_date': date,
                'entry_price': df.loc[date, 'adjusted_close'],
                'df': df,  # Reference for exit price lookup
            })

    # Sort signals by date
    all_signals.sort(key=lambda x: x['entry_date'])

    # Simulate portfolio
    portfolio_value = initial_capital
    cash = initial_capital
    positions = []  # List of active positions

    # Track daily portfolio value
    daily_values = []

    # Get all trading days in the period
    spy_dates = spy_df[(spy_df.index >= test_start_dt) &
                       (spy_df.index <= test_end_dt)].index

    signal_idx = 0

    for current_date in spy_dates:
        # 1. Check for exits (positions that have reached hold period)
        positions_to_close = []
        for i, pos in enumerate(positions):
            days_held = (current_date - pos['entry_date']).days
            if days_held >= hold_days:
                positions_to_close.append(i)

        # Close positions (in reverse order to maintain indices)
        for i in reversed(positions_to_close):
            pos = positions[i]
            # Get exit price
            df = pos['df']
            if current_date in df.index:
                exit_price = df.loc[current_date, 'adjusted_close']
            else:
                # Find nearest available date
                available = df.index[df.index <= current_date]
                if len(available) > 0:
                    exit_price = df.loc[available[-1], 'adjusted_close']
                else:
                    exit_price = pos['entry_price']

            # Calculate P&L
            position_return = exit_price / pos['entry_price'] - 1
            position_value = pos['invested'] * (1 + position_return)
            cash += position_value

            positions.pop(i)

        # 2. Enter new positions (if we have capacity and signals)
        while signal_idx < len(all_signals):
            signal = all_signals[signal_idx]

            if signal['entry_date'] > current_date:
                break  # No more signals for today

            if signal['entry_date'] == current_date:
                # Check if we have capacity
                if len(positions) < max_positions and cash > 0:
                    # Calculate position size (equal weight among max positions)
                    position_size = min(cash, portfolio_value / max_positions)

                    if position_size > 100:  # Minimum position size
                        positions.append({
                            'symbol': signal['symbol'],
                            'entry_date': signal['entry_date'],
                            'entry_price': signal['entry_price'],
                            'invested': position_size,
                            'df': signal['df'],
                        })
                        cash -= position_size

            signal_idx += 1

        # 3. Calculate current portfolio value
        total_position_value = 0
        for pos in positions:
            df = pos['df']
            if current_date in df.index:
                current_price = df.loc[current_date, 'adjusted_close']
            else:
                available = df.index[df.index <= current_date]
                if len(available) > 0:
                    current_price = df.loc[available[-1], 'adjusted_close']
                else:
                    current_price = pos['entry_price']

            position_return = current_price / pos['entry_price'] - 1
            total_position_value += pos['invested'] * (1 + position_return)

        portfolio_value = cash + total_position_value

        daily_values.append({
            'date': current_date,
            'portfolio_value': portfolio_value,
            'cash': cash,
            'num_positions': len(positions),
            'invested': total_position_value,
        })

    return pd.DataFrame(daily_values)


def main():
    print("\n" + "=" * 70)
    print("PORTFOLIO RETURN COMPARISON: 3-MONTH vs 12-MONTH HOLD")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 70)

    polygon_client = PolygonClient(POLYGON_API_KEY)

    # Get universe
    print("\n[1/3] Getting S&P 500 universe...")
    universe = get_sp500_tickers()
    universe = filter_us_exchange_stocks(universe, POLYGON_API_KEY)
    print(f"  Universe: {len(universe)} stocks")

    # Fetch prices
    print("\n[2/3] Fetching price data...")
    end_date = datetime.now().strftime('%Y-%m-%d')
    prices = polygon_client.fetch_prices_parallel(universe, '2010-01-01', end_date)
    prices['SPY'] = polygon_client.get_prices('SPY', '2010-01-01', end_date)
    print(f"  Got prices for {len(prices)} stocks")

    # Test period
    test_start = '2024-01-01'
    test_end = '2025-12-31'

    print(f"\n[3/3] Simulating portfolios ({test_start} to {test_end})...")
    print(f"  Initial capital: $100,000")
    print(f"  Max positions: 20")
    print(f"  Signal: optimal_filtered")

    # Strategy A: 3-month hold
    print("\n  Running Strategy A (3-month hold)...")
    results_3mo = simulate_portfolio(
        prices, hold_days=63,
        test_start=test_start, test_end=test_end,
        signal_type='optimal_filtered',
        max_positions=20
    )

    # Strategy B: 12-month hold
    print("  Running Strategy B (12-month hold)...")
    results_12mo = simulate_portfolio(
        prices, hold_days=252,
        test_start=test_start, test_end=test_end,
        signal_type='optimal_filtered',
        max_positions=20
    )

    # Also get SPY for comparison
    spy_df = prices['SPY']
    spy_df = spy_df[(spy_df.index >= pd.to_datetime(test_start)) &
                    (spy_df.index <= pd.to_datetime(test_end))]
    spy_start = spy_df['adjusted_close'].iloc[0]
    spy_end = spy_df['adjusted_close'].iloc[-1]
    spy_return = (spy_end / spy_start - 1) * 100

    # Results
    print(f"\n{'='*70}")
    print("RESULTS: 2024-2025 PORTFOLIO PERFORMANCE")
    print(f"{'='*70}")

    initial = 100000

    final_3mo = results_3mo['portfolio_value'].iloc[-1]
    final_12mo = results_12mo['portfolio_value'].iloc[-1]

    return_3mo = (final_3mo / initial - 1) * 100
    return_12mo = (final_12mo / initial - 1) * 100

    # Calculate max drawdown
    def max_drawdown(values):
        peak = values.expanding().max()
        dd = (values - peak) / peak
        return dd.min() * 100

    dd_3mo = max_drawdown(results_3mo['portfolio_value'])
    dd_12mo = max_drawdown(results_12mo['portfolio_value'])

    # Average positions
    avg_pos_3mo = results_3mo['num_positions'].mean()
    avg_pos_12mo = results_12mo['num_positions'].mean()

    print(f"\n{'Metric':<25} {'3-Month Hold':>15} {'12-Month Hold':>15} {'SPY B&H':>12}")
    print("-" * 70)
    print(f"{'Final Value':<25} ${final_3mo:>14,.0f} ${final_12mo:>14,.0f} ${initial*(1+spy_return/100):>11,.0f}")
    print(f"{'Total Return':<25} {return_3mo:>14.1f}% {return_12mo:>14.1f}% {spy_return:>11.1f}%")
    print(f"{'Max Drawdown':<25} {dd_3mo:>14.1f}% {dd_12mo:>14.1f}%")
    print(f"{'Avg Positions':<25} {avg_pos_3mo:>14.1f} {avg_pos_12mo:>14.1f}")

    # Monthly breakdown
    print(f"\n{'='*70}")
    print("MONTHLY PORTFOLIO VALUE")
    print(f"{'='*70}")

    results_3mo['month'] = results_3mo['date'].dt.to_period('M')
    results_12mo['month'] = results_12mo['date'].dt.to_period('M')

    monthly_3mo = results_3mo.groupby('month').last()
    monthly_12mo = results_12mo.groupby('month').last()

    print(f"\n{'Month':<12} {'3-Mo Hold':>12} {'Return':>10} {'12-Mo Hold':>12} {'Return':>10}")
    print("-" * 60)

    prev_3mo = initial
    prev_12mo = initial

    for month in monthly_3mo.index:
        val_3mo = monthly_3mo.loc[month, 'portfolio_value']
        val_12mo = monthly_12mo.loc[month, 'portfolio_value'] if month in monthly_12mo.index else prev_12mo

        ret_3mo = (val_3mo / prev_3mo - 1) * 100
        ret_12mo = (val_12mo / prev_12mo - 1) * 100

        print(f"{str(month):<12} ${val_3mo:>11,.0f} {ret_3mo:>+9.1f}% ${val_12mo:>11,.0f} {ret_12mo:>+9.1f}%")

        prev_3mo = val_3mo
        prev_12mo = val_12mo

    # Equity curve comparison
    print(f"\n{'='*70}")
    print("KEY DATES - PORTFOLIO VALUE")
    print(f"{'='*70}")

    key_dates = ['2024-01-02', '2024-04-01', '2024-07-01', '2024-10-01',
                 '2025-01-02', '2025-04-01', '2025-07-01', '2025-10-01']

    print(f"\n{'Date':<12} {'3-Mo Hold':>15} {'12-Mo Hold':>15} {'Positions (3/12)':>18}")
    print("-" * 65)

    for date_str in key_dates:
        date = pd.to_datetime(date_str)

        # Find closest date in results
        idx_3mo = results_3mo[results_3mo['date'] >= date].index
        idx_12mo = results_12mo[results_12mo['date'] >= date].index

        if len(idx_3mo) > 0 and len(idx_12mo) > 0:
            row_3mo = results_3mo.loc[idx_3mo[0]]
            row_12mo = results_12mo.loc[idx_12mo[0]]

            print(f"{date_str:<12} ${row_3mo['portfolio_value']:>14,.0f} ${row_12mo['portfolio_value']:>14,.0f} {int(row_3mo['num_positions']):>8} / {int(row_12mo['num_positions']):<8}")


if __name__ == '__main__':
    main()

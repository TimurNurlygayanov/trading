#!/usr/bin/env python3
"""
Test different hold periods and longer test windows.

Since we're not using ML (just historical win rate filtering),
we can test on longer periods without overfitting concerns.
"""

import sys
from pathlib import Path
from datetime import datetime
import warnings

import pandas as pd
import numpy as np
import pandas_ta as ta
from tqdm import tqdm

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
warnings.filterwarnings('ignore')

from backtest import (
    PolygonClient, get_sp500_tickers, filter_us_exchange_stocks,
    calculate_signals, POLYGON_API_KEY, SECTOR_MAP, BAD_SECTORS
)


def run_hold_period_test(prices, hold_days, test_start, test_end=None,
                         signal_type='optimal_filtered', min_price=10, max_price=400):
    """Run backtest for a specific hold period."""

    test_start_dt = pd.to_datetime(test_start)
    test_end_dt = pd.to_datetime(test_end) if test_end else datetime.now()

    spy_df = prices.get('SPY')
    trades = []

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
            entry_price = df.loc[date, 'adjusted_close']

            # Calculate exit
            future_dates = df.index[df.index > date]

            if len(future_dates) >= hold_days:
                exit_date = future_dates[hold_days - 1]
                exit_price = df.loc[exit_date, 'adjusted_close']
                is_complete = True
            else:
                continue  # Skip incomplete trades for fair comparison

            exit_return = exit_price / entry_price - 1

            trades.append({
                'symbol': symbol,
                'entry_date': date,
                'exit_date': exit_date,
                'return': exit_return,
                'hold_days': hold_days,
            })

    return trades


def main():
    print("\n" + "=" * 70)
    print("HOLD PERIOD COMPARISON TEST")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 70)

    polygon_client = PolygonClient(POLYGON_API_KEY)

    # Get universe
    print("\n[1/3] Getting S&P 500 universe...")
    universe = get_sp500_tickers()
    universe = filter_us_exchange_stocks(universe, POLYGON_API_KEY)
    print(f"  Universe: {len(universe)} stocks")

    # Fetch prices (need more history for longer holds)
    print("\n[2/3] Fetching price data...")
    end_date = datetime.now().strftime('%Y-%m-%d')
    prices = polygon_client.fetch_prices_parallel(universe, '2010-01-01', end_date)
    prices['SPY'] = polygon_client.get_prices('SPY', '2010-01-01', end_date)
    print(f"  Got prices for {len(prices)} stocks")

    # Test configurations
    hold_periods = [63, 126, 189, 252, 378, 504]  # ~3mo, 6mo, 9mo, 1yr, 1.5yr, 2yr
    test_periods = [
        ('2015-01-01', '2024-06-30', '2015-2024 (10 years)'),
        ('2020-01-01', '2024-06-30', '2020-2024 (5 years)'),
    ]

    signal_types = [
        ('optimal_signal', 'Optimal'),
        ('optimal_filtered', 'Optimal+Filters'),
    ]

    print("\n[3/3] Running hold period tests...")

    all_results = []

    for test_start, test_end, period_name in test_periods:
        print(f"\n{'='*70}")
        print(f"TEST PERIOD: {period_name}")
        print(f"{'='*70}")

        for signal_type, signal_name in signal_types:
            print(f"\n  Signal: {signal_name}")
            print(f"  {'Hold':<10} {'Trades':>8} {'Win Rate':>10} {'Avg Ret':>10} {'Median':>10} {'Sharpe':>8}")
            print(f"  {'-'*56}")

            for hold_days in hold_periods:
                trades = run_hold_period_test(
                    prices, hold_days, test_start, test_end,
                    signal_type=signal_type
                )

                if len(trades) < 10:
                    print(f"  {hold_days:>3}d       {'<10 trades':>8}")
                    continue

                returns = [t['return'] for t in trades]
                wins = sum(1 for r in returns if r > 0)
                win_rate = wins / len(trades)
                avg_ret = np.mean(returns)
                med_ret = np.median(returns)

                # Annualized Sharpe (rough approximation)
                ann_factor = 252 / hold_days
                sharpe = (np.mean(returns) / np.std(returns)) * np.sqrt(ann_factor) if np.std(returns) > 0 else 0

                print(f"  {hold_days:>3}d ({hold_days//21:>2}mo) {len(trades):>8} {win_rate*100:>9.1f}% {avg_ret*100:>+9.1f}% {med_ret*100:>+9.1f}% {sharpe:>7.2f}")

                all_results.append({
                    'period': period_name,
                    'signal': signal_name,
                    'hold_days': hold_days,
                    'hold_months': hold_days // 21,
                    'trades': len(trades),
                    'win_rate': win_rate,
                    'avg_return': avg_ret,
                    'median_return': med_ret,
                    'sharpe': sharpe,
                })

    # Summary: Best hold period
    print(f"\n{'='*70}")
    print("SUMMARY: BEST HOLD PERIOD BY METRIC")
    print(f"{'='*70}")

    df_results = pd.DataFrame(all_results)

    # For each signal type, find best hold period
    for signal_name in ['Optimal', 'Optimal+Filters']:
        signal_df = df_results[df_results['signal'] == signal_name]

        if len(signal_df) == 0:
            continue

        print(f"\n{signal_name}:")

        # Average across test periods
        avg_by_hold = signal_df.groupby('hold_days').agg({
            'win_rate': 'mean',
            'avg_return': 'mean',
            'median_return': 'mean',
            'sharpe': 'mean',
            'trades': 'sum',
        }).round(3)

        best_winrate = avg_by_hold['win_rate'].idxmax()
        best_avg = avg_by_hold['avg_return'].idxmax()
        best_median = avg_by_hold['median_return'].idxmax()
        best_sharpe = avg_by_hold['sharpe'].idxmax()

        print(f"  Best Win Rate:    {best_winrate}d ({best_winrate//21}mo) = {avg_by_hold.loc[best_winrate, 'win_rate']*100:.1f}%")
        print(f"  Best Avg Return:  {best_avg}d ({best_avg//21}mo) = {avg_by_hold.loc[best_avg, 'avg_return']*100:+.1f}%")
        print(f"  Best Median:      {best_median}d ({best_median//21}mo) = {avg_by_hold.loc[best_median, 'median_return']*100:+.1f}%")
        print(f"  Best Sharpe:      {best_sharpe}d ({best_sharpe//21}mo) = {avg_by_hold.loc[best_sharpe, 'sharpe']:.2f}")

    # Detailed table
    print(f"\n{'='*70}")
    print("DETAILED RESULTS (AVERAGED ACROSS TEST PERIODS)")
    print(f"{'='*70}")

    for signal_name in ['Optimal', 'Optimal+Filters']:
        signal_df = df_results[df_results['signal'] == signal_name]

        if len(signal_df) == 0:
            continue

        print(f"\n{signal_name}:")
        print(f"{'Hold':<12} {'Trades':>8} {'Win Rate':>10} {'Avg Ret':>10} {'Median':>10} {'Sharpe':>8}")
        print("-" * 60)

        avg_by_hold = signal_df.groupby('hold_days').agg({
            'win_rate': 'mean',
            'avg_return': 'mean',
            'median_return': 'mean',
            'sharpe': 'mean',
            'trades': 'sum',
        })

        for hold_days in hold_periods:
            if hold_days in avg_by_hold.index:
                row = avg_by_hold.loc[hold_days]
                print(f"{hold_days:>3}d ({hold_days//21:>2}mo)  {int(row['trades']):>8} {row['win_rate']*100:>9.1f}% {row['avg_return']*100:>+9.1f}% {row['median_return']*100:>+9.1f}% {row['sharpe']:>7.2f}")


if __name__ == '__main__':
    main()

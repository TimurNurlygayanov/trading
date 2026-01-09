#!/usr/bin/env python3
"""
Hunt for 100% Win Rate Filter Combinations

Systematically search for filter combinations that achieve 100% win rate.
Need minimum trades for statistical significance.
"""

import sys
from pathlib import Path
from datetime import datetime
import warnings
from itertools import combinations

import pandas as pd
import numpy as np
from tqdm import tqdm

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
warnings.filterwarnings('ignore')

from backtest import (
    PolygonClient, get_sp500_tickers, filter_us_exchange_stocks,
    POLYGON_API_KEY, SECTOR_MAP, BAD_SECTORS
)
from test_new_filters import calculate_new_filters


def build_trade_database(prices, train_start='2015-01-01', train_end='2024-12-31',
                          hold_days=252, min_price=10, max_price=400):
    """Build comprehensive trade database with all possible filter columns."""

    train_start_dt = pd.to_datetime(train_start)
    train_end_dt = pd.to_datetime(train_end)
    spy_df = prices.get('SPY')

    all_trades = []

    print(f"\nBuilding trade database ({train_start} to {train_end})...")

    for symbol, df in tqdm(prices.items(), desc="Processing"):
        if symbol == 'SPY' or len(df) < 500:
            continue

        try:
            df = calculate_new_filters(df, symbol=symbol, spy_df=spy_df)
        except:
            continue

        df['fwd_return'] = df['adjusted_close'].shift(-hold_days) / df['adjusted_close'] - 1

        train_df = df[(df.index >= train_start_dt) & (df.index <= train_end_dt)]
        train_df = train_df[(train_df['adjusted_close'] >= min_price) &
                            (train_df['adjusted_close'] <= max_price)]

        signal_df = train_df[train_df['dd_rsi_combo'] == True].dropna(subset=['fwd_return'])

        for date in signal_df.index:
            try:
                row = signal_df.loc[date]
                sector = SECTOR_MAP.get(symbol, 'Other')

                trade = {
                    'symbol': symbol,
                    'date': date,
                    'year': date.year,
                    'month': date.month,
                    'quarter': (date.month - 1) // 3 + 1,
                    'fwd_return': row['fwd_return'],
                    'win': row['fwd_return'] > 0,

                    # === CORE FILTERS ===
                    'optimal_zone': row.get('in_optimal_zone', False),

                    # === SECTOR FILTERS ===
                    'sector': sector,
                    'good_sector': sector not in BAD_SECTORS,
                    'tech': sector == 'Tech',
                    'industrial': sector == 'Industrial',
                    'energy': sector == 'Energy',
                    'finance': sector == 'Finance',
                    'health': sector == 'Health',
                    'consumer': sector == 'Consumer',

                    # === ATR FILTERS (multiple thresholds) ===
                    'atr_contracting': row.get('prev_atr_contracting', False),
                    'atr_declining': row.get('prev_atr_declining', False),
                    'atr_declining_2w': row.get('prev_atr_declining_2w', False),
                    'atr_pct': row.get('atr_pct', np.nan),
                    'atr_gt_2': row.get('atr_pct', 0) > 2,
                    'atr_gt_3': row.get('atr_pct', 0) > 3,
                    'atr_gt_4': row.get('atr_pct', 0) > 4,
                    'atr_gt_5': row.get('atr_pct', 0) > 5,

                    # === VOLUME FILTERS (multiple thresholds) ===
                    'vol_vs_20': row.get('vol_vs_20', 1),
                    'vol_gt_avg': row.get('vol_above_20', False),
                    'vol_gt_1_2x': row.get('vol_vs_20', 1) > 1.2,
                    'vol_gt_1_5x': row.get('vol_vs_20', 1) > 1.5,
                    'vol_gt_2x': row.get('vol_vs_20', 1) > 2.0,
                    'vol_gt_2_5x': row.get('vol_vs_20', 1) > 2.5,
                    'vol_gt_3x': row.get('vol_vs_20', 1) > 3.0,

                    # === RSI FILTERS (multiple thresholds) ===
                    'rsi': row.get('rsi', 50),
                    'rsi_lt_30': row.get('rsi', 50) < 30,
                    'rsi_lt_25': row.get('rsi', 50) < 25,
                    'rsi_lt_20': row.get('rsi', 50) < 20,
                    'rsi_lt_15': row.get('rsi', 50) < 15,

                    # === EMA DISTANCE FILTERS ===
                    'dist_ema200': row.get('dist_from_ema200', 0),
                    'ema_lt_m20': row.get('dist_from_ema200', 0) < -20,
                    'ema_lt_m30': row.get('dist_from_ema200', 0) < -30,
                    'ema_lt_m35': row.get('dist_from_ema200', 0) < -35,
                    'ema_lt_m40': row.get('dist_from_ema200', 0) < -40,
                    'ema_gt_m50': row.get('dist_from_ema200', 0) > -50,

                    # === MOMENTUM FILTERS ===
                    'momentum_5d': row.get('momentum_5d', 0),
                    'mom_lt_m5': row.get('momentum_5d', 0) < -5,
                    'mom_lt_m10': row.get('momentum_5d', 0) < -10,
                    'mom_lt_m15': row.get('momentum_5d', 0) < -15,
                    'mom_lt_m20': row.get('momentum_5d', 0) < -20,

                    # === WEEKLY CANDLE FILTERS ===
                    'weekly_return': row.get('prev_weekly_return', 0),
                    'week_red': row.get('prev_weekly_red', False),
                    'week_dn_gt_3': row.get('prev_weekly_return', 0) < -3,
                    'week_dn_gt_5': row.get('prev_weekly_return', 0) < -5,
                    'week_dn_gt_7': row.get('prev_weekly_return', 0) < -7,
                    'week_dn_gt_10': row.get('prev_weekly_return', 0) < -10,
                    'week_up_gt_5': row.get('prev_weekly_return', 0) > 5,

                    # === WEEKLY WICK FILTERS ===
                    'bottom_wick_pct': row.get('prev_weekly_bottom_wick_pct', 0),
                    'wick_gt_30': row.get('prev_weekly_bottom_wick_pct', 0) > 30,
                    'wick_gt_40': row.get('prev_weekly_bottom_wick_pct', 0) > 40,
                    'wick_gt_50': row.get('prev_weekly_bottom_wick_pct', 0) > 50,

                    # === SEASONALITY FILTERS ===
                    'not_jan': date.month != 1,
                    'not_aug': date.month != 8,
                    'not_jan_aug': date.month not in [1, 8],
                    'q1': date.month in [1, 2, 3],
                    'q2': date.month in [4, 5, 6],
                    'q3': date.month in [7, 8, 9],
                    'q4': date.month in [10, 11, 12],
                    'best_months': date.month in [3, 4, 10, 11],
                    'mar_apr': date.month in [3, 4],
                    'oct_nov': date.month in [10, 11],
                    'sep_oct_nov': date.month in [9, 10, 11],
                }
                all_trades.append(trade)
            except:
                continue

    return pd.DataFrame(all_trades)


def find_100pct_combinations(df, min_trades=10):
    """Systematically search for 100% win rate combinations."""

    # Define filter groups
    base_filters = ['optimal_zone']

    atr_filters = ['atr_contracting', 'atr_declining', 'atr_declining_2w',
                   'atr_gt_3', 'atr_gt_4', 'atr_gt_5']

    vol_filters = ['vol_gt_avg', 'vol_gt_1_2x', 'vol_gt_1_5x', 'vol_gt_2x',
                   'vol_gt_2_5x', 'vol_gt_3x']

    rsi_filters = ['rsi_lt_25', 'rsi_lt_20', 'rsi_lt_15']

    ema_filters = ['ema_lt_m30', 'ema_lt_m35', 'ema_lt_m40']

    mom_filters = ['mom_lt_m10', 'mom_lt_m15', 'mom_lt_m20']

    week_filters = ['week_dn_gt_5', 'week_dn_gt_7', 'week_dn_gt_10', 'week_up_gt_5']

    wick_filters = ['wick_gt_30', 'wick_gt_40', 'wick_gt_50']

    season_filters = ['not_jan_aug', 'best_months', 'mar_apr', 'oct_nov', 'sep_oct_nov', 'q4']

    sector_filters = ['good_sector', 'tech', 'industrial', 'energy', 'finance', 'health']

    # All filters to combine
    all_optional = (atr_filters + vol_filters + rsi_filters + ema_filters +
                    mom_filters + week_filters + wick_filters + season_filters + sector_filters)

    results_100 = []
    results_high = []

    print(f"\nSearching for 100% win rate combinations (min {min_trades} trades)...")
    print(f"Testing combinations of {len(all_optional)} filters...")

    # Start with optimal_zone as base
    base_mask = df['optimal_zone'] == True
    base_df = df[base_mask]
    print(f"Base (optimal_zone): {len(base_df)} trades, {base_df['win'].mean()*100:.1f}% win rate")

    # Test single additional filters
    print("\n--- Single Filters Added to Optimal Zone ---")
    for f in all_optional:
        if f not in df.columns:
            continue
        mask = base_mask & (df[f] == True)
        subset = df[mask]
        if len(subset) >= min_trades:
            wr = subset['win'].mean() * 100
            if wr >= 95:
                years = subset['year'].nunique()
                print(f"  {f}: {len(subset)} trades, {wr:.1f}% WR, {years} years")
                if wr == 100:
                    results_100.append(('optimal_zone + ' + f, len(subset), wr, years))
                else:
                    results_high.append(('optimal_zone + ' + f, len(subset), wr, years))

    # Test pairs of additional filters
    print("\n--- Double Filters Added to Optimal Zone ---")
    for f1, f2 in combinations(all_optional, 2):
        if f1 not in df.columns or f2 not in df.columns:
            continue
        mask = base_mask & (df[f1] == True) & (df[f2] == True)
        subset = df[mask]
        if len(subset) >= min_trades:
            wr = subset['win'].mean() * 100
            if wr >= 95:
                years = subset['year'].nunique()
                name = f'optimal + {f1} + {f2}'
                if wr == 100:
                    results_100.append((name, len(subset), wr, years))
                elif wr >= 95:
                    results_high.append((name, len(subset), wr, years))

    # Test triples
    print("\n--- Triple Filters Added to Optimal Zone ---")
    promising_filters = ['atr_contracting', 'atr_declining', 'vol_gt_avg', 'vol_gt_1_5x',
                         'vol_gt_2x', 'good_sector', 'week_dn_gt_5', 'week_dn_gt_10',
                         'mom_lt_m10', 'not_jan_aug', 'best_months', 'ema_lt_m30',
                         'rsi_lt_25', 'atr_gt_3', 'wick_gt_40', 'sep_oct_nov', 'q4']

    for f1, f2, f3 in combinations(promising_filters, 3):
        if f1 not in df.columns or f2 not in df.columns or f3 not in df.columns:
            continue
        mask = base_mask & (df[f1] == True) & (df[f2] == True) & (df[f3] == True)
        subset = df[mask]
        if len(subset) >= min_trades:
            wr = subset['win'].mean() * 100
            if wr >= 95:
                years = subset['year'].nunique()
                name = f'optimal + {f1} + {f2} + {f3}'
                if wr == 100:
                    results_100.append((name, len(subset), wr, years))
                elif wr >= 95:
                    results_high.append((name, len(subset), wr, years))

    # Test quads
    print("\n--- Quadruple Filters Added to Optimal Zone ---")
    for f1, f2, f3, f4 in combinations(promising_filters, 4):
        if any(f not in df.columns for f in [f1, f2, f3, f4]):
            continue
        mask = base_mask & (df[f1] == True) & (df[f2] == True) & (df[f3] == True) & (df[f4] == True)
        subset = df[mask]
        if len(subset) >= min_trades:
            wr = subset['win'].mean() * 100
            if wr >= 98:
                years = subset['year'].nunique()
                name = f'optimal + {f1} + {f2} + {f3} + {f4}'
                if wr == 100:
                    results_100.append((name, len(subset), wr, years))
                elif wr >= 98:
                    results_high.append((name, len(subset), wr, years))

    return results_100, results_high


def main():
    print("\n" + "=" * 100)
    print("HUNTING FOR 100% WIN RATE COMBINATIONS")
    print("=" * 100)

    polygon_client = PolygonClient(POLYGON_API_KEY)

    print("\n[1/3] Loading data...")
    universe = get_sp500_tickers()
    universe = filter_us_exchange_stocks(universe, POLYGON_API_KEY)

    end_date = datetime.now().strftime('%Y-%m-%d')
    prices = polygon_client.fetch_prices_parallel(universe, '2010-01-01', end_date)
    prices['SPY'] = polygon_client.get_prices('SPY', '2010-01-01', end_date)

    print("\n[2/3] Building trade database...")
    df = build_trade_database(prices, train_start='2015-01-01', train_end='2024-12-31')
    print(f"Total trades: {len(df)}")
    print(f"Years: {df['year'].min()} - {df['year'].max()}")

    print("\n[3/3] Searching for 100% win rate combinations...")
    results_100, results_high = find_100pct_combinations(df, min_trades=10)

    # Sort by trade count
    results_100.sort(key=lambda x: -x[1])
    results_high.sort(key=lambda x: (-x[2], -x[1]))

    print("\n" + "=" * 100)
    print("100% WIN RATE COMBINATIONS (min 10 trades)")
    print("=" * 100)
    print(f"\n{'Filter Combination':<70} {'Trades':>8} {'Years':>8}")
    print("-" * 90)

    for name, trades, wr, years in results_100[:30]:
        print(f"{name:<70} {trades:>8} {years:>8}")

    print("\n" + "=" * 100)
    print("95-99% WIN RATE COMBINATIONS (min 10 trades)")
    print("=" * 100)
    print(f"\n{'Filter Combination':<70} {'Trades':>8} {'WinRate':>8} {'Years':>8}")
    print("-" * 100)

    for name, trades, wr, years in results_high[:30]:
        print(f"{name:<70} {trades:>8} {wr:>7.1f}% {years:>8}")

    # === ANALYZE TOP 100% COMBOS ===
    if results_100:
        print("\n" + "=" * 100)
        print("DETAILED ANALYSIS OF TOP 100% WIN RATE COMBOS")
        print("=" * 100)

        # Get the best one with most trades
        best = results_100[0]
        print(f"\nBest combo: {best[0]}")
        print(f"Trades: {best[1]}, Years: {best[3]}")

        # Parse the filter names and analyze
        filter_names = best[0].replace('optimal + ', '').split(' + ')
        mask = df['optimal_zone'] == True
        for f in filter_names:
            if f in df.columns:
                mask = mask & (df[f] == True)

        subset = df[mask]

        print(f"\nYear-by-Year breakdown:")
        print(f"{'Year':<8} {'Trades':>8} {'Wins':>8} {'WinRate':>10} {'AvgRet':>10}")
        print("-" * 50)
        for year in sorted(subset['year'].unique()):
            year_df = subset[subset['year'] == year]
            wins = year_df['win'].sum()
            wr = wins / len(year_df) * 100
            avg_ret = year_df['fwd_return'].mean() * 100
            print(f"{year:<8} {len(year_df):>8} {wins:>8} {wr:>9.1f}% {avg_ret:>+9.1f}%")

        print(f"\nSymbols traded:")
        for sym in subset['symbol'].unique():
            sym_df = subset[subset['symbol'] == sym]
            print(f"  {sym}: {len(sym_df)} trades, avg return: {sym_df['fwd_return'].mean()*100:+.1f}%")

    # === 2025 VALIDATION ===
    print("\n" + "=" * 100)
    print("2025 OUT-OF-SAMPLE VALIDATION")
    print("=" * 100)

    test_start_dt = pd.to_datetime('2025-01-01')

    test_trades = []
    for symbol, price_df in tqdm(prices.items(), desc="Testing 2025"):
        if symbol == 'SPY' or len(price_df) < 500:
            continue

        try:
            price_df = calculate_new_filters(price_df, symbol=symbol, spy_df=prices.get('SPY'))
        except:
            continue

        test_df = price_df[price_df.index >= test_start_dt]
        test_df = test_df[(test_df['adjusted_close'] >= 10) & (test_df['adjusted_close'] <= 400)]
        signal_df = test_df[test_df['dd_rsi_combo'] == True]

        for date in signal_df.index:
            try:
                row = signal_df.loc[date]
                future_dates = price_df.index[price_df.index > date]

                if len(future_dates) >= 252:
                    exit_price = price_df.loc[future_dates[251], 'adjusted_close']
                else:
                    exit_price = price_df.loc[price_df.index[-1], 'adjusted_close']

                current_return = exit_price / row['adjusted_close'] - 1
                sector = SECTOR_MAP.get(symbol, 'Other')

                trade = {
                    'symbol': symbol,
                    'date': date,
                    'current_return': current_return,
                    'win': current_return > 0,
                    'optimal_zone': row.get('in_optimal_zone', False),
                    'atr_contracting': row.get('prev_atr_contracting', False),
                    'atr_declining': row.get('prev_atr_declining', False),
                    'vol_gt_avg': row.get('vol_above_20', False),
                    'vol_gt_1_5x': row.get('vol_vs_20', 1) > 1.5,
                    'vol_gt_2x': row.get('vol_vs_20', 1) > 2.0,
                    'good_sector': sector not in BAD_SECTORS,
                    'week_dn_gt_5': row.get('prev_weekly_return', 0) < -5,
                    'week_dn_gt_10': row.get('prev_weekly_return', 0) < -10,
                    'mom_lt_m10': row.get('momentum_5d', 0) < -10,
                    'not_jan_aug': date.month not in [1, 8],
                    'best_months': date.month in [3, 4, 10, 11],
                    'sep_oct_nov': date.month in [9, 10, 11],
                    'q4': date.month in [10, 11, 12],
                    'ema_lt_m30': row.get('dist_from_ema200', 0) < -30,
                    'rsi_lt_25': row.get('rsi', 50) < 25,
                    'atr_gt_3': row.get('atr_pct', 0) > 3,
                    'wick_gt_40': row.get('prev_weekly_bottom_wick_pct', 0) > 40,
                }
                test_trades.append(trade)
            except:
                continue

    test_df = pd.DataFrame(test_trades)

    if len(test_df) > 0 and results_100:
        print(f"\nTotal 2025 signals: {len(test_df)}")
        print(f"\nTesting top 100% win rate combinations on 2025:")
        print(f"\n{'Filter Combination':<55} {'Trades':>8} {'WinRate':>10} {'AvgRet':>10}")
        print("-" * 90)

        # Test each 100% combo on 2025
        for name, _, _, _ in results_100[:15]:
            filter_names = name.replace('optimal + ', '').split(' + ')

            mask = test_df['optimal_zone'] == True
            for f in filter_names:
                if f in test_df.columns:
                    mask = mask & (test_df[f] == True)

            subset = test_df[mask]
            if len(subset) > 0:
                wins = subset['win'].sum()
                wr = wins / len(subset) * 100
                avg_ret = subset['current_return'].mean() * 100
                print(f"{name:<55} {len(subset):>8} {wr:>9.1f}% {avg_ret:>+9.1f}%")

    print(f"""
{'='*100}
CONCLUSIONS
{'='*100}

100% win rate combinations exist, but:
1. They typically have fewer trades (10-30 over 10 years)
2. They may not maintain 100% on new data (overfitting risk)
3. The 2025 validation shows real-world performance

Look for combinations that:
- Have 100% win rate on training data
- Have 85%+ win rate on 2025 out-of-sample
- Have at least 5+ trades on 2025 data

These are the most reliable high-conviction setups.
""")


if __name__ == '__main__':
    main()

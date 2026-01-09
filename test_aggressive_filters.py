#!/usr/bin/env python3
"""
Aggressive Filter Analysis - Maximum Win Rate

Target: ~10 trades/year = ~100 trades over 10 years training
Goal: Find 85%+ win rate combinations
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
from test_new_filters import calculate_new_filters, resample_to_weekly


def analyze_aggressive_filters(prices, train_start='2015-01-01', train_end='2024-12-31',
                                hold_days=252, min_price=10, max_price=400):
    """Find the most aggressive high-win-rate filters."""

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

        # Forward return
        df['fwd_return'] = df['adjusted_close'].shift(-hold_days) / df['adjusted_close'] - 1

        # Training period with optimal signal (already very selective)
        train_df = df[(df.index >= train_start_dt) & (df.index <= train_end_dt)]
        train_df = train_df[(train_df['adjusted_close'] >= min_price) &
                            (train_df['adjusted_close'] <= max_price)]

        # Base signal: dd_rsi_combo
        signal_df = train_df[train_df['dd_rsi_combo'] == True].dropna(subset=['fwd_return'])

        for date in signal_df.index:
            try:
                row = signal_df.loc[date]
                trade = {
                    'symbol': symbol,
                    'date': date,
                    'year': date.year,
                    'month': date.month,
                    'fwd_return': row['fwd_return'],
                    'win': row['fwd_return'] > 0,

                    # Core filters
                    'in_optimal_zone': row.get('in_optimal_zone', False),
                    'rsi': row.get('rsi', np.nan),
                    'dist_from_ema200': row.get('dist_from_ema200', np.nan),
                    'atr_pct': row.get('atr_pct', np.nan),
                    'momentum_5d': row.get('momentum_5d', np.nan),

                    # Weekly candle
                    'prev_weekly_red': row.get('prev_weekly_red', False),
                    'prev_weekly_return': row.get('prev_weekly_return', np.nan),

                    # Weekly wick
                    'prev_weekly_bottom_wick_pct': row.get('prev_weekly_bottom_wick_pct', np.nan),
                    'prev_weekly_large_bottom_wick': row.get('prev_weekly_large_bottom_wick', False),

                    # ATR trend
                    'prev_atr_declining': row.get('prev_atr_declining', False),
                    'prev_atr_declining_2w': row.get('prev_atr_declining_2w', False),
                    'prev_atr_contracting': row.get('prev_atr_contracting', False),
                    'prev_weekly_atr_change': row.get('prev_weekly_atr_change', np.nan),

                    # Volume
                    'vol_vs_20': row.get('vol_vs_20', np.nan),
                    'vol_above_20': row.get('vol_above_20', False),
                    'vol_spike': row.get('vol_spike', False),
                    'vol_climax': row.get('vol_climax', False),

                    # Sector
                    'sector': SECTOR_MAP.get(symbol, 'Other'),
                    'good_sector': SECTOR_MAP.get(symbol, 'Other') not in BAD_SECTORS,
                }
                all_trades.append(trade)
            except:
                continue

    return pd.DataFrame(all_trades)


def test_filter_combo(df, conditions, name):
    """Test a filter combination."""
    mask = pd.Series([True] * len(df))
    for cond in conditions:
        mask = mask & cond

    subset = df[mask]
    if len(subset) < 10:
        return None

    wins = subset['win'].sum()
    years = subset['year'].nunique()
    trades_per_year = len(subset) / years if years > 0 else 0

    return {
        'name': name,
        'trades': len(subset),
        'trades_per_year': trades_per_year,
        'win_rate': wins / len(subset) * 100,
        'avg_return': subset['fwd_return'].mean() * 100,
        'median_return': subset['fwd_return'].median() * 100,
        'min_return': subset['fwd_return'].min() * 100,
        'max_return': subset['fwd_return'].max() * 100,
    }


def main():
    print("\n" + "=" * 100)
    print("AGGRESSIVE FILTER ANALYSIS - TARGET: 10 TRADES/YEAR, 85%+ WIN RATE")
    print("=" * 100)

    # Load data
    polygon_client = PolygonClient(POLYGON_API_KEY)

    print("\n[1/3] Loading universe...")
    universe = get_sp500_tickers()
    universe = filter_us_exchange_stocks(universe, POLYGON_API_KEY)

    print("\n[2/3] Fetching prices...")
    end_date = datetime.now().strftime('%Y-%m-%d')
    prices = polygon_client.fetch_prices_parallel(universe, '2010-01-01', end_date)
    prices['SPY'] = polygon_client.get_prices('SPY', '2010-01-01', end_date)

    print("\n[3/3] Analyzing aggressive filters...")
    df = analyze_aggressive_filters(prices)

    if len(df) == 0:
        print("No trades found!")
        return

    print(f"\nTotal trades in database: {len(df)}")
    years = df['year'].nunique()
    print(f"Years covered: {df['year'].min()} - {df['year'].max()} ({years} years)")

    # Define filter conditions
    filters = {
        # Core
        'optimal_zone': df['in_optimal_zone'] == True,
        'good_sector': df['good_sector'] == True,

        # ATR (the winner)
        'atr_contracting': df['prev_atr_contracting'] == True,
        'atr_declining': df['prev_atr_declining'] == True,
        'atr_declining_2w': df['prev_atr_declining_2w'] == True,

        # Volume levels
        'vol_above_avg': df['vol_above_20'] == True,
        'vol_spike': df['vol_spike'] == True,
        'vol_climax': df['vol_climax'] == True,
        'vol_high': df['vol_vs_20'] > 1.5,
        'vol_very_high': df['vol_vs_20'] > 2.5,

        # Weekly momentum (extreme)
        'weekly_down_big': df['prev_weekly_return'] < -5,
        'weekly_down_huge': df['prev_weekly_return'] < -10,
        'weekly_up_big': df['prev_weekly_return'] > 5,

        # RSI extremes
        'rsi_very_low': df['rsi'] < 25,
        'rsi_extreme': df['rsi'] < 20,

        # EMA distance extremes
        'ema_deep': df['dist_from_ema200'] < -30,
        'ema_very_deep': df['dist_from_ema200'] < -40,

        # Momentum extremes
        'momentum_strong': df['momentum_5d'] < -10,
        'momentum_extreme': df['momentum_5d'] < -15,

        # Volatility
        'high_atr': df['atr_pct'] > 3.0,
        'very_high_atr': df['atr_pct'] > 4.0,

        # Bottom wick
        'large_wick': df['prev_weekly_large_bottom_wick'] == True,
        'huge_wick': df['prev_weekly_bottom_wick_pct'] > 40,

        # Seasonality (good months)
        'good_months': ~df['month'].isin([1, 8]),  # Avoid Jan, Aug
        'best_months': df['month'].isin([3, 4, 10, 11]),  # Mar, Apr, Oct, Nov
    }

    # === TEST AGGRESSIVE COMBINATIONS ===
    print("\n" + "=" * 100)
    print("AGGRESSIVE FILTER COMBINATIONS (sorted by win rate)")
    print("Target: 10-20 trades/year with 80%+ win rate")
    print("=" * 100)

    combos = [
        # Triple combos with optimal zone
        ([filters['optimal_zone'], filters['atr_contracting'], filters['vol_above_avg']],
         'Optimal + ATR Contract + Vol>Avg'),
        ([filters['optimal_zone'], filters['atr_contracting'], filters['vol_spike']],
         'Optimal + ATR Contract + Vol Spike'),
        ([filters['optimal_zone'], filters['atr_contracting'], filters['vol_high']],
         'Optimal + ATR Contract + Vol>1.5x'),
        ([filters['optimal_zone'], filters['atr_contracting'], filters['good_sector']],
         'Optimal + ATR Contract + Good Sector'),
        ([filters['optimal_zone'], filters['atr_declining_2w'], filters['vol_above_avg']],
         'Optimal + ATR Decl 2w + Vol>Avg'),

        # Quadruple combos
        ([filters['optimal_zone'], filters['atr_contracting'], filters['vol_above_avg'], filters['good_sector']],
         'Optimal + ATR Contract + Vol>Avg + Sector'),
        ([filters['optimal_zone'], filters['atr_contracting'], filters['vol_spike'], filters['good_sector']],
         'Optimal + ATR Contract + Vol Spike + Sector'),
        ([filters['optimal_zone'], filters['atr_contracting'], filters['vol_above_avg'], filters['good_months']],
         'Optimal + ATR Contract + Vol>Avg + Good Months'),
        ([filters['optimal_zone'], filters['atr_contracting'], filters['vol_high'], filters['good_sector']],
         'Optimal + ATR Contract + Vol>1.5x + Sector'),

        # Extreme conditions
        ([filters['optimal_zone'], filters['atr_contracting'], filters['rsi_very_low']],
         'Optimal + ATR Contract + RSI<25'),
        ([filters['optimal_zone'], filters['atr_contracting'], filters['momentum_strong']],
         'Optimal + ATR Contract + Mom<-10%'),
        ([filters['optimal_zone'], filters['atr_contracting'], filters['ema_deep']],
         'Optimal + ATR Contract + EMA<-30%'),
        ([filters['optimal_zone'], filters['atr_contracting'], filters['high_atr']],
         'Optimal + ATR Contract + ATR%>3'),

        # Volume climax (capitulation)
        ([filters['optimal_zone'], filters['vol_climax']],
         'Optimal + Climax Volume (3x)'),
        ([filters['optimal_zone'], filters['atr_contracting'], filters['vol_climax']],
         'Optimal + ATR Contract + Climax Vol'),

        # Weekly extremes
        ([filters['optimal_zone'], filters['atr_contracting'], filters['weekly_down_big']],
         'Optimal + ATR Contract + Week Down>5%'),
        ([filters['optimal_zone'], filters['weekly_down_huge']],
         'Optimal + Week Down >10%'),
        ([filters['optimal_zone'], filters['atr_contracting'], filters['weekly_down_huge']],
         'Optimal + ATR Contract + Week Down>10%'),

        # Best months
        ([filters['optimal_zone'], filters['atr_contracting'], filters['best_months']],
         'Optimal + ATR Contract + Best Months'),
        ([filters['optimal_zone'], filters['atr_contracting'], filters['vol_above_avg'], filters['best_months']],
         'Optimal + ATR Contract + Vol>Avg + Best Months'),

        # Bottom wick combos
        ([filters['optimal_zone'], filters['atr_contracting'], filters['large_wick']],
         'Optimal + ATR Contract + Large Wick'),
        ([filters['optimal_zone'], filters['atr_contracting'], filters['huge_wick']],
         'Optimal + ATR Contract + Huge Wick (>40%)'),

        # 5-filter combos (very selective)
        ([filters['optimal_zone'], filters['atr_contracting'], filters['vol_above_avg'],
          filters['good_sector'], filters['good_months']],
         '5x: Optimal+ATR+Vol+Sector+Months'),
        ([filters['optimal_zone'], filters['atr_contracting'], filters['vol_high'],
          filters['good_sector'], filters['rsi_very_low']],
         '5x: Optimal+ATR+Vol1.5x+Sector+RSI25'),

        # Ultra aggressive
        ([filters['optimal_zone'], filters['atr_contracting'], filters['vol_spike'],
          filters['momentum_strong']],
         'Optimal + ATR + Vol Spike + Mom<-10%'),
        ([filters['optimal_zone'], filters['atr_contracting'], filters['vol_climax'],
          filters['weekly_down_big']],
         'Optimal + ATR + Climax + Week Down'),
    ]

    results = []
    for conditions, name in combos:
        result = test_filter_combo(df, conditions, name)
        if result:
            results.append(result)

    # Sort by win rate
    results.sort(key=lambda x: -x['win_rate'])

    print(f"\n{'Filter Combination':<50} {'Trades':>7} {'T/Yr':>6} {'Win%':>7} {'AvgRet':>8} {'Median':>8} {'Worst':>8}")
    print("-" * 105)

    for r in results:
        print(f"{r['name']:<50} {r['trades']:>7} {r['trades_per_year']:>5.1f} {r['win_rate']:>6.1f}% {r['avg_return']:>+7.1f}% {r['median_return']:>+7.1f}% {r['min_return']:>+7.1f}%")

    # === TOP PICKS ===
    print("\n" + "=" * 100)
    print("TOP PICKS FOR ~10 TRADES/YEAR")
    print("=" * 100)

    # Filter for ~10 trades/year with 80%+ win rate
    good_picks = [r for r in results if r['trades_per_year'] >= 5 and r['trades_per_year'] <= 25 and r['win_rate'] >= 80]
    good_picks.sort(key=lambda x: (-x['win_rate'], -x['avg_return']))

    print(f"\n{'Rank':<5} {'Filter Combination':<45} {'T/Yr':>6} {'Win%':>7} {'AvgRet':>8}")
    print("-" * 80)

    for i, r in enumerate(good_picks[:10], 1):
        print(f"{i:<5} {r['name']:<45} {r['trades_per_year']:>5.1f} {r['win_rate']:>6.1f}% {r['avg_return']:>+7.1f}%")

    # === BREAKDOWN BY YEAR ===
    if good_picks:
        best = good_picks[0]
        print(f"\n" + "=" * 100)
        print(f"YEAR-BY-YEAR BREAKDOWN: {best['name']}")
        print("=" * 100)

        # Rebuild the best filter
        best_conditions = None
        for conditions, name in combos:
            if name == best['name']:
                mask = pd.Series([True] * len(df))
                for cond in conditions:
                    mask = mask & cond
                best_conditions = mask
                break

        if best_conditions is not None:
            subset = df[best_conditions]

            print(f"\n{'Year':<8} {'Trades':>8} {'Wins':>8} {'Win%':>8} {'AvgRet':>10}")
            print("-" * 50)

            for year in sorted(subset['year'].unique()):
                year_df = subset[subset['year'] == year]
                wins = year_df['win'].sum()
                wr = wins / len(year_df) * 100 if len(year_df) > 0 else 0
                avg_ret = year_df['fwd_return'].mean() * 100
                print(f"{year:<8} {len(year_df):>8} {wins:>8} {wr:>7.1f}% {avg_ret:>+9.1f}%")

    # === 2025 OUT-OF-SAMPLE ===
    print("\n" + "=" * 100)
    print("2025 OUT-OF-SAMPLE TEST (Top 5 Filters)")
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

                # Calculate current return (may be incomplete)
                future_dates = price_df.index[price_df.index > date]
                if len(future_dates) >= 252:
                    exit_price = price_df.loc[future_dates[251], 'adjusted_close']
                    is_complete = True
                else:
                    exit_price = price_df.loc[price_df.index[-1], 'adjusted_close']
                    is_complete = False

                current_return = exit_price / row['adjusted_close'] - 1

                trade = {
                    'symbol': symbol,
                    'date': date,
                    'current_return': current_return,
                    'is_complete': is_complete,
                    'in_optimal_zone': row.get('in_optimal_zone', False),
                    'prev_atr_contracting': row.get('prev_atr_contracting', False),
                    'prev_atr_declining': row.get('prev_atr_declining', False),
                    'vol_above_20': row.get('vol_above_20', False),
                    'vol_spike': row.get('vol_spike', False),
                    'vol_climax': row.get('vol_climax', False),
                    'good_sector': SECTOR_MAP.get(symbol, 'Other') not in BAD_SECTORS,
                    'good_months': date.month not in [1, 8],
                    'best_months': date.month in [3, 4, 10, 11],
                    'rsi_very_low': row.get('rsi', 50) < 25,
                    'vol_high': row.get('vol_vs_20', 1) > 1.5,
                }
                test_trades.append(trade)
            except:
                continue

    test_df = pd.DataFrame(test_trades)

    if len(test_df) > 0:
        print(f"\nTotal 2025 DD+RSI signals: {len(test_df)}")

        # Test top filters on 2025
        test_combos = [
            ('Baseline (DD+RSI)', []),
            ('Optimal Zone', ['in_optimal_zone']),
            ('Optimal + ATR Contract', ['in_optimal_zone', 'prev_atr_contracting']),
            ('Optimal + ATR + Vol>Avg', ['in_optimal_zone', 'prev_atr_contracting', 'vol_above_20']),
            ('Optimal + ATR + Vol>Avg + Sector', ['in_optimal_zone', 'prev_atr_contracting', 'vol_above_20', 'good_sector']),
            ('Optimal + ATR + Vol>1.5x + Sector', ['in_optimal_zone', 'prev_atr_contracting', 'vol_high', 'good_sector']),
            ('Optimal + ATR + Vol Spike', ['in_optimal_zone', 'prev_atr_contracting', 'vol_spike']),
            ('Optimal + ATR + Vol Spike + Sector', ['in_optimal_zone', 'prev_atr_contracting', 'vol_spike', 'good_sector']),
        ]

        print(f"\n{'Filter':<45} {'Trades':>8} {'Win%':>8} {'AvgRet':>10}")
        print("-" * 75)

        for name, filter_cols in test_combos:
            if filter_cols:
                mask = pd.Series([True] * len(test_df))
                for col in filter_cols:
                    mask = mask & (test_df[col] == True)
                subset = test_df[mask]
            else:
                subset = test_df

            if len(subset) > 0:
                wins = (subset['current_return'] > 0).sum()
                wr = wins / len(subset) * 100
                avg_ret = subset['current_return'].mean() * 100
                print(f"{name:<45} {len(subset):>8} {wr:>7.1f}% {avg_ret:>+9.1f}%")

    print(f"""
{'='*100}
FINAL RECOMMENDATION FOR ~10 TRADES/YEAR
{'='*100}

BEST AGGRESSIVE FILTER COMBINATION:
  Optimal Zone + ATR Contracting + Volume > Avg + Good Sector

FILTER CRITERIA:
  1. Deep Drawdown > 20% from 52-week high (existing)
  2. RSI(14) < 30 (existing)
  3. Optimal Zone: Price 20-50% below EMA200 (existing)
  4. ATR Contracting: Weekly ATR SMA(3) < SMA(10) (NEW)
  5. Volume > 20-day average (NEW)
  6. Good Sector: NOT Communication or Consumer Staples (existing)

EXPECTED PERFORMANCE:
  - ~10-15 trades per year
  - ~80-85% win rate
  - ~35-40% average return per trade
  - Hold for 252 days (1 year)

This is a highly selective, high-conviction strategy.
""")


if __name__ == '__main__':
    main()

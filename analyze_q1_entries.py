#!/usr/bin/env python3
"""
Analyze Q1 Entry Performance

Why does Q4 work so well? And what works for Q1 entries?
"""

import sys
from pathlib import Path
from datetime import datetime
import warnings

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


def main():
    print("\n" + "=" * 100)
    print("Q1 ENTRY ANALYSIS - Finding Best Filters for January-March Entries")
    print("=" * 100)

    polygon_client = PolygonClient(POLYGON_API_KEY)

    print("\n[1/3] Loading data...")
    universe = get_sp500_tickers()
    universe = filter_us_exchange_stocks(universe, POLYGON_API_KEY)

    end_date = datetime.now().strftime('%Y-%m-%d')
    prices = polygon_client.fetch_prices_parallel(universe, '2010-01-01', end_date)
    prices['SPY'] = polygon_client.get_prices('SPY', '2010-01-01', end_date)

    print("\n[2/3] Building trade database with monthly breakdown...")

    train_start_dt = pd.to_datetime('2015-01-01')
    train_end_dt = pd.to_datetime('2024-12-31')
    spy_df = prices.get('SPY')
    hold_days = 252

    all_trades = []

    for symbol, df in tqdm(prices.items(), desc="Processing"):
        if symbol == 'SPY' or len(df) < 500:
            continue

        try:
            df = calculate_new_filters(df, symbol=symbol, spy_df=spy_df)
        except:
            continue

        df['fwd_return'] = df['adjusted_close'].shift(-hold_days) / df['adjusted_close'] - 1

        train_df = df[(df.index >= train_start_dt) & (df.index <= train_end_dt)]
        train_df = train_df[(train_df['adjusted_close'] >= 10) &
                            (train_df['adjusted_close'] <= 400)]

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

                    # Filters
                    'optimal_zone': row.get('in_optimal_zone', False),
                    'sector': sector,
                    'good_sector': sector not in BAD_SECTORS,
                    'atr_contracting': row.get('prev_atr_contracting', False),
                    'atr_declining': row.get('prev_atr_declining', False),
                    'vol_gt_avg': row.get('vol_above_20', False),
                    'vol_gt_1_5x': row.get('vol_vs_20', 1) > 1.5,
                    'vol_gt_2x': row.get('vol_vs_20', 1) > 2.0,
                    'mom_lt_m5': row.get('momentum_5d', 0) < -5,
                    'mom_lt_m10': row.get('momentum_5d', 0) < -10,
                    'mom_lt_m15': row.get('momentum_5d', 0) < -15,
                    'rsi_lt_25': row.get('rsi', 50) < 25,
                    'rsi_lt_20': row.get('rsi', 50) < 20,
                    'ema_lt_m30': row.get('dist_from_ema200', 0) < -30,
                    'ema_lt_m35': row.get('dist_from_ema200', 0) < -35,
                    'atr_gt_3': row.get('atr_pct', 0) > 3,
                    'atr_gt_4': row.get('atr_pct', 0) > 4,
                    'atr_gt_5': row.get('atr_pct', 0) > 5,
                    'week_dn_gt_5': row.get('prev_weekly_return', 0) < -5,
                    'week_dn_gt_10': row.get('prev_weekly_return', 0) < -10,
                    'wick_gt_40': row.get('prev_weekly_bottom_wick_pct', 0) > 40,

                    # Sector specific
                    'tech': sector == 'Tech',
                    'industrial': sector == 'Industrial',
                    'energy': sector == 'Energy',
                    'finance': sector == 'Finance',
                    'health': sector == 'Health',
                    'consumer': sector == 'Consumer',
                }
                all_trades.append(trade)
            except:
                continue

    df = pd.DataFrame(all_trades)
    print(f"\nTotal trades: {len(df)}")

    # === WHY Q4 WORKS ===
    print("\n" + "=" * 100)
    print("WHY Q4 ENTRY WORKS SO WELL")
    print("=" * 100)

    print("\n--- Performance by Entry Quarter (All DD+RSI Signals) ---")
    print(f"{'Quarter':<10} {'Trades':>8} {'Win Rate':>10} {'Avg Return':>12} {'Median':>10}")
    print("-" * 55)

    for q in [1, 2, 3, 4]:
        q_df = df[df['quarter'] == q]
        if len(q_df) > 0:
            wr = q_df['win'].mean() * 100
            avg_ret = q_df['fwd_return'].mean() * 100
            med_ret = q_df['fwd_return'].median() * 100
            print(f"Q{q:<9} {len(q_df):>8} {wr:>9.1f}% {avg_ret:>+11.1f}% {med_ret:>+9.1f}%")

    print("\n--- Performance by Entry Month (All DD+RSI Signals) ---")
    print(f"{'Month':<12} {'Trades':>8} {'Win Rate':>10} {'Avg Return':>12}")
    print("-" * 45)

    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    for m in range(1, 13):
        m_df = df[df['month'] == m]
        if len(m_df) > 0:
            wr = m_df['win'].mean() * 100
            avg_ret = m_df['fwd_return'].mean() * 100
            marker = " <-- BEST" if wr >= 75 else " <-- WORST" if wr < 60 else ""
            print(f"{month_names[m-1]:<12} {len(m_df):>8} {wr:>9.1f}% {avg_ret:>+11.1f}%{marker}")

    # === Q1 DEEP DIVE ===
    print("\n" + "=" * 100)
    print("Q1 ENTRY ANALYSIS (January, February, March)")
    print("=" * 100)

    q1_df = df[df['quarter'] == 1]
    print(f"\nQ1 Baseline: {len(q1_df)} trades, {q1_df['win'].mean()*100:.1f}% win rate, {q1_df['fwd_return'].mean()*100:+.1f}% avg return")

    # Q1 with optimal zone
    q1_opt = q1_df[q1_df['optimal_zone'] == True]
    print(f"Q1 + Optimal Zone: {len(q1_opt)} trades, {q1_opt['win'].mean()*100:.1f}% win rate, {q1_opt['fwd_return'].mean()*100:+.1f}% avg return")

    print("\n--- Q1 by Month ---")
    for m in [1, 2, 3]:
        m_df = q1_df[q1_df['month'] == m]
        if len(m_df) > 0:
            wr = m_df['win'].mean() * 100
            avg_ret = m_df['fwd_return'].mean() * 100
            print(f"  {month_names[m-1]}: {len(m_df)} trades, {wr:.1f}% WR, {avg_ret:+.1f}% avg")

    # === FIND BEST Q1 FILTERS ===
    print("\n" + "=" * 100)
    print("BEST FILTERS FOR Q1 ENTRIES")
    print("=" * 100)

    # Test filters on Q1 with optimal zone
    q1_opt = q1_df[q1_df['optimal_zone'] == True]
    base_wr = q1_opt['win'].mean() * 100

    filters_to_test = [
        ('atr_contracting', 'ATR Contracting'),
        ('atr_declining', 'ATR Declining'),
        ('vol_gt_avg', 'Volume > Avg'),
        ('vol_gt_1_5x', 'Volume > 1.5x'),
        ('vol_gt_2x', 'Volume > 2x'),
        ('mom_lt_m5', 'Momentum < -5%'),
        ('mom_lt_m10', 'Momentum < -10%'),
        ('mom_lt_m15', 'Momentum < -15%'),
        ('rsi_lt_25', 'RSI < 25'),
        ('rsi_lt_20', 'RSI < 20'),
        ('ema_lt_m30', 'EMA Dist < -30%'),
        ('ema_lt_m35', 'EMA Dist < -35%'),
        ('atr_gt_3', 'ATR% > 3'),
        ('atr_gt_4', 'ATR% > 4'),
        ('atr_gt_5', 'ATR% > 5'),
        ('week_dn_gt_5', 'Weekly Down > 5%'),
        ('week_dn_gt_10', 'Weekly Down > 10%'),
        ('wick_gt_40', 'Bottom Wick > 40%'),
        ('good_sector', 'Good Sector'),
        ('tech', 'Tech Sector'),
        ('industrial', 'Industrial Sector'),
        ('energy', 'Energy Sector'),
        ('finance', 'Finance Sector'),
        ('health', 'Health Sector'),
    ]

    print(f"\nQ1 + Optimal Zone Baseline: {len(q1_opt)} trades, {base_wr:.1f}% win rate")
    print(f"\n{'Filter':<25} {'Trades':>8} {'Win Rate':>10} {'Avg Return':>12} {'vs Base':>10}")
    print("-" * 70)

    results = []
    for col, name in filters_to_test:
        if col in q1_opt.columns:
            subset = q1_opt[q1_opt[col] == True]
            if len(subset) >= 5:
                wr = subset['win'].mean() * 100
                avg_ret = subset['fwd_return'].mean() * 100
                delta = wr - base_wr
                results.append((name, len(subset), wr, avg_ret, delta))

    results.sort(key=lambda x: -x[2])
    for name, trades, wr, avg_ret, delta in results:
        marker = " ***" if wr >= 90 else ""
        print(f"{name:<25} {trades:>8} {wr:>9.1f}% {avg_ret:>+11.1f}% {delta:>+9.1f}%{marker}")

    # === BEST Q1 COMBINATIONS ===
    print("\n" + "=" * 100)
    print("BEST Q1 FILTER COMBINATIONS")
    print("=" * 100)

    from itertools import combinations

    promising = ['atr_contracting', 'atr_declining', 'vol_gt_avg', 'vol_gt_1_5x',
                 'mom_lt_m10', 'rsi_lt_25', 'ema_lt_m30', 'atr_gt_3', 'atr_gt_4',
                 'good_sector', 'week_dn_gt_5', 'tech', 'industrial', 'energy']

    combo_results = []

    # Double combos
    for f1, f2 in combinations(promising, 2):
        if f1 in q1_opt.columns and f2 in q1_opt.columns:
            subset = q1_opt[(q1_opt[f1] == True) & (q1_opt[f2] == True)]
            if len(subset) >= 5:
                wr = subset['win'].mean() * 100
                avg_ret = subset['fwd_return'].mean() * 100
                if wr >= 85:
                    combo_results.append((f'{f1} + {f2}', len(subset), wr, avg_ret))

    # Triple combos
    for f1, f2, f3 in combinations(promising, 3):
        if all(f in q1_opt.columns for f in [f1, f2, f3]):
            subset = q1_opt[(q1_opt[f1] == True) & (q1_opt[f2] == True) & (q1_opt[f3] == True)]
            if len(subset) >= 5:
                wr = subset['win'].mean() * 100
                avg_ret = subset['fwd_return'].mean() * 100
                if wr >= 90:
                    combo_results.append((f'{f1} + {f2} + {f3}', len(subset), wr, avg_ret))

    combo_results.sort(key=lambda x: (-x[2], -x[1]))

    print(f"\n{'Combination (Q1 + Optimal +)':<55} {'Trades':>8} {'Win%':>8} {'AvgRet':>10}")
    print("-" * 85)

    for name, trades, wr, avg_ret in combo_results[:25]:
        marker = " <-- 100%" if wr == 100 else ""
        print(f"{name:<55} {trades:>8} {wr:>7.1f}% {avg_ret:>+9.1f}%{marker}")

    # === FEBRUARY & MARCH FOCUS ===
    print("\n" + "=" * 100)
    print("FEBRUARY & MARCH FOCUS (Avoid January)")
    print("=" * 100)

    feb_mar = q1_df[(q1_df['month'] == 2) | (q1_df['month'] == 3)]
    feb_mar_opt = feb_mar[feb_mar['optimal_zone'] == True]

    print(f"\nFeb-Mar + Optimal Zone: {len(feb_mar_opt)} trades, {feb_mar_opt['win'].mean()*100:.1f}% win rate")

    print(f"\n{'Filter':<25} {'Trades':>8} {'Win Rate':>10} {'Avg Return':>12}")
    print("-" * 60)

    for col, name in filters_to_test:
        if col in feb_mar_opt.columns:
            subset = feb_mar_opt[feb_mar_opt[col] == True]
            if len(subset) >= 5:
                wr = subset['win'].mean() * 100
                avg_ret = subset['fwd_return'].mean() * 100
                if wr >= 85:
                    print(f"{name:<25} {trades:>8} {wr:>9.1f}% {avg_ret:>+11.1f}%")

    # Best Feb-Mar combos
    print("\n--- Best Feb-Mar Combinations ---")
    feb_mar_combos = []

    for f1, f2 in combinations(promising, 2):
        if f1 in feb_mar_opt.columns and f2 in feb_mar_opt.columns:
            subset = feb_mar_opt[(feb_mar_opt[f1] == True) & (feb_mar_opt[f2] == True)]
            if len(subset) >= 5:
                wr = subset['win'].mean() * 100
                avg_ret = subset['fwd_return'].mean() * 100
                if wr >= 90:
                    feb_mar_combos.append((f'{f1} + {f2}', len(subset), wr, avg_ret))

    feb_mar_combos.sort(key=lambda x: (-x[2], -x[1]))

    print(f"\n{'Combination (Feb-Mar + Optimal +)':<50} {'Trades':>8} {'Win%':>8} {'AvgRet':>10}")
    print("-" * 80)

    for name, trades, wr, avg_ret in feb_mar_combos[:15]:
        print(f"{name:<50} {trades:>8} {wr:>7.1f}% {avg_ret:>+9.1f}%")

    # === 2025 Q1 TEST ===
    print("\n" + "=" * 100)
    print("2025 Q1 VALIDATION (Current Period)")
    print("=" * 100)

    test_start_dt = pd.to_datetime('2025-01-01')

    test_trades = []
    for symbol, price_df in tqdm(prices.items(), desc="Testing 2025 Q1"):
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
            if date.month > 3:  # Q1 only
                continue

            try:
                row = signal_df.loc[date]
                current_price = price_df.loc[price_df.index[-1], 'adjusted_close']
                current_return = current_price / row['adjusted_close'] - 1
                sector = SECTOR_MAP.get(symbol, 'Other')

                trade = {
                    'symbol': symbol,
                    'date': date,
                    'entry_price': row['adjusted_close'],
                    'current_price': current_price,
                    'current_return': current_return,
                    'win': current_return > 0,
                    'month': date.month,
                    'optimal_zone': row.get('in_optimal_zone', False),
                    'atr_contracting': row.get('prev_atr_contracting', False),
                    'atr_declining': row.get('prev_atr_declining', False),
                    'vol_gt_avg': row.get('vol_above_20', False),
                    'vol_gt_1_5x': row.get('vol_vs_20', 1) > 1.5,
                    'mom_lt_m10': row.get('momentum_5d', 0) < -10,
                    'good_sector': sector not in BAD_SECTORS,
                    'atr_gt_3': row.get('atr_pct', 0) > 3,
                    'atr_gt_4': row.get('atr_pct', 0) > 4,
                    'tech': sector == 'Tech',
                    'industrial': sector == 'Industrial',
                    'energy': sector == 'Energy',
                }
                test_trades.append(trade)
            except:
                continue

    if test_trades:
        test_df = pd.DataFrame(test_trades)
        print(f"\n2025 Q1 Signals: {len(test_df)} trades")
        print(f"Current win rate: {test_df['win'].mean()*100:.1f}%")
        print(f"Current avg return: {test_df['current_return'].mean()*100:+.1f}%")

        # Test filters
        print(f"\n{'Filter':<40} {'Trades':>8} {'Win%':>8} {'CurrRet':>10}")
        print("-" * 70)

        filter_tests = [
            ('All Q1 Signals', pd.Series([True] * len(test_df))),
            ('Optimal Zone', test_df['optimal_zone'] == True),
            ('Optimal + ATR Contract', (test_df['optimal_zone']) & (test_df['atr_contracting'])),
            ('Optimal + ATR Decline', (test_df['optimal_zone']) & (test_df['atr_declining'])),
            ('Optimal + Vol > Avg', (test_df['optimal_zone']) & (test_df['vol_gt_avg'])),
            ('Optimal + Vol > 1.5x', (test_df['optimal_zone']) & (test_df['vol_gt_1_5x'])),
            ('Optimal + Good Sector', (test_df['optimal_zone']) & (test_df['good_sector'])),
            ('Optimal + ATR% > 3', (test_df['optimal_zone']) & (test_df['atr_gt_3'])),
            ('Optimal + ATR% > 4', (test_df['optimal_zone']) & (test_df['atr_gt_4'])),
            ('Optimal + ATR Contract + Vol>Avg', (test_df['optimal_zone']) & (test_df['atr_contracting']) & (test_df['vol_gt_avg'])),
            ('Optimal + ATR Contract + Sector', (test_df['optimal_zone']) & (test_df['atr_contracting']) & (test_df['good_sector'])),
            ('Optimal + ATR + Vol + Sector', (test_df['optimal_zone']) & (test_df['atr_contracting']) & (test_df['vol_gt_avg']) & (test_df['good_sector'])),
        ]

        for name, mask in filter_tests:
            subset = test_df[mask]
            if len(subset) > 0:
                wr = subset['win'].mean() * 100
                avg_ret = subset['current_return'].mean() * 100
                print(f"{name:<40} {len(subset):>8} {wr:>7.1f}% {avg_ret:>+9.1f}%")

        # Show actual trades
        print("\n--- Current 2025 Q1 Optimal Zone Trades ---")
        opt_trades = test_df[test_df['optimal_zone'] == True].sort_values('current_return', ascending=False)
        print(f"\n{'Symbol':<8} {'Entry Date':<12} {'Entry$':>10} {'Current$':>10} {'Return':>10}")
        print("-" * 55)
        for _, t in opt_trades.head(20).iterrows():
            print(f"{t['symbol']:<8} {t['date'].strftime('%Y-%m-%d'):<12} ${t['entry_price']:>8.2f} ${t['current_price']:>8.2f} {t['current_return']*100:>+9.1f}%")

    print(f"""
{'='*100}
Q1 ENTRY RECOMMENDATIONS
{'='*100}

WHY Q4 WORKS:
- You buy beaten-down stocks in Oct-Nov
- Hold through "Santa Rally" and new year optimism
- Exit after 1 year with strong recovery

Q1 ENTRY STRATEGY:
- January is historically WEAK (26% win rate in some analysis)
- February-March are BETTER for entries
- Best Q1 filters: ATR Contracting + Volume > Avg + Good Sector

RECOMMENDED Q1 FILTERS:
  1. Optimal Zone (EMA -20% to -50%)
  2. ATR Contracting (volatility settling)
  3. Volume > Average (institutional interest)
  4. Good Sector (avoid Comm, Staples)
  5. Consider waiting for February if possible

ALTERNATIVE: If Q1 looks weak, wait for Sep-Nov entries for highest conviction.
""")


if __name__ == '__main__':
    main()

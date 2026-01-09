#!/usr/bin/env python3
"""
Test New Filter Ideas for Drawdown Recovery Strategy

NEW FILTER IDEAS TO TEST:
1. Red weekly candle (if we went down, big chance to go up)
2. Large bottom wick on weekly/daily candles (support level)
3. Weekly ATR declining (decreasing volatility = ready to change direction)
4. Volume correlation (daily and average for 10-20-50 candles)

NO FUTURE DATA LEAKAGE - all indicators use only past data.
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


def resample_to_weekly(df):
    """Convert daily OHLCV data to weekly."""
    weekly = df.resample('W').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'adjusted_close': 'last',
        'volume': 'sum'
    }).dropna()
    return weekly


def calculate_new_filters(df, symbol=None, spy_df=None):
    """
    Calculate new experimental filters on top of existing signals.

    NEW FILTERS:
    1. weekly_red: Previous weekly candle was red (close < open)
    2. weekly_bottom_wick: Large bottom wick on weekly candle (support)
    3. daily_bottom_wick: Large bottom wick on daily candle (support)
    4. weekly_atr_declining: Weekly ATR is declining (ready for reversal)
    5. volume_above_avg_10/20/50: Volume above N-period average
    6. volume_spike: Volume > 2x 20-period average
    """
    df = df.copy()

    # First calculate existing signals
    df = calculate_signals(df, symbol=symbol, spy_df=spy_df)

    # === NEW FILTER 1: RED WEEKLY CANDLE ===
    # Resample to weekly
    weekly = resample_to_weekly(df)

    # Weekly candle color (red = close < open)
    weekly['weekly_red'] = weekly['close'] < weekly['open']
    weekly['weekly_green'] = weekly['close'] >= weekly['open']

    # Weekly return
    weekly['weekly_return'] = weekly['close'].pct_change() * 100

    # Map weekly data back to daily (use last completed week)
    # Shift by 1 to avoid future leakage - we use LAST COMPLETED week's data
    weekly_shifted = weekly.shift(1)

    # Reindex to daily, forward fill
    df['prev_weekly_red'] = weekly_shifted['weekly_red'].reindex(df.index, method='ffill')
    df['prev_weekly_return'] = weekly_shifted['weekly_return'].reindex(df.index, method='ffill')

    # === NEW FILTER 2: WEEKLY BOTTOM WICK (SUPPORT) ===
    # Bottom wick = low to min(open, close)
    # As % of total candle range (high - low)
    weekly['body_low'] = weekly[['open', 'close']].min(axis=1)
    weekly['body_high'] = weekly[['open', 'close']].max(axis=1)
    weekly['candle_range'] = weekly['high'] - weekly['low']
    weekly['bottom_wick'] = weekly['body_low'] - weekly['low']
    weekly['top_wick'] = weekly['high'] - weekly['body_high']

    # Bottom wick as percentage of total range
    weekly['bottom_wick_pct'] = np.where(
        weekly['candle_range'] > 0,
        weekly['bottom_wick'] / weekly['candle_range'] * 100,
        0
    )

    # Large bottom wick = > 30% of total range
    weekly['large_bottom_wick'] = weekly['bottom_wick_pct'] > 30

    # Map to daily
    weekly_wick_shifted = weekly[['bottom_wick_pct', 'large_bottom_wick']].shift(1)
    df['prev_weekly_bottom_wick_pct'] = weekly_wick_shifted['bottom_wick_pct'].reindex(df.index, method='ffill')
    df['prev_weekly_large_bottom_wick'] = weekly_wick_shifted['large_bottom_wick'].reindex(df.index, method='ffill')

    # === NEW FILTER 3: DAILY BOTTOM WICK ===
    df['body_low'] = df[['open', 'close']].min(axis=1)
    df['body_high'] = df[['open', 'close']].max(axis=1)
    df['daily_range'] = df['high'] - df['low']
    df['daily_bottom_wick'] = df['body_low'] - df['low']

    df['daily_bottom_wick_pct'] = np.where(
        df['daily_range'] > 0,
        df['daily_bottom_wick'] / df['daily_range'] * 100,
        0
    )

    # Large daily bottom wick > 30%
    df['daily_large_bottom_wick'] = df['daily_bottom_wick_pct'] > 30

    # 3-day average bottom wick (recent support buying)
    df['avg_bottom_wick_3d'] = df['daily_bottom_wick_pct'].rolling(3).mean()
    df['strong_recent_support'] = df['avg_bottom_wick_3d'] > 25

    # === NEW FILTER 4: WEEKLY ATR DECLINING ===
    # Weekly ATR
    weekly['weekly_atr'] = ta.atr(weekly['high'], weekly['low'], weekly['close'], length=14)

    # ATR change (declining = negative)
    weekly['weekly_atr_change'] = weekly['weekly_atr'].pct_change() * 100

    # ATR declining for 2+ weeks
    weekly['atr_declining'] = weekly['weekly_atr_change'] < 0
    weekly['atr_declining_2w'] = weekly['atr_declining'] & weekly['atr_declining'].shift(1)

    # ATR trend (3-week SMA vs 10-week SMA)
    weekly['atr_sma3'] = weekly['weekly_atr'].rolling(3).mean()
    weekly['atr_sma10'] = weekly['weekly_atr'].rolling(10).mean()
    weekly['atr_contracting'] = weekly['atr_sma3'] < weekly['atr_sma10']

    # Map to daily
    weekly_atr_shifted = weekly[['weekly_atr', 'weekly_atr_change', 'atr_declining',
                                  'atr_declining_2w', 'atr_contracting']].shift(1)
    for col in weekly_atr_shifted.columns:
        df[f'prev_{col}'] = weekly_atr_shifted[col].reindex(df.index, method='ffill')

    # === NEW FILTER 5: VOLUME ANALYSIS ===
    # Volume moving averages
    df['vol_sma_10'] = df['volume'].rolling(10).mean()
    df['vol_sma_20'] = df['volume'].rolling(20).mean()
    df['vol_sma_50'] = df['volume'].rolling(50).mean()

    # Volume relative to averages
    df['vol_vs_10'] = df['volume'] / df['vol_sma_10']
    df['vol_vs_20'] = df['volume'] / df['vol_sma_20']
    df['vol_vs_50'] = df['volume'] / df['vol_sma_50']

    # Volume conditions
    df['vol_above_10'] = df['volume'] > df['vol_sma_10']
    df['vol_above_20'] = df['volume'] > df['vol_sma_20']
    df['vol_above_50'] = df['volume'] > df['vol_sma_50']
    df['vol_spike'] = df['vol_vs_20'] > 2.0  # Volume spike = 2x average
    df['vol_climax'] = df['vol_vs_20'] > 3.0  # Climax volume = 3x average

    # Weekly volume trend
    weekly['vol_sma_4w'] = weekly['volume'].rolling(4).mean()
    weekly['vol_increasing'] = weekly['volume'] > weekly['vol_sma_4w']
    weekly_vol_shifted = weekly[['vol_increasing']].shift(1)
    df['prev_weekly_vol_increasing'] = weekly_vol_shifted['vol_increasing'].reindex(df.index, method='ffill')

    return df


def analyze_filter_effectiveness(prices, train_start='2015-01-01', train_end='2024-12-31',
                                  hold_days=252, min_price=10, max_price=400):
    """
    Analyze how new filters correlate with trade outcomes.
    Uses training period only to avoid forward bias.
    """

    train_start_dt = pd.to_datetime(train_start)
    train_end_dt = pd.to_datetime(train_end)

    spy_df = prices.get('SPY')

    all_trades = []

    print(f"\nAnalyzing filter effectiveness ({train_start} to {train_end})...")

    for symbol, df in tqdm(prices.items(), desc="Processing stocks"):
        if symbol == 'SPY':
            continue

        if len(df) < 500:
            continue

        try:
            df = calculate_new_filters(df, symbol=symbol, spy_df=spy_df)
        except Exception as e:
            continue

        # Forward return (label - expected to use future data)
        df['fwd_return'] = df['adjusted_close'].shift(-hold_days) / df['adjusted_close'] - 1

        # Get training period with optimal signal
        train_df = df[(df.index >= train_start_dt) & (df.index <= train_end_dt)]
        train_df = train_df[(train_df['adjusted_close'] >= min_price) &
                            (train_df['adjusted_close'] <= max_price)]

        # Get days with dd_rsi_combo signal (base signal)
        signal_df = train_df[train_df['dd_rsi_combo'] == True].dropna(subset=['fwd_return'])

        for date in signal_df.index:
            try:
                row = signal_df.loc[date]

                trade = {
                    'symbol': symbol,
                    'date': date,
                    'entry_price': row['adjusted_close'],
                    'fwd_return': row['fwd_return'],
                    'win': row['fwd_return'] > 0,

                    # Existing filters
                    'rsi': row.get('rsi', np.nan),
                    'dist_from_ema200': row.get('dist_from_ema200', np.nan),
                    'atr_pct': row.get('atr_pct', np.nan),
                    'momentum_5d': row.get('momentum_5d', np.nan),
                    'in_optimal_zone': row.get('in_optimal_zone', False),

                    # NEW FILTER 1: Red weekly candle
                    'prev_weekly_red': row.get('prev_weekly_red', False),
                    'prev_weekly_return': row.get('prev_weekly_return', np.nan),

                    # NEW FILTER 2: Weekly bottom wick
                    'prev_weekly_bottom_wick_pct': row.get('prev_weekly_bottom_wick_pct', np.nan),
                    'prev_weekly_large_bottom_wick': row.get('prev_weekly_large_bottom_wick', False),

                    # NEW FILTER 3: Daily bottom wick
                    'daily_bottom_wick_pct': row.get('daily_bottom_wick_pct', np.nan),
                    'daily_large_bottom_wick': row.get('daily_large_bottom_wick', False),
                    'strong_recent_support': row.get('strong_recent_support', False),

                    # NEW FILTER 4: Weekly ATR declining
                    'prev_weekly_atr_change': row.get('prev_weekly_atr_change', np.nan),
                    'prev_atr_declining': row.get('prev_atr_declining', False),
                    'prev_atr_declining_2w': row.get('prev_atr_declining_2w', False),
                    'prev_atr_contracting': row.get('prev_atr_contracting', False),

                    # NEW FILTER 5: Volume
                    'vol_vs_20': row.get('vol_vs_20', np.nan),
                    'vol_above_20': row.get('vol_above_20', False),
                    'vol_spike': row.get('vol_spike', False),
                    'vol_climax': row.get('vol_climax', False),
                    'prev_weekly_vol_increasing': row.get('prev_weekly_vol_increasing', False),

                    # Sector
                    'sector': SECTOR_MAP.get(symbol, 'Other'),
                }
                all_trades.append(trade)
            except:
                continue

    return pd.DataFrame(all_trades)


def print_filter_analysis(trades_df):
    """Print analysis of each filter's effectiveness."""

    total_trades = len(trades_df)
    total_wins = trades_df['win'].sum()
    base_win_rate = total_wins / total_trades * 100 if total_trades > 0 else 0
    base_avg_return = trades_df['fwd_return'].mean() * 100

    print(f"\n{'='*80}")
    print(f"FILTER EFFECTIVENESS ANALYSIS")
    print(f"{'='*80}")
    print(f"\nBASELINE (DD+RSI Combo signal):")
    print(f"  Total Trades: {total_trades}")
    print(f"  Win Rate: {base_win_rate:.1f}%")
    print(f"  Avg Return: {base_avg_return:+.1f}%")

    # === FILTER 1: RED WEEKLY CANDLE ===
    print(f"\n{'='*80}")
    print("FILTER 1: PREVIOUS WEEKLY CANDLE COLOR")
    print("Hypothesis: Red weekly candle means bounce is more likely")
    print(f"{'='*80}")

    for condition, name in [(True, 'Red Weekly (Down)'), (False, 'Green Weekly (Up)')]:
        subset = trades_df[trades_df['prev_weekly_red'] == condition]
        if len(subset) > 0:
            wr = subset['win'].mean() * 100
            avg_ret = subset['fwd_return'].mean() * 100
            count = len(subset)
            print(f"  {name:<25} Trades: {count:>5}  Win Rate: {wr:>5.1f}%  Avg Return: {avg_ret:>+6.1f}%")

    # Analyze by weekly return buckets
    print(f"\n  By Previous Weekly Return:")
    bins = [(-100, -5), (-5, -2), (-2, 0), (0, 2), (2, 5), (5, 100)]
    for low, high in bins:
        subset = trades_df[(trades_df['prev_weekly_return'] >= low) &
                           (trades_df['prev_weekly_return'] < high)]
        if len(subset) >= 10:
            wr = subset['win'].mean() * 100
            avg_ret = subset['fwd_return'].mean() * 100
            print(f"    Weekly Return {low:>+4}% to {high:>+4}%: Trades: {len(subset):>4}  Win Rate: {wr:>5.1f}%  Avg Return: {avg_ret:>+6.1f}%")

    # === FILTER 2: WEEKLY BOTTOM WICK ===
    print(f"\n{'='*80}")
    print("FILTER 2: WEEKLY BOTTOM WICK (SUPPORT)")
    print("Hypothesis: Large bottom wick = buyers stepping in = support")
    print(f"{'='*80}")

    for condition, name in [(True, 'Large Bottom Wick (>30%)'), (False, 'Small/No Bottom Wick')]:
        subset = trades_df[trades_df['prev_weekly_large_bottom_wick'] == condition]
        if len(subset) > 0:
            wr = subset['win'].mean() * 100
            avg_ret = subset['fwd_return'].mean() * 100
            count = len(subset)
            print(f"  {name:<30} Trades: {count:>5}  Win Rate: {wr:>5.1f}%  Avg Return: {avg_ret:>+6.1f}%")

    # By bottom wick percentage
    print(f"\n  By Bottom Wick Percentage:")
    wick_bins = [(0, 10), (10, 20), (20, 30), (30, 40), (40, 100)]
    for low, high in wick_bins:
        subset = trades_df[(trades_df['prev_weekly_bottom_wick_pct'] >= low) &
                           (trades_df['prev_weekly_bottom_wick_pct'] < high)]
        if len(subset) >= 10:
            wr = subset['win'].mean() * 100
            avg_ret = subset['fwd_return'].mean() * 100
            print(f"    Bottom Wick {low:>2}%-{high:>3}%: Trades: {len(subset):>4}  Win Rate: {wr:>5.1f}%  Avg Return: {avg_ret:>+6.1f}%")

    # === FILTER 3: DAILY BOTTOM WICK ===
    print(f"\n{'='*80}")
    print("FILTER 3: DAILY BOTTOM WICK (INTRADAY SUPPORT)")
    print("Hypothesis: Daily bottom wick = intraday buyers = support")
    print(f"{'='*80}")

    for condition, name in [(True, 'Large Daily Bottom Wick'), (False, 'Small/No Daily Wick')]:
        subset = trades_df[trades_df['daily_large_bottom_wick'] == condition]
        if len(subset) > 0:
            wr = subset['win'].mean() * 100
            avg_ret = subset['fwd_return'].mean() * 100
            count = len(subset)
            print(f"  {name:<30} Trades: {count:>5}  Win Rate: {wr:>5.1f}%  Avg Return: {avg_ret:>+6.1f}%")

    for condition, name in [(True, 'Strong Recent Support (3d)'), (False, 'Weak Recent Support')]:
        subset = trades_df[trades_df['strong_recent_support'] == condition]
        if len(subset) > 0:
            wr = subset['win'].mean() * 100
            avg_ret = subset['fwd_return'].mean() * 100
            count = len(subset)
            print(f"  {name:<30} Trades: {count:>5}  Win Rate: {wr:>5.1f}%  Avg Return: {avg_ret:>+6.1f}%")

    # === FILTER 4: WEEKLY ATR DECLINING ===
    print(f"\n{'='*80}")
    print("FILTER 4: WEEKLY ATR TREND (VOLATILITY)")
    print("Hypothesis: Declining ATR = volatility contraction = ready for reversal")
    print(f"{'='*80}")

    for condition, name in [(True, 'ATR Declining (1 week)'), (False, 'ATR Increasing')]:
        subset = trades_df[trades_df['prev_atr_declining'] == condition]
        if len(subset) > 0:
            wr = subset['win'].mean() * 100
            avg_ret = subset['fwd_return'].mean() * 100
            count = len(subset)
            print(f"  {name:<30} Trades: {count:>5}  Win Rate: {wr:>5.1f}%  Avg Return: {avg_ret:>+6.1f}%")

    for condition, name in [(True, 'ATR Declining 2+ weeks'), (False, 'Not Declining 2+ weeks')]:
        subset = trades_df[trades_df['prev_atr_declining_2w'] == condition]
        if len(subset) > 0:
            wr = subset['win'].mean() * 100
            avg_ret = subset['fwd_return'].mean() * 100
            count = len(subset)
            print(f"  {name:<30} Trades: {count:>5}  Win Rate: {wr:>5.1f}%  Avg Return: {avg_ret:>+6.1f}%")

    for condition, name in [(True, 'ATR Contracting (SMA3<SMA10)'), (False, 'ATR Expanding')]:
        subset = trades_df[trades_df['prev_atr_contracting'] == condition]
        if len(subset) > 0:
            wr = subset['win'].mean() * 100
            avg_ret = subset['fwd_return'].mean() * 100
            count = len(subset)
            print(f"  {name:<30} Trades: {count:>5}  Win Rate: {wr:>5.1f}%  Avg Return: {avg_ret:>+6.1f}%")

    # === FILTER 5: VOLUME ===
    print(f"\n{'='*80}")
    print("FILTER 5: VOLUME ANALYSIS")
    print("Hypothesis: High volume on down days = capitulation = reversal")
    print(f"{'='*80}")

    for condition, name in [(True, 'Volume > 20-day Avg'), (False, 'Volume < 20-day Avg')]:
        subset = trades_df[trades_df['vol_above_20'] == condition]
        if len(subset) > 0:
            wr = subset['win'].mean() * 100
            avg_ret = subset['fwd_return'].mean() * 100
            count = len(subset)
            print(f"  {name:<30} Trades: {count:>5}  Win Rate: {wr:>5.1f}%  Avg Return: {avg_ret:>+6.1f}%")

    for condition, name in [(True, 'Volume Spike (>2x avg)'), (False, 'Normal Volume')]:
        subset = trades_df[trades_df['vol_spike'] == condition]
        if len(subset) > 0:
            wr = subset['win'].mean() * 100
            avg_ret = subset['fwd_return'].mean() * 100
            count = len(subset)
            print(f"  {name:<30} Trades: {count:>5}  Win Rate: {wr:>5.1f}%  Avg Return: {avg_ret:>+6.1f}%")

    for condition, name in [(True, 'Climax Volume (>3x avg)'), (False, 'Non-Climax Volume')]:
        subset = trades_df[trades_df['vol_climax'] == condition]
        if len(subset) > 0:
            wr = subset['win'].mean() * 100
            avg_ret = subset['fwd_return'].mean() * 100
            count = len(subset)
            print(f"  {name:<30} Trades: {count:>5}  Win Rate: {wr:>5.1f}%  Avg Return: {avg_ret:>+6.1f}%")

    # Volume buckets
    print(f"\n  By Volume Relative to 20-day Avg:")
    vol_bins = [(0, 0.5), (0.5, 0.75), (0.75, 1.0), (1.0, 1.5), (1.5, 2.0), (2.0, 3.0), (3.0, 10.0)]
    for low, high in vol_bins:
        subset = trades_df[(trades_df['vol_vs_20'] >= low) & (trades_df['vol_vs_20'] < high)]
        if len(subset) >= 10:
            wr = subset['win'].mean() * 100
            avg_ret = subset['fwd_return'].mean() * 100
            print(f"    Volume {low:.1f}x-{high:.1f}x avg: Trades: {len(subset):>4}  Win Rate: {wr:>5.1f}%  Avg Return: {avg_ret:>+6.1f}%")

    # === COMBINED FILTERS ===
    print(f"\n{'='*80}")
    print("COMBINED FILTER ANALYSIS")
    print("Testing combinations of promising filters")
    print(f"{'='*80}")

    # Test various combinations
    combinations = [
        # Red weekly + large bottom wick
        (trades_df['prev_weekly_red'] & trades_df['prev_weekly_large_bottom_wick'],
         'Red Weekly + Large Bottom Wick'),

        # Red weekly + volume spike
        (trades_df['prev_weekly_red'] & trades_df['vol_spike'],
         'Red Weekly + Volume Spike'),

        # ATR declining + large bottom wick
        (trades_df['prev_atr_declining'] & trades_df['prev_weekly_large_bottom_wick'],
         'ATR Declining + Large Bottom Wick'),

        # ATR contracting + red weekly
        (trades_df['prev_atr_contracting'] & trades_df['prev_weekly_red'],
         'ATR Contracting + Red Weekly'),

        # Triple: Red weekly + bottom wick + ATR declining
        (trades_df['prev_weekly_red'] & trades_df['prev_weekly_large_bottom_wick'] & trades_df['prev_atr_declining'],
         'Red Weekly + Bottom Wick + ATR Declining'),

        # Volume spike + strong support
        (trades_df['vol_spike'] & trades_df['strong_recent_support'],
         'Volume Spike + Strong Support'),

        # Best weekly conditions
        ((trades_df['prev_weekly_return'] < -3) & trades_df['prev_weekly_large_bottom_wick'],
         'Weekly Down >3% + Large Bottom Wick'),

        # Climax selling
        (trades_df['vol_climax'] & trades_df['prev_weekly_red'],
         'Climax Volume + Red Weekly'),
    ]

    print(f"\n{'Filter Combination':<45} {'Trades':>8} {'Win Rate':>10} {'Avg Return':>12} {'vs Base':>10}")
    print("-" * 90)

    for condition, name in combinations:
        subset = trades_df[condition]
        if len(subset) >= 5:
            wr = subset['win'].mean() * 100
            avg_ret = subset['fwd_return'].mean() * 100
            delta = wr - base_win_rate
            print(f"{name:<45} {len(subset):>8} {wr:>9.1f}% {avg_ret:>+11.1f}% {delta:>+9.1f}%")

    # === CORRELATION ANALYSIS ===
    print(f"\n{'='*80}")
    print("CORRELATION WITH RETURNS")
    print("Pearson correlation of each factor with forward returns")
    print(f"{'='*80}")

    numeric_cols = [
        'prev_weekly_return', 'prev_weekly_bottom_wick_pct',
        'daily_bottom_wick_pct', 'prev_weekly_atr_change',
        'vol_vs_20', 'rsi', 'dist_from_ema200', 'atr_pct', 'momentum_5d'
    ]

    print(f"\n{'Factor':<35} {'Correlation':>12} {'Interpretation':<30}")
    print("-" * 80)

    for col in numeric_cols:
        if col in trades_df.columns:
            corr = trades_df[col].corr(trades_df['fwd_return'])
            if not np.isnan(corr):
                interpretation = "POSITIVE" if corr > 0.05 else "NEGATIVE" if corr < -0.05 else "NEUTRAL"
                strength = "strong" if abs(corr) > 0.15 else "moderate" if abs(corr) > 0.05 else "weak"
                print(f"{col:<35} {corr:>+11.3f}  {strength} {interpretation}")

    return trades_df


def test_on_2025(prices, best_filters, hold_days=252, min_price=10, max_price=400):
    """Test best filters on 2025 out-of-sample data."""

    test_start_dt = pd.to_datetime('2025-01-01')
    spy_df = prices.get('SPY')

    trades = []

    print(f"\n{'='*80}")
    print("2025 OUT-OF-SAMPLE TEST")
    print(f"{'='*80}")

    for symbol, df in tqdm(prices.items(), desc="Testing on 2025"):
        if symbol == 'SPY':
            continue

        if len(df) < 500:
            continue

        try:
            df = calculate_new_filters(df, symbol=symbol, spy_df=spy_df)
        except:
            continue

        # Get 2025 signals
        test_df = df[df.index >= test_start_dt]
        test_df = test_df[(test_df['adjusted_close'] >= min_price) &
                          (test_df['adjusted_close'] <= max_price)]

        # Base signal: dd_rsi_combo
        signal_df = test_df[test_df['dd_rsi_combo'] == True]

        for date in signal_df.index:
            try:
                row = signal_df.loc[date]

                # Calculate exit
                future_dates = df.index[df.index > date]
                if len(future_dates) >= hold_days:
                    exit_date = future_dates[hold_days - 1]
                    exit_price = df.loc[exit_date, 'adjusted_close']
                    is_complete = True
                else:
                    exit_date = df.index[-1]
                    exit_price = df.loc[exit_date, 'adjusted_close']
                    is_complete = False

                exit_return = exit_price / row['adjusted_close'] - 1

                trade = {
                    'symbol': symbol,
                    'date': date,
                    'entry_price': row['adjusted_close'],
                    'exit_date': exit_date,
                    'exit_price': exit_price,
                    'return': exit_return,
                    'is_complete': is_complete,
                    'in_optimal_zone': row.get('in_optimal_zone', False),
                    'prev_weekly_red': row.get('prev_weekly_red', False),
                    'prev_weekly_large_bottom_wick': row.get('prev_weekly_large_bottom_wick', False),
                    'prev_atr_declining': row.get('prev_atr_declining', False),
                    'vol_spike': row.get('vol_spike', False),
                    'sector': SECTOR_MAP.get(symbol, 'Other'),
                }
                trades.append(trade)
            except:
                continue

    if not trades:
        print("No trades found in 2025")
        return pd.DataFrame()

    trades_df = pd.DataFrame(trades)

    # Print results for different filter combinations
    print(f"\n{'Filter Combination':<50} {'Trades':>8} {'Win Rate':>10} {'Avg Return':>12}")
    print("-" * 85)

    # Base: all DD+RSI signals
    wr = (trades_df['return'] > 0).mean() * 100
    avg_ret = trades_df['return'].mean() * 100
    print(f"{'All DD+RSI Signals':<50} {len(trades_df):>8} {wr:>9.1f}% {avg_ret:>+11.1f}%")

    # With optimal zone
    subset = trades_df[trades_df['in_optimal_zone']]
    if len(subset) > 0:
        wr = (subset['return'] > 0).mean() * 100
        avg_ret = subset['return'].mean() * 100
        print(f"{'+ Optimal Zone (EMA200 -20% to -50%)':<50} {len(subset):>8} {wr:>9.1f}% {avg_ret:>+11.1f}%")

    # Test new filters
    filter_tests = [
        (trades_df['prev_weekly_red'], '+ Red Weekly Candle'),
        (trades_df['prev_weekly_large_bottom_wick'], '+ Large Weekly Bottom Wick'),
        (trades_df['prev_atr_declining'], '+ ATR Declining'),
        (trades_df['vol_spike'], '+ Volume Spike'),
        (trades_df['prev_weekly_red'] & trades_df['prev_weekly_large_bottom_wick'],
         '+ Red Weekly + Bottom Wick'),
        (trades_df['prev_weekly_red'] & trades_df['prev_atr_declining'],
         '+ Red Weekly + ATR Declining'),
    ]

    for condition, name in filter_tests:
        subset = trades_df[condition]
        if len(subset) > 0:
            wr = (subset['return'] > 0).mean() * 100
            avg_ret = subset['return'].mean() * 100
            print(f"{name:<50} {len(subset):>8} {wr:>9.1f}% {avg_ret:>+11.1f}%")

    return trades_df


def main():
    print("\n" + "=" * 80)
    print("NEW FILTER IDEAS TEST")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 80)

    print("""
    TESTING NEW FILTER IDEAS:
    1. Red weekly candle (down week = bounce more likely)
    2. Large bottom wick (support buying)
    3. Weekly ATR declining (volatility contraction)
    4. Volume analysis (capitulation signals)
    """)

    # Initialize client
    polygon_client = PolygonClient(POLYGON_API_KEY)

    # Get universe
    print("\n[1/4] Getting S&P 500 universe...")
    universe = get_sp500_tickers()
    universe = filter_us_exchange_stocks(universe, POLYGON_API_KEY)
    print(f"  Universe: {len(universe)} stocks")

    # Fetch prices
    print("\n[2/4] Fetching price data...")
    end_date = datetime.now().strftime('%Y-%m-%d')
    prices = polygon_client.fetch_prices_parallel(universe, '2010-01-01', end_date)
    prices['SPY'] = polygon_client.get_prices('SPY', '2010-01-01', end_date)
    print(f"  Got prices for {len(prices)} stocks")

    # Analyze filter effectiveness on training data
    print("\n[3/4] Analyzing filter effectiveness on training data (2015-2024)...")
    trades_df = analyze_filter_effectiveness(
        prices,
        train_start='2015-01-01',
        train_end='2024-12-31',
        hold_days=252
    )

    if len(trades_df) > 0:
        print_filter_analysis(trades_df)

        # Save for further analysis
        trades_df.to_csv('filter_analysis_trades.csv', index=False)
        print(f"\nSaved {len(trades_df)} trades to filter_analysis_trades.csv")

    # Test on 2025
    print("\n[4/4] Testing on 2025 out-of-sample data...")
    test_trades = test_on_2025(prices, best_filters=None, hold_days=252)

    if len(test_trades) > 0:
        test_trades.to_csv('filter_analysis_2025.csv', index=False)
        print(f"\nSaved {len(test_trades)} trades to filter_analysis_2025.csv")

    print(f"""
{'='*80}
SUMMARY
{'='*80}

Analysis complete! Check the output above to see:

1. RED WEEKLY CANDLE: Does a down week predict better bounces?
2. WEEKLY BOTTOM WICK: Does support buying improve win rates?
3. ATR DECLINING: Does volatility contraction signal reversals?
4. VOLUME: Does high volume on down days = capitulation?

Key findings should show:
- Win rate and avg return for each filter condition
- Combined filter performance
- Correlation of each factor with forward returns
- 2025 out-of-sample validation

Look for filters that:
- Improve win rate by 5%+ over baseline
- Have enough trades (>30) for statistical significance
- Work on both training AND 2025 out-of-sample data
""")


if __name__ == '__main__':
    main()

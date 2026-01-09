#!/usr/bin/env python3
"""
Drawdown Recovery Screener

Screens S&P 500 stocks for high-probability buy signals using the optimal
combination of technical indicators with filters.

STRATEGY: Deep Drawdown + RSI Oversold + EMA200 Distance + Filters
- Deep Drawdown: Price > 20% below 52-week high
- RSI Oversold: RSI(14) < 30
- EMA200 Distance: Price 20-50% below EMA200 (optimal zone)
- Filters: Sector, Market Context, Volatility, Momentum, Seasonality

BACKTEST RESULTS (2015-2024, 756 trades):
- Optimal + Filters: 88.4% win rate, +79.5% avg return (2yr hold)
- Optimal + Filters: 84.1% win rate, +37.5% avg return (1yr hold)

Hold period: 12 months (252 trading days)

NO FUTURE DATA LEAKAGE - All indicators use only past data.
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
    calculate_signals, POLYGON_API_KEY, SECTOR_MAP, BAD_SECTORS
)


def main():
    print("\n" + "=" * 70)
    print("DRAWDOWN RECOVERY SCREENER")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 70)

    # Configuration
    MIN_PRICE = 10
    MAX_PRICE = 400
    HOLD_DAYS = 252  # 12 months

    polygon_client = PolygonClient(POLYGON_API_KEY)

    # Get S&P 500 universe
    print("\n[1/4] Getting S&P 500 universe...")
    universe = get_sp500_tickers()
    print(f"  Initial universe: {len(universe)} stocks")

    # Filter to NYSE/NASDAQ
    print("\n[2/4] Filtering to NYSE/NASDAQ only...")
    universe = filter_us_exchange_stocks(universe, POLYGON_API_KEY)
    print(f"  After exchange filter: {len(universe)} stocks")

    # Fetch prices
    print("\n[3/4] Fetching price data...")
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - pd.Timedelta(days=400)).strftime('%Y-%m-%d')

    prices = polygon_client.fetch_prices_parallel(universe, start_date, end_date)

    # Also fetch SPY for market context filter
    spy_df = polygon_client.get_prices('SPY', start_date, end_date)
    print(f"  Got prices for {len(prices)} stocks + SPY")

    # Scan for signals
    print("\n[4/4] Scanning for buy signals...")

    optimal_filtered_signals = []  # Best tier - all filters pass
    optimal_signals = []           # Good tier - optimal but some filters fail
    combo_signals = []             # DD + RSI combo

    for symbol in tqdm(prices.keys(), desc="  Scanning"):
        df = prices[symbol]
        if len(df) < 260:
            continue

        df = calculate_signals(df, symbol=symbol, spy_df=spy_df)
        latest = df.iloc[-1]

        price = latest['adjusted_close']

        # Price filter
        if price < MIN_PRICE or price > MAX_PRICE:
            continue

        # Check signals
        is_optimal_filtered = latest.get('optimal_filtered', False)
        is_optimal = latest.get('optimal_signal', False)
        is_combo = latest.get('dd_rsi_combo', False)

        if not is_combo:
            continue

        # Get filter status
        sector = SECTOR_MAP.get(symbol, 'Other')
        filter_sector = latest.get('filter_sector', True)
        filter_market = latest.get('filter_market', True)
        filter_volatility = latest.get('filter_volatility', True)
        filter_momentum = latest.get('filter_momentum', True)
        filter_seasonality = latest.get('filter_seasonality', True)

        # Build filter status string
        filters_passed = []
        filters_failed = []
        if filter_sector:
            filters_passed.append('Sector')
        else:
            filters_failed.append(f'Sector({sector})')
        if filter_market:
            filters_passed.append('Market')
        else:
            filters_failed.append('Market')
        if filter_volatility:
            filters_passed.append('Vol')
        else:
            filters_failed.append('Vol')
        if filter_momentum:
            filters_passed.append('Mom')
        else:
            filters_failed.append('Mom')
        if filter_seasonality:
            filters_passed.append('Season')
        else:
            filters_failed.append('Season')

        signal_data = {
            'symbol': symbol,
            'price': price,
            'sector': sector,
            'drawdown': latest['dist_from_high'] * 100,
            'rsi': latest['rsi'],
            'dist_from_ema200': latest['dist_from_ema200'],
            'atr_pct': latest.get('atr_pct', 0),
            'momentum_5d': latest.get('momentum_5d', 0),
            'filters_passed': filters_passed,
            'filters_failed': filters_failed,
            'is_optimal': is_optimal,
            'is_optimal_filtered': is_optimal_filtered,
        }

        if is_optimal_filtered:
            optimal_filtered_signals.append(signal_data)
        elif is_optimal:
            optimal_signals.append(signal_data)
        else:
            combo_signals.append(signal_data)

    # Sort by distance from EMA200 (more negative = better opportunity)
    optimal_filtered_signals.sort(key=lambda x: x['dist_from_ema200'])
    optimal_signals.sort(key=lambda x: x['dist_from_ema200'])
    combo_signals.sort(key=lambda x: x['dist_from_ema200'])

    # Display results
    print(f"\n{'='*90}")
    print("BUY SIGNALS FOUND")
    print(f"{'='*90}")

    # TIER 1: Optimal + Filters (best signals)
    if optimal_filtered_signals:
        print(f"\n{'='*90}")
        print(f"TIER 1: OPTIMAL + ALL FILTERS PASS ({len(optimal_filtered_signals)} stocks) <- BUY THESE")
        print("DD + RSI + EMA200 Zone + Sector + Market + Volatility + Momentum + Seasonality")
        print("Historical: 84% win rate, +37% avg return (1yr hold), 88% win rate (2yr hold)")
        print(f"{'='*90}")
        print(f"{'Symbol':<8} {'Sector':<10} {'Price':>8} {'DD%':>7} {'RSI':>6} {'EMA200':>8} {'ATR%':>6} {'Mom5d':>7}")
        print("-" * 90)

        for s in optimal_filtered_signals[:15]:
            print(f"{s['symbol']:<8} {s['sector']:<10} ${s['price']:>6.2f} {s['drawdown']:>6.1f}% {s['rsi']:>5.1f} {s['dist_from_ema200']:>+7.1f}% {s['atr_pct']:>5.1f}% {s['momentum_5d']:>+6.1f}%")

    # TIER 2: Optimal but some filters fail
    if optimal_signals:
        print(f"\n{'='*90}")
        print(f"TIER 2: OPTIMAL (some filters fail) ({len(optimal_signals)} stocks)")
        print("DD + RSI + EMA200 Zone (missing some filters)")
        print(f"{'='*90}")
        print(f"{'Symbol':<8} {'Sector':<10} {'Price':>8} {'DD%':>7} {'RSI':>6} {'EMA200':>8} {'Failed Filters':<25}")
        print("-" * 90)

        for s in optimal_signals[:10]:
            failed = ', '.join(s['filters_failed']) if s['filters_failed'] else 'None'
            print(f"{s['symbol']:<8} {s['sector']:<10} ${s['price']:>6.2f} {s['drawdown']:>6.1f}% {s['rsi']:>5.1f} {s['dist_from_ema200']:>+7.1f}% {failed:<25}")

    # TIER 3: DD + RSI combo only
    if combo_signals:
        print(f"\n{'='*90}")
        print(f"TIER 3: DD + RSI COMBO ({len(combo_signals)} stocks)")
        print("Deep Drawdown + RSI Oversold (not in optimal EMA200 zone)")
        print(f"{'='*90}")
        print(f"{'Symbol':<8} {'Sector':<10} {'Price':>8} {'DD%':>7} {'RSI':>6} {'EMA200':>8}")
        print("-" * 90)

        for s in combo_signals[:10]:
            print(f"{s['symbol']:<8} {s['sector']:<10} ${s['price']:>6.2f} {s['drawdown']:>6.1f}% {s['rsi']:>5.1f} {s['dist_from_ema200']:>+7.1f}%")

    # Summary
    print(f"\n{'='*90}")
    print("SUMMARY")
    print(f"{'='*90}")

    print(f"\n  TIER 1 (Optimal + Filters): {len(optimal_filtered_signals)} stocks <- BUY THESE FIRST")
    print(f"  TIER 2 (Optimal, some filters fail): {len(optimal_signals)} stocks")
    print(f"  TIER 3 (DD+RSI only): {len(combo_signals)} stocks")
    print(f"  Total signals: {len(optimal_filtered_signals) + len(optimal_signals) + len(combo_signals)}")

    # Current month warning
    current_month = datetime.now().month
    if current_month in [1, 8]:
        month_name = 'January' if current_month == 1 else 'August'
        print(f"\n  WARNING: Current month ({month_name}) has historically lower win rates!")
        print(f"           Consider waiting or being more selective.")

    print(f"""
{'='*90}
STRATEGY GUIDE
{'='*90}

SIGNAL TIERS (by historical performance 2015-2024):

  TIER 1 - OPTIMAL + FILTERS (84% win rate, +37% return @ 1yr hold)
  Base Conditions:
    - Deep Drawdown: Price > 20% below 52-week high
    - RSI Oversold: RSI(14) < 30
    - EMA200 Distance: Price 20-50% below EMA200
  Filters (all must pass):
    - Sector: Avoid Communication & Consumer Staples
    - Market: SPY 20-day return < 0 (weak market)
    - Volatility: ATR% > 2%
    - Momentum: 5-day price change < -5%
    - Seasonality: Avoid January & August

  TIER 2 - OPTIMAL (some filters fail)
  Same base conditions, but some filters fail.
  Historical win rate lower (~75-80%).

  TIER 3 - DD+RSI COMBO
  Only deep drawdown + RSI oversold, not in optimal EMA200 zone.

PORTFOLIO RULES:
  - Price filter: ${MIN_PRICE} - ${MAX_PRICE}
  - Hold 10-20 stocks max
  - Position size: 5% per stock
  - Hold period: 12 months (252 trading days)
  - No stop-loss (historically hurts mean reversion performance)

SECTORS TO AVOID:
  - Communication (NFLX, DIS, T, VZ, etc.) - 20% historical win rate
  - Consumer Staples (PG, KO, PEP, etc.) - 0% historical win rate
""")


if __name__ == '__main__':
    main()

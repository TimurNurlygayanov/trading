#!/usr/bin/env python3
"""
Drawdown Recovery Screener

Screens S&P 500 stocks for high-probability buy signals using strategy presets.
Now includes ML confidence scores to rank signals.

STRATEGY PRESETS:
  ULTRA       - 94%+ win rate, Sep-Nov entries only
  AGGRESSIVE  - 90%+ win rate, year-round
  Q1_SPECIAL  - 93%+ win rate, optimized for Jan-Mar entries
  BALANCED    - 88%+ win rate, more opportunities
  MOMENTUM    - 90%+ win rate, requires strong weekly bounce

Usage:
  python screener.py                    # Use auto-selected strategy based on month
  python screener.py --strategy AGGRESSIVE
  python screener.py --min-confidence 70  # Only show signals with 70%+ ML confidence
  python screener.py --list

NO FUTURE DATA LEAKAGE - All indicators use only past data.
"""

import sys
from pathlib import Path
from datetime import datetime
import argparse
import warnings
import pickle

import pandas as pd
import numpy as np
from tqdm import tqdm

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
warnings.filterwarnings('ignore')

from backtest import (
    PolygonClient, get_sp500_tickers, filter_us_exchange_stocks,
    calculate_signals, POLYGON_API_KEY, SECTOR_MAP, BAD_SECTORS,
    STRATEGY_PRESETS
)

# ML model imports
ML_MODEL_PATH = project_root / 'ml_model.pkl'
ML_MODEL = None
ML_SCALER = None
ML_FEATURE_NAMES = None

# Feature columns for ML (must match ml_model.py)
FEATURE_COLS = [
    'dist_from_ema200', 'rsi', 'atr_pct', 'vol_vs_avg', 'dist_from_high',
    'momentum_5d', 'prev_weekly_return', 'atr_contracting', 'spy_20d_return',
]
SECTORS = ['Tech', 'Industrial', 'Energy', 'Consumer', 'Health',
           'Finance', 'Utilities', 'RealEstate', 'Staples', 'Comm', 'Materials', 'Other']


def load_ml_model():
    """Load the trained ML model for confidence scoring."""
    global ML_MODEL, ML_SCALER, ML_FEATURE_NAMES

    if not ML_MODEL_PATH.exists():
        print("  ML model not found. Run 'python ml_model.py --save' to train.")
        return False

    try:
        with open(ML_MODEL_PATH, 'rb') as f:
            data = pickle.load(f)
        ML_MODEL = data['model']
        ML_SCALER = data['scaler']
        ML_FEATURE_NAMES = data['feature_names']
        return True
    except Exception as e:
        print(f"  Error loading ML model: {e}")
        return False


def get_ml_confidence(row, symbol, spy_20d_return=0):
    """Calculate ML confidence score for a signal."""
    if ML_MODEL is None:
        return None

    # Extract features
    features = {}
    for col in FEATURE_COLS:
        if col in row:
            val = row[col]
            if isinstance(val, bool):
                val = int(val)
            features[col] = val if pd.notna(val) else 0
        else:
            features[col] = 0

    features['spy_20d_return'] = spy_20d_return

    # One-hot encode sector
    sector = SECTOR_MAP.get(symbol, 'Other')
    for s in SECTORS:
        features[f'sector_{s}'] = 1 if sector == s else 0

    # Month encoding
    month = datetime.now().month
    features['month_sin'] = np.sin(2 * np.pi * month / 12)
    features['month_cos'] = np.cos(2 * np.pi * month / 12)

    # Create feature vector
    X = np.array([[features.get(col, 0) for col in ML_FEATURE_NAMES]])

    # Scale and predict
    X_scaled = ML_SCALER.transform(X)
    proba = ML_MODEL.predict_proba(X_scaled)[0, 1]

    return proba * 100  # Return as percentage


def get_recommended_strategy():
    """Get recommended strategy based on current month."""
    month = datetime.now().month

    if month in [9, 10, 11]:  # Sep-Nov: Best months
        return 'ULTRA'
    elif month in [1, 2, 3]:  # Q1: Use Q1-optimized
        return 'Q1_SPECIAL'
    else:  # Rest of year
        return 'AGGRESSIVE'


def main():
    parser = argparse.ArgumentParser(description='Drawdown Recovery Screener')
    parser.add_argument('--strategy', '-s', type=str, default=None,
                        choices=['ULTRA', 'AGGRESSIVE', 'Q1_SPECIAL', 'BALANCED', 'MOMENTUM', 'ALL'],
                        help='Strategy preset (default: auto-select based on month)')
    parser.add_argument('--min-confidence', '-c', type=float, default=0,
                        help='Minimum ML confidence score (0-100, default: 0 = show all)')
    parser.add_argument('--list', '-l', action='store_true',
                        help='List available strategy presets')
    args = parser.parse_args()

    # List strategies if requested
    if args.list:
        print("\n" + "=" * 80)
        print("AVAILABLE STRATEGY PRESETS")
        print("=" * 80)
        for key, preset in STRATEGY_PRESETS.items():
            print(f"\n{key}:")
            print(f"  {preset['name']}")
            print(f"  {preset['description']}")
            print(f"  Expected Win Rate: {preset['expected_win_rate']}")
            print(f"  Expected Return: {preset['expected_return']}")
        print(f"\nCurrent month recommendation: {get_recommended_strategy()}")
        return

    # Determine strategy
    if args.strategy and args.strategy != 'ALL':
        strategies = [args.strategy]
    elif args.strategy == 'ALL':
        strategies = list(STRATEGY_PRESETS.keys())
    else:
        # Auto-select based on month
        strategies = [get_recommended_strategy()]

    print("\n" + "=" * 80)
    print("DRAWDOWN RECOVERY SCREENER")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 80)

    # Load ML model for confidence scoring
    print("\n[0/4] Loading ML model...")
    ml_loaded = load_ml_model()
    if ml_loaded:
        print("  ML model loaded - confidence scores enabled")
    else:
        print("  ML model not available - running without confidence scores")

    # Configuration
    MIN_PRICE = 10
    MAX_PRICE = 400

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
    spy_df = polygon_client.get_prices('SPY', start_date, end_date)
    print(f"  Got prices for {len(prices)} stocks + SPY")

    # Calculate SPY 20d return for ML features
    spy_20d_return = 0
    if spy_df is not None and len(spy_df) > 20:
        spy_20d_return = (spy_df['adjusted_close'].iloc[-1] / spy_df['adjusted_close'].iloc[-21] - 1) * 100

    # Scan for signals
    print("\n[4/4] Scanning for buy signals...")

    for strategy_name in strategies:
        preset = STRATEGY_PRESETS[strategy_name]

        print(f"\n{'='*80}")
        print(f"STRATEGY: {preset['name']}")
        print(f"Expected: {preset['expected_win_rate']} win rate, {preset['expected_return']} return")
        if args.min_confidence > 0:
            print(f"ML Confidence Filter: >= {args.min_confidence:.0f}%")
        print(f"{'='*80}")

        signals = []

        for symbol in tqdm(prices.keys(), desc=f"  Scanning ({strategy_name})"):
            df = prices[symbol]
            if len(df) < 260:
                continue

            df = calculate_signals(df, symbol=symbol, spy_df=spy_df, strategy=strategy_name)
            latest = df.iloc[-1]

            price = latest['adjusted_close']

            # Price filter
            if price < MIN_PRICE or price > MAX_PRICE:
                continue

            # Check strategy signal
            if not latest.get('strategy_signal', False):
                continue

            sector = SECTOR_MAP.get(symbol, 'Other')

            # Calculate ML confidence score
            confidence = get_ml_confidence(latest, symbol, spy_20d_return) if ml_loaded else None

            # Apply confidence filter
            if args.min_confidence > 0 and confidence is not None:
                if confidence < args.min_confidence:
                    continue

            signal_data = {
                'symbol': symbol,
                'price': price,
                'sector': sector,
                'drawdown': latest['dist_from_high'] * 100,
                'rsi': latest['rsi'],
                'dist_from_ema200': latest['dist_from_ema200'],
                'atr_pct': latest.get('atr_pct', 0),
                'vol_vs_avg': latest.get('vol_vs_avg', 1),
                'atr_contracting': latest.get('atr_contracting', False),
                'prev_weekly_return': latest.get('prev_weekly_return', 0),
                'confidence': confidence,
            }
            signals.append(signal_data)

        # Sort by ML confidence (highest first) if available, else by EMA200 distance
        if ml_loaded and signals and signals[0].get('confidence') is not None:
            signals.sort(key=lambda x: -(x['confidence'] or 0))
        else:
            signals.sort(key=lambda x: x['dist_from_ema200'])

        if signals:
            print(f"\n  Found {len(signals)} signals!")

            if ml_loaded:
                print(f"\n  {'Symbol':<8} {'Sector':<10} {'Price':>8} {'Conf':>6} {'DD%':>7} {'RSI':>6} {'EMA200':>8} {'ATR%':>6} {'Vol':>5}")
                print("  " + "-" * 85)

                for s in signals[:15]:
                    conf_str = f"{s['confidence']:.0f}%" if s['confidence'] is not None else "N/A"
                    atr_flag = "*" if s['atr_contracting'] else " "
                    print(f"  {s['symbol']:<8} {s['sector']:<10} ${s['price']:>6.2f} {conf_str:>6} {s['drawdown']:>6.1f}% {s['rsi']:>5.1f} {s['dist_from_ema200']:>+7.1f}% {s['atr_pct']:>5.1f}%{atr_flag} {s['vol_vs_avg']:>4.1f}x")
            else:
                print(f"\n  {'Symbol':<8} {'Sector':<12} {'Price':>8} {'DD%':>7} {'RSI':>6} {'EMA200':>8} {'ATR%':>6} {'Vol':>6} {'WkRet':>7}")
                print("  " + "-" * 85)

                for s in signals[:15]:
                    atr_flag = "*" if s['atr_contracting'] else " "
                    print(f"  {s['symbol']:<8} {s['sector']:<12} ${s['price']:>6.2f} {s['drawdown']:>6.1f}% {s['rsi']:>5.1f} {s['dist_from_ema200']:>+7.1f}% {s['atr_pct']:>5.1f}%{atr_flag} {s['vol_vs_avg']:>5.1f}x {s['prev_weekly_return']:>+6.1f}%")

            if len(signals) > 15:
                print(f"\n  ... and {len(signals) - 15} more signals")
        else:
            print(f"\n  No signals found for {strategy_name}")
            if args.min_confidence > 0:
                print(f"  Try lowering --min-confidence below {args.min_confidence:.0f}%")
            else:
                print("  This is normal - the strategy is selective to achieve high win rates.")

    # Summary and recommendations
    current_month = datetime.now().month
    month_name = datetime.now().strftime('%B')

    print(f"""
{'='*80}
RECOMMENDATIONS
{'='*80}

Current Month: {month_name}
Recommended Strategy: {get_recommended_strategy()}

STRATEGY GUIDE:
  ULTRA       - Best for Sep-Nov, highest win rate (94%+)
  AGGRESSIVE  - Year-round, high win rate (90%+)
  Q1_SPECIAL  - Optimized for Jan-Mar entries (93%+)
  BALANCED    - More signals, good win rate (88%+)
  MOMENTUM    - When prev week was up >5% (90%+)

POSITION SIZING:
  - 5% per position
  - Max 10-20 positions
  - Hold 12 months (252 trading days)
  - No stop-loss (mean reversion needs room)

ML CONFIDENCE SCORES:
  - Higher confidence = higher probability of win
  - Use --min-confidence 70 for high confidence only
  - Confidence >= 80% historically had 94%+ win rate

SECTORS TO AVOID:
  - Communication (T, VZ, DIS, etc.)
  - Consumer Staples (PG, KO, PEP, etc.)

* = ATR Contracting (volatility compression, good for reversal)
""")


if __name__ == '__main__':
    main()

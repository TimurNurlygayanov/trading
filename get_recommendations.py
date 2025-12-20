#!/usr/bin/env python3
"""
Get current stock recommendations from the trained model.

Usage:
    python3 get_recommendations.py

Prerequisites:
    - Run backtest.py first to train and save the model
    - Or have a saved model in data_cache_eodhd/momentum_model.cbm

This will:
1. Load the trained CatBoost model
2. Fetch FRESH price data from Polygon (always up-to-date)
3. Use cached fundamentals from EODHD (if fetched this month)
4. Generate top 20 stock recommendations
5. Save recommendations to a JSON file
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
import json

import pandas as pd
import numpy as np
from catboost import CatBoostClassifier

from run_strategy import (
    EODHDClient,
    PolygonClient,
    EODHD_API_KEY,
    POLYGON_API_KEY,
    CACHE_DIR,
    get_feature_columns,
    calculate_momentum_features,
    extract_fundamentals
)


def get_recommendations(top_n: int = 20) -> list:
    """Get current stock recommendations using saved model."""

    # Check for saved model
    model_path = CACHE_DIR / "momentum_model.cbm"
    if not model_path.exists():
        print("ERROR: No trained model found!")
        print("Please run 'python3 backtest.py' first to train the model.")
        sys.exit(1)

    print("=" * 60)
    print("MOMENTUM + FUNDAMENTALS - STOCK RECOMMENDATIONS")
    print("=" * 60)
    print(f"\nDate: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

    # Load model
    print("\nLoading trained model...")
    model = CatBoostClassifier()
    model.load_model(str(model_path))
    print(f"  Model loaded from: {model_path}")

    # Initialize clients
    eodhd_client = EODHDClient(EODHD_API_KEY)
    polygon_client = PolygonClient(POLYGON_API_KEY)
    feature_cols = get_feature_columns()

    today = datetime.now()
    start_date = (today - timedelta(days=400)).strftime('%Y-%m-%d')
    end_date = today.strftime('%Y-%m-%d')

    # Get universe
    print("\nFetching market data...")
    stocks = eodhd_client.get_us_stocks()
    symbols = [s['Code'] for s in stocks]

    # Fetch fundamentals (uses cache if fetched this month, otherwise refreshes)
    print("  Loading fundamentals (cached if fetched this month)...")
    fundamentals = eodhd_client.fetch_fundamentals_parallel(symbols)

    # Filter by market cap and EPS
    valid_symbols = []
    for symbol, fund in fundamentals.items():
        highlights = fund.get('Highlights', {})
        market_cap = highlights.get('MarketCapitalization', 0) or 0
        eps = highlights.get('EarningsShare')
        if market_cap >= 1e9 and eps is not None and eps > 0:
            valid_symbols.append(symbol)

    print(f"  Valid symbols: {len(valid_symbols)}")

    # Fetch FRESH prices from Polygon (no rate limits, no cache!)
    # This ensures we have the latest prices for accurate recommendations
    print("\n  Fetching fresh price data from Polygon...")
    prices = polygon_client.fetch_prices_parallel(valid_symbols, start_date, end_date, force_refresh=True)
    print(f"  Got fresh prices for: {len(prices)} symbols")

    # Build features for today
    rows = []
    for symbol in prices.keys():
        if symbol not in fundamentals:
            continue

        price_df = prices[symbol]
        fund_data = fundamentals[symbol]

        if len(price_df) < 260:
            continue

        try:
            momentum_df = calculate_momentum_features(price_df)
            latest = momentum_df.iloc[-1]

            fund_features = extract_fundamentals(fund_data, today)
            if fund_features.get('eps') is None or fund_features['eps'] <= 0:
                continue

            row = {
                'symbol': symbol,
                'date': today,
                'close': latest.get('adjusted_close', latest.get('close')),
            }

            for col in ['return_21d', 'return_63d', 'return_126d', 'return_252d',
                       'volatility_21d', 'volatility_63d', 'dist_from_high', 'dist_from_low',
                       'price_to_sma_20', 'price_to_sma_50', 'price_to_sma_200',
                       'rsi_14', 'volume_ratio', 'trend_strength']:
                row[col] = latest.get(col, 0)

            for key, value in fund_features.items():
                row[key] = value if value is not None else 0

            row['log_market_cap'] = np.log(fund_features.get('market_cap', 1e9) + 1)

            # Check momentum filter (must have positive 12-month return)
            if row['return_252d'] <= 0:
                continue

            rows.append(row)
        except Exception:
            continue

    if not rows:
        print("ERROR: No valid stocks found")
        return []

    df = pd.DataFrame(rows)

    # Predict
    X = df[feature_cols].fillna(0).replace([np.inf, -np.inf], 0).values
    df['pred_prob'] = model.predict_proba(X)[:, 1]

    # Select top N
    selected = df.sort_values('pred_prob', ascending=False).head(top_n)

    # Display results
    print(f"\n{'='*60}")
    print(f"TOP {top_n} STOCKS TO BUY")
    print(f"{'='*60}")
    print(f"\n{'Rank':<6} {'Symbol':<8} {'Score':>8} {'12M Ret':>10} {'EPS':>8} {'Price':>12}")
    print("-" * 60)

    trades = []
    for i, (_, row) in enumerate(selected.iterrows(), 1):
        print(f"{i:<6} {row['symbol']:<8} {row['pred_prob']:>7.1%} {row['return_252d']:>9.1%} {row['eps']:>8.2f} ${row['close']:>10.2f}")
        trades.append({
            'rank': i,
            'symbol': row['symbol'],
            'score': float(row['pred_prob']),
            'momentum_12m': float(row['return_252d']),
            'eps': float(row['eps']),
            'price': float(row['close']),
            'weight': 1.0 / top_n
        })

    print(f"\n{'='*60}")
    print("PORTFOLIO ALLOCATION")
    print(f"{'='*60}")
    print(f"  Stocks: {top_n}")
    print(f"  Weight per stock: {100/top_n:.1f}%")
    print(f"  Strategy: Equal weight, hold for 3 months")
    print(f"  Next rebalance: ~{(today + timedelta(days=90)).strftime('%Y-%m-%d')}")

    # Save to file
    output_file = CACHE_DIR / f"recommendations_{today.strftime('%Y%m%d')}.json"
    with open(output_file, 'w') as f:
        json.dump({
            'date': today.strftime('%Y-%m-%d'),
            'stocks': trades
        }, f, indent=2)
    print(f"\n  Saved to: {output_file}")

    print(f"\nTotal EODHD API calls: {eodhd_client.api_calls}")

    return trades


if __name__ == '__main__':
    get_recommendations()

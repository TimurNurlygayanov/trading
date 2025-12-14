#!/usr/bin/env python3
"""
Strategy Comparison - Backtest all strategies on 2024 and 2025 data.

Uses REAL data from Polygon.io API.

Strategies:
1. Mean Reversion (Bollinger Bands + RSI)
2. Momentum (EMA + MACD + ADX)
3. RL PPO Simple (momentum-based)
4. VWAP ML (VWAP crossover + CatBoost filter)
"""
import warnings
warnings.filterwarnings('ignore')

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
from datetime import datetime
import logging

logging.basicConfig(level=logging.WARNING)

from core.backtester import Backtester
from strategies.next_candle.strategy import NextCandleStrategy, NextCandleConfig
from strategies.mean_reversion.strategy import MeanReversionStrategy
from data.downloaders.forex_downloader import ForexDownloader


def download_real_forex_data(start_date: str, end_date: str, symbol: str = 'EURUSD', timeframe: str = '1h') -> pd.DataFrame:
    """
    Download real forex data from Polygon.io API.

    Args:
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        symbol: Forex pair (default EURUSD)
        timeframe: Candle timeframe (default 5min)

    Returns:
        OHLCV DataFrame with real market data
    """
    downloader = ForexDownloader(source='polygon')

    print(f"    Downloading {symbol} {timeframe} data from Polygon.io...")
    print(f"    Date range: {start_date} to {end_date}")

    data = downloader.download(
        symbol=symbol,
        timeframe=timeframe,
        start_date=start_date,
        end_date=end_date
    )

    return data


def run_backtest(strategy, data: pd.DataFrame, name: str) -> dict:
    """Run backtest and return key metrics."""
    backtester = Backtester(
        initial_capital=100_000,
        leverage=30.0,
        commission=0.00007,
        slippage=0.00003,
        max_daily_drawdown=5_000,
        risk_per_trade=2_000,
        stop_on_daily_breach=False  # Continue even after DD breach
    )

    try:
        results = backtester.run(strategy, data)
        metrics = results.metrics

        return {
            'strategy': name,
            'total_return': metrics['total_return'],
            'sharpe_ratio': metrics['sharpe_ratio'],
            'max_drawdown': metrics['max_drawdown'],
            'win_rate': metrics.get('win_rate', 0),
            'profit_factor': metrics.get('profit_factor', 0),
            'num_trades': metrics.get('num_trades', 0),
            'final_equity': results.equity_curve.iloc[-1],
            'account_blown': metrics.get('account_blown', False),
            'daily_dd_breaches': metrics.get('daily_dd_breaches', 0)
        }
    except Exception as e:
        return {
            'strategy': name,
            'total_return': 0,
            'sharpe_ratio': 0,
            'max_drawdown': 0,
            'win_rate': 0,
            'profit_factor': 0,
            'num_trades': 0,
            'final_equity': 100_000,
            'account_blown': False,
            'daily_dd_breaches': 0,
            'error': str(e)
        }


def main():
    print("=" * 70)
    print("STRATEGY COMPARISON - EURUSD 1-HOUR DATA (2020-2025)")
    print("=" * 70)

    # Download REAL data from Polygon.io
    print("\n[1/4] Downloading REAL data from Polygon.io...")

    print("  - 2020 data...")
    data_2020 = download_real_forex_data('2020-01-01', '2020-12-31', timeframe='1h')
    print(f"    Downloaded {len(data_2020):,} bars")

    print("  - 2021 data...")
    data_2021 = download_real_forex_data('2021-01-01', '2021-12-31', timeframe='1h')
    print(f"    Downloaded {len(data_2021):,} bars")

    print("  - 2022 data...")
    data_2022 = download_real_forex_data('2022-01-01', '2022-12-31', timeframe='1h')
    print(f"    Downloaded {len(data_2022):,} bars")

    print("  - 2023 data...")
    data_2023 = download_real_forex_data('2023-01-01', '2023-12-31', timeframe='1h')
    print(f"    Downloaded {len(data_2023):,} bars")

    print("  - 2024 data...")
    data_2024 = download_real_forex_data('2024-01-01', '2024-12-31', timeframe='1h')
    print(f"    Downloaded {len(data_2024):,} bars")

    print("  - 2025 data...")
    data_2025 = download_real_forex_data('2025-01-01', '2025-12-14', timeframe='1h')
    print(f"    Downloaded {len(data_2025):,} bars")

    # Combine 2020-2024 for training
    train_data = pd.concat([data_2020, data_2021, data_2022, data_2023, data_2024], ignore_index=False)
    print(f"\n  TRAINING DATA: 2020-2024 = {len(train_data):,} bars")
    print(f"  TEST DATA: 2025 = {len(data_2025):,} bars")

    # Initialize strategies
    print("\n[2/4] Initializing strategies...")

    # CatBoost WITHOUT tsfresh
    config_no_tsfresh = NextCandleConfig(
        min_bars_between_trades=12,
        min_probability=0.58,
        use_tsfresh=False
    )

    # CatBoost WITH tsfresh
    config_with_tsfresh = NextCandleConfig(
        min_bars_between_trades=12,
        min_probability=0.58,
        use_tsfresh=True,
        tsfresh_window=10
    )

    strategies = {
        'CatBoost (no tsfresh)': NextCandleStrategy(config_no_tsfresh),
        'CatBoost + tsfresh': NextCandleStrategy(config_with_tsfresh),
        'Mean Reversion (BB+RSI)': MeanReversionStrategy(),
    }

    print(f"  - Total strategies: {len(strategies)}")

    # Run backtests
    print("\n[3/4] Running backtests...")
    print("\n  >>> MODEL LEARNS ON: 2023 + 2024 (combined)")
    print("  >>> MODEL TESTED ON: 2025 (out-of-sample)\n")

    results_train = []
    results_2025 = []

    for name, strategy in strategies.items():
        print(f"\n  Testing: {name}")

        # Train on combined 2023+2024 data
        if hasattr(strategy, 'train'):
            print(f"    - Training on 2020-2024 data ({len(train_data):,} bars)...")
            processed = strategy.preprocess_data(train_data.copy())
            metrics = strategy.train(processed)
            print(f"    - Train accuracy: {metrics['accuracy']:.2%}, High-conf: {metrics['high_conf_accuracy']:.2%}")
            print(f"    - High-conf samples: {metrics['high_conf_pct']:.1%} of data")
            print(f"    - Features used: {len(strategy.feature_names)}")

        # Test on training data (2020-2024) - to see in-sample performance
        print(f"    - 2020-2024 (in-sample)...", end=" ")
        r_train = run_backtest(strategy, train_data, name)
        r_train['period'] = '2020-2024'
        results_train.append(r_train)
        print(f"Return: {r_train['total_return']:.2%}, Trades: {r_train['num_trades']}")

        # Test on 2025 (out-of-sample)
        print(f"    - 2025 (OUT-OF-SAMPLE)...", end=" ")
        r2025 = run_backtest(strategy, data_2025, name)
        r2025['period'] = '2025'
        results_2025.append(r2025)
        print(f"Return: {r2025['total_return']:.2%}, Trades: {r2025['num_trades']}")

    # Display results
    print("\n" + "=" * 70)
    print("RESULTS - 2020-2024 (IN-SAMPLE / Training Period)")
    print("=" * 70)
    df_train = pd.DataFrame(results_train)
    print(df_train[['strategy', 'total_return', 'sharpe_ratio', 'max_drawdown',
                   'win_rate', 'num_trades', 'account_blown']].to_string(index=False))

    print("\n" + "=" * 70)
    print("RESULTS - 2025 (OUT-OF-SAMPLE / Test Period)")
    print("=" * 70)
    df_2025 = pd.DataFrame(results_2025)
    print(df_2025[['strategy', 'total_return', 'sharpe_ratio', 'max_drawdown',
                   'win_rate', 'num_trades', 'account_blown']].to_string(index=False))

    # Summary comparison
    print("\n" + "=" * 70)
    print("SUMMARY COMPARISON")
    print("=" * 70)
    print("\n  TRAINING: 2020-2024 (5 years)")
    print("  TESTING:  2025 (completely unseen data)\n")

    for name in strategies.keys():
        r_tr = next(r for r in results_train if r['strategy'] == name)
        r25 = next(r for r in results_2025 if r['strategy'] == name)

        print(f"\n{name}:")
        print(f"  In-Sample (2020-24):  {r_tr['total_return']:+.2%} return, {r_tr['sharpe_ratio']:.2f} Sharpe, {r_tr['num_trades']} trades")
        print(f"  Out-of-Sample (2025): {r25['total_return']:+.2%} return, {r25['sharpe_ratio']:.2f} Sharpe, {r25['num_trades']} trades")

        # Check for overfitting
        if r_tr['total_return'] > 0 and r25['total_return'] < 0:
            print(f"  ⚠️  OVERFITTING: positive in-sample, negative out-of-sample")
        elif r_tr['total_return'] > 0 and r25['total_return'] > 0:
            print(f"  ✓  CONSISTENT: profitable in both periods")

    print("\n" + "=" * 70)
    print("BEST STRATEGY (Based on 2025 Out-of-Sample)")
    print("=" * 70)

    # Best by 2025 Sharpe (out-of-sample)
    best_2025 = df_2025.loc[df_2025['sharpe_ratio'].idxmax()]
    print(f"\nBest 2025 Sharpe Ratio: {best_2025['strategy']} ({best_2025['sharpe_ratio']:.2f})")

    # Best by 2025 Return
    best_return = df_2025.loc[df_2025['total_return'].idxmax()]
    print(f"Best 2025 Return: {best_return['strategy']} ({best_return['total_return']:.2%})")

    # tsfresh improvement
    try:
        no_ts = next(r for r in results_2025 if 'no tsfresh' in r['strategy'])
        with_ts = next(r for r in results_2025 if 'tsfresh' in r['strategy'] and 'no' not in r['strategy'])
        improvement = with_ts['total_return'] - no_ts['total_return']
        print(f"\ntsfresh Impact: {improvement:+.2%} return difference")
    except StopIteration:
        pass

    print("\n" + "=" * 70)


if __name__ == '__main__':
    main()

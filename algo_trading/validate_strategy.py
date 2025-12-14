#!/usr/bin/env python3
"""
Strategy Validation - Check for future leaks and other issues.

Tests:
1. Feature look-ahead bias detection
2. Train/test data leakage
3. Realistic performance bounds check
4. Random baseline comparison
"""
import warnings
warnings.filterwarnings('ignore')

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
from datetime import datetime

from strategies.next_candle.strategy import NextCandleStrategy, NextCandleConfig
from data.downloaders.forex_downloader import ForexDownloader
from core.backtester import Backtester


def download_data(start_date: str, end_date: str) -> pd.DataFrame:
    """Download real forex data."""
    downloader = ForexDownloader(source='polygon')
    return downloader.download(
        symbol='EURUSD',
        timeframe='1h',
        start_date=start_date,
        end_date=end_date
    )


def test_feature_lookahead(strategy, data: pd.DataFrame) -> dict:
    """
    Test for look-ahead bias in features.

    Method: For each feature, check correlation with FUTURE returns.
    If feature correlates more with future than past, it's suspect.
    """
    print("\n[TEST 1] Feature Look-Ahead Bias Detection")
    print("-" * 50)

    processed = strategy.preprocess_data(data.copy())

    # Calculate future returns at different horizons
    processed['future_1'] = processed['close'].pct_change().shift(-1)
    processed['future_5'] = processed['close'].pct_change(5).shift(-5)
    processed['past_1'] = processed['close'].pct_change().shift(1)
    processed['past_5'] = processed['close'].pct_change(5)

    feature_cols = strategy.get_feature_columns(include_tsfresh=False)

    suspicious = []
    for feat in feature_cols:
        if feat not in processed.columns:
            continue

        valid = processed[[feat, 'future_1', 'future_5', 'past_1', 'past_5']].dropna()
        if len(valid) < 100:
            continue

        # Correlation with future vs past
        corr_future_1 = abs(valid[feat].corr(valid['future_1']))
        corr_future_5 = abs(valid[feat].corr(valid['future_5']))
        corr_past_1 = abs(valid[feat].corr(valid['past_1']))
        corr_past_5 = abs(valid[feat].corr(valid['past_5']))

        # Suspicious if correlates more with future than past
        if corr_future_5 > corr_past_5 * 2 and corr_future_5 > 0.1:
            suspicious.append({
                'feature': feat,
                'corr_future_5': corr_future_5,
                'corr_past_5': corr_past_5,
                'ratio': corr_future_5 / (corr_past_5 + 1e-10)
            })

    if suspicious:
        print(f"  WARNING: {len(suspicious)} suspicious features found:")
        for s in sorted(suspicious, key=lambda x: -x['ratio'])[:5]:
            print(f"    - {s['feature']}: future_corr={s['corr_future_5']:.3f}, past_corr={s['corr_past_5']:.3f}")
    else:
        print("  OK: No obvious look-ahead bias detected in features")

    return {'suspicious_features': suspicious}


def test_random_baseline(data: pd.DataFrame) -> dict:
    """
    Compare against random trading baseline.

    If strategy beats random by >10x, it might be too good to be true.
    """
    print("\n[TEST 2] Random Baseline Comparison")
    print("-" * 50)

    backtester = Backtester(
        initial_capital=100_000,
        leverage=30.0,
        commission=0.00007,
        slippage=0.00003
    )

    # Generate random signals with same frequency as strategy
    np.random.seed(42)
    n = len(data)

    # Random strategy: ~20 trades per 1000 bars (similar to our strategy)
    random_signals = pd.Series(0, index=data.index)
    trade_prob = 20 / 1000

    for i in range(60, n - 1):
        if np.random.random() < trade_prob:
            random_signals.iloc[i] = np.random.choice([1, -1])

    # Create a simple wrapper
    class RandomStrategy:
        name = "Random"
        required_history = 60
        def generate_signals(self, data):
            return random_signals
        def get_position_size(self, signal, portfolio_value, current_price):
            return portfolio_value * 0.002 / 0.0005 if signal != 0 else 0

    random_strat = RandomStrategy()

    try:
        results = backtester.run(random_strat, data)
        random_return = results.metrics['total_return']
        random_sharpe = results.metrics['sharpe_ratio']
        print(f"  Random baseline: {random_return:.2%} return, {random_sharpe:.2f} Sharpe")
        return {'random_return': random_return, 'random_sharpe': random_sharpe}
    except Exception as e:
        print(f"  Random baseline error: {e}")
        return {'random_return': 0, 'random_sharpe': 0}


def test_shuffle_validation(strategy, train_data: pd.DataFrame, test_data: pd.DataFrame) -> dict:
    """
    Shuffle test: if we shuffle the test data labels, performance should drop.

    This validates that the model is actually learning patterns, not just lucky.
    """
    print("\n[TEST 3] Shuffle Validation (Model Learning Check)")
    print("-" * 50)

    backtester = Backtester(
        initial_capital=100_000,
        leverage=30.0,
        commission=0.00007,
        slippage=0.00003
    )

    # Train on real data
    processed_train = strategy.preprocess_data(train_data.copy())
    strategy.train(processed_train)

    # Test on real data
    try:
        results_real = backtester.run(strategy, test_data)
        real_return = results_real.metrics['total_return']
        print(f"  Real test data: {real_return:.2%} return")
    except Exception as e:
        print(f"  Real test error: {e}")
        real_return = 0

    # Test on shuffled data (shuffle the prices within each day)
    shuffled_test = test_data.copy()
    shuffled_test['close'] = np.random.permutation(shuffled_test['close'].values)
    shuffled_test['open'] = np.random.permutation(shuffled_test['open'].values)
    shuffled_test['high'] = shuffled_test[['open', 'close']].max(axis=1) * 1.001
    shuffled_test['low'] = shuffled_test[['open', 'close']].min(axis=1) * 0.999

    try:
        results_shuffled = backtester.run(strategy, shuffled_test)
        shuffled_return = results_shuffled.metrics['total_return']
        print(f"  Shuffled test data: {shuffled_return:.2%} return")
    except Exception as e:
        print(f"  Shuffled test error: {e}")
        shuffled_return = 0

    if real_return > 0 and shuffled_return < real_return * 0.5:
        print("  OK: Model performance drops significantly on shuffled data")
    elif real_return > 0 and shuffled_return > real_return * 0.8:
        print("  WARNING: Model performs similarly on shuffled data - possible issue!")

    return {'real_return': real_return, 'shuffled_return': shuffled_return}


def test_walk_forward(strategy_class, config, data: pd.DataFrame) -> dict:
    """
    Walk-forward validation: train on expanding window, test on next period.

    More realistic than single train/test split.
    """
    print("\n[TEST 4] Walk-Forward Validation")
    print("-" * 50)

    backtester = Backtester(
        initial_capital=100_000,
        leverage=30.0,
        commission=0.00007,
        slippage=0.00003
    )

    # Split data into 5 periods
    n = len(data)
    period_size = n // 5

    results = []

    for i in range(3):  # 3 test periods
        train_end = (i + 2) * period_size
        test_start = train_end
        test_end = test_start + period_size

        if test_end > n:
            break

        train_data = data.iloc[:train_end]
        test_data = data.iloc[test_start:test_end]

        # Fresh strategy for each fold
        strategy = strategy_class(config)
        processed_train = strategy.preprocess_data(train_data.copy())
        strategy.train(processed_train)

        try:
            fold_results = backtester.run(strategy, test_data)
            fold_return = fold_results.metrics['total_return']
            results.append(fold_return)
            print(f"  Fold {i+1}: {fold_return:.2%} return (test period {test_start}-{test_end})")
        except Exception as e:
            print(f"  Fold {i+1} error: {e}")
            results.append(0)

    if results:
        avg_return = np.mean(results)
        std_return = np.std(results)
        print(f"\n  Average: {avg_return:.2%} +/- {std_return:.2%}")

        # Check consistency
        positive_folds = sum(1 for r in results if r > 0)
        print(f"  Positive folds: {positive_folds}/{len(results)}")

        if positive_folds == len(results):
            print("  OK: Consistent positive returns across all folds")
        elif positive_folds >= len(results) * 0.6:
            print("  MODERATE: Mostly positive returns")
        else:
            print("  WARNING: Inconsistent returns across folds")

    return {'fold_returns': results}


def test_realistic_bounds(strategy, train_data: pd.DataFrame, test_data: pd.DataFrame) -> dict:
    """
    Check if returns are within realistic bounds.

    Professional quant funds rarely exceed 50% annual returns consistently.
    """
    print("\n[TEST 5] Realistic Performance Bounds")
    print("-" * 50)

    backtester = Backtester(
        initial_capital=100_000,
        leverage=30.0,
        commission=0.00007,
        slippage=0.00003
    )

    # Train and test
    processed_train = strategy.preprocess_data(train_data.copy())
    strategy.train(processed_train)

    try:
        results = backtester.run(strategy, test_data)
        total_return = results.metrics['total_return']
        sharpe = results.metrics['sharpe_ratio']
        win_rate = results.metrics.get('win_rate', 0)
        num_trades = results.metrics.get('num_trades', 0)

        # Calculate annualized return (assuming ~250 trading days)
        test_days = (test_data.index[-1] - test_data.index[0]).days
        annual_return = (1 + total_return) ** (365 / test_days) - 1 if test_days > 0 else total_return

        print(f"  Test period return: {total_return:.2%}")
        print(f"  Annualized return: {annual_return:.2%}")
        print(f"  Sharpe ratio: {sharpe:.2f}")
        print(f"  Win rate: {win_rate:.1%}")
        print(f"  Number of trades: {num_trades}")

        # Realistic bounds check
        warnings = []

        if annual_return > 5.0:  # >500% annual
            warnings.append(f"Annual return {annual_return:.0%} is extremely high - verify no data leakage")
        elif annual_return > 2.0:  # >200% annual
            warnings.append(f"Annual return {annual_return:.0%} is very high - exercise caution")

        if sharpe > 3.0:
            warnings.append(f"Sharpe {sharpe:.2f} is unusually high - verify calculations")

        if win_rate > 0.85:
            warnings.append(f"Win rate {win_rate:.1%} is very high - check for look-ahead bias")

        if warnings:
            print("\n  WARNINGS:")
            for w in warnings:
                print(f"    - {w}")
        else:
            print("\n  OK: Performance within realistic bounds")

        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'sharpe': sharpe,
            'win_rate': win_rate,
            'warnings': warnings
        }

    except Exception as e:
        print(f"  Error: {e}")
        return {'error': str(e)}


def main():
    print("=" * 70)
    print("STRATEGY VALIDATION - Checking for Issues")
    print("=" * 70)

    # Download data
    print("\nDownloading validation data...")

    print("  - 2023-2024 (training)...")
    train_data = download_data('2023-01-01', '2024-12-31')
    print(f"    {len(train_data):,} bars")

    print("  - 2025 (testing)...")
    test_data = download_data('2025-01-01', '2025-12-14')
    print(f"    {len(test_data):,} bars")

    # Initialize strategy
    config = NextCandleConfig(use_tsfresh=False)  # Faster without tsfresh
    strategy = NextCandleStrategy(config)

    # Run all tests
    results = {}

    results['lookahead'] = test_feature_lookahead(strategy, train_data)
    results['random'] = test_random_baseline(test_data)

    # Need fresh strategy for remaining tests
    strategy = NextCandleStrategy(config)
    results['shuffle'] = test_shuffle_validation(strategy, train_data, test_data)

    # Combine all data for walk-forward
    all_data = pd.concat([train_data, test_data])
    results['walk_forward'] = test_walk_forward(NextCandleStrategy, config, all_data)

    # Final bounds check
    strategy = NextCandleStrategy(config)
    results['bounds'] = test_realistic_bounds(strategy, train_data, test_data)

    # Summary
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)

    issues = []

    if results['lookahead']['suspicious_features']:
        issues.append(f"- {len(results['lookahead']['suspicious_features'])} suspicious features with potential look-ahead bias")

    if results['bounds'].get('warnings'):
        issues.extend([f"- {w}" for w in results['bounds']['warnings']])

    if results['shuffle']['shuffled_return'] > results['shuffle']['real_return'] * 0.8:
        issues.append("- Model performs similarly on shuffled data")

    walk_forward_positive = sum(1 for r in results['walk_forward'].get('fold_returns', []) if r > 0)
    walk_forward_total = len(results['walk_forward'].get('fold_returns', []))
    if walk_forward_total > 0 and walk_forward_positive < walk_forward_total * 0.5:
        issues.append(f"- Walk-forward: only {walk_forward_positive}/{walk_forward_total} positive folds")

    if issues:
        print("\nISSUES FOUND:")
        for issue in issues:
            print(f"  {issue}")
    else:
        print("\nNO MAJOR ISSUES FOUND")
        print("Strategy appears valid for further testing.")

    print("\n" + "=" * 70)


if __name__ == '__main__':
    main()

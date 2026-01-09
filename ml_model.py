#!/usr/bin/env python3
"""
ML Confidence Score Model for Drawdown Recovery Strategy

Uses Logistic Regression to predict probability of trade success (confidence score).
Simple, interpretable, and low overfit risk.

FEATURES (all backward-looking, no future leakage):
- Technical: RSI, ATR%, distance from EMA200, drawdown depth
- Volume: volume vs 20-day average
- Volatility: ATR contracting flag
- Momentum: 5-day momentum, previous weekly return
- Context: sector, month, SPY 20d return

TRAINING:
- Period: 2015-2024 (historical signals)
- Target: Binary win/loss (forward return > 0)
- Model outputs P(win) as confidence score 0-100%

USAGE:
  python ml_model.py              # Train and evaluate model
  python ml_model.py --save       # Train and save model to disk

NO FUTURE DATA LEAKAGE:
- All features use only past data
- Forward return only used as TARGET (label), not feature
"""

import sys
from pathlib import Path
from datetime import datetime
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
    calculate_signals, POLYGON_API_KEY, SECTOR_MAP, BAD_SECTORS
)

# ML imports
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, roc_auc_score


# Feature columns to use (all backward-looking)
FEATURE_COLS = [
    'dist_from_ema200',      # Distance from EMA200 (%)
    'rsi',                    # RSI (14)
    'atr_pct',               # ATR as % of price
    'vol_vs_avg',            # Volume vs 20-day average
    'dist_from_high',        # Distance from 52-week high
    'momentum_5d',           # 5-day price momentum
    'prev_weekly_return',    # Previous week's return
    'atr_contracting',       # Weekly ATR contracting (binary)
    'spy_20d_return',        # SPY 20-day return (market context)
]

# Sectors for one-hot encoding
SECTORS = ['Tech', 'Industrial', 'Energy', 'Consumer', 'Health',
           'Finance', 'Utilities', 'RealEstate', 'Staples', 'Comm', 'Materials', 'Other']


def extract_features(df, symbol, spy_df=None):
    """
    Extract ML features from a signal DataFrame.

    All features use only past data - NO FUTURE LEAKAGE.
    """
    # Calculate signals (this adds all our indicators)
    df = calculate_signals(df, symbol=symbol, spy_df=spy_df, strategy='AGGRESSIVE')

    # Add SPY context if available
    if spy_df is not None and len(spy_df) > 0:
        spy_20d_return = spy_df['adjusted_close'].pct_change(20) * 100
        df['spy_20d_return'] = spy_20d_return.reindex(df.index, method='ffill')
    else:
        df['spy_20d_return'] = 0

    # Get sector
    sector = SECTOR_MAP.get(symbol, 'Other')
    df['sector'] = sector

    return df


def prepare_training_data(prices, spy_df, signal_type='optimal_signal',
                          start_date='2015-01-01', end_date='2024-12-31',
                          hold_days=252, min_price=10, max_price=400):
    """
    Prepare training data from historical signals.

    Returns:
        X: Feature matrix
        y: Target vector (1=win, 0=loss)
        metadata: DataFrame with symbol, date, return info
    """
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)

    all_features = []
    all_targets = []
    all_metadata = []

    for symbol, df in tqdm(prices.items(), desc="Extracting features"):
        if symbol == 'SPY' or len(df) < 500:
            continue

        # Extract features
        df = extract_features(df, symbol, spy_df)

        # Forward return (TARGET only - this is what we predict)
        df['fwd_return'] = df['adjusted_close'].shift(-hold_days) / df['adjusted_close'] - 1
        df['target'] = (df['fwd_return'] > 0).astype(int)

        # Filter to training period
        train_df = df[(df.index >= start_dt) & (df.index <= end_dt)]

        # Apply price filter
        train_df = train_df[(train_df['adjusted_close'] >= min_price) &
                            (train_df['adjusted_close'] <= max_price)]

        # Get signals only
        signal_df = train_df[train_df[signal_type] == True].dropna(subset=['fwd_return'])

        if len(signal_df) == 0:
            continue

        # Extract feature values
        for idx, row in signal_df.iterrows():
            features = {}

            # Numeric features
            for col in FEATURE_COLS:
                if col in row:
                    val = row[col]
                    # Handle boolean
                    if isinstance(val, bool):
                        val = int(val)
                    features[col] = val if pd.notna(val) else 0
                else:
                    features[col] = 0

            # One-hot encode sector
            sector = row.get('sector', 'Other')
            for s in SECTORS:
                features[f'sector_{s}'] = 1 if sector == s else 0

            # Month (cyclical encoding)
            month = idx.month
            features['month_sin'] = np.sin(2 * np.pi * month / 12)
            features['month_cos'] = np.cos(2 * np.pi * month / 12)

            all_features.append(features)
            all_targets.append(row['target'])
            all_metadata.append({
                'symbol': symbol,
                'date': idx,
                'entry_price': row['adjusted_close'],
                'fwd_return': row['fwd_return'],
                'sector': sector,
            })

    # Convert to arrays
    feature_names = list(all_features[0].keys())
    X = np.array([[f[col] for col in feature_names] for f in all_features])
    y = np.array(all_targets)
    metadata = pd.DataFrame(all_metadata)

    return X, y, metadata, feature_names


def train_model(X, y, feature_names):
    """
    Train logistic regression model.

    Returns:
        model: Trained LogisticRegression
        scaler: Fitted StandardScaler
        cv_scores: Cross-validation scores
    """
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train logistic regression
    model = LogisticRegression(
        penalty='l2',           # L2 regularization to prevent overfit
        C=1.0,                  # Regularization strength
        class_weight='balanced', # Handle class imbalance
        max_iter=1000,
        random_state=42
    )

    # Cross-validation
    cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring='roc_auc')

    # Fit final model
    model.fit(X_scaled, y)

    # Feature importance (coefficients)
    importance = pd.DataFrame({
        'feature': feature_names,
        'coefficient': model.coef_[0],
        'abs_importance': np.abs(model.coef_[0])
    }).sort_values('abs_importance', ascending=False)

    return model, scaler, cv_scores, importance


def predict_confidence(model, scaler, features_dict, feature_names):
    """
    Predict confidence score (probability of win) for a single signal.

    Args:
        model: Trained LogisticRegression
        scaler: Fitted StandardScaler
        features_dict: Dictionary of feature values
        feature_names: List of feature names in correct order

    Returns:
        confidence: Probability of win (0-100%)
    """
    # Create feature vector
    X = np.array([[features_dict.get(col, 0) for col in feature_names]])

    # Scale
    X_scaled = scaler.transform(X)

    # Predict probability
    proba = model.predict_proba(X_scaled)[0, 1]  # P(win)

    return proba * 100  # Convert to percentage


def evaluate_on_test(model, scaler, prices, spy_df, feature_names,
                     signal_type='optimal_signal', start_date='2025-01-01',
                     hold_days=252, min_price=10, max_price=400):
    """
    Evaluate model on out-of-sample test data.
    """
    start_dt = pd.to_datetime(start_date)

    test_results = []

    for symbol, df in tqdm(prices.items(), desc="Testing"):
        if symbol == 'SPY' or len(df) < 500:
            continue

        df = extract_features(df, symbol, spy_df)

        # Forward return for evaluation
        df['fwd_return'] = df['adjusted_close'].shift(-hold_days) / df['adjusted_close'] - 1

        # Filter to test period
        test_df = df[df.index >= start_dt]
        test_df = test_df[(test_df['adjusted_close'] >= min_price) &
                          (test_df['adjusted_close'] <= max_price)]

        # Get signals
        signal_df = test_df[test_df[signal_type] == True]

        for idx, row in signal_df.iterrows():
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

            sector = row.get('sector', 'Other')
            for s in SECTORS:
                features[f'sector_{s}'] = 1 if sector == s else 0

            month = idx.month
            features['month_sin'] = np.sin(2 * np.pi * month / 12)
            features['month_cos'] = np.cos(2 * np.pi * month / 12)

            # Predict confidence
            confidence = predict_confidence(model, scaler, features, feature_names)

            # Get actual return (may be NaN if trade not complete)
            actual_return = row['fwd_return'] if pd.notna(row.get('fwd_return')) else None
            is_complete = actual_return is not None

            test_results.append({
                'symbol': symbol,
                'date': idx,
                'confidence': confidence,
                'actual_return': actual_return,
                'is_complete': is_complete,
                'sector': sector,
            })

    return pd.DataFrame(test_results)


def save_model(model, scaler, feature_names, filepath='ml_model.pkl'):
    """Save trained model to disk."""
    with open(filepath, 'wb') as f:
        pickle.dump({
            'model': model,
            'scaler': scaler,
            'feature_names': feature_names,
        }, f)
    print(f"Model saved to {filepath}")


def load_model(filepath='ml_model.pkl'):
    """Load trained model from disk."""
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    return data['model'], data['scaler'], data['feature_names']


def main():
    import argparse

    parser = argparse.ArgumentParser(description='ML Confidence Score Model')
    parser.add_argument('--save', action='store_true', help='Save trained model to disk')
    parser.add_argument('--threshold', type=float, default=60,
                        help='Confidence threshold for filtering (default: 60%%)')
    args = parser.parse_args()

    print("\n" + "=" * 80)
    print("ML CONFIDENCE SCORE MODEL")
    print("=" * 80)
    print("\nModel: Logistic Regression")
    print("Target: P(win) - Probability of positive return after 1 year hold")
    print("Features: Technical + Volume + Volatility + Sector + Seasonality")

    # Load data
    print("\n[1/5] Loading price data...")
    polygon_client = PolygonClient(POLYGON_API_KEY)

    universe = get_sp500_tickers()
    universe = filter_us_exchange_stocks(universe, POLYGON_API_KEY)

    end_date = datetime.now().strftime('%Y-%m-%d')
    prices = polygon_client.fetch_prices_parallel(universe, '2010-01-01', end_date)
    spy_df = polygon_client.get_prices('SPY', '2010-01-01', end_date)
    prices['SPY'] = spy_df

    print(f"  Loaded {len(prices)} stocks")

    # Prepare training data
    print("\n[2/5] Preparing training data (2015-2024)...")
    X_train, y_train, metadata, feature_names = prepare_training_data(
        prices, spy_df,
        signal_type='optimal_signal',
        start_date='2015-01-01',
        end_date='2024-12-31'
    )

    print(f"  Training samples: {len(X_train)}")
    print(f"  Features: {len(feature_names)}")
    print(f"  Win rate in training: {y_train.mean()*100:.1f}%")

    # Train model
    print("\n[3/5] Training logistic regression...")
    model, scaler, cv_scores, importance = train_model(X_train, y_train, feature_names)

    print(f"\n  Cross-validation ROC-AUC: {cv_scores.mean():.3f} (+/- {cv_scores.std()*2:.3f})")

    print("\n  Top 10 Most Important Features:")
    print("  " + "-" * 50)
    for _, row in importance.head(10).iterrows():
        direction = "+" if row['coefficient'] > 0 else "-"
        print(f"  {direction} {row['feature']:<25} {row['coefficient']:+.3f}")

    # Training set predictions
    X_train_scaled = scaler.transform(X_train)
    train_proba = model.predict_proba(X_train_scaled)[:, 1] * 100

    print("\n  Training Set Performance by Confidence Threshold:")
    print("  " + "-" * 60)
    print(f"  {'Threshold':<12} {'Trades':<10} {'Win Rate':<12} {'Avg Return':<12}")
    print("  " + "-" * 60)

    for threshold in [50, 60, 70, 80, 90]:
        mask = train_proba >= threshold
        if mask.sum() > 0:
            wins = y_train[mask].sum()
            total = mask.sum()
            win_rate = wins / total * 100
            avg_ret = metadata.loc[mask, 'fwd_return'].mean() * 100
            print(f"  >= {threshold}%{'':<7} {total:<10} {win_rate:.1f}%{'':<6} {avg_ret:+.1f}%")

    # Test on 2025 data
    print("\n[4/5] Testing on 2025 data (out-of-sample)...")
    test_results = evaluate_on_test(
        model, scaler, prices, spy_df, feature_names,
        signal_type='optimal_signal',
        start_date='2025-01-01'
    )

    if len(test_results) > 0:
        print(f"\n  2025 Signals Found: {len(test_results)}")

        # Filter to completed trades only for accurate metrics
        completed = test_results[test_results['is_complete'] == True]

        if len(completed) > 0:
            print(f"  Completed Trades: {len(completed)}")

            print("\n  2025 Out-of-Sample Performance by Confidence Threshold:")
            print("  " + "-" * 60)
            print(f"  {'Threshold':<12} {'Trades':<10} {'Win Rate':<12} {'Avg Return':<12}")
            print("  " + "-" * 60)

            for threshold in [50, 60, 70, 80, 90]:
                mask = completed['confidence'] >= threshold
                if mask.sum() > 0:
                    subset = completed[mask]
                    wins = (subset['actual_return'] > 0).sum()
                    total = len(subset)
                    win_rate = wins / total * 100
                    avg_ret = subset['actual_return'].mean() * 100
                    print(f"  >= {threshold}%{'':<7} {total:<10} {win_rate:.1f}%{'':<6} {avg_ret:+.1f}%")

        # Show all 2025 signals with confidence
        print("\n  2025 Signals with Confidence Scores:")
        print("  " + "-" * 70)
        test_results_sorted = test_results.sort_values('confidence', ascending=False)

        for _, row in test_results_sorted.head(15).iterrows():
            ret_str = f"{row['actual_return']*100:+.1f}%" if row['is_complete'] else "pending"
            print(f"  {row['symbol']:<6} {row['date'].strftime('%Y-%m-%d')} "
                  f"Confidence: {row['confidence']:.0f}%  Return: {ret_str}")
    else:
        print("  No signals found in 2025")

    # Save model if requested
    if args.save:
        print("\n[5/5] Saving model...")
        save_model(model, scaler, feature_names, project_root / 'ml_model.pkl')
    else:
        print("\n[5/5] Skipping save (use --save to save model)")

    # Summary
    print(f"""
{'='*80}
USAGE
{'='*80}

The model outputs a confidence score (0-100%) for each signal.
Higher confidence = higher probability of profitable trade.

RECOMMENDED THRESHOLDS:
  >= 70%: High confidence only (fewer trades, higher win rate)
  >= 60%: Balanced (moderate trades, good win rate)
  >= 50%: All signals (more trades, baseline win rate)

INTEGRATION:
  The screener will automatically show confidence scores.
  Filter signals by minimum confidence threshold.

NO FUTURE DATA LEAKAGE:
  - All features use only past data
  - Model trained on 2015-2024, tested on 2025
  - Forward return only used as training TARGET
""")


if __name__ == '__main__':
    main()

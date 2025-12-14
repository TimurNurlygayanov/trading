#!/usr/bin/env python3
"""
Training Pipeline for VWAP ML Strategy.

This script:
1. Downloads/loads historical price data (2024 for training, 2025 for testing)
2. Downloads news data from Polygon.io if configured
3. Trains CatBoost filter and Transformer model
4. Evaluates on 2025 test data
5. Saves trained models

Usage:
    python train_vwap_ml.py --symbol EURUSD --train-year 2024 --test-year 2025

Requirements:
    - Set POLYGON_API_KEY environment variable for news data
    - Set OANDA_API_KEY for price data (or use other data source)
"""
import argparse
import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from strategies.vwap_ml.strategy import VWAPMLStrategy, VWAPMLStrategyTrainer, VWAPMLConfig
from strategies.vwap_ml.catboost_filter import FilterConfig
from strategies.vwap_ml.features import FeatureEngineering
from strategies.vwap_ml.vwap_signal import VWAPSignalGenerator
from core.backtester import Backtester
from core.metrics import MetricsCalculator

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def download_price_data(
    symbol: str,
    start_date: str,
    end_date: str,
    timeframe: str = '5min'
) -> pd.DataFrame:
    """
    Download price data from various sources.

    Tries OANDA first, then falls back to yfinance or generates synthetic data.
    """
    logger.info(f"Downloading {symbol} data from {start_date} to {end_date}")

    # Try OANDA
    try:
        from data.downloaders.forex_downloader import ForexDownloader
        downloader = ForexDownloader()
        data = downloader.download(
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date
        )
        if data is not None and len(data) > 0:
            logger.info(f"Downloaded {len(data)} bars from OANDA")
            return data
    except Exception as e:
        logger.warning(f"OANDA download failed: {e}")

    # Try yfinance for forex
    try:
        import yfinance as yf

        # Convert symbol format (EURUSD -> EURUSD=X)
        yf_symbol = f"{symbol[:3]}{symbol[3:]}=X"

        ticker = yf.Ticker(yf_symbol)
        data = ticker.history(start=start_date, end=end_date, interval='5m')

        if len(data) > 0:
            data.columns = data.columns.str.lower()
            logger.info(f"Downloaded {len(data)} bars from yfinance")
            return data
    except Exception as e:
        logger.warning(f"yfinance download failed: {e}")

    # Generate synthetic data for testing
    logger.warning("Using synthetic data for demonstration")
    return generate_synthetic_data(start_date, end_date, symbol)


def generate_synthetic_data(
    start_date: str,
    end_date: str,
    symbol: str = 'EURUSD'
) -> pd.DataFrame:
    """
    Generate synthetic forex data for testing.

    Uses realistic parameters for EURUSD.
    """
    # Create date range (5-minute bars, only trading hours)
    dates = pd.date_range(
        start=start_date,
        end=end_date,
        freq='5min'
    )

    # Filter to trading hours (approximate forex hours)
    # Remove weekends and keep most hours
    dates = dates[dates.dayofweek < 5]

    np.random.seed(42)
    n = len(dates)

    # Simulate price movement with mean reversion and trends
    base_price = 1.10  # EURUSD typical price
    volatility = 0.0003  # ~3 pips per 5-min bar

    # Generate returns with autocorrelation
    returns = np.random.normal(0, volatility, n)

    # Add some autocorrelation (momentum)
    for i in range(1, n):
        returns[i] += 0.2 * returns[i-1]

    # Add day-of-week effects
    dow = dates.dayofweek
    returns[dow == 0] *= 0.8  # Lower volatility Monday
    returns[dow == 4] *= 1.2  # Higher volatility Friday

    # Generate OHLC from returns
    close = base_price * np.exp(np.cumsum(returns))
    open_prices = np.roll(close, 1)
    open_prices[0] = base_price

    # Generate high/low with realistic range
    range_mult = np.random.uniform(1.0, 2.0, n)
    half_range = np.abs(returns) * range_mult

    high = np.maximum(open_prices, close) + half_range
    low = np.minimum(open_prices, close) - half_range

    # Generate volume (synthetic)
    base_volume = 1000000
    volume = base_volume * np.random.lognormal(0, 0.5, n)

    # Add time-of-day volume patterns
    hour = dates.hour
    volume[hour < 7] *= 0.3  # Low volume Asian session
    volume[(hour >= 7) & (hour < 16)] *= 1.5  # High volume London
    volume[(hour >= 13) & (hour < 16)] *= 2.0  # Very high overlap

    data = pd.DataFrame({
        'open': open_prices,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    }, index=dates)

    logger.info(f"Generated {len(data)} synthetic bars")
    return data


def download_news_data(
    ticker: str,
    start_date: str,
    end_date: str
) -> Optional[pd.DataFrame]:
    """
    Download news data from Polygon.io.

    Returns None if API key not configured.
    """
    api_key = os.environ.get('POLYGON_API_KEY', '')

    if not api_key:
        logger.warning(
            "POLYGON_API_KEY not set. News features will not be available.\n"
            "Set it with: export POLYGON_API_KEY='your_key_here'\n"
            "Get a free API key at: https://polygon.io/"
        )
        return None

    try:
        from data.downloaders.news_downloader import PolygonNewsDownloader

        downloader = PolygonNewsDownloader()
        news_df = downloader.download_news(ticker, start_date, end_date)

        if news_df is not None and len(news_df) > 0:
            logger.info(f"Downloaded {len(news_df)} news articles")
            return news_df
        else:
            logger.warning("No news data found")
            return None

    except Exception as e:
        logger.error(f"News download failed: {e}")
        return None


def prepare_data(
    symbol: str,
    train_year: int,
    test_year: int,
    include_news: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.Series], Optional[pd.Series]]:
    """
    Prepare training and test data.

    Returns:
        Tuple of (train_data, test_data, train_news_sentiment, test_news_sentiment)
    """
    # Download price data
    train_data = download_price_data(
        symbol,
        f"{train_year}-01-01",
        f"{train_year}-12-31"
    )

    test_data = download_price_data(
        symbol,
        f"{test_year}-01-01",
        f"{test_year}-12-31"
    )

    # Download news data if configured
    train_news_sentiment = None
    test_news_sentiment = None

    if include_news:
        news_ticker = symbol.replace('/', '')  # EURUSD format for Polygon

        train_news = download_news_data(
            news_ticker,
            f"{train_year}-01-01",
            f"{train_year}-12-31"
        )

        if train_news is not None:
            from data.downloaders.news_downloader import PolygonNewsDownloader
            downloader = PolygonNewsDownloader()
            train_news_sentiment = downloader.get_sentiment_timeseries(
                train_news,
                freq='5min',
                start_date=f"{train_year}-01-01",
                end_date=f"{train_year}-12-31"
            )

        test_news = download_news_data(
            news_ticker,
            f"{test_year}-01-01",
            f"{test_year}-12-31"
        )

        if test_news is not None:
            test_news_sentiment = downloader.get_sentiment_timeseries(
                test_news,
                freq='5min',
                start_date=f"{test_year}-01-01",
                end_date=f"{test_year}-12-31"
            )

    return train_data, test_data, train_news_sentiment, test_news_sentiment


def train_models(
    train_data: pd.DataFrame,
    val_data: Optional[pd.DataFrame],
    news_sentiment: Optional[pd.Series],
    config: VWAPMLConfig,
    catboost_config: FilterConfig
) -> Dict[str, Any]:
    """
    Train VWAP ML strategy models.
    """
    trainer = VWAPMLStrategyTrainer(config, catboost_config)

    results = trainer.train(
        train_data,
        val_data,
        news_sentiment=news_sentiment
    )

    return results


def evaluate_strategy(
    test_data: pd.DataFrame,
    config: VWAPMLConfig,
    news_sentiment: Optional[pd.Series] = None
) -> Dict[str, Any]:
    """
    Evaluate trained strategy on test data.
    """
    # Load trained strategy
    strategy = VWAPMLStrategy(config=config)

    # Run backtest
    backtester = Backtester(
        initial_capital=100_000,
        leverage=30.0,
        commission=0.00007,
        max_daily_drawdown=5_000,
        risk_per_trade=2_000
    )

    results = backtester.run(strategy, test_data)

    return results


def print_results(train_results: Dict, test_results) -> None:
    """Print training and evaluation results."""
    print("\n" + "="*60)
    print("TRAINING RESULTS")
    print("="*60)

    if 'catboost' in train_results:
        cb = train_results['catboost']
        print("\nCatBoost Filter Metrics:")
        print(f"  Train Accuracy: {cb.get('train_accuracy', 0):.4f}")
        print(f"  Train Precision: {cb.get('train_precision', 0):.4f}")
        print(f"  Train Recall: {cb.get('train_recall', 0):.4f}")

        if 'val_accuracy' in cb:
            print(f"\n  Val Accuracy: {cb.get('val_accuracy', 0):.4f}")
            print(f"  Val Precision: {cb.get('val_precision', 0):.4f}")
            print(f"  Val Recall: {cb.get('val_recall', 0):.4f}")

        if 'feature_importance' in cb:
            print("\n  Top 10 Features:")
            importance = cb['feature_importance']
            sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]
            for feat, imp in sorted_features:
                print(f"    {feat}: {imp:.4f}")

    print("\n" + "="*60)
    print("BACKTEST RESULTS (Test Data)")
    print("="*60)

    metrics = test_results.metrics

    print(f"\n  Initial Capital: ${100_000:,.2f}")
    print(f"  Final Equity: ${test_results.equity_curve.iloc[-1]:,.2f}")
    print(f"\n  Total Return: {metrics['total_return']:.2%}")
    print(f"  Annualized Return: {metrics.get('annualized_return', 0):.2%}")
    print(f"\n  Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    print(f"  Sortino Ratio: {metrics.get('sortino_ratio', 0):.2f}")
    print(f"  Calmar Ratio: {metrics.get('calmar_ratio', 0):.2f}")
    print(f"\n  Max Drawdown: {metrics['max_drawdown']:.2%}")
    print(f"  Avg Drawdown: {metrics.get('avg_drawdown', 0):.2%}")
    print(f"\n  Win Rate: {metrics.get('win_rate', 0):.2%}")
    print(f"  Profit Factor: {metrics.get('profit_factor', 0):.2f}")
    print(f"  Num Trades: {metrics.get('num_trades', 0)}")

    print(f"\n  Account Blown: {metrics.get('account_blown', False)}")
    print(f"  Daily DD Breaches: {metrics.get('daily_dd_breaches', 0)}")

    print("="*60 + "\n")


def main():
    """Main training pipeline."""
    parser = argparse.ArgumentParser(description='Train VWAP ML Strategy')

    parser.add_argument('--symbol', type=str, default='EURUSD',
                        help='Trading symbol (default: EURUSD)')
    parser.add_argument('--train-year', type=int, default=2024,
                        help='Training year (default: 2024)')
    parser.add_argument('--test-year', type=int, default=2025,
                        help='Test year (default: 2025)')
    parser.add_argument('--no-news', action='store_true',
                        help='Disable news features')
    parser.add_argument('--model-dir', type=str, default='models/vwap_ml',
                        help='Directory to save models')
    parser.add_argument('--min-probability', type=float, default=0.55,
                        help='Minimum probability threshold (default: 0.55)')
    parser.add_argument('--use-transformer', action='store_true',
                        help='Train and use transformer model')

    args = parser.parse_args()

    # Configuration
    config = VWAPMLConfig(
        include_news=not args.no_news,
        use_transformer=args.use_transformer,
        min_probability=args.min_probability,
        model_dir=args.model_dir
    )

    catboost_config = FilterConfig(
        min_probability=args.min_probability,
        iterations=1000,
        learning_rate=0.03,
        depth=6
    )

    logger.info("="*60)
    logger.info("VWAP ML Strategy Training Pipeline")
    logger.info("="*60)
    logger.info(f"Symbol: {args.symbol}")
    logger.info(f"Train Year: {args.train_year}")
    logger.info(f"Test Year: {args.test_year}")
    logger.info(f"Include News: {not args.no_news}")
    logger.info(f"Use Transformer: {args.use_transformer}")
    logger.info(f"Min Probability: {args.min_probability}")
    logger.info("="*60)

    # Step 1: Prepare data
    logger.info("\n[1/4] Preparing data...")
    train_data, test_data, train_news, test_news = prepare_data(
        args.symbol,
        args.train_year,
        args.test_year,
        include_news=not args.no_news
    )

    logger.info(f"Train data: {len(train_data)} bars")
    logger.info(f"Test data: {len(test_data)} bars")

    # Step 2: Split training data for validation
    logger.info("\n[2/4] Splitting train/val data...")
    val_size = len(train_data) // 5  # 20% validation
    val_data = train_data.iloc[-val_size:]
    train_data_subset = train_data.iloc[:-val_size]

    logger.info(f"Training samples: {len(train_data_subset)}")
    logger.info(f"Validation samples: {len(val_data)}")

    # Step 3: Train models
    logger.info("\n[3/4] Training models...")
    train_results = train_models(
        train_data_subset,
        val_data,
        train_news,
        config,
        catboost_config
    )

    # Step 4: Evaluate on test data
    logger.info("\n[4/4] Evaluating on test data...")
    test_results = evaluate_strategy(
        test_data,
        config,
        test_news
    )

    # Print results
    print_results(train_results, test_results)

    # Save results summary
    results_path = Path(args.model_dir) / 'results_summary.json'
    results_path.parent.mkdir(parents=True, exist_ok=True)

    import json
    summary = {
        'symbol': args.symbol,
        'train_year': args.train_year,
        'test_year': args.test_year,
        'train_samples': len(train_data_subset),
        'val_samples': len(val_data),
        'test_samples': len(test_data),
        'metrics': {
            'total_return': float(test_results.metrics['total_return']),
            'sharpe_ratio': float(test_results.metrics['sharpe_ratio']),
            'max_drawdown': float(test_results.metrics['max_drawdown']),
            'win_rate': float(test_results.metrics.get('win_rate', 0)),
            'num_trades': int(test_results.metrics.get('num_trades', 0))
        }
    }

    with open(results_path, 'w') as f:
        json.dump(summary, f, indent=2)

    logger.info(f"\nResults saved to {results_path}")
    logger.info("Training complete!")


if __name__ == '__main__':
    main()

"""
Backtest runner for Mean Reversion Strategy.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from strategies.mean_reversion.strategy import MeanReversionStrategy
from core.backtester import Backtester
from data.downloaders.forex_downloader import ForexDownloader


def run_backtest(
    symbol: str = 'EURUSD',
    timeframe: str = '5min',
    start_date: str = '2023-01-01',
    end_date: str = '2024-01-01',
    initial_capital: float = 10000.0,
    **strategy_params
):
    """
    Run backtest for Mean Reversion Strategy.

    Args:
        symbol: Trading symbol
        timeframe: Data timeframe
        start_date: Backtest start date
        end_date: Backtest end date
        initial_capital: Starting capital
        **strategy_params: Strategy-specific parameters
    """
    print(f"\n{'='*60}")
    print(f"Mean Reversion Strategy Backtest")
    print(f"{'='*60}")
    print(f"Symbol: {symbol}")
    print(f"Timeframe: {timeframe}")
    print(f"Period: {start_date} to {end_date}")
    print(f"Initial Capital: ${initial_capital:,.2f}")
    print(f"{'='*60}\n")

    # Download data
    print("Downloading data...")
    downloader = ForexDownloader(source='yfinance')

    try:
        data = downloader.download(
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date
        )
    except Exception as e:
        print(f"Error downloading data: {e}")
        print("Using synthetic data instead...")
        data = downloader.generate_sample_data(
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date
        )

    print(f"Data shape: {data.shape}")
    print(f"Date range: {data.index[0]} to {data.index[-1]}")

    # Initialize strategy
    strategy = MeanReversionStrategy(params=strategy_params)
    print(f"\nStrategy: {strategy.name}")
    print(f"Parameters: {strategy.params}")

    # Run backtest
    print("\nRunning backtest...")
    backtester = Backtester(
        initial_capital=initial_capital,
        commission=0.0001,
        slippage=0.0001
    )

    results = backtester.run(strategy, data, start_date, end_date)

    # Print results
    print(results.summary())

    # Trade analysis
    if len(results.trades) > 0:
        print(f"\nRecent Trades:")
        print(results.trades.tail(10).to_string())

    return results


def main():
    parser = argparse.ArgumentParser(description='Mean Reversion Strategy Backtest')
    parser.add_argument('--symbol', type=str, default='EURUSD', help='Trading symbol')
    parser.add_argument('--timeframe', type=str, default='1h', help='Data timeframe')
    parser.add_argument('--start', type=str, default='2023-01-01', help='Start date')
    parser.add_argument('--end', type=str, default='2024-01-01', help='End date')
    parser.add_argument('--capital', type=float, default=10000.0, help='Initial capital')
    parser.add_argument('--bb-period', type=int, default=20, help='Bollinger Band period')
    parser.add_argument('--bb-std', type=float, default=2.0, help='Bollinger Band std')
    parser.add_argument('--rsi-period', type=int, default=14, help='RSI period')

    args = parser.parse_args()

    run_backtest(
        symbol=args.symbol,
        timeframe=args.timeframe,
        start_date=args.start,
        end_date=args.end,
        initial_capital=args.capital,
        bb_period=args.bb_period,
        bb_std=args.bb_std,
        rsi_period=args.rsi_period
    )


if __name__ == '__main__':
    main()

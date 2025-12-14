"""
Benchmark runner for comparing multiple strategies.

Features:
- Run backtests for multiple strategies
- Compare metrics across strategies
- Generate reports
"""
import pandas as pd
import importlib
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Type
import logging

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.backtester import Backtester
from core.metrics import BacktestResults
from data.downloaders.forex_downloader import ForexDownloader
from strategies.base_strategy import Strategy

logger = logging.getLogger(__name__)


class BenchmarkRunner:
    """
    Runner for benchmarking multiple trading strategies.

    Provides:
    - Automatic strategy discovery and loading
    - Parallel backtest execution
    - Metric comparison and ranking
    - Report generation
    """

    def __init__(
        self,
        initial_capital: float = 10000.0,
        commission: float = 0.0001,
        slippage: float = 0.0001,
        symbol: str = "EURUSD",
        timeframe: str = "1h"
    ):
        """
        Initialize benchmark runner.

        Args:
            initial_capital: Starting capital for backtests
            commission: Commission rate
            slippage: Slippage rate
            symbol: Trading symbol
            timeframe: Data timeframe
        """
        self.backtester = Backtester(
            initial_capital=initial_capital,
            commission=commission,
            slippage=slippage
        )
        self.downloader = ForexDownloader(source='yfinance')
        self.symbol = symbol
        self.timeframe = timeframe
        self.initial_capital = initial_capital

    def run_all(
        self,
        strategy_names: List[str],
        start_date: str,
        end_date: str,
        data: Optional[pd.DataFrame] = None
    ) -> Dict[str, BacktestResults]:
        """
        Run backtests for a list of strategies.

        Args:
            strategy_names: List of strategy folder names
            start_date: Backtest start date
            end_date: Backtest end date
            data: Optional pre-loaded data (downloads if not provided)

        Returns:
            Dictionary mapping strategy names to BacktestResults
        """
        # Download data if not provided
        if data is None:
            logger.info(f"Downloading {self.symbol} data...")
            try:
                data = self.downloader.download(
                    symbol=self.symbol,
                    timeframe=self.timeframe,
                    start_date=start_date,
                    end_date=end_date
                )
            except Exception as e:
                logger.warning(f"Download failed: {e}. Using synthetic data.")
                data = self.downloader.generate_sample_data(
                    symbol=self.symbol,
                    timeframe=self.timeframe,
                    start_date=start_date,
                    end_date=end_date
                )

        logger.info(f"Data loaded: {len(data)} bars")

        results = {}

        for strategy_name in strategy_names:
            logger.info(f"Running backtest for {strategy_name}...")
            try:
                strategy = self._load_strategy(strategy_name)
                result = self.backtester.run(strategy, data.copy(), start_date, end_date)
                results[strategy_name] = result
                logger.info(
                    f"{strategy_name}: Return={result.metrics['total_return']:.2%}, "
                    f"Sharpe={result.metrics['sharpe_ratio']:.2f}"
                )
            except Exception as e:
                logger.error(f"Error running {strategy_name}: {e}")

        return results

    def _load_strategy(self, strategy_name: str) -> Strategy:
        """
        Dynamically load a strategy by folder name.

        Args:
            strategy_name: Name of strategy folder (e.g., 'mean_reversion')

        Returns:
            Instantiated Strategy object
        """
        try:
            module = importlib.import_module(f"strategies.{strategy_name}.strategy")
        except ImportError as e:
            raise ImportError(f"Could not import strategy {strategy_name}: {e}")

        # Find the Strategy class in the module
        strategy_class = None
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if (isinstance(attr, type) and
                issubclass(attr, Strategy) and
                attr is not Strategy):
                strategy_class = attr
                break

        if strategy_class is None:
            raise ValueError(f"No Strategy subclass found in {strategy_name}")

        return strategy_class({})

    def get_available_strategies(self) -> List[str]:
        """
        Discover available strategies.

        Returns:
            List of strategy folder names
        """
        strategies_dir = Path(__file__).parent.parent / "strategies"
        strategies = []

        for item in strategies_dir.iterdir():
            if item.is_dir() and not item.name.startswith('_'):
                strategy_file = item / 'strategy.py'
                if strategy_file.exists():
                    strategies.append(item.name)

        return strategies

    def generate_report(self, results: Dict[str, BacktestResults]) -> pd.DataFrame:
        """
        Generate comparison report DataFrame.

        Args:
            results: Dictionary of backtest results

        Returns:
            DataFrame with metrics for all strategies
        """
        report_data = []

        for name, result in results.items():
            row = {'Strategy': name}
            row.update(result.metrics)
            report_data.append(row)

        df = pd.DataFrame(report_data)

        # Reorder columns
        priority_cols = [
            'Strategy', 'total_return', 'annualized_return', 'sharpe_ratio',
            'sortino_ratio', 'calmar_ratio', 'max_drawdown', 'win_rate',
            'profit_factor', 'num_trades'
        ]
        other_cols = [c for c in df.columns if c not in priority_cols]
        df = df[[c for c in priority_cols if c in df.columns] + other_cols]

        # Sort by Sharpe ratio
        if 'sharpe_ratio' in df.columns:
            df = df.sort_values('sharpe_ratio', ascending=False)

        return df

    def rank_strategies(
        self,
        results: Dict[str, BacktestResults],
        metrics: List[str] = None
    ) -> pd.DataFrame:
        """
        Rank strategies by multiple metrics.

        Args:
            results: Backtest results
            metrics: Metrics to rank by (default: sharpe, sortino, calmar)

        Returns:
            DataFrame with rankings
        """
        if metrics is None:
            metrics = ['sharpe_ratio', 'sortino_ratio', 'calmar_ratio', 'total_return']

        df = self.generate_report(results)

        rankings = pd.DataFrame({'Strategy': df['Strategy']})

        for metric in metrics:
            if metric in df.columns:
                # Higher is better for most metrics, except drawdown
                ascending = 'drawdown' in metric
                rankings[f'{metric}_rank'] = df[metric].rank(ascending=ascending)

        # Calculate average rank
        rank_cols = [c for c in rankings.columns if c.endswith('_rank')]
        if rank_cols:
            rankings['avg_rank'] = rankings[rank_cols].mean(axis=1)
            rankings = rankings.sort_values('avg_rank')

        return rankings


class ReportGenerator:
    """Generate formatted reports from backtest results."""

    @staticmethod
    def to_markdown(results: Dict[str, BacktestResults]) -> str:
        """Generate Markdown report."""
        runner = BenchmarkRunner()
        df = runner.generate_report(results)

        lines = [
            "# Strategy Benchmark Report",
            "",
            "## Summary Metrics",
            "",
            df.to_markdown(index=False),
            "",
            "## Individual Strategy Details",
            ""
        ]

        for name, result in results.items():
            lines.extend([
                f"### {name}",
                "",
                f"- Total Return: {result.metrics['total_return']:.2%}",
                f"- Sharpe Ratio: {result.metrics['sharpe_ratio']:.2f}",
                f"- Max Drawdown: {result.metrics['max_drawdown']:.2%}",
                f"- Number of Trades: {result.metrics.get('num_trades', 0):.0f}",
                ""
            ])

        return "\n".join(lines)

    @staticmethod
    def to_html(results: Dict[str, BacktestResults]) -> str:
        """Generate HTML report."""
        runner = BenchmarkRunner()
        df = runner.generate_report(results)

        # Format percentages
        pct_cols = ['total_return', 'annualized_return', 'max_drawdown',
                    'avg_drawdown', 'win_rate', 'volatility']
        for col in pct_cols:
            if col in df.columns:
                df[col] = df[col].apply(lambda x: f"{x:.2%}")

        # Format ratios
        ratio_cols = ['sharpe_ratio', 'sortino_ratio', 'calmar_ratio', 'profit_factor']
        for col in ratio_cols:
            if col in df.columns:
                df[col] = df[col].apply(lambda x: f"{x:.2f}")

        html = f"""
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #4CAF50; color: white; }}
                tr:nth-child(even) {{ background-color: #f2f2f2; }}
                h1 {{ color: #333; }}
            </style>
        </head>
        <body>
            <h1>Strategy Benchmark Report</h1>
            {df.to_html(index=False)}
        </body>
        </html>
        """
        return html


def main():
    """Run benchmark from command line."""
    import argparse

    parser = argparse.ArgumentParser(description='Run strategy benchmarks')
    parser.add_argument('--start', type=str, default='2023-01-01')
    parser.add_argument('--end', type=str, default='2024-01-01')
    parser.add_argument('--symbol', type=str, default='EURUSD')
    parser.add_argument('--timeframe', type=str, default='1h')
    parser.add_argument('--capital', type=float, default=10000.0)
    parser.add_argument('--strategies', nargs='+', default=None)

    args = parser.parse_args()

    runner = BenchmarkRunner(
        initial_capital=args.capital,
        symbol=args.symbol,
        timeframe=args.timeframe
    )

    # Get strategies to test
    strategies = args.strategies
    if strategies is None:
        strategies = runner.get_available_strategies()
        print(f"Found strategies: {strategies}")

    if not strategies:
        print("No strategies found!")
        return

    # Run benchmarks
    results = runner.run_all(strategies, args.start, args.end)

    # Print report
    report = runner.generate_report(results)
    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS")
    print("=" * 80)
    print(report.to_string())

    # Print rankings
    rankings = runner.rank_strategies(results)
    print("\n" + "=" * 80)
    print("STRATEGY RANKINGS")
    print("=" * 80)
    print(rankings.to_string(index=False))


if __name__ == '__main__':
    main()

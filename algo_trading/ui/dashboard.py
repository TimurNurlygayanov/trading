"""
Streamlit dashboard for algorithmic trading analysis.

Features:
- Strategy selection and comparison
- Backtest execution with progress
- Interactive metrics visualization
- Equity curve and drawdown charts
- Trade analysis
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime, timedelta

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.backtester import Backtester
from core.metrics import BacktestResults
from benchmarks.runner import BenchmarkRunner
from data.downloaders.forex_downloader import ForexDownloader


class TradingDashboard:
    """Streamlit dashboard for trading strategy analysis."""

    def __init__(self):
        """Initialize dashboard."""
        st.set_page_config(
            page_title="Algo Trading Dashboard",
            page_icon="📈",
            layout="wide",
            initial_sidebar_state="expanded"
        )

        # Initialize session state
        if 'results' not in st.session_state:
            st.session_state.results = None
        if 'data' not in st.session_state:
            st.session_state.data = None

    def run(self):
        """Run the dashboard."""
        st.title("📈 Algorithmic Trading Dashboard")

        # Sidebar configuration
        with st.sidebar:
            self._render_sidebar()

        # Main content
        if st.session_state.results:
            self._render_results()
        else:
            self._render_welcome()

    def _render_sidebar(self):
        """Render sidebar configuration."""
        st.header("Configuration")

        # Strategy selection
        runner = BenchmarkRunner()
        available_strategies = runner.get_available_strategies()

        if not available_strategies:
            st.warning("No strategies found!")
            available_strategies = ['mean_reversion', 'momentum', 'rl_ppo']

        selected_strategies = st.multiselect(
            "Select Strategies",
            available_strategies,
            default=available_strategies[:min(3, len(available_strategies))]
        )

        # Symbol and timeframe
        col1, col2 = st.columns(2)
        with col1:
            symbol = st.selectbox(
                "Symbol",
                ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD'],
                index=0
            )
        with col2:
            timeframe = st.selectbox(
                "Timeframe",
                ['5min', '15min', '30min', '1h', '4h', '1d'],
                index=3
            )

        # Date range
        st.subheader("Date Range")
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input(
                "Start Date",
                value=datetime.now() - timedelta(days=365)
            )
        with col2:
            end_date = st.date_input(
                "End Date",
                value=datetime.now()
            )

        # Parameters
        st.subheader("Parameters")
        initial_capital = st.number_input(
            "Initial Capital ($)",
            value=10000,
            min_value=1000,
            step=1000
        )

        commission = st.number_input(
            "Commission (pips)",
            value=1.0,
            min_value=0.0,
            max_value=10.0,
            step=0.1
        ) / 10000

        # Run button
        st.markdown("---")
        run_button = st.button(
            "🚀 Run Backtest",
            type="primary",
            use_container_width=True
        )

        if run_button and selected_strategies:
            with st.spinner("Running backtests..."):
                self._run_backtests(
                    selected_strategies,
                    symbol,
                    timeframe,
                    str(start_date),
                    str(end_date),
                    initial_capital,
                    commission
                )

    def _run_backtests(
        self,
        strategies: List[str],
        symbol: str,
        timeframe: str,
        start_date: str,
        end_date: str,
        initial_capital: float,
        commission: float
    ):
        """Execute backtests."""
        runner = BenchmarkRunner(
            initial_capital=initial_capital,
            commission=commission,
            symbol=symbol,
            timeframe=timeframe
        )

        results = runner.run_all(strategies, start_date, end_date)
        st.session_state.results = results

    def _render_welcome(self):
        """Render welcome message when no results."""
        st.markdown("""
        ## Welcome to the Algo Trading Dashboard

        This dashboard allows you to:
        - **Compare** multiple trading strategies
        - **Analyze** backtest results with comprehensive metrics
        - **Visualize** equity curves and drawdowns
        - **Examine** individual trades

        ### Getting Started

        1. Select one or more strategies from the sidebar
        2. Choose your symbol and timeframe
        3. Set the date range for backtesting
        4. Click "Run Backtest" to see results

        ### Available Strategies

        - **Mean Reversion**: Trades based on price returning to mean (Bollinger Bands + RSI)
        - **Momentum**: Follows trends using EMA crossovers and ADX
        - **RL PPO**: Reinforcement learning strategy using PPO algorithm
        """)

    def _render_results(self):
        """Render backtest results."""
        results = st.session_state.results

        # Metrics comparison
        st.header("📊 Strategy Comparison")
        self._render_metrics_table(results)

        # Charts
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📈 Equity Curves")
            fig_equity = self._plot_equity_curves(results)
            st.plotly_chart(fig_equity, use_container_width=True)

        with col2:
            st.subheader("📉 Drawdown Analysis")
            fig_dd = self._plot_drawdowns(results)
            st.plotly_chart(fig_dd, use_container_width=True)

        # Returns distribution
        st.subheader("📊 Returns Distribution")
        fig_returns = self._plot_returns_distribution(results)
        st.plotly_chart(fig_returns, use_container_width=True)

        # Individual strategy details
        st.header("🔍 Strategy Details")
        selected_strategy = st.selectbox(
            "Select Strategy for Details",
            list(results.keys())
        )

        if selected_strategy:
            self._render_strategy_details(selected_strategy, results[selected_strategy])

    def _render_metrics_table(self, results: Dict[str, BacktestResults]):
        """Render metrics comparison table."""
        data = []
        for name, result in results.items():
            row = {'Strategy': name}
            row.update(result.metrics)
            data.append(row)

        df = pd.DataFrame(data)

        # Format columns
        pct_cols = ['total_return', 'annualized_return', 'max_drawdown',
                    'avg_drawdown', 'win_rate', 'volatility', 'downside_volatility']
        ratio_cols = ['sharpe_ratio', 'sortino_ratio', 'calmar_ratio',
                      'profit_factor', 'omega_ratio']

        # Apply styling
        def highlight_best(s):
            """Highlight best value in column."""
            if s.name in ['max_drawdown', 'avg_drawdown', 'volatility', 'downside_volatility']:
                is_best = s == s.min()
            else:
                is_best = s == s.max()
            return ['background-color: lightgreen' if v else '' for v in is_best]

        # Format values
        format_dict = {}
        for col in df.columns:
            if col in pct_cols:
                format_dict[col] = '{:.2%}'
            elif col in ratio_cols:
                format_dict[col] = '{:.2f}'
            elif col == 'num_trades':
                format_dict[col] = '{:.0f}'

        styled_df = df.style.apply(
            highlight_best,
            subset=[c for c in df.columns if c not in ['Strategy']]
        ).format(format_dict)

        st.dataframe(styled_df, use_container_width=True)

        # Sort options
        col1, col2 = st.columns(2)
        with col1:
            sort_by = st.selectbox(
                "Sort by",
                ['sharpe_ratio', 'total_return', 'max_drawdown', 'win_rate', 'profit_factor']
            )
        with col2:
            sort_order = st.radio("Order", ["Descending", "Ascending"], horizontal=True)

        ascending = sort_order == "Ascending"
        if 'drawdown' in sort_by:
            ascending = not ascending

        sorted_df = df.sort_values(sort_by, ascending=ascending)
        st.write(f"**Best by {sort_by}:** {sorted_df['Strategy'].iloc[0]}")

    def _plot_equity_curves(self, results: Dict[str, BacktestResults]) -> go.Figure:
        """Plot equity curves for all strategies."""
        fig = go.Figure()

        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

        for i, (name, result) in enumerate(results.items()):
            color = colors[i % len(colors)]
            fig.add_trace(go.Scatter(
                x=result.equity_curve.index,
                y=result.equity_curve.values,
                name=name,
                mode='lines',
                line=dict(color=color, width=2)
            ))

        fig.update_layout(
            title="Portfolio Value Over Time",
            xaxis_title="Date",
            yaxis_title="Portfolio Value ($)",
            hovermode='x unified',
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            )
        )

        return fig

    def _plot_drawdowns(self, results: Dict[str, BacktestResults]) -> go.Figure:
        """Plot drawdown curves for all strategies."""
        fig = go.Figure()

        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

        for i, (name, result) in enumerate(results.items()):
            color = colors[i % len(colors)]

            # Calculate drawdown
            peak = result.equity_curve.expanding().max()
            dd = ((result.equity_curve - peak) / peak) * 100

            fig.add_trace(go.Scatter(
                x=dd.index,
                y=dd.values,
                name=name,
                mode='lines',
                fill='tozeroy',
                line=dict(color=color, width=1)
            ))

        fig.update_layout(
            title="Drawdown Over Time",
            xaxis_title="Date",
            yaxis_title="Drawdown (%)",
            hovermode='x unified',
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            )
        )

        return fig

    def _plot_returns_distribution(self, results: Dict[str, BacktestResults]) -> go.Figure:
        """Plot returns distribution for all strategies."""
        fig = make_subplots(
            rows=1,
            cols=len(results),
            subplot_titles=list(results.keys())
        )

        for i, (name, result) in enumerate(results.items(), 1):
            returns = result.returns.dropna() * 100

            fig.add_trace(
                go.Histogram(
                    x=returns,
                    nbinsx=50,
                    name=name,
                    showlegend=False
                ),
                row=1,
                col=i
            )

        fig.update_layout(
            title="Daily Returns Distribution",
            height=300
        )

        for i in range(1, len(results) + 1):
            fig.update_xaxes(title_text="Return (%)", row=1, col=i)
            fig.update_yaxes(title_text="Frequency", row=1, col=i)

        return fig

    def _render_strategy_details(self, name: str, result: BacktestResults):
        """Render detailed strategy information."""
        col1, col2, col3 = st.columns(3)

        # Key metrics
        with col1:
            st.metric(
                "Total Return",
                f"{result.metrics['total_return']:.2%}",
                delta=f"Sharpe: {result.metrics['sharpe_ratio']:.2f}"
            )

        with col2:
            st.metric(
                "Max Drawdown",
                f"{result.metrics['max_drawdown']:.2%}",
                delta=f"Duration: {result.metrics.get('max_drawdown_duration', 0):.0f} bars"
            )

        with col3:
            st.metric(
                "Win Rate",
                f"{result.metrics.get('win_rate', 0):.2%}",
                delta=f"Trades: {result.metrics.get('num_trades', 0):.0f}"
            )

        # All metrics
        st.subheader("All Metrics")

        metrics_df = pd.DataFrame([
            {'Metric': k, 'Value': v}
            for k, v in result.metrics.items()
        ])

        # Format values
        def format_metric(row):
            name = row['Metric']
            value = row['Value']

            if 'ratio' in name or 'factor' in name:
                return f"{value:.2f}"
            elif 'rate' in name or 'return' in name or 'drawdown' in name or 'volatility' in name:
                return f"{value:.2%}"
            elif 'duration' in name or 'trades' in name:
                return f"{value:.0f}"
            else:
                return f"{value:.4f}"

        metrics_df['Formatted'] = metrics_df.apply(format_metric, axis=1)

        # Display in columns
        col1, col2 = st.columns(2)
        mid = len(metrics_df) // 2

        with col1:
            for _, row in metrics_df.iloc[:mid].iterrows():
                st.write(f"**{row['Metric']}:** {row['Formatted']}")

        with col2:
            for _, row in metrics_df.iloc[mid:].iterrows():
                st.write(f"**{row['Metric']}:** {row['Formatted']}")

        # Recent trades
        if len(result.trades) > 0:
            st.subheader("Recent Trades")
            trades_display = result.trades.tail(20).copy()

            # Format columns
            if 'pnl' in trades_display.columns:
                trades_display['pnl'] = trades_display['pnl'].apply(lambda x: f"${x:.2f}")

            st.dataframe(trades_display, use_container_width=True)

            # Trade statistics
            st.subheader("Trade Statistics")
            total_trades = len(result.trades)
            winning_trades = (result.trades['pnl'] > 0).sum()
            losing_trades = (result.trades['pnl'] < 0).sum()

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Trades", total_trades)
            with col2:
                st.metric("Winning Trades", winning_trades)
            with col3:
                st.metric("Losing Trades", losing_trades)


def main():
    """Main entry point for dashboard."""
    dashboard = TradingDashboard()
    dashboard.run()


if __name__ == "__main__":
    main()

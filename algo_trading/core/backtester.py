"""
Universal backtester for all trading strategies.

CRITICAL RISK PARAMETERS:
- Account: $100,000
- Leverage: 30x
- Max Daily Drawdown: $5,000 (5%) - ACCOUNT BLOWN IF BREACHED
- Risk per Trade: $2,000 (2%)

Features:
- Event-driven simulation
- Daily drawdown tracking
- Risk-managed position sizing
- Commission and slippage modeling
"""
import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from datetime import date
import logging

from strategies.base_strategy import Strategy
from core.metrics import MetricsCalculator, BacktestResults
from core.risk_manager import RiskManager, RiskManagerConfig, RiskStatus

logger = logging.getLogger(__name__)


@dataclass
class Trade:
    """Represents a single trade."""
    entry_time: pd.Timestamp
    exit_time: Optional[pd.Timestamp]
    entry_price: float
    exit_price: Optional[float]
    position: int  # 1 for long, -1 for short
    units: int  # Number of currency units
    lots: float  # Lot size
    stop_loss: float
    take_profit: float
    pnl: float = 0.0
    pnl_pips: float = 0.0
    commission: float = 0.0
    risk_amount: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'entry_time': self.entry_time,
            'exit_time': self.exit_time,
            'entry_price': self.entry_price,
            'exit_price': self.exit_price,
            'position': self.position,
            'units': self.units,
            'lots': self.lots,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'pnl': self.pnl,
            'pnl_pips': self.pnl_pips,
            'commission': self.commission,
            'risk_amount': self.risk_amount,
        }


class Backtester:
    """
    Universal backtester with integrated risk management.

    ACCOUNT CONFIGURATION:
    - Initial Capital: $100,000
    - Leverage: 30x
    - Max Daily Drawdown: $5,000 (5%)
    - Risk per Trade: $2,000 (2%)

    If daily drawdown exceeds $5,000, the account is considered BLOWN
    and the backtest stops.
    """

    def __init__(
        self,
        initial_capital: float = 100_000.0,
        leverage: float = 30.0,
        commission: float = 0.00007,  # 0.7 pips
        slippage: float = 0.00003,    # 0.3 pips
        max_daily_drawdown: float = 5_000.0,
        risk_per_trade: float = 2_000.0,
        stop_on_daily_breach: bool = True
    ):
        """
        Initialize backtester with risk parameters.

        Args:
            initial_capital: Starting capital ($100,000 default)
            leverage: Account leverage (30x default)
            commission: Commission rate
            slippage: Slippage rate
            max_daily_drawdown: Max daily loss before stopping ($5,000)
            risk_per_trade: Max risk per trade ($2,000)
            stop_on_daily_breach: Stop trading if daily limit breached
        """
        self.initial_capital = initial_capital
        self.leverage = leverage
        self.commission = commission
        self.slippage = slippage
        self.max_daily_drawdown = max_daily_drawdown
        self.risk_per_trade = risk_per_trade
        self.stop_on_daily_breach = stop_on_daily_breach

        # Initialize risk manager
        self.risk_config = RiskManagerConfig(
            initial_capital=initial_capital,
            leverage=leverage,
            max_daily_drawdown=max_daily_drawdown,
            risk_per_trade=risk_per_trade,
            stop_on_daily_breach=stop_on_daily_breach
        )

    def run(
        self,
        strategy: Strategy,
        data: pd.DataFrame,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> BacktestResults:
        """
        Run backtest for a strategy with full risk management.

        Args:
            strategy: Strategy instance to test
            data: OHLCV data
            start_date: Start date
            end_date: End date

        Returns:
            BacktestResults with metrics and risk analysis
        """
        # Prepare data
        data = self._prepare_data(data, start_date, end_date)
        strategy.validate_data(data)

        # Preprocess data
        processed_data = strategy.preprocess_data(data.copy())

        # Generate signals
        logger.info(f"Generating signals for {strategy.name}...")
        signals = strategy.generate_signals(processed_data)

        # Simulate trading with risk management
        logger.info("Running trading simulation with risk management...")
        equity_curve, trades, risk_events = self._simulate_trading_with_risk(
            data, processed_data, signals, strategy
        )

        # Calculate returns
        returns = equity_curve.pct_change().fillna(0)

        # Convert trades to DataFrame
        trades_df = self._trades_to_dataframe(trades)

        # Calculate metrics
        logger.info("Calculating metrics...")
        metrics = MetricsCalculator.calculate_all(
            equity_curve, returns, trades_df, self.commission
        )

        # Add risk-specific metrics
        metrics['initial_capital'] = self.initial_capital
        metrics['leverage'] = self.leverage
        metrics['max_daily_drawdown_limit'] = self.max_daily_drawdown
        metrics['account_blown'] = risk_events.get('account_blown', False)
        metrics['days_traded'] = risk_events.get('days_traded', 0)
        metrics['max_daily_loss'] = risk_events.get('max_daily_loss', 0)
        metrics['daily_dd_breaches'] = risk_events.get('dd_breaches', 0)

        result = BacktestResults(
            equity_curve=equity_curve,
            returns=returns,
            trades=trades_df,
            metrics=metrics
        )

        # Log summary
        self._log_summary(result, risk_events)

        return result

    def _prepare_data(
        self,
        data: pd.DataFrame,
        start_date: Optional[str],
        end_date: Optional[str]
    ) -> pd.DataFrame:
        """Prepare and filter data."""
        data = data.copy()

        if not isinstance(data.index, pd.DatetimeIndex):
            data.index = pd.to_datetime(data.index)

        if start_date:
            data = data[data.index >= start_date]
        if end_date:
            data = data[data.index <= end_date]

        data.columns = data.columns.str.lower()
        return data

    def _simulate_trading_with_risk(
        self,
        data: pd.DataFrame,
        processed_data: pd.DataFrame,
        signals: pd.Series,
        strategy: Strategy
    ) -> tuple:
        """
        Simulate trading with full risk management.

        Returns:
            Tuple of (equity_curve, trades, risk_events)
        """
        # Initialize risk manager
        risk_manager = RiskManager(self.risk_config)

        equity = [self.initial_capital]
        cash = self.initial_capital
        position = 0
        entry_price = 0.0
        entry_time = None
        units = 0
        stop_loss = 0.0
        take_profit = 0.0
        trades: List[Trade] = []

        # Risk tracking
        current_day = None
        risk_events = {
            'account_blown': False,
            'days_traded': 0,
            'max_daily_loss': 0,
            'dd_breaches': 0,
            'daily_pnl': {}
        }

        # Calculate ATR for stop losses
        atr = self._calculate_atr(data, period=14)

        for i in range(1, len(data)):
            current_time = data.index[i]
            current_price = data['close'].iloc[i]
            current_atr = atr.iloc[i] if i < len(atr) and not pd.isna(atr.iloc[i]) else 0.0010

            # Track daily changes
            day = current_time.date()
            if day != current_day:
                current_day = day
                risk_manager.new_day(day, cash)
                risk_events['days_traded'] += 1

            # Update risk manager
            risk_state = risk_manager.update(cash, current_time)

            # Check if account is blown
            if risk_state.status == RiskStatus.BREACHED:
                risk_events['account_blown'] = True
                risk_events['dd_breaches'] += 1
                if self.stop_on_daily_breach:
                    logger.critical(f"ACCOUNT BLOWN at {current_time}! Daily loss: ${abs(risk_state.daily_drawdown):,.2f}")
                    break

            # Track max daily loss
            if risk_state.daily_drawdown < 0:
                risk_events['max_daily_loss'] = min(
                    risk_events['max_daily_loss'],
                    risk_state.daily_drawdown
                )

            # Get signal
            if i < len(signals):
                signal = signals.iloc[i]
            else:
                signal = 0

            if hasattr(signal, 'item'):
                signal = signal.item()
            signal = int(signal) if not pd.isna(signal) else 0

            # Check stop loss / take profit for open position
            if position != 0:
                hit_sl = (position == 1 and current_price <= stop_loss) or \
                         (position == -1 and current_price >= stop_loss)
                hit_tp = (position == 1 and current_price >= take_profit) or \
                         (position == -1 and current_price <= take_profit)

                if hit_sl or hit_tp:
                    exit_price = stop_loss if hit_sl else take_profit
                    exit_price = self._apply_slippage(exit_price, -position)

                    # Calculate P&L
                    pnl_pips = (exit_price - entry_price) * position * 10000
                    pnl_value = pnl_pips * (units / 100_000) * 10  # $10 per pip per lot

                    commission_cost = (units * exit_price + units * entry_price) * self.commission
                    pnl_value -= commission_cost

                    trade = Trade(
                        entry_time=entry_time,
                        exit_time=current_time,
                        entry_price=entry_price,
                        exit_price=exit_price,
                        position=position,
                        units=units,
                        lots=units / 100_000,
                        stop_loss=stop_loss,
                        take_profit=take_profit,
                        pnl=pnl_value,
                        pnl_pips=pnl_pips,
                        commission=commission_cost,
                        risk_amount=self.risk_per_trade
                    )
                    trades.append(trade)

                    cash += pnl_value
                    position = 0
                    units = 0
                    signal = 0  # Don't open new position on same bar

            # Close position on signal change
            if position != 0 and signal != position and signal != 0:
                exit_price = self._apply_slippage(current_price, -position)

                pnl_pips = (exit_price - entry_price) * position * 10000
                pnl_value = pnl_pips * (units / 100_000) * 10

                commission_cost = (units * exit_price) * self.commission
                pnl_value -= commission_cost

                trade = Trade(
                    entry_time=entry_time,
                    exit_time=current_time,
                    entry_price=entry_price,
                    exit_price=exit_price,
                    position=position,
                    units=units,
                    lots=units / 100_000,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    pnl=pnl_value,
                    pnl_pips=pnl_pips,
                    commission=commission_cost,
                    risk_amount=self.risk_per_trade
                )
                trades.append(trade)

                cash += pnl_value
                position = 0
                units = 0

            # Open new position
            if signal != 0 and position == 0:
                # Check if we can trade
                can_trade, reason = risk_manager.can_trade()

                if can_trade:
                    # Calculate stop loss and take profit
                    stop_loss = self._calculate_stop_loss(
                        current_price, current_atr, signal, multiplier=2.0
                    )
                    take_profit = self._calculate_take_profit(
                        current_price, stop_loss, signal, rr_ratio=2.0
                    )

                    # Calculate position size
                    pos_info = risk_manager.calculate_position_size(
                        entry_price=current_price,
                        stop_loss_price=stop_loss,
                        direction=signal
                    )

                    if pos_info['can_trade'] and pos_info['units'] > 0:
                        position = signal
                        entry_price = self._apply_slippage(current_price, signal)
                        entry_time = current_time
                        units = pos_info['units']

                        # Commission on entry
                        commission_cost = units * entry_price * self.commission
                        cash -= commission_cost

            # Update equity (mark-to-market)
            if position != 0:
                unrealized_pips = (current_price - entry_price) * position * 10000
                unrealized_pnl = unrealized_pips * (units / 100_000) * 10
                equity.append(cash + unrealized_pnl)
            else:
                equity.append(cash)

        # Create equity curve
        equity_series = pd.Series(
            equity[:len(data)],
            index=data.index[:len(equity)]
        )

        return equity_series, trades, risk_events

    def _calculate_atr(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range."""
        high = data['high']
        low = data['low']
        close = data['close']

        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window=period).mean()

    def _calculate_stop_loss(
        self,
        entry_price: float,
        atr: float,
        direction: int,
        multiplier: float = 2.0
    ) -> float:
        """Calculate stop loss based on ATR."""
        stop_distance = max(atr * multiplier, 0.0005)  # Min 5 pips
        return entry_price - (stop_distance * direction)

    def _calculate_take_profit(
        self,
        entry_price: float,
        stop_loss: float,
        direction: int,
        rr_ratio: float = 2.0
    ) -> float:
        """Calculate take profit based on R:R ratio."""
        risk = abs(entry_price - stop_loss)
        reward = risk * rr_ratio
        return entry_price + (reward * direction)

    def _apply_slippage(self, price: float, direction: int) -> float:
        """Apply slippage to execution price."""
        return price * (1 + self.slippage * direction)

    def _trades_to_dataframe(self, trades: List[Trade]) -> pd.DataFrame:
        """Convert trades to DataFrame."""
        if not trades:
            return pd.DataFrame(columns=[
                'entry_time', 'exit_time', 'entry_price', 'exit_price',
                'position', 'units', 'lots', 'stop_loss', 'take_profit',
                'pnl', 'pnl_pips', 'commission', 'risk_amount'
            ])
        return pd.DataFrame([t.to_dict() for t in trades])

    def _log_summary(self, result: BacktestResults, risk_events: Dict):
        """Log backtest summary."""
        metrics = result.metrics

        logger.info("=" * 60)
        logger.info("BACKTEST SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Initial Capital: ${self.initial_capital:,.2f}")
        logger.info(f"Final Equity: ${result.equity_curve.iloc[-1]:,.2f}")
        logger.info(f"Total Return: {metrics['total_return']:.2%}")
        logger.info(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        logger.info(f"Max Drawdown: {metrics['max_drawdown']:.2%}")
        logger.info("-" * 60)
        logger.info("RISK METRICS:")
        logger.info(f"Account Blown: {risk_events['account_blown']}")
        logger.info(f"Days Traded: {risk_events['days_traded']}")
        logger.info(f"Max Daily Loss: ${abs(risk_events['max_daily_loss']):,.2f}")
        logger.info(f"Daily DD Breaches: {risk_events['dd_breaches']}")
        logger.info("=" * 60)

    def run_walk_forward(
        self,
        strategy: Strategy,
        data: pd.DataFrame,
        train_period: int = 252,
        test_period: int = 63,
        step: int = 63
    ) -> List[BacktestResults]:
        """Run walk-forward analysis."""
        results = []
        total_bars = len(data)

        start_idx = train_period
        while start_idx + test_period <= total_bars:
            test_data = data.iloc[start_idx:start_idx + test_period]
            result = self.run(strategy, test_data)
            results.append(result)
            start_idx += step

        return results

    def monte_carlo_simulation(
        self,
        trades_pnl: pd.Series,
        n_simulations: int = 1000,
        n_trades: Optional[int] = None
    ) -> Dict[str, float]:
        """Run Monte Carlo simulation on trade PnLs."""
        if len(trades_pnl) == 0:
            return {}

        n_trades = n_trades or len(trades_pnl)
        final_pnls = []

        for _ in range(n_simulations):
            sampled = np.random.choice(trades_pnl.values, size=n_trades, replace=True)
            cumulative = np.cumsum(sampled)

            # Check for account blow (daily DD > $5000 at any point)
            # Simplified: check if cumsum ever drops below -5000 from peak
            peak = np.maximum.accumulate(cumulative)
            dd = cumulative - peak
            blown = (dd < -self.max_daily_drawdown).any()

            if blown:
                # Find blow point
                blow_idx = np.where(dd < -self.max_daily_drawdown)[0][0]
                final_pnls.append(cumulative[blow_idx])
            else:
                final_pnls.append(cumulative[-1])

        final_pnls = np.array(final_pnls)

        return {
            'mean_final_pnl': np.mean(final_pnls),
            'std_final_pnl': np.std(final_pnls),
            'percentile_5': np.percentile(final_pnls, 5),
            'percentile_25': np.percentile(final_pnls, 25),
            'percentile_50': np.percentile(final_pnls, 50),
            'percentile_75': np.percentile(final_pnls, 75),
            'percentile_95': np.percentile(final_pnls, 95),
            'prob_profit': np.mean(final_pnls > 0),
            'prob_account_blown': np.mean(final_pnls < -self.max_daily_drawdown),
        }

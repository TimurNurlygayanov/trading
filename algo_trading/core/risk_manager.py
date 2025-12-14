"""
Risk Manager for controlling position sizing and daily drawdown.

CRITICAL: This module enforces the $5,000 daily drawdown limit.
If breached, the account is considered lost and trading stops.

Account Configuration:
- Initial Capital: $100,000
- Leverage: 30x
- Max Daily Drawdown: $5,000 (5%)
- Risk per Trade: $2,000 (2%)
"""
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Optional, List, Tuple
from datetime import datetime, date
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class RiskStatus(Enum):
    """Risk status levels."""
    NORMAL = "normal"           # < 50% of daily limit
    WARNING = "warning"         # 50-80% of daily limit
    CRITICAL = "critical"       # 80-100% of daily limit
    BREACHED = "breached"       # Daily limit exceeded - STOP TRADING


@dataclass
class DailyRiskState:
    """Tracks daily risk metrics."""
    date: date
    starting_equity: float
    current_equity: float
    daily_pnl: float = 0.0
    daily_drawdown: float = 0.0
    peak_equity: float = 0.0
    trades_today: int = 0
    status: RiskStatus = RiskStatus.NORMAL

    @property
    def drawdown_remaining(self) -> float:
        """Remaining drawdown before breach."""
        return max(0, 5000.0 - abs(self.daily_drawdown))

    @property
    def drawdown_pct_used(self) -> float:
        """Percentage of daily drawdown limit used."""
        return abs(self.daily_drawdown) / 5000.0


@dataclass
class RiskManagerConfig:
    """Risk management configuration."""
    initial_capital: float = 100_000.0
    leverage: float = 30.0
    max_daily_drawdown: float = 5_000.0
    risk_per_trade: float = 2_000.0
    max_position_risk_pct: float = 0.02  # 2% of capital
    max_open_positions: int = 3
    stop_on_daily_breach: bool = True


class RiskManager:
    """
    Risk Manager for controlling trading risk.

    CRITICAL RULES:
    1. Max daily drawdown: $5,000 - if breached, STOP ALL TRADING
    2. Max risk per trade: $2,000 (2% of account)
    3. Position sizing based on stop loss distance
    4. Leverage: 30x maximum
    """

    def __init__(self, config: Optional[RiskManagerConfig] = None):
        """
        Initialize risk manager.

        Args:
            config: Risk configuration (uses defaults if not provided)
        """
        self.config = config or RiskManagerConfig()
        self.daily_states: Dict[date, DailyRiskState] = {}
        self.current_date: Optional[date] = None
        self.account_blown = False
        self.total_equity = self.config.initial_capital

    def new_day(self, trading_date: date, equity: float) -> DailyRiskState:
        """
        Start a new trading day.

        Args:
            trading_date: The trading date
            equity: Current account equity

        Returns:
            DailyRiskState for the new day
        """
        state = DailyRiskState(
            date=trading_date,
            starting_equity=equity,
            current_equity=equity,
            peak_equity=equity
        )
        self.daily_states[trading_date] = state
        self.current_date = trading_date
        self.total_equity = equity

        logger.info(f"New trading day: {trading_date}, Starting equity: ${equity:,.2f}")
        return state

    def update(self, equity: float, timestamp: Optional[datetime] = None) -> DailyRiskState:
        """
        Update risk state with new equity value.

        Args:
            equity: Current account equity
            timestamp: Optional timestamp (uses current date if not provided)

        Returns:
            Updated DailyRiskState
        """
        if timestamp:
            current_date = timestamp.date()
        else:
            current_date = self.current_date or date.today()

        # Initialize new day if needed
        if current_date not in self.daily_states:
            self.new_day(current_date, equity)

        state = self.daily_states[current_date]
        state.current_equity = equity
        state.peak_equity = max(state.peak_equity, equity)

        # Calculate daily P&L and drawdown
        state.daily_pnl = equity - state.starting_equity
        state.daily_drawdown = min(0, equity - state.peak_equity)

        # Also track intraday drawdown from day start
        intraday_dd = equity - state.starting_equity
        if intraday_dd < state.daily_drawdown:
            state.daily_drawdown = intraday_dd

        # Update status
        state.status = self._calculate_status(state)

        # Check for breach
        if state.status == RiskStatus.BREACHED:
            self._handle_breach(state)

        self.total_equity = equity
        return state

    def _calculate_status(self, state: DailyRiskState) -> RiskStatus:
        """Calculate risk status based on daily drawdown."""
        dd_abs = abs(state.daily_drawdown)
        max_dd = self.config.max_daily_drawdown

        if dd_abs >= max_dd:
            return RiskStatus.BREACHED
        elif dd_abs >= max_dd * 0.8:
            return RiskStatus.CRITICAL
        elif dd_abs >= max_dd * 0.5:
            return RiskStatus.WARNING
        else:
            return RiskStatus.NORMAL

    def _handle_breach(self, state: DailyRiskState):
        """Handle daily drawdown breach."""
        self.account_blown = True
        logger.critical(
            f"DAILY DRAWDOWN BREACH! Date: {state.date}, "
            f"Drawdown: ${abs(state.daily_drawdown):,.2f}, "
            f"Limit: ${self.config.max_daily_drawdown:,.2f}"
        )
        logger.critical("ACCOUNT BLOWN - STOP ALL TRADING")

    def can_trade(self) -> Tuple[bool, str]:
        """
        Check if trading is allowed.

        Returns:
            Tuple of (can_trade, reason)
        """
        if self.account_blown:
            return False, "Account blown - daily drawdown limit breached"

        if self.current_date and self.current_date in self.daily_states:
            state = self.daily_states[self.current_date]

            if state.status == RiskStatus.BREACHED:
                return False, f"Daily drawdown limit breached: ${abs(state.daily_drawdown):,.2f}"

            if state.status == RiskStatus.CRITICAL:
                return True, f"WARNING: Critical drawdown level: ${abs(state.daily_drawdown):,.2f}"

        return True, "OK"

    def calculate_position_size(
        self,
        entry_price: float,
        stop_loss_price: float,
        direction: int = 1  # 1 for long, -1 for short
    ) -> Dict[str, float]:
        """
        Calculate position size based on risk management rules.

        Uses fixed fractional position sizing:
        Position Size = Risk Amount / (Entry - Stop Loss)

        Args:
            entry_price: Planned entry price
            stop_loss_price: Stop loss price
            direction: 1 for long, -1 for short

        Returns:
            Dict with position sizing details
        """
        # Check if we can trade
        can_trade, reason = self.can_trade()
        if not can_trade:
            return {
                'can_trade': False,
                'reason': reason,
                'units': 0,
                'lots': 0,
                'notional': 0,
                'risk_amount': 0
            }

        # Calculate pip distance to stop loss
        pip_distance = abs(entry_price - stop_loss_price) * 10000  # Convert to pips

        if pip_distance == 0:
            return {
                'can_trade': False,
                'reason': 'Stop loss too close to entry',
                'units': 0,
                'lots': 0,
                'notional': 0,
                'risk_amount': 0
            }

        # Risk amount (reduced if we're in warning/critical status)
        risk_amount = self.config.risk_per_trade

        if self.current_date and self.current_date in self.daily_states:
            state = self.daily_states[self.current_date]
            remaining = state.drawdown_remaining

            # Don't risk more than remaining daily limit
            risk_amount = min(risk_amount, remaining * 0.5)  # Max 50% of remaining

            if state.status == RiskStatus.WARNING:
                risk_amount *= 0.5  # Reduce by 50%
            elif state.status == RiskStatus.CRITICAL:
                risk_amount *= 0.25  # Reduce by 75%

        # Calculate position size
        # For EURUSD: 1 pip = $10 per standard lot (100,000 units)
        pip_value_per_lot = 10.0
        lots = risk_amount / (pip_distance * pip_value_per_lot)

        # Convert to units
        units = lots * 100_000

        # Calculate notional value
        notional = units * entry_price

        # Check leverage constraint
        max_notional = self.total_equity * self.config.leverage
        if notional > max_notional:
            # Scale down to max leverage
            units = max_notional / entry_price
            lots = units / 100_000
            notional = max_notional

        return {
            'can_trade': True,
            'reason': 'OK',
            'units': int(units),
            'lots': round(lots, 2),
            'notional': round(notional, 2),
            'risk_amount': round(risk_amount, 2),
            'pip_distance': round(pip_distance, 1),
            'leverage_used': round(notional / self.total_equity, 1)
        }

    def calculate_stop_loss(
        self,
        entry_price: float,
        atr: float,
        direction: int,
        atr_multiplier: float = 2.0
    ) -> float:
        """
        Calculate stop loss price based on ATR.

        Args:
            entry_price: Entry price
            atr: Average True Range value
            direction: 1 for long, -1 for short
            atr_multiplier: Multiplier for ATR

        Returns:
            Stop loss price
        """
        stop_distance = atr * atr_multiplier
        return entry_price - (stop_distance * direction)

    def calculate_take_profit(
        self,
        entry_price: float,
        stop_loss_price: float,
        direction: int,
        risk_reward_ratio: float = 2.0
    ) -> float:
        """
        Calculate take profit based on risk/reward ratio.

        Args:
            entry_price: Entry price
            stop_loss_price: Stop loss price
            direction: 1 for long, -1 for short
            risk_reward_ratio: Target R:R ratio

        Returns:
            Take profit price
        """
        risk = abs(entry_price - stop_loss_price)
        reward = risk * risk_reward_ratio
        return entry_price + (reward * direction)

    def get_current_state(self) -> Optional[DailyRiskState]:
        """Get current day's risk state."""
        if self.current_date and self.current_date in self.daily_states:
            return self.daily_states[self.current_date]
        return None

    def get_summary(self) -> Dict:
        """Get risk management summary."""
        state = self.get_current_state()

        return {
            'account_blown': self.account_blown,
            'total_equity': self.total_equity,
            'initial_capital': self.config.initial_capital,
            'total_pnl': self.total_equity - self.config.initial_capital,
            'total_pnl_pct': (self.total_equity / self.config.initial_capital - 1) * 100,
            'daily_state': {
                'date': str(state.date) if state else None,
                'daily_pnl': state.daily_pnl if state else 0,
                'daily_drawdown': state.daily_drawdown if state else 0,
                'drawdown_remaining': state.drawdown_remaining if state else 5000,
                'status': state.status.value if state else 'unknown',
            } if state else None,
            'max_daily_drawdown_limit': self.config.max_daily_drawdown,
            'risk_per_trade': self.config.risk_per_trade,
            'leverage': self.config.leverage,
        }

    def reset(self):
        """Reset risk manager to initial state."""
        self.daily_states.clear()
        self.current_date = None
        self.account_blown = False
        self.total_equity = self.config.initial_capital
        logger.info("Risk manager reset")


def calculate_lot_size(
    account_balance: float,
    risk_percent: float,
    stop_loss_pips: float,
    pip_value: float = 10.0
) -> float:
    """
    Utility function to calculate lot size.

    Args:
        account_balance: Account balance in USD
        risk_percent: Risk per trade as decimal (e.g., 0.02 for 2%)
        stop_loss_pips: Distance to stop loss in pips
        pip_value: Value per pip per lot (default $10 for EURUSD)

    Returns:
        Lot size
    """
    risk_amount = account_balance * risk_percent
    lot_size = risk_amount / (stop_loss_pips * pip_value)
    return round(lot_size, 2)

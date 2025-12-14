"""
Mean Reversion Trading Strategy.

ACCOUNT PARAMETERS:
- Initial Capital: $100,000
- Leverage: 30x
- Max Daily Drawdown: $5,000 (5%)
- Risk per Trade: $2,000 (2%)

This strategy trades based on the assumption that prices tend to revert
to their mean over time. Uses Bollinger Bands and RSI for signal generation.
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from strategies.base_strategy import Strategy


class MeanReversionStrategy(Strategy):
    """
    Mean Reversion Strategy using Bollinger Bands and RSI.

    RISK MANAGEMENT:
    - Uses ATR-based stop losses (1.5x ATR for mean reversion)
    - Tighter stops for mean reversion vs trend following
    - Best suited for range-bound, low volatility conditions

    Signal Logic:
    - Buy when price touches lower Bollinger Band AND RSI < oversold
    - Sell when price touches upper Bollinger Band AND RSI > overbought
    - Exit when price returns to middle band OR stop loss hit

    Best suited for:
    - Range-bound markets (low ADX < 25)
    - Low volatility periods (Asian session)
    - EURUSD 01:00-04:00 GMT
    """

    DEFAULT_PARAMS = {
        # Bollinger Bands
        'bb_period': 20,
        'bb_std': 2.0,

        # RSI
        'rsi_period': 14,
        'rsi_oversold': 30,
        'rsi_overbought': 70,

        # ATR for stops
        'atr_period': 14,
        'atr_stop_multiplier': 1.5,  # Tighter for mean reversion

        # Risk Management (aligned with $100k account)
        'risk_per_trade_pct': 0.02,  # 2% = $2,000
        'take_profit_rr': 1.5,       # 1.5:1 R:R for mean reversion (higher win rate expected)

        # Filters
        'max_adx': 25,  # Only trade when ADX below this (ranging market)
        'min_atr_pips': 3,  # Minimum volatility to trade
        'max_atr_pips': 30,  # Maximum volatility to trade
    }

    def __init__(self, params: Dict[str, Any] = None):
        """Initialize Mean Reversion Strategy."""
        merged_params = {**self.DEFAULT_PARAMS, **(params or {})}
        super().__init__(merged_params)

    @property
    def required_history(self) -> int:
        """Minimum bars needed."""
        return max(
            self.params['bb_period'],
            self.params['rsi_period'],
            self.params['atr_period']
        ) + 20

    def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators for mean reversion trading."""
        df = data.copy()

        # Bollinger Bands
        bb_period = self.params['bb_period']
        bb_std = self.params['bb_std']

        df['bb_middle'] = df['close'].rolling(window=bb_period).mean()
        df['bb_std'] = df['close'].rolling(window=bb_period).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * df['bb_std'])
        df['bb_lower'] = df['bb_middle'] - (bb_std * df['bb_std'])

        # Percent B (position within Bollinger Bands)
        bb_width = df['bb_upper'] - df['bb_lower']
        df['percent_b'] = (df['close'] - df['bb_lower']) / bb_width.replace(0, np.nan)

        # RSI
        df['rsi'] = self._calculate_rsi(df['close'], self.params['rsi_period'])

        # ATR for stop loss calculation
        df['atr'] = self._calculate_atr(df, self.params['atr_period'])
        df['atr_pips'] = df['atr'] * 10000  # Convert to pips

        # ADX for trend filter
        df['adx'] = self._calculate_adx(df, 14)

        return df

    def _calculate_rsi(self, prices: pd.Series, period: int) -> pd.Series:
        """Calculate RSI."""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        avg_gain = gain.rolling(window=period, min_periods=period).mean()
        avg_loss = loss.rolling(window=period, min_periods=period).mean()

        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    def _calculate_atr(self, df: pd.DataFrame, period: int) -> pd.Series:
        """Calculate ATR."""
        high = df['high']
        low = df['low']
        close = df['close']

        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window=period).mean()

    def _calculate_adx(self, df: pd.DataFrame, period: int) -> pd.Series:
        """Calculate ADX (trend strength indicator)."""
        high = df['high']
        low = df['low']
        close = df['close']

        # True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # Directional Movement
        plus_dm = high.diff()
        minus_dm = -low.diff()
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)

        # Smoothed values
        atr = tr.rolling(window=period).mean()
        plus_dm_smooth = plus_dm.rolling(window=period).mean()
        minus_dm_smooth = minus_dm.rolling(window=period).mean()

        # Directional Indicators
        plus_di = 100 * (plus_dm_smooth / atr)
        minus_di = 100 * (minus_dm_smooth / atr)

        # DX and ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.nan)
        return dx.rolling(window=period).mean()

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate trading signals based on mean reversion logic.

        Returns:
            Series with values: 1 (buy), -1 (sell), 0 (hold/flat)
        """
        signals = pd.Series(index=data.index, data=0)

        # Ensure indicators are present
        if 'bb_upper' not in data.columns:
            data = self.preprocess_data(data)

        rsi_oversold = self.params['rsi_oversold']
        rsi_overbought = self.params['rsi_overbought']
        max_adx = self.params['max_adx']
        min_atr = self.params['min_atr_pips']
        max_atr = self.params['max_atr_pips']

        for i in range(self.required_history, len(data)):
            close = data['close'].iloc[i]
            bb_lower = data['bb_lower'].iloc[i]
            bb_upper = data['bb_upper'].iloc[i]
            bb_middle = data['bb_middle'].iloc[i]
            rsi = data['rsi'].iloc[i]
            adx = data['adx'].iloc[i] if 'adx' in data.columns else 20
            atr_pips = data['atr_pips'].iloc[i] if 'atr_pips' in data.columns else 10

            # Skip if NaN values
            if pd.isna(bb_lower) or pd.isna(rsi):
                continue

            prev_signal = signals.iloc[i-1] if i > 0 else 0

            # Volatility filter
            valid_volatility = min_atr <= atr_pips <= max_atr

            # Trend filter (only trade in ranging markets)
            ranging_market = pd.isna(adx) or adx < max_adx

            # Buy signal: price at lower band + RSI oversold + ranging market
            if (close <= bb_lower and
                rsi < rsi_oversold and
                ranging_market and
                valid_volatility):
                signals.iloc[i] = 1

            # Sell signal: price at upper band + RSI overbought + ranging market
            elif (close >= bb_upper and
                  rsi > rsi_overbought and
                  ranging_market and
                  valid_volatility):
                signals.iloc[i] = -1

            # Exit long: price crosses above middle band (profit target area)
            elif prev_signal == 1 and close >= bb_middle:
                signals.iloc[i] = 0

            # Exit short: price crosses below middle band
            elif prev_signal == -1 and close <= bb_middle:
                signals.iloc[i] = 0

            # Exit if trend develops (ADX rises)
            elif prev_signal != 0 and not pd.isna(adx) and adx > max_adx + 5:
                signals.iloc[i] = 0

            # Hold current position
            else:
                signals.iloc[i] = prev_signal

        return signals

    def get_stop_loss(self, entry_price: float, atr: float, direction: int) -> float:
        """
        Calculate stop loss price.

        For mean reversion, use tighter stops (1.5x ATR).
        """
        multiplier = self.params['atr_stop_multiplier']
        stop_distance = max(atr * multiplier, 0.0003)  # Min 3 pips
        return entry_price - (stop_distance * direction)

    def get_take_profit(self, entry_price: float, stop_loss: float, direction: int) -> float:
        """Calculate take profit based on R:R ratio."""
        risk = abs(entry_price - stop_loss)
        reward = risk * self.params['take_profit_rr']
        return entry_price + (reward * direction)

    def get_position_size(
        self,
        signal: int,
        portfolio_value: float,
        current_price: float,
        stop_loss_price: float = None,
        atr: float = None
    ) -> float:
        """
        Calculate position size based on risk management rules.

        Uses fixed fractional position sizing:
        Risk Amount / (Entry - Stop) = Position Size

        With $100k account and 2% risk:
        $2,000 / (stop_distance_in_pips * $10_per_pip) = lots
        """
        if signal == 0:
            return 0.0

        risk_pct = self.params['risk_per_trade_pct']
        risk_amount = portfolio_value * risk_pct  # $2,000 with $100k

        if stop_loss_price is not None and stop_loss_price != current_price:
            stop_pips = abs(current_price - stop_loss_price) * 10000
        elif atr is not None:
            stop_pips = atr * 10000 * self.params['atr_stop_multiplier']
        else:
            stop_pips = 15  # Default 15 pips

        # $10 per pip per standard lot
        pip_value = 10.0
        lots = risk_amount / (stop_pips * pip_value)

        # Convert to notional value
        units = lots * 100_000
        return units * current_price

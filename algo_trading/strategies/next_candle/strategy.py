"""
Next Candle Prediction Strategy - CatBoost ML + tsfresh

Concept:
- Predict if next candle will be GREEN (close > open)
- If green predicted with high confidence: BUY now, SELL at candle close
- If red predicted with high confidence: SELL now, BUY at candle close
- Simple 1-bar holding period

Features:
- tsfresh automated time series features
- Recent price patterns (last N candles)
- Candle body/wick ratios
- EMAs and their slopes
- RSI, MACD
- Volume patterns
- Time of day/week features
"""
import numpy as np
import pandas as pd
from typing import Optional, Dict, List
from dataclasses import dataclass
from catboost import CatBoostClassifier
from tsfresh import extract_features
from tsfresh.feature_extraction import MinimalFCParameters
from tsfresh.utilities.dataframe_functions import impute
import warnings
warnings.filterwarnings('ignore')


@dataclass
class NextCandleConfig:
    """Configuration for Next Candle prediction strategy."""
    # Model parameters - balanced
    iterations: int = 300
    depth: int = 4
    learning_rate: float = 0.05
    l2_leaf_reg: float = 5.0

    # tsfresh features
    use_tsfresh: bool = True
    tsfresh_window: int = 10  # Window for tsfresh extraction

    # Trading parameters
    min_probability: float = 0.58   # Balanced confidence

    # Lookback for features
    lookback_candles: int = 10

    # Cooldown
    min_bars_between_trades: int = 12  # 1 hour at 5min TF

    # Risk management (1-bar trades, so minimal)
    max_trades_per_day: int = 50


class NextCandleStrategy:
    """
    Next Candle Color Prediction Strategy.

    Uses CatBoost to predict if next candle will be green.
    Trades: Buy at current close, sell at next close (or vice versa).
    """

    def __init__(self, config: Optional[NextCandleConfig] = None):
        self.config = config or NextCandleConfig()
        self.name = "Next Candle"
        self.params = {}

        self.model: Optional[CatBoostClassifier] = None
        self.feature_names: List[str] = []
        self._tsfresh_features: List[str] = []
        self._is_trained = False

    @property
    def required_history(self) -> int:
        return max(60, self.config.lookback_candles + 30, self.config.tsfresh_window + 30)

    def validate_data(self, data: pd.DataFrame) -> bool:
        required = ['open', 'high', 'low', 'close']
        missing = [c for c in required if c not in data.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")
        return True

    def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate all features for prediction."""
        df = data.copy()

        # Basic candle features
        df['body'] = df['close'] - df['open']
        df['body_pct'] = df['body'] / df['open']
        df['upper_wick'] = df['high'] - df[['open', 'close']].max(axis=1)
        df['lower_wick'] = df[['open', 'close']].min(axis=1) - df['low']
        df['range'] = df['high'] - df['low']
        df['body_to_range'] = df['body'].abs() / (df['range'] + 1e-10)

        # Is green candle
        df['is_green'] = (df['close'] > df['open']).astype(int)

        # Recent candle patterns
        for i in range(1, self.config.lookback_candles + 1):
            df[f'green_{i}'] = df['is_green'].shift(i)
            df[f'body_pct_{i}'] = df['body_pct'].shift(i)
            df[f'range_{i}'] = df['range'].shift(i)

        # Consecutive greens/reds
        df['consec_green'] = self._count_consecutive(df['is_green'], 1)
        df['consec_red'] = self._count_consecutive(df['is_green'], 0)

        # Price momentum
        for period in [3, 5, 10, 20]:
            df[f'return_{period}'] = df['close'].pct_change(period)

        # EMAs
        df['ema5'] = df['close'].ewm(span=5, adjust=False).mean()
        df['ema10'] = df['close'].ewm(span=10, adjust=False).mean()
        df['ema20'] = df['close'].ewm(span=20, adjust=False).mean()
        df['ema50'] = df['close'].ewm(span=50, adjust=False).mean()

        df['price_to_ema5'] = df['close'] / df['ema5'] - 1
        df['price_to_ema10'] = df['close'] / df['ema10'] - 1
        df['price_to_ema20'] = df['close'] / df['ema20'] - 1
        df['price_to_ema50'] = df['close'] / df['ema50'] - 1

        df['ema5_slope'] = df['ema5'].pct_change(3)
        df['ema10_slope'] = df['ema10'].pct_change(3)
        df['ema20_slope'] = df['ema20'].pct_change(3)
        df['ema50_slope'] = df['ema50'].pct_change(3)

        # EMA50 direction: 1 = up, 0 = down
        df['ema50_up'] = (df['ema50'] > df['ema50'].shift(1)).astype(int)

        # RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.ewm(span=14, adjust=False).mean()
        avg_loss = loss.ewm(span=14, adjust=False).mean()
        rs = avg_gain / (avg_loss + 1e-10)
        df['rsi'] = 100 - (100 / (1 + rs))
        df['rsi_slope'] = df['rsi'].diff(3)

        # MACD
        ema12 = df['close'].ewm(span=12, adjust=False).mean()
        ema26 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = ema12 - ema26
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']

        # Bollinger Bands
        df['bb_mid'] = df['close'].rolling(20).mean()
        bb_std = df['close'].rolling(20).std()
        df['bb_upper'] = df['bb_mid'] + 2 * bb_std
        df['bb_lower'] = df['bb_mid'] - 2 * bb_std
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-10)

        # Volatility
        df['volatility'] = df['close'].pct_change().rolling(20).std()
        df['atr'] = self._calculate_atr(df, 14)
        df['atr_pct'] = df['atr'] / df['close']

        # Volume features (if available)
        if 'volume' in df.columns and df['volume'].sum() > 0:
            df['volume_ma'] = df['volume'].rolling(20).mean()
            df['volume_ratio'] = df['volume'] / (df['volume_ma'] + 1)
            df['volume_trend'] = df['volume'].rolling(5).mean() / (df['volume'].rolling(20).mean() + 1)
        else:
            df['volume_ratio'] = 1.0
            df['volume_trend'] = 1.0

        # Time features (if datetime index)
        if isinstance(df.index, pd.DatetimeIndex):
            df['hour'] = df.index.hour
            df['day_of_week'] = df.index.dayofweek
            # Session indicators
            df['asian_session'] = ((df['hour'] >= 0) & (df['hour'] < 8)).astype(int)
            df['london_session'] = ((df['hour'] >= 8) & (df['hour'] < 16)).astype(int)
            df['ny_session'] = ((df['hour'] >= 13) & (df['hour'] < 22)).astype(int)

            # Daily candle features (current day and previous day green/red)
            # Get daily open for each day
            daily_open = df.resample('D')['open'].first().dropna()

            # Previous day: use completed daily candles (no future leak)
            daily_close = df.resample('D')['close'].last().dropna()
            daily_green = (daily_close > daily_open).astype(int)
            prev_daily_green = daily_green.shift(1)

            # Map previous day green to intraday bars
            df['date'] = df.index.date
            prev_daily_map = prev_daily_green.to_dict()
            df['prev_day_green'] = df['date'].map(lambda d: prev_daily_map.get(pd.Timestamp(d), np.nan))

            # Current day green: compare daily open vs CURRENT close (no future leak)
            # This is "is the day green SO FAR at this point in time"
            daily_open_map = daily_open.to_dict()
            df['daily_open'] = df['date'].map(lambda d: daily_open_map.get(pd.Timestamp(d), np.nan))
            df['current_day_green'] = (df['close'] > df['daily_open']).astype(int)
            df.drop(columns=['date', 'daily_open'], inplace=True)
        else:
            df['hour'] = 12
            df['day_of_week'] = 2
            df['asian_session'] = 0
            df['london_session'] = 1
            df['ny_session'] = 0
            df['current_day_green'] = 0
            df['prev_day_green'] = 0

        # Candlestick patterns
        body_abs = df['body'].abs()
        avg_body = body_abs.rolling(20).mean()

        # Doji (small body)
        df['is_doji'] = (body_abs < avg_body * 0.1).astype(int)

        # Hammer/Shooting star (small body, long wick)
        df['is_hammer'] = ((df['lower_wick'] > body_abs * 2) & (df['upper_wick'] < body_abs)).astype(int)
        df['is_shooting_star'] = ((df['upper_wick'] > body_abs * 2) & (df['lower_wick'] < body_abs)).astype(int)

        # Engulfing patterns
        df['bullish_engulf'] = ((df['body'] > 0) & (df['body'].shift(1) < 0) &
                                (df['close'] > df['open'].shift(1)) &
                                (df['open'] < df['close'].shift(1))).astype(int)
        df['bearish_engulf'] = ((df['body'] < 0) & (df['body'].shift(1) > 0) &
                                (df['close'] < df['open'].shift(1)) &
                                (df['open'] > df['close'].shift(1))).astype(int)

        # Price distance from recent high/low
        df['dist_from_high'] = (df['high'].rolling(20).max() - df['close']) / df['close']
        df['dist_from_low'] = (df['close'] - df['low'].rolling(20).min()) / df['close']

        # Min/Max for 20 recent candles
        df['high_20'] = df['high'].rolling(20).max()
        df['low_20'] = df['low'].rolling(20).min()
        df['close_max_20'] = df['close'].rolling(20).max()
        df['close_min_20'] = df['close'].rolling(20).min()

        # Position within 20-candle range
        range_20 = df['high_20'] - df['low_20']
        df['position_in_range_20'] = (df['close'] - df['low_20']) / (range_20 + 1e-10)

        # Current candle structure as % of total range
        total_range = df['range'] + 1e-10
        df['body_pct_of_range'] = body_abs / total_range  # Body as % of range
        df['upper_wick_pct'] = df['upper_wick'] / total_range  # Upper wick as % of range
        df['lower_wick_pct'] = df['lower_wick'] / total_range  # Lower wick as % of range

        # Body vs average body
        df['body_vs_avg'] = body_abs / (avg_body + 1e-10)

        # Range vs average range
        avg_range = df['range'].rolling(20).mean()
        df['range_vs_avg'] = df['range'] / (avg_range + 1e-10)

        # Min/Max body in last 20 candles
        df['max_body_20'] = body_abs.rolling(20).max()
        df['min_body_20'] = body_abs.rolling(20).min()
        df['body_rank_20'] = body_abs / (df['max_body_20'] + 1e-10)  # How big is this candle vs recent

        # Target: next candle is green
        df['target'] = df['is_green'].shift(-1)

        # Next candle return (for analysis)
        df['next_return'] = df['close'].pct_change().shift(-1)

        # Extract tsfresh features
        df = self._extract_tsfresh_features(df)

        return df

    def _count_consecutive(self, series: pd.Series, value: int) -> pd.Series:
        """Count consecutive occurrences of a value."""
        result = pd.Series(0, index=series.index)
        count = 0
        for i in range(len(series)):
            if series.iloc[i] == value:
                count += 1
            else:
                count = 0
            result.iloc[i] = count
        return result

    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate ATR."""
        tr = pd.concat([
            df['high'] - df['low'],
            abs(df['high'] - df['close'].shift(1)),
            abs(df['low'] - df['close'].shift(1))
        ], axis=1).max(axis=1)
        return tr.rolling(period).mean()

    def _extract_tsfresh_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract tsfresh features from rolling windows."""
        if not self.config.use_tsfresh:
            return df

        print("      Extracting tsfresh features...")
        window = self.config.tsfresh_window

        # Columns to use for tsfresh
        ts_cols = ['close', 'body', 'range']
        available_cols = [c for c in ts_cols if c in df.columns]

        if not available_cols:
            return df

        # Reset index to integer for easier handling
        df_reset = df.reset_index(drop=True)

        # Build long-format data for tsfresh
        records = []
        valid_indices = []

        for i in range(window, len(df_reset)):
            for col in available_cols:
                for t in range(window):
                    val = df_reset[col].iloc[i - window + t]
                    if pd.notna(val):
                        records.append({
                            'id': i,
                            'time': t,
                            'kind': col,
                            'value': float(val)
                        })
            valid_indices.append(i)

        if not records:
            return df

        long_df = pd.DataFrame(records)

        try:
            # Extract minimal features
            extracted = extract_features(
                long_df,
                column_id='id',
                column_sort='time',
                column_kind='kind',
                column_value='value',
                default_fc_parameters=MinimalFCParameters(),
                n_jobs=1,
                disable_progressbar=True
            )
            impute(extracted)

            # Prefix columns
            extracted.columns = [f'ts_{c}' for c in extracted.columns]
            print(f"      Extracted {len(extracted.columns)} tsfresh features")

            # Store tsfresh feature names
            self._tsfresh_features = list(extracted.columns)

            # Add tsfresh features to df using iloc (integer position)
            for col in extracted.columns:
                df_reset[col] = np.nan
                for idx, val in zip(extracted.index, extracted[col].values):
                    if idx < len(df_reset):
                        df_reset.loc[idx, col] = val

            # Restore original index
            df_reset.index = df.index
            return df_reset

        except Exception as e:
            print(f"      tsfresh extraction failed: {e}")
            self._tsfresh_features = []

        return df

    def get_feature_columns(self, include_tsfresh: bool = True) -> List[str]:
        """Get list of feature columns for model."""
        base_features = [
            'body_pct', 'body_to_range', 'upper_wick', 'lower_wick',
            'consec_green', 'consec_red',
            'return_3', 'return_5', 'return_10', 'return_20',
            'price_to_ema5', 'price_to_ema10', 'price_to_ema20', 'price_to_ema50',
            'ema5_slope', 'ema10_slope', 'ema20_slope', 'ema50_slope',
            'ema50_up',  # EMA50 direction: 1 = up, 0 = down
            'rsi', 'rsi_slope',
            'macd', 'macd_signal', 'macd_hist',
            'bb_position',
            'volatility', 'atr_pct',
            'volume_ratio', 'volume_trend',
            'hour', 'day_of_week',
            'asian_session', 'london_session', 'ny_session',
            # Daily candle features
            'current_day_green', 'prev_day_green',
            # Candlestick patterns
            'is_doji', 'is_hammer', 'is_shooting_star',
            'bullish_engulf', 'bearish_engulf',
            'dist_from_high', 'dist_from_low',
            # 20-candle min/max features
            'position_in_range_20',
            'body_pct_of_range', 'upper_wick_pct', 'lower_wick_pct',
            'body_vs_avg', 'range_vs_avg', 'body_rank_20'
        ]

        # Add lookback features
        for i in range(1, self.config.lookback_candles + 1):
            base_features.extend([f'green_{i}', f'body_pct_{i}'])

        # Add tsfresh features if available
        if include_tsfresh and self._tsfresh_features:
            base_features.extend(self._tsfresh_features)

        return base_features

    def train(self, data: pd.DataFrame) -> Dict:
        """
        Train the next candle prediction model.

        Args:
            data: Training data (preprocessed or raw)

        Returns:
            Dictionary with training metrics
        """
        if 'target' not in data.columns:
            data = self.preprocess_data(data)

        feature_cols = self.get_feature_columns()
        self.feature_names = feature_cols

        # Prepare training data
        df = data.dropna(subset=feature_cols + ['target'])

        X = df[feature_cols].values
        y = df['target'].values

        # Remove any remaining NaN/inf
        valid_mask = np.isfinite(X).all(axis=1)
        X = X[valid_mask]
        y = y[valid_mask]

        if len(X) < 500:
            raise ValueError(f"Insufficient training data: {len(X)} samples")

        # Train model
        self.model = CatBoostClassifier(
            iterations=self.config.iterations,
            depth=self.config.depth,
            learning_rate=self.config.learning_rate,
            l2_leaf_reg=self.config.l2_leaf_reg,
            random_state=42,
            verbose=False
        )

        self.model.fit(X, y)
        self._is_trained = True

        # Calculate training metrics
        train_pred = self.model.predict(X)
        train_proba = self.model.predict_proba(X)[:, 1]

        accuracy = (train_pred == y).mean()

        # High confidence accuracy
        high_conf_mask = (train_proba >= self.config.min_probability) | (train_proba <= 1 - self.config.min_probability)
        high_conf_acc = (train_pred[high_conf_mask] == y[high_conf_mask]).mean() if high_conf_mask.sum() > 0 else 0

        # Feature importance
        importance = dict(zip(feature_cols, self.model.feature_importances_))

        return {
            'samples': len(X),
            'accuracy': accuracy,
            'high_conf_accuracy': high_conf_acc,
            'high_conf_samples': int(high_conf_mask.sum()),
            'high_conf_pct': high_conf_mask.mean(),
            'class_balance': y.mean(),
            'top_features': sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]
        }

    def predict(self, data: pd.DataFrame) -> tuple:
        """
        Predict next candle color.

        Returns:
            Tuple of (predictions, probabilities)
        """
        if not self._is_trained:
            raise ValueError("Model not trained. Call train() first.")

        if 'target' not in data.columns:
            data = self.preprocess_data(data)

        feature_cols = self.feature_names
        X = data[feature_cols].values

        # Handle NaN/inf
        X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)

        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)[:, 1]

        return predictions, probabilities

    def get_position_size(self, signal: int, portfolio_value: float, current_price: float) -> float:
        """Calculate position size - fixed fraction for 1-bar trades."""
        if signal == 0:
            return 0.0
        # Use very small position for frequent short-term trades
        risk_pct = 0.002  # 0.2% risk per trade (very conservative)
        risk_amount = portfolio_value * risk_pct
        # Assume ~5 pip stop for 5-min candle
        stop_distance = 0.0005
        return risk_amount / stop_distance

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate trading signals based on next candle prediction.

        Signal = 1: Predict green, buy now
        Signal = -1: Predict red, sell now
        """
        signals = pd.Series(0, index=data.index)

        # Preprocess
        if 'target' not in data.columns:
            data = self.preprocess_data(data)

        n = len(data)

        # Train if needed (use first 70%)
        if not self._is_trained:
            train_end = int(n * 0.7)
            train_data = data.iloc[:train_end]
            self.train(train_data)
            start_idx = train_end
        else:
            start_idx = self.required_history

        # Generate predictions
        try:
            predictions, probabilities = self.predict(data)
        except Exception as e:
            print(f"Prediction error: {e}")
            return signals

        min_prob = self.config.min_probability
        last_signal_bar = -100
        min_bars = self.config.min_bars_between_trades

        for i in range(start_idx, n - 1):  # -1 because we hold for 1 bar
            # Cooldown check
            if i - last_signal_bar < min_bars:
                continue

            prob = probabilities[i]

            # High confidence GREEN prediction - BUY
            if prob >= min_prob:
                signals.iloc[i] = 1
                last_signal_bar = i
            # High confidence RED prediction - SELL
            elif prob <= (1 - min_prob):
                signals.iloc[i] = -1
                last_signal_bar = i

        return signals

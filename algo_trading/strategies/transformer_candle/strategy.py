"""
Transformer Candle Prediction Strategy

Combines:
1. tsfresh automated time series features
2. Manual features from Next Candle strategy (EMAs, RSI, MACD, etc.)
3. Transformer encoder for sequence classification

Predicts if next candle will be green, trades accordingly.
"""
import numpy as np
import pandas as pd
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tsfresh import extract_features
from tsfresh.feature_extraction import MinimalFCParameters, EfficientFCParameters
from tsfresh.utilities.dataframe_functions import impute
import warnings
warnings.filterwarnings('ignore')


@dataclass
class TransformerCandleConfig:
    """Configuration for Transformer Candle strategy."""
    # Feature extraction
    lookback_window: int = 20  # Window for tsfresh features
    use_tsfresh: bool = True
    tsfresh_minimal: bool = True  # Use minimal features for speed

    # Transformer parameters
    d_model: int = 64
    nhead: int = 4
    num_layers: int = 2
    dim_feedforward: int = 128
    dropout: float = 0.1

    # Training parameters
    epochs: int = 50
    batch_size: int = 64
    learning_rate: float = 0.001
    train_split: float = 0.8

    # Trading parameters
    min_probability: float = 0.58
    min_bars_between_trades: int = 6


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer."""

    def __init__(self, d_model: int, max_len: int = 100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class TransformerClassifier(nn.Module):
    """Transformer encoder for binary classification."""

    def __init__(self, input_dim: int, config: TransformerCandleConfig):
        super().__init__()
        self.config = config

        # Input projection
        self.input_proj = nn.Linear(input_dim, config.d_model)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(config.d_model)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(config.d_model, config.d_model // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x shape: (batch, seq_len, features)
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        # Use last token for classification
        x = x[:, -1, :]
        return self.classifier(x)


class TransformerCandleStrategy:
    """
    Transformer-based next candle prediction.

    Uses tsfresh for automated feature extraction and
    Transformer encoder for sequence classification.
    """

    def __init__(self, config: Optional[TransformerCandleConfig] = None):
        self.config = config or TransformerCandleConfig()
        self.name = "Transformer Candle"
        self.params = {}

        self.model: Optional[TransformerClassifier] = None
        self.feature_names: List[str] = []
        self.feature_mean: Optional[np.ndarray] = None
        self.feature_std: Optional[np.ndarray] = None
        self._is_trained = False

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    @property
    def required_history(self) -> int:
        return self.config.lookback_window + 50

    def validate_data(self, data: pd.DataFrame) -> bool:
        required = ['open', 'high', 'low', 'close']
        missing = [c for c in required if c not in data.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")
        return True

    def _extract_manual_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract manual features similar to Next Candle strategy."""
        # Basic candle features
        df['body'] = df['close'] - df['open']
        df['body_pct'] = df['body'] / df['open']
        df['upper_wick'] = df['high'] - df[['open', 'close']].max(axis=1)
        df['lower_wick'] = df[['open', 'close']].min(axis=1) - df['low']
        df['range'] = df['high'] - df['low']
        df['body_to_range'] = df['body'].abs() / (df['range'] + 1e-10)

        # Is green
        df['is_green'] = (df['close'] > df['open']).astype(int)

        # Price momentum
        for period in [3, 5, 10, 20]:
            df[f'return_{period}'] = df['close'].pct_change(period)

        # EMAs
        df['ema5'] = df['close'].ewm(span=5, adjust=False).mean()
        df['ema10'] = df['close'].ewm(span=10, adjust=False).mean()
        df['ema20'] = df['close'].ewm(span=20, adjust=False).mean()

        df['price_to_ema5'] = df['close'] / df['ema5'] - 1
        df['price_to_ema10'] = df['close'] / df['ema10'] - 1
        df['price_to_ema20'] = df['close'] / df['ema20'] - 1

        df['ema5_slope'] = df['ema5'].pct_change(3)
        df['ema10_slope'] = df['ema10'].pct_change(3)
        df['ema20_slope'] = df['ema20'].pct_change(3)

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

        # ATR
        tr = pd.concat([
            df['high'] - df['low'],
            abs(df['high'] - df['close'].shift(1)),
            abs(df['low'] - df['close'].shift(1))
        ], axis=1).max(axis=1)
        df['atr'] = tr.rolling(14).mean()
        df['atr_pct'] = df['atr'] / df['close']

        # Volume features (if available)
        if 'volume' in df.columns and df['volume'].sum() > 0:
            df['volume_ma'] = df['volume'].rolling(20).mean()
            df['volume_ratio'] = df['volume'] / (df['volume_ma'] + 1)
        else:
            df['volume_ratio'] = 1.0

        # Time features
        if isinstance(df.index, pd.DatetimeIndex):
            df['hour'] = df.index.hour
            df['day_of_week'] = df.index.dayofweek
        else:
            df['hour'] = 12
            df['day_of_week'] = 2

        # Target
        df['target'] = df['is_green'].shift(-1)

        return df

    def _extract_tsfresh_features(self, df: pd.DataFrame, window: int) -> pd.DataFrame:
        """Extract tsfresh features from rolling windows."""
        print("    Extracting tsfresh features...")

        # Prepare data for tsfresh (long format)
        features_list = []

        # Use subset of columns for tsfresh
        cols_for_tsfresh = ['close', 'high', 'low', 'body', 'range']
        available_cols = [c for c in cols_for_tsfresh if c in df.columns]

        for i in range(window, len(df)):
            window_data = df.iloc[i-window:i][available_cols].copy()
            window_data['id'] = i
            window_data['time'] = range(window)
            features_list.append(window_data)

        if not features_list:
            return pd.DataFrame()

        long_df = pd.concat(features_list, ignore_index=True)

        # Melt to long format
        melted = long_df.melt(id_vars=['id', 'time'], var_name='kind', value_name='value')

        # Extract features
        if self.config.tsfresh_minimal:
            fc_params = MinimalFCParameters()
        else:
            fc_params = EfficientFCParameters()

        try:
            extracted = extract_features(
                melted,
                column_id='id',
                column_sort='time',
                column_kind='kind',
                column_value='value',
                default_fc_parameters=fc_params,
                n_jobs=1,
                disable_progressbar=True
            )
            impute(extracted)
        except Exception as e:
            print(f"    tsfresh extraction failed: {e}")
            return pd.DataFrame()

        print(f"    Extracted {extracted.shape[1]} tsfresh features")
        return extracted

    def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate all features."""
        df = data.copy()
        df = self._extract_manual_features(df)
        return df

    def _get_manual_feature_cols(self) -> List[str]:
        """Get manual feature column names."""
        return [
            'body_pct', 'body_to_range', 'upper_wick', 'lower_wick',
            'return_3', 'return_5', 'return_10', 'return_20',
            'price_to_ema5', 'price_to_ema10', 'price_to_ema20',
            'ema5_slope', 'ema10_slope', 'ema20_slope',
            'rsi', 'rsi_slope',
            'macd', 'macd_signal', 'macd_hist',
            'bb_position', 'volatility', 'atr_pct',
            'volume_ratio', 'hour', 'day_of_week'
        ]

    def _prepare_sequences(self, df: pd.DataFrame, seq_len: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare sequences for transformer input."""
        feature_cols = self._get_manual_feature_cols()
        available_cols = [c for c in feature_cols if c in df.columns]

        X_list = []
        y_list = []

        for i in range(seq_len, len(df) - 1):
            # Get sequence of features
            seq = df.iloc[i-seq_len:i][available_cols].values
            target = df['target'].iloc[i]

            if not np.isnan(target) and np.isfinite(seq).all():
                X_list.append(seq)
                y_list.append(target)

        if not X_list:
            return np.array([]), np.array([])

        X = np.array(X_list, dtype=np.float32)
        y = np.array(y_list, dtype=np.float32)

        return X, y

    def train(self, data: pd.DataFrame) -> Dict:
        """Train the Transformer model."""
        print("    Preprocessing data...")
        if 'target' not in data.columns:
            data = self.preprocess_data(data)

        print("    Preparing sequences...")
        seq_len = 10  # Use 10 bars as input sequence
        X, y = self._prepare_sequences(data, seq_len)

        if len(X) < 100:
            raise ValueError(f"Insufficient data: {len(X)} samples")

        print(f"    Dataset: {len(X)} samples, {X.shape[2]} features per timestep")

        # Normalize features
        self.feature_mean = X.mean(axis=(0, 1))
        self.feature_std = X.std(axis=(0, 1)) + 1e-8
        X = (X - self.feature_mean) / self.feature_std

        # Train/val split
        split_idx = int(len(X) * self.config.train_split)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        # Create dataloaders
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train),
            torch.FloatTensor(y_train).unsqueeze(1)
        )
        val_dataset = TensorDataset(
            torch.FloatTensor(X_val),
            torch.FloatTensor(y_val).unsqueeze(1)
        )

        train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.config.batch_size)

        # Initialize model
        input_dim = X.shape[2]
        self.model = TransformerClassifier(input_dim, self.config).to(self.device)

        # Training setup
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

        best_val_acc = 0
        best_state = None

        print(f"    Training for {self.config.epochs} epochs...")
        for epoch in range(self.config.epochs):
            # Training
            self.model.train()
            train_loss = 0
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            # Validation
            self.model.eval()
            val_preds = []
            val_targets = []
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X = batch_X.to(self.device)
                    outputs = self.model(batch_X)
                    val_preds.extend(outputs.cpu().numpy())
                    val_targets.extend(batch_y.numpy())

            val_preds = np.array(val_preds).flatten()
            val_targets = np.array(val_targets).flatten()
            val_acc = ((val_preds > 0.5) == val_targets).mean()

            scheduler.step(1 - val_acc)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = self.model.state_dict().copy()

            if (epoch + 1) % 10 == 0:
                print(f"      Epoch {epoch+1}: train_loss={train_loss/len(train_loader):.4f}, val_acc={val_acc:.4f}")

        # Load best model
        if best_state:
            self.model.load_state_dict(best_state)

        self._is_trained = True
        self.feature_names = self._get_manual_feature_cols()

        # Calculate metrics
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            all_preds = self.model(X_tensor).cpu().numpy().flatten()

        accuracy = ((all_preds > 0.5) == y).mean()
        high_conf_mask = (all_preds >= self.config.min_probability) | (all_preds <= 1 - self.config.min_probability)
        high_conf_acc = ((all_preds[high_conf_mask] > 0.5) == y[high_conf_mask]).mean() if high_conf_mask.sum() > 0 else 0

        return {
            'samples': len(X),
            'accuracy': float(accuracy),
            'high_conf_accuracy': float(high_conf_acc),
            'high_conf_pct': float(high_conf_mask.mean()),
            'best_val_accuracy': float(best_val_acc),
            'class_balance': float(y.mean())
        }

    def predict(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Predict next candle direction."""
        if not self._is_trained:
            raise ValueError("Model not trained")

        if 'target' not in data.columns:
            data = self.preprocess_data(data)

        seq_len = 10
        feature_cols = [c for c in self._get_manual_feature_cols() if c in data.columns]

        predictions = np.zeros(len(data))
        probabilities = np.full(len(data), 0.5)

        self.model.eval()
        with torch.no_grad():
            for i in range(seq_len, len(data)):
                seq = data.iloc[i-seq_len:i][feature_cols].values.astype(np.float32)

                if not np.isfinite(seq).all():
                    continue

                # Normalize
                seq = (seq - self.feature_mean) / self.feature_std

                # Predict
                X_tensor = torch.FloatTensor(seq).unsqueeze(0).to(self.device)
                prob = self.model(X_tensor).cpu().numpy()[0, 0]

                probabilities[i] = prob
                predictions[i] = 1 if prob > 0.5 else 0

        return predictions, probabilities

    def get_position_size(self, signal: int, portfolio_value: float, current_price: float) -> float:
        """Calculate position size."""
        if signal == 0:
            return 0.0
        risk_pct = 0.002
        risk_amount = portfolio_value * risk_pct
        stop_distance = 0.0005
        return risk_amount / stop_distance

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate trading signals."""
        signals = pd.Series(0, index=data.index)

        if 'target' not in data.columns:
            data = self.preprocess_data(data)

        n = len(data)

        # Train if needed
        if not self._is_trained:
            train_end = int(n * 0.7)
            train_data = data.iloc[:train_end]
            self.train(train_data)
            start_idx = train_end
        else:
            start_idx = self.required_history

        # Predict
        try:
            predictions, probabilities = self.predict(data)
        except Exception as e:
            print(f"Prediction error: {e}")
            return signals

        min_prob = self.config.min_probability
        last_signal_bar = -100
        min_bars = self.config.min_bars_between_trades

        for i in range(start_idx, n - 1):
            if i - last_signal_bar < min_bars:
                continue

            prob = probabilities[i]

            if prob >= min_prob:
                signals.iloc[i] = 1
                last_signal_bar = i
            elif prob <= (1 - min_prob):
                signals.iloc[i] = -1
                last_signal_bar = i

        return signals

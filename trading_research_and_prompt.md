# Анализ научных статей по алгоритмическому трейдингу и промпт для создания проекта

## 1. Обзор научных исследований

### 1.1 Ключевые статьи и источники

| Статья | Год | Основные темы |
|--------|-----|---------------|
| Deep learning for algorithmic trading: A systematic review (ScienceDirect) | 2025 | DL архитектуры (RNN, LSTM, CNN), гибридные модели, прогнозирование волатильности |
| Algorithmic Trading and AI: A Review of Strategies and Market Impact | 2024 | ML/DL стратегии, HFT, supervised/unsupervised/reinforcement learning |
| Reinforcement Learning Framework for Quantitative Trading (arXiv) | 2024 | RL с индикаторами, DQN, PPO, Backtesting.py |
| FinRL: A Deep Reinforcement Learning Library (NeurIPS 2020) | 2020 | DRL для автоматизированной торговли, Stable Baselines 3 |
| Reinforcement Learning for Quantitative Trading (ACM) | 2023 | Value-based, Policy-based, Actor-Critic методы |
| Technology-driven advancements in algorithmic trading | 2024 | Portfolio optimization, HFT, ML applications |

### 1.2 Основные выводы из исследований

**Архитектуры Deep Learning:**
- LSTM/GRU - наиболее эффективны для временных рядов
- CNN - извлечение паттернов из ценовых данных
- Transformer-based модели - захват долгосрочных зависимостей
- Hybrid models (CNN-LSTM) - комбинация для лучших результатов

**Reinforcement Learning подходы:**
- Value-based: DQN, Double DQN, Dueling DQN
- Policy-based: PPO (наиболее стабильный), A2C, TRPO
- Actor-Critic: DDPG, TD3, SAC

**Ключевые проблемы:**
- Overfitting на исторических данных
- Low signal-to-noise ratio в финансовых данных
- Distribution shift между периодами
- Чувствительность к гиперпараметрам

---

## 2. Фреймворки и библиотеки

### 2.1 Backtesting фреймворки

| Фреймворк | Тип | Скорость | Live Trading | Особенности |
|-----------|-----|----------|--------------|-------------|
| **VectorBT** | Vectorized | ⭐⭐⭐⭐⭐ | Через StrateQueue | Numba-оптимизация, параллельные бэктесты |
| **Backtrader** | Event-driven | ⭐⭐⭐ | Да (IB, Oanda) | Простота, хорошая документация |
| **Backtesting.py** | Vectorized | ⭐⭐⭐⭐ | Нет | Интуитивный API, визуализация |
| **Zipline** | Event-driven | ⭐⭐⭐ | Ограниченно | Quantopian legacy |
| **QuantConnect/Lean** | Hybrid | ⭐⭐⭐⭐ | Да | Cloud-based, C#/Python |

### 2.2 Reinforcement Learning

| Библиотека | Алгоритмы | Интеграция |
|------------|-----------|------------|
| **Stable Baselines 3** | PPO, A2C, DQN, SAC, TD3, DDPG | Gymnasium/Gym |
| **FinRL** | Все SB3 + ensemble | Yahoo Finance, Alpaca |
| **RLlib** | Distributed RL | Ray ecosystem |
| **ElegantRL** | Parallel training | GPU-optimized |

### 2.3 Trading Environments

| Environment | Тип рынка | Особенности |
|-------------|-----------|-------------|
| **gym-anytrading** | Forex, Stocks | Простой API, EURUSD включён |
| **FinRL Environments** | Multi-asset | Technical indicators |
| **Custom Gymnasium** | Any | Полная кастомизация |

---

## 3. Метрики оценки стратегий

### 3.1 Risk-Adjusted Returns

| Метрика | Формула | Интерпретация |
|---------|---------|---------------|
| **Sharpe Ratio** | (R - Rf) / σ | > 1.0 хорошо, > 2.0 отлично |
| **Sortino Ratio** | (R - Rf) / σ_downside | Фокус на downside risk |
| **Calmar Ratio** | Annual Return / Max DD | > 1.0 хорошо, > 2.0 отлично |
| **Information Ratio** | (R - Rb) / Tracking Error | Относительно бенчмарка |
| **Omega Ratio** | ∫(1-F(r))dr / ∫F(r)dr | > 1.0 прибыльно |

### 3.2 Drawdown метрики

| Метрика | Описание | Целевое значение |
|---------|----------|------------------|
| **Maximum Drawdown** | Максимальное падение от пика | < 15-20% консервативно |
| **Average Drawdown** | Среднее падение | Чем меньше, тем лучше |
| **Recovery Time** | Время восстановления | Чем короче, тем лучше |
| **Ulcer Index** | Глубина и длительность DD | Чем меньше, тем лучше |

### 3.3 Trade метрики

| Метрика | Описание |
|---------|----------|
| **Win Rate** | % прибыльных сделок |
| **Profit Factor** | Gross Profit / Gross Loss (> 1.5 хорошо) |
| **Average Win/Loss** | Средний размер выигрыша/проигрыша |
| **Expectancy** | (Win% × Avg Win) - (Loss% × Avg Loss) |
| **Number of Trades** | Статистическая значимость |

### 3.4 Robustness Tests

- **Walk-Forward Analysis** - rolling window validation
- **Monte Carlo Simulation** - random resampling
- **Parameter Sensitivity** - stability across parameter changes
- **Out-of-Sample Testing** - unseen data performance

---

## 4. Временные окна для EURUSD

### 4.1 Торговые сессии (GMT)

| Сессия | Время | Волатильность | Характеристика |
|--------|-------|---------------|----------------|
| Sydney | 22:00-07:00 | Низкая | Спокойный старт |
| Tokyo | 00:00-09:00 | Низкая-средняя | JPY пары активны |
| London | 07:00-16:00 | Высокая | 35-40% объёма |
| New York | 13:00-22:00 | Высокая | US data releases |

### 4.2 Оптимальные периоды для EURUSD

| Период (GMT) | Характеристика | Рекомендация |
|--------------|----------------|--------------|
| **07:00-09:00** | London open, высокая волатильность | Momentum стратегии |
| **12:00-16:00** | London-NY overlap, максимум | Breakout, trend-following |
| **01:00-04:00** | Asian session, низкая волатильность | Mean-reversion, range |

### 4.3 Средняя волатильность EURUSD

- **Daily range**: ~50-70 pips (normal), ~80+ pips (high)
- **5-minute candle**: ~3-8 pips (normal session)
- **ATR(14)**: используется для динамической адаптации

---

## 5. Промпт для LLM

```
Создай Python проект для алгоритмического трейдинга с модульной архитектурой.

## СТРУКТУРА ПРОЕКТА

```
algo_trading/
├── README.md
├── requirements.txt
├── setup.py
├── config/
│   ├── __init__.py
│   └── settings.py              # Глобальные настройки, API ключи
├── data/
│   ├── __init__.py
│   ├── downloaders/
│   │   ├── __init__.py
│   │   ├── base.py              # Абстрактный класс DataDownloader
│   │   └── forex_downloader.py  # OANDA/MetaTrader/yfinance
│   └── storage/
│       └── historical/          # Кэш исторических данных
├── strategies/
│   ├── __init__.py
│   ├── base_strategy.py         # Абстрактный класс Strategy
│   ├── mean_reversion/
│   │   ├── __init__.py
│   │   ├── strategy.py          # Логика стратегии
│   │   ├── backtest.py          # Бэктест runner
│   │   └── live.py              # Live trading runner
│   ├── momentum/
│   │   ├── __init__.py
│   │   ├── strategy.py
│   │   ├── backtest.py
│   │   └── live.py
│   └── rl_ppo/                  # RL стратегия
│       ├── __init__.py
│       ├── environment.py       # Gymnasium environment
│       ├── training.py          # Обучение модели
│       ├── strategy.py          # Обёртка для использования
│       ├── backtest.py
│       └── live.py
├── core/
│   ├── __init__.py
│   ├── backtester.py            # Универсальный backtester
│   ├── metrics.py               # Все метрики
│   ├── portfolio.py             # Управление портфелем
│   └── risk_manager.py          # Risk management
├── brokers/
│   ├── __init__.py
│   ├── base_broker.py           # Абстрактный класс Broker
│   ├── paper_broker.py          # Paper trading
│   └── oanda_broker.py          # OANDA API integration
├── benchmarks/
│   ├── __init__.py
│   ├── runner.py                # Запуск бенчмарков
│   └── report_generator.py      # Генерация отчётов
├── ui/
│   ├── __init__.py
│   ├── dashboard.py             # Streamlit/Dash UI
│   └── templates/
└── tests/
    ├── __init__.py
    ├── test_strategies.py
    └── test_metrics.py
```

## ТРЕБОВАНИЯ К РЕАЛИЗАЦИИ

### 1. base_strategy.py (strategies/base_strategy.py)
```python
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import pandas as pd

class Strategy(ABC):
    """Базовый класс для всех стратегий"""
    
    def __init__(self, params: Dict[str, Any]):
        self.params = params
        self.name = self.__class__.__name__
    
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Генерирует торговые сигналы.
        Returns: Series с значениями 1 (buy), -1 (sell), 0 (hold)
        """
        pass
    
    @abstractmethod
    def get_position_size(self, signal: int, portfolio_value: float, 
                          current_price: float) -> float:
        """Определяет размер позиции"""
        pass
    
    def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Предобработка данных (индикаторы и т.д.)"""
        return data
    
    @property
    @abstractmethod
    def required_history(self) -> int:
        """Минимальное количество баров для генерации сигнала"""
        pass
```

### 2. metrics.py (core/metrics.py)
```python
import numpy as np
import pandas as pd
from typing import Dict, Any
from dataclasses import dataclass

@dataclass
class BacktestResults:
    equity_curve: pd.Series
    returns: pd.Series
    trades: pd.DataFrame
    metrics: Dict[str, float]

class MetricsCalculator:
    """Калькулятор всех метрик"""
    
    RISK_FREE_RATE = 0.02  # 2% годовых
    TRADING_DAYS = 252
    
    @staticmethod
    def calculate_all(equity_curve: pd.Series, 
                      returns: pd.Series,
                      trades: pd.DataFrame,
                      commission_rate: float = 0.0001) -> Dict[str, float]:
        """Вычисляет все метрики"""
        
        metrics = {}
        
        # Returns
        metrics['total_return'] = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1
        metrics['annualized_return'] = MetricsCalculator._annualize_return(
            metrics['total_return'], len(returns))
        
        # Volatility
        metrics['volatility'] = returns.std() * np.sqrt(MetricsCalculator.TRADING_DAYS)
        metrics['downside_volatility'] = returns[returns < 0].std() * np.sqrt(252)
        
        # Risk-adjusted
        metrics['sharpe_ratio'] = MetricsCalculator._sharpe_ratio(returns)
        metrics['sortino_ratio'] = MetricsCalculator._sortino_ratio(returns)
        metrics['calmar_ratio'] = MetricsCalculator._calmar_ratio(
            metrics['annualized_return'], equity_curve)
        
        # Drawdown
        dd = MetricsCalculator._calculate_drawdowns(equity_curve)
        metrics['max_drawdown'] = dd['max_drawdown']
        metrics['avg_drawdown'] = dd['avg_drawdown']
        metrics['max_drawdown_duration'] = dd['max_duration_days']
        
        # Trade metrics
        if len(trades) > 0:
            metrics['win_rate'] = (trades['pnl'] > 0).mean()
            metrics['profit_factor'] = (
                trades[trades['pnl'] > 0]['pnl'].sum() / 
                abs(trades[trades['pnl'] < 0]['pnl'].sum())
                if (trades['pnl'] < 0).any() else float('inf')
            )
            metrics['avg_win'] = trades[trades['pnl'] > 0]['pnl'].mean()
            metrics['avg_loss'] = trades[trades['pnl'] < 0]['pnl'].mean()
            metrics['expectancy'] = trades['pnl'].mean()
            metrics['num_trades'] = len(trades)
        
        return metrics
    
    @staticmethod
    def _sharpe_ratio(returns: pd.Series) -> float:
        excess_returns = returns.mean() - MetricsCalculator.RISK_FREE_RATE / 252
        if returns.std() == 0:
            return 0
        return (excess_returns / returns.std()) * np.sqrt(252)
    
    @staticmethod
    def _sortino_ratio(returns: pd.Series) -> float:
        excess_returns = returns.mean() - MetricsCalculator.RISK_FREE_RATE / 252
        downside_std = returns[returns < 0].std()
        if downside_std == 0:
            return 0
        return (excess_returns / downside_std) * np.sqrt(252)
    
    @staticmethod
    def _calmar_ratio(annualized_return: float, equity_curve: pd.Series) -> float:
        max_dd = MetricsCalculator._calculate_drawdowns(equity_curve)['max_drawdown']
        if max_dd == 0:
            return 0
        return annualized_return / abs(max_dd)
    
    @staticmethod
    def _calculate_drawdowns(equity_curve: pd.Series) -> Dict[str, float]:
        peak = equity_curve.expanding().max()
        drawdown = (equity_curve - peak) / peak
        
        return {
            'max_drawdown': drawdown.min(),
            'avg_drawdown': drawdown[drawdown < 0].mean() if (drawdown < 0).any() else 0,
            'max_duration_days': MetricsCalculator._max_drawdown_duration(drawdown)
        }
    
    @staticmethod
    def _max_drawdown_duration(drawdown: pd.Series) -> int:
        is_in_dd = drawdown < 0
        groups = (~is_in_dd).cumsum()
        durations = is_in_dd.groupby(groups).cumsum()
        return int(durations.max()) if len(durations) > 0 else 0
    
    @staticmethod
    def _annualize_return(total_return: float, periods: int) -> float:
        if periods == 0:
            return 0
        years = periods / MetricsCalculator.TRADING_DAYS
        if years <= 0:
            return 0
        return (1 + total_return) ** (1 / years) - 1
```

### 3. backtester.py (core/backtester.py)
```python
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from strategies.base_strategy import Strategy
from core.metrics import MetricsCalculator, BacktestResults

class Backtester:
    """Универсальный бэктестер для всех стратегий"""
    
    def __init__(self, 
                 initial_capital: float = 10000,
                 commission: float = 0.0001,  # 1 pip spread
                 slippage: float = 0.0001):
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
    
    def run(self, 
            strategy: Strategy, 
            data: pd.DataFrame,
            start_date: Optional[str] = None,
            end_date: Optional[str] = None) -> BacktestResults:
        """
        Запускает бэктест стратегии
        
        Args:
            strategy: Экземпляр стратегии
            data: OHLCV данные (columns: open, high, low, close, volume)
            start_date: Начало периода
            end_date: Конец периода
        """
        # Фильтрация по датам
        if start_date:
            data = data[data.index >= start_date]
        if end_date:
            data = data[data.index <= end_date]
        
        # Предобработка
        data = strategy.preprocess_data(data)
        
        # Генерация сигналов
        signals = strategy.generate_signals(data)
        
        # Симуляция торговли
        equity_curve, trades = self._simulate_trading(
            data, signals, strategy
        )
        
        # Расчёт returns
        returns = equity_curve.pct_change().fillna(0)
        
        # Расчёт метрик
        metrics = MetricsCalculator.calculate_all(
            equity_curve, returns, trades, self.commission
        )
        
        return BacktestResults(
            equity_curve=equity_curve,
            returns=returns,
            trades=trades,
            metrics=metrics
        )
    
    def _simulate_trading(self, data: pd.DataFrame, signals: pd.Series,
                          strategy: Strategy) -> tuple:
        """Симуляция торговли"""
        equity = [self.initial_capital]
        position = 0
        entry_price = 0
        trades = []
        
        for i in range(1, len(data)):
            current_price = data['close'].iloc[i]
            signal = signals.iloc[i] if i < len(signals) else 0
            
            # Закрытие позиции
            if position != 0 and signal != position:
                pnl = (current_price - entry_price) * position
                pnl -= abs(position) * current_price * (self.commission + self.slippage)
                
                trades.append({
                    'entry_time': entry_time,
                    'exit_time': data.index[i],
                    'entry_price': entry_price,
                    'exit_price': current_price,
                    'position': position,
                    'pnl': pnl
                })
                
                equity.append(equity[-1] + pnl)
                position = 0
            
            # Открытие позиции
            if signal != 0 and position == 0:
                position = signal
                entry_price = current_price * (1 + self.slippage * signal)
                entry_time = data.index[i]
                equity.append(equity[-1])  # Нет изменения equity при входе
            elif position == 0:
                equity.append(equity[-1])
        
        equity_series = pd.Series(equity[:len(data)], index=data.index[:len(equity)])
        trades_df = pd.DataFrame(trades) if trades else pd.DataFrame(
            columns=['entry_time', 'exit_time', 'entry_price', 'exit_price', 'position', 'pnl']
        )
        
        return equity_series, trades_df
```

### 4. RL стратегия с Stable Baselines 3 (strategies/rl_ppo/)

#### environment.py
```python
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict, Any

class ForexTradingEnv(gym.Env):
    """
    Gymnasium environment для forex trading (EURUSD 5min)
    """
    
    metadata = {'render_modes': ['human']}
    
    def __init__(self, 
                 df: pd.DataFrame,
                 initial_balance: float = 10000,
                 commission: float = 0.0001,
                 window_size: int = 60,  # 5 hours of 5-min bars
                 max_position: float = 1.0):
        super().__init__()
        
        self.df = df
        self.initial_balance = initial_balance
        self.commission = commission
        self.window_size = window_size
        self.max_position = max_position
        
        # Нормализация данных
        self._preprocess_data()
        
        # Action space: -1 (sell), 0 (hold), 1 (buy)
        self.action_space = spaces.Discrete(3)
        
        # Observation space: OHLCV + indicators + position + balance
        n_features = self.df_normalized.shape[1]
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(window_size, n_features + 2),  # +2 for position and balance
            dtype=np.float32
        )
        
        self.reset()
    
    def _preprocess_data(self):
        """Добавление индикаторов и нормализация"""
        df = self.df.copy()
        
        # Technical indicators
        df['returns'] = df['close'].pct_change()
        df['ema_20'] = df['close'].ewm(span=20).mean()
        df['ema_50'] = df['close'].ewm(span=50).mean()
        df['rsi'] = self._calculate_rsi(df['close'], 14)
        df['atr'] = self._calculate_atr(df, 14)
        
        # Hour of day (для определения торговой сессии)
        df['hour'] = pd.to_datetime(df.index).hour
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        
        # Day of week
        df['dow'] = pd.to_datetime(df.index).dayofweek
        df['dow_sin'] = np.sin(2 * np.pi * df['dow'] / 5)
        df['dow_cos'] = np.cos(2 * np.pi * df['dow'] / 5)
        
        df.dropna(inplace=True)
        
        # Normalize
        self.df_normalized = (df - df.mean()) / df.std()
        self.df_prices = df['close'].values
    
    def _calculate_rsi(self, prices: pd.Series, period: int) -> pd.Series:
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _calculate_atr(self, df: pd.DataFrame, period: int) -> pd.Series:
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return tr.rolling(period).mean()
    
    def reset(self, seed: Optional[int] = None, 
              options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        
        self.current_step = self.window_size
        self.balance = self.initial_balance
        self.position = 0  # -1, 0, or 1
        self.entry_price = 0
        self.total_profit = 0
        self.trades = []
        
        return self._get_observation(), {}
    
    def _get_observation(self) -> np.ndarray:
        """Формирование наблюдения"""
        start = self.current_step - self.window_size
        end = self.current_step
        
        obs = self.df_normalized.iloc[start:end].values
        
        # Добавляем позицию и нормализованный баланс
        position_arr = np.full((self.window_size, 1), self.position)
        balance_arr = np.full((self.window_size, 1), 
                             self.balance / self.initial_balance)
        
        obs = np.concatenate([obs, position_arr, balance_arr], axis=1)
        
        return obs.astype(np.float32)
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Action: 0 = sell, 1 = hold, 2 = buy
        """
        action = action - 1  # Convert to -1, 0, 1
        
        current_price = self.df_prices[self.current_step]
        reward = 0
        
        # Закрытие позиции
        if self.position != 0 and action != self.position:
            pnl = (current_price - self.entry_price) * self.position
            pnl *= self.initial_balance / self.entry_price  # Leverage
            pnl -= abs(self.position) * self.initial_balance * self.commission
            
            self.balance += pnl
            self.total_profit += pnl
            
            self.trades.append({
                'step': self.current_step,
                'pnl': pnl,
                'position': self.position
            })
            
            reward = pnl / self.initial_balance  # Normalized reward
            self.position = 0
        
        # Открытие позиции
        if action != 0 and self.position == 0:
            self.position = action
            self.entry_price = current_price
            reward -= self.commission  # Commission penalty
        
        # Holding reward/penalty (small negative for holding without position)
        if self.position == 0 and action == 0:
            reward -= 0.0001
        
        self.current_step += 1
        
        # Check termination
        terminated = self.current_step >= len(self.df_normalized) - 1
        truncated = self.balance <= 0
        
        info = {
            'balance': self.balance,
            'position': self.position,
            'total_profit': self.total_profit,
            'num_trades': len(self.trades)
        }
        
        return self._get_observation(), reward, terminated, truncated, info
    
    def get_trading_hours_mask(self) -> np.ndarray:
        """
        Маска для стабильных торговых часов (London-NY overlap)
        Возвращает True для часов 12:00-16:00 GMT
        """
        hours = pd.to_datetime(self.df.index).hour
        # London-NY overlap: 12:00-16:00 GMT
        return (hours >= 12) & (hours <= 16)
```

#### training.py
```python
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement
from stable_baselines3.common.monitor import Monitor
import os
from typing import Optional, Dict, Any

from .environment import ForexTradingEnv

class RLTrainer:
    """Trainer для RL стратегий"""
    
    def __init__(self, 
                 model_dir: str = "models/rl",
                 log_dir: str = "logs/rl"):
        self.model_dir = model_dir
        self.log_dir = log_dir
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
    
    def train(self,
              train_df: pd.DataFrame,
              val_df: pd.DataFrame,
              total_timesteps: int = 100000,
              learning_rate: float = 3e-4,
              n_steps: int = 2048,
              batch_size: int = 64,
              n_epochs: int = 10,
              gamma: float = 0.99,
              clip_range: float = 0.2,
              filter_trading_hours: bool = True,
              **env_kwargs) -> PPO:
        """
        Обучение PPO модели
        
        Args:
            train_df: Training data
            val_df: Validation data
            filter_trading_hours: Фильтровать только стабильные часы
        """
        
        # Фильтрация по торговым часам если нужно
        if filter_trading_hours:
            train_df = self._filter_trading_hours(train_df)
            val_df = self._filter_trading_hours(val_df)
        
        # Создание environments
        train_env = DummyVecEnv([lambda: Monitor(
            ForexTradingEnv(train_df, **env_kwargs),
            self.log_dir
        )])
        train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True)
        
        eval_env = DummyVecEnv([lambda: ForexTradingEnv(val_df, **env_kwargs)])
        eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, training=False)
        
        # Callbacks
        stop_callback = StopTrainingOnNoModelImprovement(
            max_no_improvement_evals=10,
            min_evals=20,
            verbose=1
        )
        
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=self.model_dir,
            log_path=self.log_dir,
            eval_freq=5000,
            n_eval_episodes=5,
            callback_after_eval=stop_callback,
            deterministic=True,
            verbose=1
        )
        
        # Модель PPO
        model = PPO(
            "MlpPolicy",
            train_env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            clip_range=clip_range,
            verbose=1,
            tensorboard_log=self.log_dir
        )
        
        # Training
        model.learn(
            total_timesteps=total_timesteps,
            callback=eval_callback,
            progress_bar=True
        )
        
        # Save final model
        model.save(os.path.join(self.model_dir, "ppo_forex_final"))
        train_env.save(os.path.join(self.model_dir, "vec_normalize.pkl"))
        
        return model
    
    def _filter_trading_hours(self, df: pd.DataFrame) -> pd.DataFrame:
        """Фильтрация данных по стабильным торговым часам"""
        hours = pd.to_datetime(df.index).hour
        # London-NY overlap: 12:00-16:00 GMT (наиболее стабильный период)
        mask = (hours >= 12) & (hours <= 16)
        return df[mask]
    
    def load_model(self, model_path: str) -> PPO:
        """Загрузка обученной модели"""
        return PPO.load(model_path)
```

#### strategy.py
```python
import pandas as pd
import numpy as np
from typing import Dict, Any
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import os

from strategies.base_strategy import Strategy
from .environment import ForexTradingEnv

class RLPPOStrategy(Strategy):
    """RL стратегия на основе PPO"""
    
    def __init__(self, 
                 model_path: str,
                 vec_normalize_path: str,
                 params: Dict[str, Any] = None):
        params = params or {}
        super().__init__(params)
        
        self.model = PPO.load(model_path)
        
        # Загрузка нормализатора
        self.vec_normalize = VecNormalize.load(
            vec_normalize_path, 
            DummyVecEnv([lambda: ForexTradingEnv(pd.DataFrame())])
        )
    
    @property
    def required_history(self) -> int:
        return 60  # window_size
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Генерация сигналов с помощью обученной модели"""
        signals = pd.Series(index=data.index, data=0)
        
        env = ForexTradingEnv(data)
        obs, _ = env.reset()
        
        for i in range(len(data) - self.required_history):
            obs_normalized = self.vec_normalize.normalize_obs(obs.reshape(1, -1, obs.shape[-1]))
            action, _ = self.model.predict(obs_normalized, deterministic=True)
            
            # Convert action to signal
            signal = int(action[0]) - 1  # 0->-1, 1->0, 2->1
            signals.iloc[self.required_history + i] = signal
            
            obs, _, done, _, _ = env.step(action[0])
            if done:
                break
        
        return signals
    
    def get_position_size(self, signal: int, portfolio_value: float,
                          current_price: float) -> float:
        """Фиксированный размер позиции"""
        return 0.1 * portfolio_value  # 10% от портфеля
    
    def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Данные уже обрабатываются в environment"""
        return data
```

### 5. Dashboard UI (ui/dashboard.py)
```python
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import sys
from typing import List, Dict

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.backtester import Backtester
from core.metrics import BacktestResults
from benchmarks.runner import BenchmarkRunner

class TradingDashboard:
    """Streamlit dashboard для анализа стратегий"""
    
    def __init__(self):
        st.set_page_config(
            page_title="Algo Trading Dashboard",
            page_icon="📈",
            layout="wide"
        )
    
    def run(self):
        st.title("📈 Algorithmic Trading Dashboard")
        
        # Sidebar
        with st.sidebar:
            st.header("Configuration")
            
            # Strategy selection
            strategies = self._get_available_strategies()
            selected_strategies = st.multiselect(
                "Select Strategies",
                strategies,
                default=strategies[:3] if len(strategies) >= 3 else strategies
            )
            
            # Date range
            col1, col2 = st.columns(2)
            with col1:
                start_date = st.date_input("Start Date", value=pd.to_datetime("2023-01-01"))
            with col2:
                end_date = st.date_input("End Date", value=pd.to_datetime("2024-01-01"))
            
            # Parameters
            initial_capital = st.number_input("Initial Capital", value=10000, min_value=1000)
            commission = st.number_input("Commission (pips)", value=1.0, min_value=0.0) / 10000
            
            run_backtest = st.button("🚀 Run Backtest", type="primary")
        
        # Main content
        if run_backtest and selected_strategies:
            results = self._run_backtests(
                selected_strategies, 
                str(start_date), 
                str(end_date),
                initial_capital,
                commission
            )
            
            self._display_results(results)
    
    def _get_available_strategies(self) -> List[str]:
        """Получение списка доступных стратегий"""
        strategies_dir = "strategies"
        strategies = []
        
        for item in os.listdir(strategies_dir):
            item_path = os.path.join(strategies_dir, item)
            if os.path.isdir(item_path) and not item.startswith('__'):
                if os.path.exists(os.path.join(item_path, 'strategy.py')):
                    strategies.append(item)
        
        return strategies
    
    def _run_backtests(self, strategies: List[str], start_date: str, 
                       end_date: str, initial_capital: float,
                       commission: float) -> Dict[str, BacktestResults]:
        """Запуск бэктестов для выбранных стратегий"""
        runner = BenchmarkRunner(initial_capital=initial_capital, commission=commission)
        
        with st.spinner("Running backtests..."):
            results = runner.run_all(strategies, start_date, end_date)
        
        return results
    
    def _display_results(self, results: Dict[str, BacktestResults]):
        """Отображение результатов"""
        
        # Metrics comparison table
        st.header("📊 Strategy Comparison")
        
        metrics_df = self._create_metrics_dataframe(results)
        
        # Sortable table
        sort_by = st.selectbox(
            "Sort by",
            options=['sharpe_ratio', 'total_return', 'max_drawdown', 'win_rate', 'profit_factor'],
            index=0
        )
        sort_ascending = st.checkbox("Ascending", value=False)
        
        metrics_df_sorted = metrics_df.sort_values(sort_by, ascending=sort_ascending)
        
        # Color coding
        st.dataframe(
            metrics_df_sorted.style.background_gradient(
                subset=['sharpe_ratio', 'total_return', 'win_rate', 'profit_factor'],
                cmap='RdYlGn'
            ).background_gradient(
                subset=['max_drawdown'],
                cmap='RdYlGn_r'
            ).format({
                'total_return': '{:.2%}',
                'annualized_return': '{:.2%}',
                'max_drawdown': '{:.2%}',
                'win_rate': '{:.2%}',
                'sharpe_ratio': '{:.2f}',
                'sortino_ratio': '{:.2f}',
                'calmar_ratio': '{:.2f}',
                'profit_factor': '{:.2f}'
            }),
            use_container_width=True
        )
        
        # Equity curves
        st.header("📈 Equity Curves")
        fig = self._plot_equity_curves(results)
        st.plotly_chart(fig, use_container_width=True)
        
        # Drawdown analysis
        st.header("📉 Drawdown Analysis")
        fig_dd = self._plot_drawdowns(results)
        st.plotly_chart(fig_dd, use_container_width=True)
        
        # Individual strategy details
        st.header("🔍 Strategy Details")
        selected_strategy = st.selectbox("Select Strategy", list(results.keys()))
        
        if selected_strategy:
            self._display_strategy_details(selected_strategy, results[selected_strategy])
    
    def _create_metrics_dataframe(self, results: Dict[str, BacktestResults]) -> pd.DataFrame:
        """Создание DataFrame с метриками"""
        data = []
        for name, result in results.items():
            row = {'strategy': name}
            row.update(result.metrics)
            data.append(row)
        
        return pd.DataFrame(data).set_index('strategy')
    
    def _plot_equity_curves(self, results: Dict[str, BacktestResults]) -> go.Figure:
        """График equity curves"""
        fig = go.Figure()
        
        for name, result in results.items():
            fig.add_trace(go.Scatter(
                x=result.equity_curve.index,
                y=result.equity_curve.values,
                name=name,
                mode='lines'
            ))
        
        fig.update_layout(
            title="Equity Curves Comparison",
            xaxis_title="Date",
            yaxis_title="Portfolio Value",
            hovermode='x unified'
        )
        
        return fig
    
    def _plot_drawdowns(self, results: Dict[str, BacktestResults]) -> go.Figure:
        """График drawdowns"""
        fig = go.Figure()
        
        for name, result in results.items():
            peak = result.equity_curve.expanding().max()
            dd = (result.equity_curve - peak) / peak * 100
            
            fig.add_trace(go.Scatter(
                x=dd.index,
                y=dd.values,
                name=name,
                fill='tozeroy',
                mode='lines'
            ))
        
        fig.update_layout(
            title="Drawdown Analysis",
            xaxis_title="Date",
            yaxis_title="Drawdown (%)",
            hovermode='x unified'
        )
        
        return fig
    
    def _display_strategy_details(self, name: str, result: BacktestResults):
        """Детальная информация о стратегии"""
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Metrics")
            for metric, value in result.metrics.items():
                if isinstance(value, float):
                    if 'ratio' in metric or 'factor' in metric:
                        st.metric(metric, f"{value:.2f}")
                    elif 'rate' in metric or 'return' in metric or 'drawdown' in metric:
                        st.metric(metric, f"{value:.2%}")
                    else:
                        st.metric(metric, f"{value:.2f}")
                else:
                    st.metric(metric, str(value))
        
        with col2:
            st.subheader("Trade Statistics")
            if len(result.trades) > 0:
                st.write(f"Total Trades: {len(result.trades)}")
                st.write(f"Winning Trades: {(result.trades['pnl'] > 0).sum()}")
                st.write(f"Losing Trades: {(result.trades['pnl'] < 0).sum()}")
                st.dataframe(result.trades.tail(10))


if __name__ == "__main__":
    dashboard = TradingDashboard()
    dashboard.run()
```

### 6. Benchmark Runner (benchmarks/runner.py)
```python
import pandas as pd
import importlib
import os
from typing import Dict, List, Optional
from core.backtester import Backtester
from core.metrics import BacktestResults
from data.downloaders.forex_downloader import ForexDownloader

class BenchmarkRunner:
    """Запуск бенчмарков для всех стратегий"""
    
    def __init__(self, 
                 initial_capital: float = 10000,
                 commission: float = 0.0001,
                 symbol: str = "EURUSD",
                 timeframe: str = "5min"):
        self.backtester = Backtester(initial_capital=initial_capital, commission=commission)
        self.downloader = ForexDownloader()
        self.symbol = symbol
        self.timeframe = timeframe
    
    def run_all(self, 
                strategy_names: List[str],
                start_date: str,
                end_date: str) -> Dict[str, BacktestResults]:
        """Запуск бэктестов для списка стратегий"""
        
        # Загрузка данных
        data = self.downloader.download(
            symbol=self.symbol,
            timeframe=self.timeframe,
            start_date=start_date,
            end_date=end_date
        )
        
        results = {}
        
        for strategy_name in strategy_names:
            try:
                strategy = self._load_strategy(strategy_name)
                result = self.backtester.run(strategy, data, start_date, end_date)
                results[strategy_name] = result
            except Exception as e:
                print(f"Error running {strategy_name}: {e}")
        
        return results
    
    def _load_strategy(self, strategy_name: str):
        """Динамическая загрузка стратегии"""
        module = importlib.import_module(f"strategies.{strategy_name}.strategy")
        
        # Ищем класс Strategy в модуле
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if isinstance(attr, type) and hasattr(attr, 'generate_signals'):
                return attr({})  # Инстанцируем с пустыми параметрами
        
        raise ValueError(f"No strategy class found in {strategy_name}")
    
    def generate_report(self, results: Dict[str, BacktestResults]) -> pd.DataFrame:
        """Генерация сравнительного отчёта"""
        report_data = []
        
        for name, result in results.items():
            row = {
                'Strategy': name,
                **result.metrics
            }
            report_data.append(row)
        
        df = pd.DataFrame(report_data)
        
        # Сортировка по Sharpe Ratio
        df = df.sort_values('sharpe_ratio', ascending=False)
        
        return df
```

## REQUIREMENTS.TXT

```
# Core
numpy>=1.24.0
pandas>=2.0.0
scipy>=1.10.0

# Backtesting
vectorbt>=0.25.0
ta-lib>=0.4.0

# Reinforcement Learning
stable-baselines3>=2.0.0
gymnasium>=0.28.0
torch>=2.0.0

# Data
yfinance>=0.2.0
oandapyV20>=0.6.0
python-dotenv>=1.0.0

# Visualization
plotly>=5.14.0
streamlit>=1.22.0
matplotlib>=3.7.0

# Utils
joblib>=1.2.0
tqdm>=4.65.0
```

## ИНСТРУКЦИИ ПО ЗАПУСКУ

1. **Установка зависимостей:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Обучение RL модели:**
   ```bash
   python -m strategies.rl_ppo.training --symbol EURUSD --timeframe 5min
   ```

3. **Запуск бэктеста отдельной стратегии:**
   ```bash
   python -m strategies.rl_ppo.backtest --start 2023-01-01 --end 2024-01-01
   ```

4. **Запуск Dashboard:**
   ```bash
   streamlit run ui/dashboard.py
   ```

5. **Live trading:**
   ```bash
   python -m strategies.rl_ppo.live --broker oanda --symbol EURUSD
   ```

## КЛЮЧЕВЫЕ ОСОБЕННОСТИ

1. **Модульная архитектура** - каждая стратегия изолирована в своей папке
2. **Единый интерфейс** - все стратегии наследуют base_strategy
3. **Комплексные метрики** - Sharpe, Sortino, Calmar, Max DD, Profit Factor
4. **Фильтрация по времени** - торговля только в стабильные часы (12:00-16:00 GMT)
5. **UI для сравнения** - Streamlit dashboard с сортировкой
6. **Live trading support** - Paper и real broker интеграция
```

---

## 6. Рекомендации по реализации

### 6.1 Приоритет фич

1. **MVP (1-2 недели):**
   - base_strategy.py + metrics.py + backtester.py
   - Одна простая стратегия (mean reversion)
   - Базовый CLI для бэктеста

2. **RL интеграция (1-2 недели):**
   - ForexTradingEnv с Gymnasium
   - PPO training pipeline
   - Интеграция с base_strategy

3. **Dashboard (1 неделя):**
   - Streamlit UI
   - Сравнение метрик
   - Визуализация equity curves

4. **Live Trading (1-2 недели):**
   - Broker API integration (OANDA)
   - Paper trading mode
   - Real-time signal generation

### 6.2 Тестирование

- **Unit tests** для metrics.py
- **Integration tests** для backtester
- **Walk-forward validation** для RL модели
- **Paper trading** перед live

### 6.3 Мониторинг в production

- Логирование всех сделок
- Alerts при превышении drawdown
- Daily performance reports
- Model drift detection для RL

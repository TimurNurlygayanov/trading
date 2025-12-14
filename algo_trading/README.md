# Algo Trading Framework

A modular Python framework for algorithmic trading with support for traditional strategies and reinforcement learning.

## Features

- **Modular Architecture**: Each strategy is isolated in its own folder with a consistent interface
- **Multiple Strategy Types**: Mean reversion, momentum/trend following, and RL-based strategies
- **Comprehensive Metrics**: Sharpe, Sortino, Calmar ratios, max drawdown, profit factor, and more
- **Backtesting Engine**: Event-driven simulation with commission and slippage modeling
- **RL Integration**: PPO-based trading agents using Stable Baselines 3 and Gymnasium
- **Live Trading Support**: Paper trading and OANDA broker integration
- **Interactive Dashboard**: Streamlit-based UI for strategy comparison and analysis

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/algo_trading.git
cd algo_trading

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

## Baseline Strategy: Next Candle Prediction (CatBoost ML + tsfresh)

The **Next Candle** strategy is our current baseline - a CatBoost-based ML model that predicts whether the next candle will be green (close > open) and trades accordingly. Optional tsfresh automated feature extraction improves out-of-sample performance.

### Performance (EURUSD 1-Hour, Real Polygon.io Data)

**Training: 2020-2024 (5 years, 31,266 bars) | Testing: 2025 (out-of-sample, 5,878 bars)**

| Strategy | 2020-24 Return | 2025 Return | 2025 Sharpe | Win Rate | Trades |
|----------|----------------|-------------|-------------|----------|--------|
| **CatBoost + tsfresh** | +260% | **+60.99%** | **0.03** | 51.6% | 124 |
| CatBoost (no tsfresh) | +168% | +42.12% | -0.06 | 50.4% | 133 |
| Mean Reversion (BB+RSI) | +48% | -22.51% | -0.66 | 29.8% | 47 |

### Validation (No Future Leaks)
Strategy has been validated for:
- No look-ahead bias in features
- Proper train/test split
- Walk-forward validation (2/3 positive folds)
- Shuffle test passed (model learns real patterns)

### Key Features
- **70 manual features** including:
  - Candle structure (body, wicks, patterns)
  - EMAs (5, 10, 20, 50) with slopes and direction
  - RSI, MACD, Bollinger Bands
  - Daily candle context (day green so far, previous day green/red)
  - Time/session features (hour, day of week, trading sessions)
- **+30 tsfresh features** (optional): automated time series feature extraction
- **~60% training accuracy**, ~80% on high-confidence predictions
- Trades only when model confidence > 58%
- 12-bar cooldown between trades

### Run the Baseline Strategy

```bash
# Activate virtual environment
source venv/bin/activate

# Run strategy comparison (trains on 2020-2024, tests on 2025)
python run_strategy_comparison.py

# Run validation tests
python validate_strategy.py
```

### Strategy Location
- Code: `strategies/next_candle/strategy.py`
- Config: `NextCandleConfig` dataclass with adjustable parameters
- tsfresh: Enable with `use_tsfresh=True` (default)

---

## Quick Start

### 1. Run a Backtest

```bash
# Next Candle ML strategy (baseline)
python run_strategy_comparison.py

# Mean reversion strategy
python -m strategies.mean_reversion.backtest --symbol EURUSD --timeframe 1h --start 2023-01-01 --end 2024-01-01

# Momentum strategy
python -m strategies.momentum.backtest --symbol EURUSD --timeframe 1h
```

### 2. Compare Strategies

```bash
# Run benchmark comparison
python run_strategy_comparison.py
```

### 3. Launch Dashboard

```bash
streamlit run ui/dashboard.py
```

### 4. Train RL Agent

```bash
python -m strategies.rl_ppo.training --symbol EURUSD --timeframe 1h --timesteps 100000
```

## Project Structure

```
algo_trading/
├── config/
│   └── settings.py           # Global configuration
├── data/
│   └── downloaders/
│       ├── base.py           # Abstract downloader
│       └── forex_downloader.py
├── strategies/
│   ├── base_strategy.py      # Strategy interface
│   ├── next_candle/          # BASELINE - CatBoost ML
│   │   └── strategy.py       # Next candle prediction
│   ├── supertrend/
│   │   └── strategy.py       # SuperTrend with filters
│   ├── mean_reversion/
│   │   ├── strategy.py
│   │   └── backtest.py
│   ├── momentum/
│   │   ├── strategy.py
│   │   └── backtest.py
│   └── rl_ppo/
│       ├── environment.py    # Gymnasium env
│       ├── training.py       # PPO training
│       └── strategy.py       # Strategy wrapper
├── core/
│   ├── backtester.py         # Backtest engine
│   └── metrics.py            # Performance metrics
├── brokers/
│   ├── base_broker.py        # Broker interface
│   ├── paper_broker.py       # Paper trading
│   └── oanda_broker.py       # OANDA integration
├── benchmarks/
│   └── runner.py             # Strategy comparison
├── ui/
│   └── dashboard.py          # Streamlit UI
└── tests/
```

## Creating a New Strategy

1. Create a new folder under `strategies/`:

```bash
mkdir strategies/my_strategy
touch strategies/my_strategy/__init__.py
touch strategies/my_strategy/strategy.py
touch strategies/my_strategy/backtest.py
```

2. Implement the strategy in `strategy.py`:

```python
from strategies.base_strategy import Strategy

class MyStrategy(Strategy):
    DEFAULT_PARAMS = {
        'param1': 10,
        'param2': 20,
    }

    def __init__(self, params=None):
        merged = {**self.DEFAULT_PARAMS, **(params or {})}
        super().__init__(merged)

    @property
    def required_history(self) -> int:
        return 50  # Minimum bars needed

    def generate_signals(self, data):
        signals = pd.Series(index=data.index, data=0)
        # Your signal logic here
        return signals

    def get_position_size(self, signal, portfolio_value, current_price):
        return portfolio_value * 0.1  # 10% position size
```

## Metrics Explained

| Metric | Description | Target |
|--------|-------------|--------|
| Sharpe Ratio | Risk-adjusted return | > 1.0 good, > 2.0 excellent |
| Sortino Ratio | Downside risk-adjusted | > 1.5 good |
| Calmar Ratio | Return / Max Drawdown | > 1.0 good |
| Max Drawdown | Maximum peak-to-trough | < 20% conservative |
| Win Rate | % profitable trades | > 50% typical |
| Profit Factor | Gross profit / Gross loss | > 1.5 good |

## Configuration

Create a `.env` file for API keys:

```env
OANDA_API_KEY=your_api_key_here
OANDA_ACCOUNT_ID=your_account_id_here
```

### Data Source: Polygon.io

The framework uses **Polygon.io** for real forex data. Store your API key in `~/api_key2.txt`:

```bash
echo "your_polygon_api_key" > ~/api_key2.txt
```

Supported timeframes: `1min`, `5min`, `15min`, `1h`, `4h`, `1d`

## Trading Hours (EURUSD)

| Period (GMT) | Characteristic | Recommended Strategy |
|--------------|----------------|---------------------|
| 07:00-09:00 | London open, high volatility | Momentum |
| 12:00-16:00 | London-NY overlap, maximum | Breakout, Trend |
| 01:00-04:00 | Asian session, low volatility | Mean Reversion |

## License

MIT License - see LICENSE file for details.

## Disclaimer

This software is for educational and research purposes only. Trading involves substantial risk of loss. Past performance is not indicative of future results. Always test strategies thoroughly before live trading.

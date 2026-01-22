# Trading Strategies

Two quantitative trading strategies for S&P 500 stocks using Polygon API data.

## Quick Start

```bash
pip install pandas numpy pandas_ta requests tqdm
```

Set your Polygon API key in `api_config.py` or as environment variable `POLYGON_API_KEY`.

---

## Strategy 1: Drawdown Recovery (Mean Reversion)

**Files**: `screener.py`, `backtest.py`

Buy S&P 500 stocks during deep drawdowns with oversold conditions. Hold for 1 year.

### Entry Signals

| Condition | Description |
|-----------|-------------|
| Deep Drawdown | Price > 20% below 52-week high |
| RSI Oversold | RSI(14) < 30 |
| Optimal Zone | Price 20-50% below EMA200 |

### Strategy Presets

| Strategy | Win Rate | Best For |
|----------|----------|----------|
| **ULTRA** | 94%+ | Sep-Nov entries only |
| **AGGRESSIVE** | 90%+ | Year-round trading |
| **Q1_SPECIAL** | 93%+ | January-March entries |
| **BALANCED** | 88%+ | More opportunities |
| **MOMENTUM** | 90%+ | After strong weekly bounce |

### Usage

```bash
# Screen for current buy signals (auto-selects strategy by month)
python screener.py

# Use specific strategy
python screener.py --strategy AGGRESSIVE

# List available strategies
python screener.py --list

# Run backtest
python backtest.py --strategy AGGRESSIVE
python backtest.py  # runs all strategies
```

### Position Sizing

- 5% per position
- Max 10-20 concurrent positions
- Hold 1 year (252 trading days)
- No stop-loss (mean reversion needs room to recover)

### Sectors to Avoid

- Communication (20% win rate)
- Consumer Staples (0% win rate)

---

## Strategy 2: 3-Month Momentum (Trend Following)

**Files**: `sp500_3m_momentum_screener.py`, `sp500_3m_momentum_backtest.py`

Buy the S&P 500 stock with the highest 3-month return each month. Hold for 1 year.

### Entry Rules

1. Each month, rank all S&P 500 stocks by 3-month return
2. Buy the stock with the largest gain
3. Hold for exactly 1 year

### Usage

```bash
# Screen for top momentum stocks
python sp500_3m_momentum_screener.py

# Run backtest
python sp500_3m_momentum_backtest.py
```

### Position Sizing

- 10% per position ($2k of $20k default)
- Max 10 concurrent positions
- Hold 1 year

---

## File Structure

```
trading/
├── screener.py                    # Drawdown Recovery screener
├── backtest.py                    # Drawdown Recovery backtest
├── sp500_3m_momentum_screener.py  # Momentum screener
├── sp500_3m_momentum_backtest.py  # Momentum backtest
├── api_config.py                  # API key configuration
├── requirements.txt               # Python dependencies
├── CLAUDE.md                      # Development rules
└── README.md                      # This file
```

---

## API Setup

### Polygon.io (Required)

1. Sign up at [polygon.io](https://polygon.io) (free tier works)
2. Configure your key:

```bash
# Option 1: Environment variable
export POLYGON_API_KEY="your_key"

# Option 2: Edit api_config.py directly
```

---

## Disclaimer

For educational and research purposes only. Past performance does not guarantee future results. Always do your own due diligence before trading real money.

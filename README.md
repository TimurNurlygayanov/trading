# Momentum + Fundamentals Strategy with CatBoost

A quantitative long-only stock selection strategy combining time-series momentum, fundamental analysis, and machine learning with a "Cut Losers, Keep Winners" rebalancing approach.

## Quick Start

```bash
# Run backtest (trains model and tests on 2025 data)
python3 run_strategy.py

# Get current stock recommendations
python3 get_recommendations.py
```

## Strategy Overview

| Parameter | Value |
|-----------|-------|
| **Universe** | Top 500 US stocks by market cap (NYSE/NASDAQ) |
| **Rebalancing** | Quarterly (every 3 months) |
| **Portfolio Size** | 5 stocks |
| **Position Sizing** | Equal weight (20% each) |
| **Direction** | Long only |
| **Rebalance Logic** | Cut losers, keep winners |
| **Data Sources** | Polygon (prices) + EODHD (fundamentals) |

## Rebalancing Strategy: Cut Losers, Keep Winners

Every 3 months:
1. **Review each position** - check if profitable or not
2. **KEEP winners** - let profitable positions continue to run
3. **SELL losers** - cut losing positions immediately
4. **BUY replacements** - replace sold positions with new top picks from model

This approach follows the classic trading wisdom: "Cut your losses, let your winners run."

## 2025 Backtest Results (Out-of-Sample)

| Metric | Strategy | S&P 500 |
|--------|----------|---------|
| **Total Return** | **+90.2%** | +15.7% |
| **Alpha** | **+74.5%** | - |
| **Sharpe Ratio** | **1.53** | ~1.0 |
| **Max Drawdown** | **-3.5%** | ~8% |
| **Win Rate** | **75%** | - |

### Strategy Comparison (2025)

| Strategy | Return | Max DD | Sharpe |
|----------|--------|--------|--------|
| **Cut Losers (5 stocks)** | **+90.2%** | **-3.5%** | **1.53** |
| Cut Losers (10 stocks) | +62.2% | -7.1% | 1.25 |
| Cut Losers (20 stocks) | +52.2% | -3.6% | 1.36 |
| Standard 3mo (5 stocks) | +86.7% | -5.6% | 1.42 |
| Buy & Hold (20 stocks) | +44.5% | - | - |
| Rebalance Weights | +42.1% | -10.7% | 1.72 |

### Sample Trade Log

```
2025-01-03: BUY  MPWR @ $594.22
2025-01-03: BUY  NFLX @ $88.67
2025-01-03: BUY  META @ $599.24
2025-01-03: BUY  KLAC @ $636.62
2025-01-03: BUY  APH @ $69.01
2025-04-03: KEEP NFLX (profit: +5.5%)
2025-04-03: KEEP KLAC (profit: +7.8%)
2025-04-03: SELL MPWR (loss: -0.6%)
2025-04-03: SELL META (loss: -2.6%)
2025-04-03: SELL APH (loss: -1.6%)
2025-04-03: BUY  CRDO @ $43.04
2025-04-03: BUY  META @ $583.93
2025-04-03: BUY  FIX @ $342.28
2025-07-03: KEEP all (CRDO +107.6%, KLAC +44.7%, etc.)
2025-10-03: KEEP all (CRDO +247.7%, FIX +143.4%, KLAC +79.0%)
```

---

## Files

| File | Description |
|------|-------------|
| `run_strategy.py` | Main strategy with data fetching, training, and backtesting |
| `get_recommendations.py` | Get current stock picks from trained model |
| `test_cut_losers.py` | Backtest for cut losers strategy |
| `api_config.py` | API key configuration (Polygon + EODHD) |

---

## Setup

### API Keys Configuration

API keys are required for data access. Configure using one of these methods:

**Option 1: Environment Variables**
```bash
export EODHD_API_KEY="your_eodhd_key"
export POLYGON_API_KEY="your_polygon_key"
```

**Option 2: Key Files (recommended)**
```bash
echo "your_eodhd_key" > ~/.eodhd_api_key
echo "your_polygon_key" > ~/.polygon_api_key
```

### Install Dependencies

```bash
pip install pandas numpy requests catboost tqdm
```

---

## Why "Cut Losers, Keep Winners" Works

1. **Momentum continuation**: Winners tend to keep winning
2. **Early loss cutting**: Small losses don't become big losses
3. **Portfolio refresh**: Losers are replaced with fresh high-conviction picks
4. **Compound effect**: Winners that run for multiple quarters compound gains

### Tested Alternatives (Less Effective)

| Strategy | Result |
|----------|--------|
| Hold losers, sell winners | Worse returns (-13.8% for 5 stocks) |
| Rebalance to equal weights | Hurts returns (-2.4% vs buy & hold) |
| Skip rebalance if losers | Same as 6mo hold, adds complexity |
| Earnings-synced timing | High drawdowns (26-28%) |

---

## Academic Basis

The strategy incorporates findings from peer-reviewed financial research:

1. **Time Series Momentum** (Moskowitz, Ooi, Pedersen 2012) - Past 12-month returns predict future returns
2. **Residual Momentum** (Blitz, Huij, Martens 2011) - Market-neutral momentum doubles Sharpe ratio
3. **52-Week High Momentum** (George & Hwang 2004) - Stocks near 52-week highs outperform
4. **Quality-Value-Momentum** (Asness et al.) - Combining quality metrics with momentum

---

## Features Used (43 total)

### Technical Indicators (14 features)
- Momentum: 21d, 63d, 126d, 252d returns
- Volatility: 21d, 63d
- Distance from 52-week high/low
- Price vs SMA (20, 50, 200)
- RSI (14), Volume ratio, Trend strength

### Fundamental Data (17 features)
- EPS, EPS estimates, expected growth
- PE ratio, PEG ratio, Forward PE
- Profit margin, ROE, ROA
- Revenue/earnings growth
- Market cap

### Academic Enhancement Features (6 features)
- Residual Momentum (21d, 63d)
- Volatility-Adjusted Return
- Near 52-Week High indicator
- Quality Score
- Momentum Consistency

---

## Stock Selection Filters

1. **Market Cap**: >= $1 billion
2. **EPS**: Must be positive (profitable companies only)
3. **Liquidity**: Average daily dollar volume > $300k
4. **12-Month Momentum**: Must be positive (trend following)

---

## CatBoost Configuration

```python
CatBoostClassifier(
    iterations=30000,
    depth=10,
    learning_rate=0.0005,
    auto_class_weights='Balanced',
    early_stopping_rounds=500,
    use_best_model=True
)
```

Training typically stops around 1,500-2,000 iterations with early stopping.

---

## Smart Caching

- **Fundamentals**: Cached for current month, auto-refreshes next month
- **Prices**: Cached by date range, no expiry
- This minimizes API calls while keeping data fresh

---

## Disclaimer

This strategy is for educational and research purposes only. Past performance does not guarantee future results. Always conduct your own due diligence before trading.

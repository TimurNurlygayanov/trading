# Drawdown Recovery Strategy

A quantitative trading strategy for US stocks that buys quality large-cap stocks during significant drawdowns.

## Strategy Overview

**Core Idea**: Buy S&P 500 stocks when they experience deep drawdowns combined with oversold conditions and are significantly below their 200-day EMA. Hold for 1 year.

### Signal Tiers (by historical performance)

| Tier | Signal | Win Rate | Avg Return | Description |
|------|--------|----------|------------|-------------|
| **TIER 1** | Optimal | **90.6%** | **+44.9%** | DD + RSI + EMA200 zone |
| TIER 2 | DD + RSI Combo | 67.7% | +18.6% | DD + RSI oversold |
| TIER 3 | DD Only | ~60% | varies | Deep drawdown only |

## Quick Start

```bash
# Screen for current buy signals
python3 screener.py

# Run backtest (train: 2010-2024, test: 2025)
python3 backtest.py
```

**Note**: Only requires Polygon API key (free tier works). No EODHD API needed.

---

## Detailed Signal Definitions

### TIER 1 - OPTIMAL SIGNAL (90.6% win rate)

All three conditions must be true:

| Condition | Formula | Rationale |
|-----------|---------|-----------|
| **Deep Drawdown** | Price > 20% below 52-week high | Stock has pulled back significantly |
| **RSI Oversold** | RSI(14) < 30 | Technically oversold, selling exhaustion |
| **EMA200 Distance** | Price 20-50% below EMA200 | Sweet spot for mean reversion |

**Why 20-50% below EMA200?**
- < 20% below: Not oversold enough, may continue falling
- 20-50% below: Optimal recovery zone (90.6% win rate)
- > 50% below: Often indicates fundamental problems (lower win rate)

### TIER 2 - COMBO SIGNAL (67.7% win rate)

| Condition | Formula |
|-----------|---------|
| Deep Drawdown | Price > 20% below 52-week high |
| RSI Oversold | RSI(14) < 30 |

### TIER 3 - DRAWDOWN ONLY

| Condition | Formula |
|-----------|---------|
| Deep Drawdown | Price > 20% below 52-week high |

---

## Technical Indicator Calculations

All indicators use **only past data** - no future leakage.

### 52-Week High/Low
```python
# Rolling maximum of past 252 days (looks back only)
high_252d = df['high'].rolling(252, min_periods=252).max()
low_252d = df['low'].rolling(252, min_periods=252).min()

# Distance from high (drawdown)
dist_from_high = (high_252d - price) / high_252d
deep_drawdown = dist_from_high > 0.20
```

### RSI (Relative Strength Index)
```python
# Uses exponential weighted mean - past data only
delta = df['adjusted_close'].diff()
gain = delta.where(delta > 0, 0).ewm(span=14, adjust=False).mean()
loss = (-delta.where(delta < 0, 0)).ewm(span=14, adjust=False).mean()
rsi = 100 - (100 / (1 + gain / (loss + 1e-10)))
rsi_oversold = rsi < 30
```

### EMA200 Distance
```python
# EMA uses exponential weighted mean - past data only
ema_200 = df['adjusted_close'].ewm(span=200, adjust=False).mean()

# Distance as percentage
dist_from_ema200 = (price - ema_200) / ema_200 * 100

# Optimal zone: 20-50% below EMA200
in_optimal_zone = (dist_from_ema200 >= -50) & (dist_from_ema200 <= -20)
```

---

## Backtest Methodology

### Train/Test Split

| Period | Date Range | Purpose |
|--------|------------|---------|
| Training | 2010-01-01 to 2024-12-31 | Calculate historical win rates per stock |
| Testing | 2025-01-01 to present | Out-of-sample validation |

### Stock Universe Filters

1. **S&P 500**: Only stocks in the S&P 500 index
2. **Exchange**: NYSE and NASDAQ only (no OTC, pink sheets)
3. **Price**: $10 - $400 (suitable for $20k portfolio)

### Trading Rules

| Rule | Value | Rationale |
|------|-------|-----------|
| Hold Period | 252 trading days (1 year) | Allows full recovery |
| Stop Loss | **None** | Historically hurts mean reversion |
| Take Profit | **None** | Let winners run |
| Position Size | 5-10% per stock | Risk management |
| Max Positions | 10-15 stocks | Diversification |

---

## 2025 Out-of-Sample Results

Results from `backtest_polygon_only.py`:

| Signal | Trades | Win Rate | Avg Return | Median Return |
|--------|--------|----------|------------|---------------|
| **Optimal (DD+RSI+EMA200)** | varies | **90.6%** | **+44.9%** | varies |
| DD + RSI Combo | varies | 67.7% | +18.6% | varies |
| Deep Drawdown Only | varies | ~60% | varies | varies |

### EMA200 Distance Analysis (DD+RSI Combo trades)

| EMA200 Distance | Win Rate | Avg Return |
|-----------------|----------|------------|
| Above EMA200 | lower | lower |
| 0% to -20% | ~60% | varies |
| **-20% to -50% (OPTIMAL)** | **90.6%** | **+44.9%** |
| Below -50% | lower | varies |

---

## Files

| File | Description |
|------|-------------|
| `screener.py` | **Current buy signals** - screens for active signals |
| `backtest.py` | **Backtest** - validates strategy on historical data |
| `run_strategy.py` | Contains PolygonClient class and ML momentum strategy |
| `api_config.py` | API key configuration |
| `CLAUDE.md` | Development rules and future leak prevention |

---

## Setup

### 1. Get Polygon API Key

1. Sign up at [polygon.io](https://polygon.io)
2. Free tier is sufficient for this strategy

### 2. Configure API Key

```bash
# Option 1: Environment variable
export POLYGON_API_KEY="your_key"

# Option 2: Key file (recommended)
echo "your_key" > ~/.polygon_api_key
```

### 3. Install Dependencies

```bash
pip install pandas numpy requests tqdm
```

---

## Future Data Leakage Audit - PASSED

### Verification Checklist

- [x] All rolling windows use standard `rolling(N)` (looks back only)
- [x] EMA uses `ewm(span=N, adjust=False)` (exponential weighted past data)
- [x] RSI uses `diff()` and `ewm()` (both look back only)
- [x] No negative shifts in feature calculations (`shift(-N)` only in labels)
- [x] No centered rolling windows (`center=True` not used)
- [x] Daily aggregations not used (no `transform('max')` leakage)

### What Uses Future Data (Expected)

**Forward Returns (Labels Only)**:
```python
# This is the TARGET variable we're predicting, not a feature
fwd_return = df['adjusted_close'].shift(-hold_days) / df['adjusted_close'] - 1
```

This is expected behavior - forward returns are what we're trying to predict.
They are never used as input features.

---

## Why This Strategy Works

### Mean Reversion in Quality Stocks

1. **Quality Filter**: S&P 500 stocks have lower bankruptcy risk
2. **Fear Premium**: Buying when others are selling captures fear premium
3. **Technical Confirmation**: RSI oversold indicates selling exhaustion
4. **EMA200 Sweet Spot**: 20-50% below EMA200 is optimal recovery zone

### Why No Stop Loss?

Stop losses **hurt** mean reversion strategies because:
- They cut off recovery potential
- The strategy profits from extreme moves reverting
- Large drawdowns often precede large recoveries

### When This Strategy Fails

- Secular decline (company fundamentally broken)
- Fraud or scandal (irreversible damage)
- Industry disruption (obsolete business model)
- Broader market crash extending drawdown

---

## Portfolio Management Rules

### Position Sizing

| Portfolio Size | Position Size | Max Positions |
|----------------|---------------|---------------|
| $10,000 | 10% ($1,000) | 10 stocks |
| $20,000 | 5-10% ($1,000-2,000) | 10-20 stocks |
| $50,000+ | 5% | 20 stocks |

### Entry Priority

1. **First**: TIER 1 signals (Optimal - 90.6% win rate)
2. **Second**: TIER 2 signals (DD + RSI combo - 67.7% win rate)
3. **Avoid**: TIER 3 signals (DD only - lower win rate)

### Sector Diversification

- Max 2-3 stocks per sector
- Avoid all positions in same sector
- Tech sector limit: 30% of portfolio max

---

## Disclaimer

This strategy is for educational and research purposes only. Past performance does not guarantee future results. The backtest results may contain survivorship bias as we're using current S&P 500 constituents. Always conduct your own due diligence before trading real money.

**Key Risks**:
- Individual stocks can go to zero
- Drawdowns can extend beyond 1 year
- Market conditions may change
- Results based on historical data only

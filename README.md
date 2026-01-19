# Trading Strategies

A collection of quantitative trading strategies for US stocks.

---

# Strategy 1: Monthly Breakout (2-Month Hold)

A momentum breakout strategy that buys S&P 500 stocks breaking above previous month highs.

## Strategy Overview

**Core Idea**: After a green monthly candle with high volatility (ATR% > 3) and strong momentum (RSI > 55), buy when the daily close breaks above the previous month's high. Hold for 2 months.

## 2025 Backtest Results

| Metric | Value |
|--------|-------|
| **Win Rate** | 72% |
| **Profit Factor** | 5.20 |
| **Total Return** | +923.6% |
| **Avg Return/Trade** | +9.24% |
| **Trades** | 100 |
| **Max Win** | +82.2% |
| **Max Loss** | -27.0% |

## Quick Start

```bash
# Run backtest + screener
python monthly_breakout_2m_hold.py

# Run ML-enhanced version (80% win rate)
python monthly_breakout_ml.py
```

## Entry Rules

1. **Previous Month**: Must be GREEN (close > open)
2. **Volatility Filter**: ATR% > 3 (high volatility stocks only)
3. **Momentum Filter**: RSI(14) > 55 (bullish momentum)
4. **Entry Signal**: Daily candle CLOSES above previous month's high
5. **Buy**: Next day at OPEN

## Exit Rules

- **Hold Period**: 2 months from entry (no SL/TP)
- Exit at close on the day that is 2 months from entry

## Alternative Configurations

| Configuration | Win Rate | Total Return | Notes |
|--------------|----------|--------------|-------|
| 2-Month Hold (default) | 72% | +923.6% | Best risk-adjusted |
| 1-Month Hold | 78% | +676.2% | Higher WR, lower return |
| 5% Take Profit | 83% | +251.2% | Highest WR, capped gains |
| 3-Month Hold | 73% | +1,131% | More return, same WR |
| 6-Month Hold | 70% | +1,795% | Highest return |

## ML Enhancement (CatBoost)

Adding ML filter improves win rate from 72% to 80%:

| Filter | Trades | Win Rate | PF | Return |
|--------|--------|----------|-----|--------|
| Base (ATR>3 + RSI>55) | 100 | 72% | 5.20 | +923.6% |
| + ML prob >= 0.65 | 35 | **80%** | 9.29 | +375.0% |
| + ML prob >= 0.70 | 30 | **80%** | 10.22 | +328.5% |

**Top ML Features**: ATR%, 2-month momentum, body ratio, month, volume ratio

## Current Watchlist

Run the screener to see current signals:

```bash
python monthly_breakout_2m_hold.py
```

Example output:
```
>>> WATCHLIST - Pending Breakouts <<<
Symbol     Close    Target  Distance   ATR%   RSI
NVDA   $  186.50 $  192.69     +3.3%   3.4%    76
MU     $  285.41 $  298.83     +4.7%   5.0%    85
AMD    $  214.16 $  225.98     +5.5%   5.0%    69
TSLA   $  449.72 $  498.83    +10.9%   4.0%    70
```

## Files

| File | Description |
|------|-------------|
| `monthly_breakout_2m_hold.py` | Main strategy with screener |
| `monthly_breakout_ml.py` | ML-enhanced version |
| `test_2month_hold.py` | Hold period analysis |
| `test_volatility.py` | Volatility filter tests |
| `test_atr_tp.py` | ATR-based take profit tests |

---

# Strategy 2: Drawdown Recovery (1-Year Hold)

A quantitative trading strategy for US stocks that buys quality large-cap stocks during significant drawdowns.

## Strategy Overview

**Core Idea**: Buy S&P 500 stocks when they experience deep drawdowns combined with oversold conditions and are significantly below their 200-day EMA. Hold for 1 year.

## Quick Start

```bash
# Screen for current buy signals (auto-selects best strategy for current month)
python screener.py

# Screen with specific strategy
python screener.py --strategy AGGRESSIVE

# List all available strategies
python screener.py --list

# Run backtest
python backtest.py --strategy AGGRESSIVE
python backtest.py  # runs all strategies
```

**Note**: Only requires Polygon API key (free tier works).

---

## Strategy Presets

Choose the right strategy based on your goals and current month:

| Strategy | Win Rate | Avg Return | Trades/Year | Best For |
|----------|----------|------------|-------------|----------|
| **ULTRA** | 94%+ | +40-50% | 10-15 | Sep-Nov entries (highest conviction) |
| **AGGRESSIVE** | 90%+ | +35-45% | 20-30 | Year-round trading |
| **Q1_SPECIAL** | 93%+ | +30-40% | 15-25 | January-March entries |
| **BALANCED** | 88%+ | +30-40% | 40-50 | More opportunities |
| **MOMENTUM** | 90%+ | +36% | 10-20 | After strong weekly bounce |

### Strategy Details

#### ULTRA - Maximum Win Rate (Sep-Nov Only)
```
Filters: Optimal Zone + ATR Contracting + Volume > Avg + Good Sector + Sep-Nov Only
```
- Highest selectivity, highest win rate
- Only triggers during best seasonal months (September, October, November)
- Expects ~10-15 trades per year
- Best for patient investors seeking maximum probability

#### AGGRESSIVE - High Win Rate Year-Round
```
Filters: Optimal Zone + ATR Contracting + Volume > Avg + Good Sector
```
- Works throughout the year
- Good balance of selectivity and opportunity
- Expects ~20-30 trades per year
- Recommended for most users

#### Q1_SPECIAL - Optimized for Jan-Mar Entries
```
Filters: Optimal Zone + Volume > 1.5x Avg + ATR% > 3 + Good Sector
```
- Specifically tuned for Q1 market conditions
- Volume-focused filters to catch institutional interest
- Higher volatility requirement (ATR% > 3)
- Use this when entering positions in January-March

#### BALANCED - More Trades, Good Win Rate
```
Filters: Optimal Zone + ATR Contracting + Good Sector
```
- Less restrictive than AGGRESSIVE
- More trading opportunities
- Expects ~40-50 trades per year
- Good for active traders

#### MOMENTUM - Strong Weekly Bounce
```
Filters: Optimal Zone + Previous Week Up > 5% + Good Sector
```
- Counter-intuitive but effective
- Triggers when stock bounces strongly after being oversold
- Captures momentum continuation
- 90% win rate historically

---

## Filter Definitions

### Core Filters (Base Signal)

| Filter | Condition | Description |
|--------|-----------|-------------|
| **Deep Drawdown** | Price > 20% below 52-week high | Stock has pulled back significantly |
| **RSI Oversold** | RSI(14) < 30 | Technically oversold, selling exhaustion |
| **Optimal Zone** | Price 20-50% below EMA200 | Sweet spot for mean reversion |

### Advanced Filters

| Filter | Condition | Description | Win Rate Impact |
|--------|-----------|-------------|-----------------|
| **ATR Contracting** | Weekly ATR SMA(3) < SMA(10) | Volatility compression before reversal | +10% win rate |
| **Volume > Avg** | Volume > 20-day SMA | Institutional interest | +3-4% win rate |
| **Volume > 1.5x** | Volume > 1.5x 20-day SMA | Strong institutional buying | +5% win rate |
| **ATR% > 3** | ATR/Price > 3% | High volatility = bigger moves | +4% win rate |
| **Good Sector** | Not Comm or Staples | Avoid low-performing sectors | +5% win rate |
| **Weekly Up > 5%** | Previous week return > 5% | Momentum continuation | +8% win rate |
| **Seasonal Best** | Month in Sep, Oct, Nov | Best seasonal months | +10% win rate |

### Sectors to Avoid

| Sector | Historical Win Rate | Reason |
|--------|---------------------|--------|
| Communication | 20% | Structural challenges (cord-cutting, competition) |
| Consumer Staples | 0% | Slow growth, mean reversion doesn't work well |

### Best Performing Sectors

| Sector | Historical Win Rate |
|--------|---------------------|
| Technology | 87.9% |
| Industrial | 87.3% |
| Energy | 100% (small sample) |

---

## Seasonality Analysis

Entry month significantly affects win rate:

| Month | Win Rate | Recommendation |
|-------|----------|----------------|
| January | 48.5% | Use Q1_SPECIAL filters |
| February | 42.9% | Use Q1_SPECIAL filters |
| March | 74.8% | Good entry month |
| April | 47.1% | Be selective |
| May | 69.9% | Good entry month |
| June | 72.0% | Good entry month |
| July | 56.8% | Be selective |
| August | 69.1% | Avoid (filter excludes) |
| **September** | **75.8%** | **BEST - Use ULTRA** |
| **October** | **79.4%** | **BEST - Use ULTRA** |
| **November** | **79.0%** | **BEST - Use ULTRA** |
| December | 69.0% | Good entry month |

**Key Insight**: Q4 entries (Oct-Nov) benefit from "Santa Rally" and new year optimism.

---

## Technical Indicator Calculations

All indicators use **only past data** - no future leakage.

### 52-Week High (Drawdown)
```python
high_252d = df['high'].rolling(252, min_periods=252).max()
dist_from_high = (high_252d - price) / high_252d
deep_drawdown = dist_from_high > 0.20
```

### RSI (Relative Strength Index)
```python
rsi = ta.rsi(df['adjusted_close'], length=14)
rsi_oversold = rsi < 30
```

### EMA200 Distance
```python
ema_200 = ta.ema(df['adjusted_close'], length=200)
dist_from_ema200 = (price - ema_200) / ema_200 * 100
in_optimal_zone = (dist_from_ema200 >= -50) & (dist_from_ema200 <= -20)
```

### Weekly ATR Contracting (NEW)
```python
weekly = df.resample('W').agg({...})
weekly['weekly_atr'] = ta.atr(weekly['high'], weekly['low'], weekly['close'], length=14)
weekly['atr_sma3'] = weekly['weekly_atr'].rolling(3).mean()
weekly['atr_sma10'] = weekly['weekly_atr'].rolling(10).mean()
atr_contracting = atr_sma3 < atr_sma10  # Volatility compression
```

### Volume Filters (NEW)
```python
vol_sma_20 = df['volume'].rolling(20).mean()
vol_above_avg = df['volume'] > vol_sma_20
vol_above_1_5x = df['volume'] > (vol_sma_20 * 1.5)
```

---

## Usage Examples

### Screener

```bash
# Auto-select strategy based on current month
python screener.py

# Use specific strategy
python screener.py --strategy ULTRA
python screener.py --strategy AGGRESSIVE
python screener.py --strategy Q1_SPECIAL
python screener.py --strategy BALANCED
python screener.py --strategy MOMENTUM

# Scan with ALL strategies
python screener.py --strategy ALL

# List available strategies
python screener.py --list
```

### Backtest

```bash
# Run specific strategy backtest
python backtest.py --strategy AGGRESSIVE

# Run all strategies
python backtest.py

# List strategies
python backtest.py --list
```

---

## Backtest Methodology

### Train/Test Split

| Period | Date Range | Purpose |
|--------|------------|---------|
| Training | 2015-01-01 to 2024-12-31 | Calculate historical win rates |
| Testing | 2025-01-01 to present | Out-of-sample validation |

### Stock Universe

1. **S&P 500**: Only stocks in the S&P 500 index
2. **Exchange**: NYSE and NASDAQ only
3. **Price**: $10 - $400

### Trading Rules

| Rule | Value | Rationale |
|------|-------|-----------|
| Hold Period | 252 trading days (1 year) | Allows full recovery |
| Stop Loss | **None** | Historically hurts mean reversion |
| Position Size | 5% per stock | Risk management |
| Max Positions | 10-20 stocks | Diversification |

---

## Files

| File | Description |
|------|-------------|
| `screener.py` | Screen for current buy signals with strategy selection |
| `backtest.py` | Backtest strategies on historical data |
| `ml_model.py` | ML confidence score model (logistic regression) |
| `api_config.py` | API key configuration |
| `CLAUDE.md` | Development rules and future leak prevention |
| `test_new_filters.py` | Filter analysis and testing |
| `test_combined_filters.py` | Combined filter analysis |
| `analyze_q1_entries.py` | Q1 entry analysis |

---

## ML Confidence Scores

The strategy includes an optional ML model that predicts the probability of each trade being profitable.

### How It Works

- **Model**: Logistic Regression (simple, interpretable, low overfit risk)
- **Target**: P(win) - probability that 1-year return > 0
- **Output**: Confidence score 0-100%

### Features Used (No Future Leakage)

| Feature | Description |
|---------|-------------|
| `dist_from_ema200` | Distance from 200-day EMA |
| `rsi` | RSI(14) value |
| `atr_pct` | ATR as % of price |
| `vol_vs_avg` | Volume vs 20-day average |
| `atr_contracting` | Weekly ATR compression |
| `prev_weekly_return` | Previous week's return |
| `sector` | One-hot encoded sector |
| `month` | Cyclical month encoding |
| `spy_20d_return` | Market context |

### Performance by Confidence Threshold

| Threshold | Win Rate | Avg Return | Trades |
|-----------|----------|------------|--------|
| >= 50% | 86.0% | +39.1% | 429 |
| >= 60% | 90.9% | +44.5% | 331 |
| >= 70% | 92.3% | +51.9% | 234 |
| >= 80% | 94.4% | +61.2% | 71 |
| >= 90% | 95.7% | +62.7% | 23 |

### Usage

```bash
# Train and save ML model
python ml_model.py --save

# Screen with confidence filter
python screener.py --min-confidence 70  # Only 70%+ confidence signals

# Screen without filter (shows all signals with confidence scores)
python screener.py
```

### Most Important Features

| Feature | Impact | Direction |
|---------|--------|-----------|
| `sector_Comm` | -0.68 | Avoid communication stocks |
| `atr_contracting` | +0.45 | Volatility compression is good |
| `dist_from_ema200` | +0.42 | Deeper oversold is better |
| `atr_pct` | +0.38 | Higher volatility = better returns |

---

## Setup

### 1. Get Polygon API Key

1. Sign up at [polygon.io](https://polygon.io)
2. Free tier is sufficient

### 2. Configure API Key

```bash
# Option 1: Environment variable
export POLYGON_API_KEY="your_key"

# Option 2: Key file
echo "your_key" > ~/.polygon_api_key
```

### 3. Install Dependencies

```bash
pip install pandas numpy pandas_ta requests tqdm scikit-learn
```

---

## Why This Strategy Works

### Mean Reversion in Quality Stocks

1. **Quality Filter**: S&P 500 stocks have lower bankruptcy risk
2. **Fear Premium**: Buying when others are selling captures fear premium
3. **Technical Confirmation**: RSI oversold indicates selling exhaustion
4. **EMA200 Sweet Spot**: 20-50% below EMA200 is optimal recovery zone
5. **ATR Contracting**: Volatility compression often precedes reversals
6. **Volume Confirmation**: High volume on down days = capitulation

### Why No Stop Loss?

Stop losses **hurt** mean reversion strategies because:
- They cut off recovery potential
- The strategy profits from extreme moves reverting
- Large drawdowns often precede large recoveries

---

## Future Data Leakage Audit - PASSED

- [x] All rolling windows use standard `rolling(N)` (looks back only)
- [x] EMA uses `ewm(span=N)` (exponential weighted past data)
- [x] Weekly data uses `shift(1)` to avoid future leak
- [x] No negative shifts in feature calculations
- [x] Forward returns only used for labels (expected)

---

## Disclaimer

This strategy is for educational and research purposes only. Past performance does not guarantee future results. Always conduct your own due diligence before trading real money.

**Key Risks**:
- Individual stocks can go to zero
- Drawdowns can extend beyond 1 year
- Market conditions may change
- Results based on historical data only

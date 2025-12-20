# Momentum + Fundamentals Strategy with CatBoost

A quantitative long-only stock selection strategy combining time-series momentum, fundamental analysis, and machine learning.

## Quick Start

```bash
# Run backtest (trains model and tests on 2025 data)
python3 backtest.py

# Get current stock recommendations (requires trained model)
python3 get_recommendations.py
```

## Strategy Overview

| Parameter | Value |
|-----------|-------|
| **Universe** | Top 500 US stocks by market cap (NYSE/NASDAQ) |
| **Rebalancing** | Quarterly (every 3 months) |
| **Portfolio Size** | Top 20 stocks |
| **Position Sizing** | Equal weight (5% each) |
| **Direction** | Long only |
| **Holding Period** | 3 months |
| **Data Sources** | Polygon (prices) + EODHD (fundamentals) |

## 2025 Backtest Results (Out-of-Sample)

| Metric | Strategy | S&P 500 |
|--------|----------|---------|
| **Total Return** | **34.8%** | 15.7% |
| **Sharpe Ratio** | **1.13** | ~1.0 |
| **Max Drawdown** | **-8.4%** | ~8% |
| **Win Rate** | **75%** | - |
| **Alpha** | **+19.1%** | - |

### Quarterly Returns (2025)

| Quarter | Return |
|---------|--------|
| Q1 2025 (Jan-Mar) | +0.4% |
| Q2 2025 (Apr-Jun) | +24.4% |
| Q3 2025 (Jul-Sep) | +17.7% |
| Q4 2025 (Oct-Dec) | -8.4% |

### Sample Holdings (2025)

- **Q1**: WPM, APH, SAP, GE, SHOP, NOW, FICO, LLY...
- **Q2**: CRDO, GE, GOOGL, VRTX, FICO, LLY, GOOG, BKNG...
- **Q3**: SAP, GE, FICO, FTNT, AXON, V, MELI, ALAB...
- **Q4**: RDDT, CRDO, MSI, NU, IDXX, BKNG, COIN, AVGO...

---

## Files

| File | Description |
|------|-------------|
| `run_strategy.py` | Main strategy with data fetching, training, and backtesting |
| `backtest.py` | Simple runner for backtesting |
| `get_recommendations.py` | Get current stock picks from trained model |
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
pip install pandas numpy requests catboost
```

---

## Academic Basis

1. **Time Series Momentum** (Moskowitz, Ooi, Pedersen 2012) - Past 12-month returns predict future returns
2. **Fundamental Factors** - EPS, ROE, earnings growth predict stock performance
3. **Earnings Momentum** - Stocks with positive earnings surprises continue to outperform

## Features Used

### Technical Indicators (14 features)
- Momentum: 21d, 63d, 126d, 252d returns (shifted 5 days)
- Volatility: 21d, 63d
- Distance from 52-week high/low
- Price vs SMA (20, 50, 200)
- RSI (14), Volume ratio, Trend strength

### Fundamental Data (17 features)
- EPS, EPS estimates, expected growth
- PE ratio, PEG ratio, Forward PE
- Profit margin, ROE, ROA
- Revenue/earnings growth
- Price to book/sales, Dividend yield
- Earnings surprise metrics
- Market cap

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

## Smart Caching

- **Fundamentals**: Cached for current month, auto-refreshes next month
- **Prices**: Cached by date range, no expiry
- This minimizes API calls while keeping data fresh

---

## Disclaimer

This strategy is for educational and research purposes only. Past performance does not guarantee future results. Always conduct your own due diligence before trading.

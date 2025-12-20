# Trading Strategy Development Rules

This document defines mandatory rules for developing and testing trading strategies in this repository.

## 1. Future Data Leak Prevention (CRITICAL)

Before any strategy is considered complete, verify there are NO FUTURE DATA LEAKS:

### Common Leak Sources to Check:

1. **Swing High/Low Detection**
   - WRONG: Looking at bars before AND after to confirm swings
   - RIGHT: Only confirm swing after N confirmation bars have passed
   ```python
   # BAD - looks at future data
   window = highs[i - lookback:i + lookback + 1]

   # GOOD - only uses past data
   window = highs[max(0, i - lookback):i + 1]
   # Then confirm after N bars pass
   ```

2. **Session/Daily High/Low**
   - WRONG: `df.groupby('date')['high'].transform('max')` (uses entire day)
   - RIGHT: `df.groupby('date')['high'].cummax()` (cumulative, no future)

3. **Rolling Windows**
   - WRONG: Centered rolling windows `rolling(N, center=True)`
   - RIGHT: Standard rolling windows `rolling(N)` (uses past only)

4. **Shift Direction**
   - WRONG: Negative shift `df['close'].shift(-1)` (looks ahead)
   - RIGHT: Positive shift `df['close'].shift(1)` (looks back)

5. **Labeling for ML**
   - Labels (win/loss) naturally use future data - this is expected
   - But FEATURES must NEVER use future data
   - The label is what you're predicting, not what you're using to predict

### Verification Checklist:
- [ ] All technical indicators use only past data (rolling windows, EMA, etc.)
- [ ] Swing detection uses confirmation delay
- [ ] Session-based features use cumulative aggregation
- [ ] No negative shifts in feature calculation
- [ ] ML features don't include future-dependent columns

## 2. Out-of-Sample Testing (2025 Data)

Every strategy MUST be tested on 2025 data with proper train/test split:

### Required Testing Protocol:

1. **Training Period**: 2015-2024 (or subset if data limited)
2. **Test Period**: 2025 (Jan 1 to current date)
3. **NO retraining on test data**

### Reporting Requirements:

For every strategy, report these metrics on 2025 data:

| Metric | Target |
|--------|--------|
| Win Rate | > 33.3% (for 1:2 RR) |
| Profit Factor | > 1.0 |
| Max Drawdown | < 30% |
| Total Trades | > 30 (statistical significance) |

### Example Test Structure:
```python
# Download data
train_data = download('2015-01-01', '2024-12-31')
test_data = download('2025-01-01', 'today')

# Train on historical
strategy.train(train_data)

# Test on 2025 (no peeking!)
results = strategy.backtest(test_data)

# Report
print(f"2025 Trades: {results['total_trades']}")
print(f"2025 Win Rate: {results['win_rate']:.1%}")
print(f"2025 Return: {results['total_return']:.1%}")
```

## 3. Strategy Development Workflow

1. **Design** - Define entry/exit logic
2. **Implement** - Code the strategy
3. **Leak Check** - Verify no future leaks (see Section 1)
4. **Backtest** - Train on 2015-2024
5. **Validate** - Test on 2025 data (see Section 2)
6. **Iterate** - Only if 2025 results are acceptable

## 4. File Structure

```
strategies/
├── strategy_name/
│   ├── __init__.py
│   ├── strategy.py       # Main strategy logic
│   ├── ml_strategy.py    # ML-enhanced version (if applicable)
│   └── config.py         # Configuration dataclass
run_strategy_name.py      # Backtest runner
```

## 5. Quick Reference

### Minimum Requirements for Strategy Approval:
- No future data leaks
- Tested on 2025 out-of-sample data
- Win rate above breakeven for RR ratio
- Profit factor > 1.0
- At least 30 trades for statistical significance

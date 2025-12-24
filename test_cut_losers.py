#!/usr/bin/env python3
"""
CUT LOSERS, KEEP WINNERS STRATEGY TEST

Strategy: Review every 3 months
- SELL losers (cut losses early)
- KEEP winners (let profits run)
- Replace sold positions with new top picks
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
import warnings

import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from tqdm import tqdm

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

warnings.filterwarnings('ignore')

from run_strategy import (
    EODHDClient, PolygonClient,
    EODHD_API_KEY, POLYGON_API_KEY, CACHE_DIR,
    get_feature_columns, calculate_momentum_features, extract_fundamentals
)


def run_standard_backtest(test_df, prices, rebal_dates, test_end, top_n):
    """Standard backtest - sell all at rebalance."""
    portfolio_values = [100.0]
    period_returns = []

    for i, date in enumerate(rebal_dates):
        date_df = test_df[test_df['date'] == date].copy()
        if len(date_df) == 0:
            continue

        filtered = date_df[date_df['eps'] > 0].copy()
        filtered = filtered[filtered['return_252d'] > 0]

        if len(filtered) == 0:
            portfolio_values.append(portfolio_values[-1])
            period_returns.append(0)
            continue

        selected = filtered.sort_values('pred_prob', ascending=False).head(top_n)

        next_date = rebal_dates[i + 1] if i < len(rebal_dates) - 1 else test_end

        portfolio_return = 0
        n_stocks = 0

        for _, row in selected.iterrows():
            symbol = row['symbol']
            if symbol not in prices:
                continue

            price_df = prices[symbol]
            current_prices = price_df[price_df.index <= date]
            future_prices = price_df[(price_df.index > date) & (price_df.index <= next_date)]

            if len(current_prices) == 0 or len(future_prices) == 0:
                continue

            buy_price = current_prices['adjusted_close'].iloc[-1]
            sell_price = future_prices['adjusted_close'].iloc[-1]
            stock_return = (sell_price - buy_price) / buy_price
            portfolio_return += stock_return
            n_stocks += 1

        if n_stocks > 0:
            period_return = portfolio_return / n_stocks
            period_returns.append(period_return)
            portfolio_values.append(portfolio_values[-1] * (1 + period_return))
        else:
            portfolio_values.append(portfolio_values[-1])
            period_returns.append(0)

    return calculate_metrics(portfolio_values, period_returns)


def run_cut_losers_backtest(test_df, prices, rebal_dates, test_end, top_n):
    """
    CUT LOSERS, KEEP WINNERS strategy.

    Review every 3 months:
    - If position is profitable: KEEP it
    - If position is losing: SELL it and buy new top pick
    """
    portfolio_values = [100.0]
    period_returns = []
    trade_log = []

    # Holdings: {symbol: {'buy_price': float, 'buy_date': date}}
    holdings = {}

    for i, date in enumerate(rebal_dates):
        date_df = test_df[test_df['date'] == date].copy()
        if len(date_df) == 0:
            continue

        next_date = rebal_dates[i + 1] if i < len(rebal_dates) - 1 else test_end

        if i == 0:
            # Initial buy
            filtered = date_df[date_df['eps'] > 0].copy()
            filtered = filtered[filtered['return_252d'] > 0]

            if len(filtered) == 0:
                portfolio_values.append(portfolio_values[-1])
                period_returns.append(0)
                continue

            selected = filtered.sort_values('pred_prob', ascending=False).head(top_n)

            for _, row in selected.iterrows():
                symbol = row['symbol']
                if symbol not in prices:
                    continue
                price_df = prices[symbol]
                current_prices = price_df[price_df.index <= date]
                if len(current_prices) == 0:
                    continue
                buy_price = current_prices['adjusted_close'].iloc[-1]
                holdings[symbol] = {'buy_price': buy_price, 'buy_date': date}
                trade_log.append(f"{date.strftime('%Y-%m-%d')}: BUY  {symbol} @ ${buy_price:.2f}")

            # Calculate return to next period
            period_return = 0
            n_stocks = 0
            for symbol, info in holdings.items():
                if symbol not in prices:
                    continue
                price_df = prices[symbol]
                future_prices = price_df[(price_df.index > date) & (price_df.index <= next_date)]
                if len(future_prices) == 0:
                    continue
                sell_price = future_prices['adjusted_close'].iloc[-1]
                stock_return = (sell_price - info['buy_price']) / info['buy_price']
                period_return += stock_return
                n_stocks += 1

            if n_stocks > 0:
                period_return /= n_stocks
                period_returns.append(period_return)
                portfolio_values.append(portfolio_values[-1] * (1 + period_return))
            else:
                portfolio_values.append(portfolio_values[-1])
                period_returns.append(0)
            continue

        # Review positions
        winners = []
        losers = []

        for symbol, info in list(holdings.items()):
            if symbol not in prices:
                continue
            price_df = prices[symbol]
            current_prices = price_df[price_df.index <= date]
            if len(current_prices) == 0:
                continue
            current_price = current_prices['adjusted_close'].iloc[-1]
            pnl_pct = (current_price / info['buy_price'] - 1) * 100

            if pnl_pct >= 0:
                winners.append((symbol, current_price, pnl_pct, info))
            else:
                losers.append((symbol, current_price, pnl_pct, info))

        # Log winners
        for symbol, price, pnl, info in winners:
            trade_log.append(f"{date.strftime('%Y-%m-%d')}: KEEP {symbol} (profit: +{pnl:.1f}%)")

        # Sell losers
        for symbol, price, pnl, info in losers:
            trade_log.append(f"{date.strftime('%Y-%m-%d')}: SELL {symbol} (loss: {pnl:.1f}%)")
            del holdings[symbol]

        # Buy replacements
        if len(losers) > 0:
            current_holdings = set(holdings.keys())
            filtered = date_df[date_df['eps'] > 0].copy()
            filtered = filtered[filtered['return_252d'] > 0]
            filtered = filtered[~filtered['symbol'].isin(current_holdings)]

            if len(filtered) > 0:
                n_to_buy = len(losers)
                selected = filtered.sort_values('pred_prob', ascending=False).head(n_to_buy)

                for _, row in selected.iterrows():
                    symbol = row['symbol']
                    if symbol not in prices:
                        continue
                    price_df = prices[symbol]
                    current_prices = price_df[price_df.index <= date]
                    if len(current_prices) == 0:
                        continue
                    buy_price = current_prices['adjusted_close'].iloc[-1]
                    holdings[symbol] = {'buy_price': buy_price, 'buy_date': date}
                    trade_log.append(f"{date.strftime('%Y-%m-%d')}: BUY  {symbol} @ ${buy_price:.2f}")

        # Calculate period return
        period_return = 0
        n_stocks = 0
        for symbol, info in holdings.items():
            if symbol not in prices:
                continue
            price_df = prices[symbol]
            current_prices = price_df[price_df.index <= date]
            future_prices = price_df[(price_df.index > date) & (price_df.index <= next_date)]
            if len(current_prices) == 0 or len(future_prices) == 0:
                continue
            start_price = current_prices['adjusted_close'].iloc[-1]
            end_price = future_prices['adjusted_close'].iloc[-1]
            stock_return = (end_price - start_price) / start_price
            period_return += stock_return
            n_stocks += 1

        if n_stocks > 0:
            period_return /= n_stocks
            period_returns.append(period_return)
            portfolio_values.append(portfolio_values[-1] * (1 + period_return))
        else:
            portfolio_values.append(portfolio_values[-1])
            period_returns.append(0)

    metrics = calculate_metrics(portfolio_values, period_returns)
    metrics['trade_log'] = trade_log
    return metrics


def calculate_metrics(portfolio_values, period_returns):
    total_return = (portfolio_values[-1] - 100) / 100
    returns_series = pd.Series(period_returns)

    # Sharpe
    if len(period_returns) > 1 and returns_series.std() > 0:
        sharpe = returns_series.mean() / returns_series.std() * np.sqrt(4)
    else:
        sharpe = 0

    # Max drawdown
    portfolio_series = pd.Series(portfolio_values)
    running_max = portfolio_series.cummax()
    drawdown = (portfolio_series - running_max) / running_max
    max_drawdown = drawdown.min()

    # Win rate
    win_rate = (returns_series > 0).mean() if len(returns_series) > 0 else 0

    return {
        'total_return': total_return,
        'sharpe_ratio': sharpe,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
    }


def main():
    print("\n" + "=" * 80)
    print("CUT LOSERS, KEEP WINNERS STRATEGY TEST")
    print("=" * 80)
    print("\nStrategy: Review every 3 months")
    print("  - SELL losers (cut losses)")
    print("  - KEEP winners (let profits run)")
    print("  - Replace sold positions with new top picks")

    # Load data
    eodhd_client = EODHDClient(EODHD_API_KEY, max_workers=10)
    polygon_client = PolygonClient(POLYGON_API_KEY)

    print("\n[1/4] Loading data...")
    stocks = eodhd_client.get_us_stocks()
    symbols = [s['Code'] for s in stocks]
    fundamentals = eodhd_client.fetch_fundamentals_parallel(symbols)

    valid_symbols = []
    for symbol, fund in fundamentals.items():
        highlights = fund.get('Highlights', {})
        market_cap = highlights.get('MarketCapitalization', 0) or 0
        eps = highlights.get('EarningsShare')
        if market_cap >= 1e9 and eps is not None:
            valid_symbols.append((symbol, market_cap))

    valid_symbols.sort(key=lambda x: x[1], reverse=True)
    valid_symbols = [s[0] for s in valid_symbols[:500]]

    prices = polygon_client.fetch_prices_parallel(valid_symbols, '2015-01-01', '2025-12-31')
    prices['SPY'] = polygon_client.get_prices('SPY', '2015-01-01', '2025-12-31')

    liquid_symbols = []
    for symbol, df in prices.items():
        if symbol == 'SPY' or len(df) < 100:
            continue
        recent = df.tail(60)
        if (recent['close'] * recent['volume']).mean() >= 300_000:
            liquid_symbols.append(symbol)

    prices = {s: prices[s] for s in liquid_symbols if s in prices}
    prices['SPY'] = polygon_client.get_prices('SPY', '2015-01-01', '2025-12-31')
    print(f"  Loaded {len(prices) - 1} stocks")

    # Build features
    print("\n[2/4] Building features...")
    train_end = datetime(2024, 12, 31)
    test_start = datetime(2025, 1, 1)
    test_end = datetime(2025, 12, 19)

    all_dates = []
    current = datetime(2015, 1, 1).replace(day=1)
    while current <= test_end:
        all_dates.append(current.replace(day=3))
        current = (current + timedelta(days=32)).replace(day=1)

    train_dates = [d for d in all_dates if d <= train_end]
    test_dates = [d for d in all_dates if d > train_end]

    spy_df = prices.get('SPY')
    spy_returns = spy_df['adjusted_close'].pct_change() if spy_df is not None else None

    all_rows = []
    for date in tqdm(train_dates[::2] + test_dates, desc="  Features"):
        for symbol in prices.keys():
            if symbol not in fundamentals or symbol == 'SPY':
                continue
            price_df = prices[symbol]
            fund_data = fundamentals[symbol]
            price_filtered = price_df[price_df.index <= date]
            if len(price_filtered) < 260:
                continue
            try:
                momentum_df = calculate_momentum_features(price_filtered, spy_returns)
                latest = momentum_df.iloc[-1]
                fund_features = extract_fundamentals(fund_data, date)
                if fund_features.get('eps') is None:
                    continue

                row = {'symbol': symbol, 'date': date,
                       'close': latest.get('adjusted_close', latest.get('close'))}

                for col in ['return_21d', 'return_63d', 'return_126d', 'return_252d',
                           'volatility_21d', 'volatility_63d', 'dist_from_high', 'dist_from_low',
                           'price_to_sma_20', 'price_to_sma_50', 'price_to_sma_200',
                           'rsi_14', 'volume_ratio', 'trend_strength',
                           'residual_momentum_21d', 'residual_momentum_63d',
                           'volatility_adjusted_return', 'near_52w_high', 'momentum_consistency']:
                    row[col] = latest.get(col, 0)

                for key, value in fund_features.items():
                    row[key] = value if value is not None else 0
                row['log_market_cap'] = np.log(fund_features.get('market_cap', 1e9) + 1)
                all_rows.append(row)
            except:
                continue

    feature_df = pd.DataFrame(all_rows)
    print(f"  Samples: {len(feature_df)}")

    # Forward returns
    forward_returns = []
    for _, row in tqdm(feature_df.iterrows(), total=len(feature_df), desc="  Forward ret"):
        symbol, current_date, current_price = row['symbol'], row['date'], row['close']
        if symbol not in prices:
            forward_returns.append(np.nan)
            continue
        price_df = prices[symbol]
        future = price_df[(price_df.index > current_date) & (price_df.index <= current_date + timedelta(days=21))]
        if len(future) > 0:
            forward_returns.append((future['adjusted_close'].iloc[-1] - current_price) / current_price)
        else:
            forward_returns.append(np.nan)

    feature_df['forward_return'] = forward_returns

    # Train
    print("\n[3/4] Training model...")
    train_df = feature_df[feature_df['date'] <= train_end].dropna(subset=['forward_return'])
    test_df_base = feature_df[feature_df['date'] > train_end]

    market_returns = train_df.groupby('date')['forward_return'].mean()
    train_df = train_df.copy()
    train_df['market_return'] = train_df['date'].map(market_returns)
    train_df['label'] = (train_df['forward_return'] > train_df['market_return']).astype(int)

    feature_cols = get_feature_columns()
    X_train = train_df[feature_cols].fillna(0).replace([np.inf, -np.inf], 0).values
    y_train = train_df['label'].values

    val_size = int(len(X_train) * 0.1)
    model = CatBoostClassifier(iterations=30000, depth=10, learning_rate=0.0005,
                                auto_class_weights='Balanced', random_state=42,
                                verbose=1000, early_stopping_rounds=500, use_best_model=True)
    model.fit(X_train[:-val_size], y_train[:-val_size],
              eval_set=(X_train[-val_size:], y_train[-val_size:]))

    X_test = test_df_base[feature_cols].fillna(0).replace([np.inf, -np.inf], 0).values
    test_df = test_df_base.copy()
    test_df['pred_prob'] = model.predict_proba(X_test)[:, 1]

    # SPY
    spy = prices.get('SPY')
    spy_2025 = spy[(spy.index >= test_start) & (spy.index <= test_end)]
    spy_return = (spy_2025['adjusted_close'].iloc[-1] - spy_2025['adjusted_close'].iloc[0]) / spy_2025['adjusted_close'].iloc[0]

    # Test configurations
    print("\n[4/4] Testing strategies...")

    # Rebalance dates - every 3 months
    rebal_dates_3mo = [test_dates[i] for i in range(0, len(test_dates), 3)]
    rebal_dates_6mo = [test_dates[i] for i in range(0, len(test_dates), 6)]

    print("\n" + "=" * 90)
    print("RESULTS COMPARISON")
    print("=" * 90)
    print(f"\nS&P 500: +{spy_return*100:.1f}%\n")

    for n_stocks in [5, 10, 15, 20]:
        print("=" * 70)
        print(f"{n_stocks} STOCKS")
        print("=" * 70)
        print()

        # Standard 3-month
        std_3mo = run_standard_backtest(test_df, prices, rebal_dates_3mo, test_end, n_stocks)

        # Standard 6-month
        std_6mo = run_standard_backtest(test_df, prices, rebal_dates_6mo, test_end, n_stocks)

        # Cut losers strategy (review every 3mo)
        cut_losers = run_cut_losers_backtest(test_df, prices, rebal_dates_3mo, test_end, n_stocks)

        print(f"{'Strategy':<30} {'Return':>10} {'Sharpe':>10} {'MaxDD':>10} {'WinRate':>10}")
        print("-" * 70)
        print(f"{'Standard 3-month':<30} {std_3mo['total_return']*100:>+10.1f}% {std_3mo['sharpe_ratio']:>10.2f} {std_3mo['max_drawdown']*100:>9.1f}% {std_3mo['win_rate']*100:>9.0f}%")
        print(f"{'Standard 6-month':<30} {std_6mo['total_return']*100:>+10.1f}% {std_6mo['sharpe_ratio']:>10.2f} {std_6mo['max_drawdown']*100:>9.1f}% {std_6mo['win_rate']*100:>9.0f}%")
        print(f"{'Cut Losers, Keep Winners':<30} {cut_losers['total_return']*100:>+10.1f}% {cut_losers['sharpe_ratio']:>10.2f} {cut_losers['max_drawdown']*100:>9.1f}% {cut_losers['win_rate']*100:>9.0f}%")

        diff_3mo = (cut_losers['total_return'] - std_3mo['total_return']) * 100
        diff_6mo = (cut_losers['total_return'] - std_6mo['total_return']) * 100

        if diff_3mo > 0:
            print(f"\n  vs 3mo Standard: +{diff_3mo:.1f}% (better with cut losers)")
        else:
            print(f"\n  vs 3mo Standard: {diff_3mo:.1f}% (worse with cut losers)")

        # Trade log
        if cut_losers.get('trade_log'):
            print("\n  Trade log:")
            for log in cut_losers['trade_log'][:25]:
                print(f"    {log}")
            if len(cut_losers['trade_log']) > 25:
                print(f"    ... ({len(cut_losers['trade_log']) - 25} more trades)")

        print()

    print("=" * 90)


if __name__ == "__main__":
    main()

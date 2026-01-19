#!/usr/bin/env python3
"""
Monthly Breakout Strategy - 6-Month Hold

Strategy Rules:
1. Previous month must be GREEN (close > open)
2. Filter: ATR% > 3 AND RSI(14) > 55 (high volatility + momentum)
3. Entry: When daily candle CLOSES above previous month's high
4. Buy: Next day at OPEN
5. Hold: 6 months (no SL/TP)
6. Exit: After 6 months from entry

2025 Backtest Results:
- Trades: 100
- Win Rate: 72%
- Profit Factor: 9.61
- Total Return: +2209%
- Avg Return per Trade: 22.09%
"""

import pandas as pd
import numpy as np
import pandas_ta as ta
from pathlib import Path
from datetime import datetime, timedelta
from tqdm import tqdm
from catboost import CatBoostClassifier
import warnings
warnings.filterwarnings('ignore')

CACHE_DIR = Path('data_cache_weekly')


def get_sp500_tickers():
    return [
        'AAPL', 'MSFT', 'NVDA', 'GOOGL', 'AMZN', 'META', 'TSLA', 'BRK-B', 'AVGO', 'LLY',
        'JPM', 'V', 'UNH', 'XOM', 'MA', 'COST', 'HD', 'PG', 'JNJ', 'WMT',
        'ABBV', 'NFLX', 'CRM', 'BAC', 'ORCL', 'CVX', 'MRK', 'KO', 'PEP', 'AMD',
        'TMO', 'CSCO', 'LIN', 'ACN', 'MCD', 'ABT', 'ADBE', 'WFC', 'IBM', 'PM',
        'GE', 'ISRG', 'NOW', 'CAT', 'QCOM', 'GS', 'TXN', 'INTU', 'VZ', 'BKNG',
        'AXP', 'MS', 'RTX', 'SPGI', 'AMGN', 'DHR', 'NEE', 'T', 'PFE', 'BLK',
        'HON', 'UBER', 'UNP', 'ETN', 'LOW', 'AMAT', 'COP', 'PLD', 'SYK', 'C',
        'BX', 'SCHW', 'DE', 'BA', 'VRTX', 'BSX', 'TJX', 'ADP', 'LMT', 'BMY',
        'GILD', 'ADI', 'PANW', 'SBUX', 'MDT', 'CB', 'MMC', 'LRCX', 'MU',
        'CI', 'KKR', 'AMT', 'SO', 'CME', 'REGN', 'KLAC', 'DUK', 'ICE', 'INTC',
        'SHW', 'MDLZ', 'SNPS', 'PH', 'CDNS', 'EQIX', 'PNC', 'ZTS', 'PYPL', 'CMG',
        'CL', 'CTAS', 'USB', 'WM', 'MCO', 'AON', 'TT', 'APH', 'ITW', 'WELL',
        'MSI', 'TDG', 'EOG', 'CVS', 'EMR', 'MAR', 'NOC', 'MMM', 'ORLY', 'CEG',
        'FDX', 'GD', 'HCA', 'NSC', 'ABNB', 'CSX', 'FCX', 'AJG', 'CARR', 'ROP',
        'ECL', 'HLT', 'APD', 'BDX', 'TRV', 'PCAR', 'GM', 'OKE', 'AZO', 'DLR',
        'SRE', 'MPC', 'PSX', 'CPRT', 'NXPI', 'AEP', 'PSA', 'JCI', 'URI', 'TFC',
        'AFL', 'NEM', 'AIG', 'MET', 'FICO', 'SPG', 'KMI', 'FTNT', 'VLO', 'ALL',
        'HUM', 'PCG', 'SLB', 'PAYX', 'D', 'MNST', 'CCI', 'GWW', 'FAST', 'KMB',
        'O', 'DHI', 'MSCI', 'PRU', 'BK', 'CTVA', 'CMI', 'PWR', 'LEN', 'HES',
        'AME', 'A', 'ODFL', 'NUE', 'RSG', 'KR', 'EXC', 'AXON', 'FANG', 'F',
    ]


# Strategy parameters
ATR_THRESHOLD = 3.0
RSI_THRESHOLD = 55
HOLD_MONTHS = 6


def load_data():
    """Load cached data for all symbols."""
    all_data = {}
    for symbol in get_sp500_tickers():
        cache_file = CACHE_DIR / f'polygon_{symbol}_2024-01-01_2025-12-31.pkl'
        if cache_file.exists():
            all_data[symbol] = pd.read_pickle(cache_file)
    return all_data


def calculate_indicators(daily_df):
    """Calculate monthly indicators from daily data."""
    monthly = daily_df.resample('ME').agg({
        'open': 'first', 'high': 'max', 'low': 'min',
        'close': 'last', 'volume': 'sum'
    }).dropna()

    if len(monthly) < 3:
        return None, None

    monthly['is_green'] = monthly['close'] > monthly['open']
    monthly['body_pct'] = (monthly['close'] - monthly['open']) / monthly['open'] * 100
    monthly['range_pct'] = (monthly['high'] - monthly['low']) / monthly['low'] * 100

    rsi = ta.rsi(monthly['close'], length=14)
    monthly['rsi'] = rsi if rsi is not None else 50

    daily_atr = ta.atr(daily_df['high'], daily_df['low'], daily_df['close'], length=14)
    if daily_atr is None:
        return None, None

    daily_df_copy = daily_df.copy()
    daily_df_copy['atr'] = daily_atr
    monthly_atr = daily_df_copy['atr'].resample('ME').mean()
    monthly['atr'] = monthly_atr
    monthly['atr_pct'] = monthly_atr / monthly['close'] * 100

    return monthly, daily_df_copy


def check_filters(prev_month):
    """Check if previous month passes all filters."""
    if not prev_month['is_green']:
        return False

    atr_pct = prev_month['atr_pct'] if pd.notna(prev_month['atr_pct']) else 0
    rsi = prev_month['rsi'] if pd.notna(prev_month['rsi']) else 0

    return atr_pct > ATR_THRESHOLD and rsi > RSI_THRESHOLD


def run_backtest(all_data, year=2025):
    """Run backtest for specified year."""
    all_trades = []

    for symbol, daily_df in tqdm(all_data.items(), desc='Backtesting'):
        monthly, daily_with_atr = calculate_indicators(daily_df)
        if monthly is None:
            continue

        for i in range(1, len(monthly)):
            prev_month = monthly.iloc[i - 1]
            curr_month_end = monthly.index[i]

            if not check_filters(prev_month):
                continue

            prev_month_high = prev_month['high']
            prev_month_end = prev_month.name

            month_daily = daily_df[(daily_df.index > prev_month_end) &
                                   (daily_df.index <= curr_month_end)]

            if len(month_daily) < 2:
                continue

            for j in range(len(month_daily) - 1):
                day = month_daily.iloc[j]

                if day['close'] > prev_month_high:
                    next_day_idx = j + 1
                    if next_day_idx >= len(month_daily):
                        break

                    entry_day = month_daily.iloc[next_day_idx]
                    entry_price = entry_day['open']
                    entry_date = entry_day.name

                    if entry_date.year == year:
                        # Calculate exit (2 months later)
                        exit_target = entry_date + pd.DateOffset(months=HOLD_MONTHS)
                        future_data = daily_df[daily_df.index > entry_date]

                        if len(future_data) == 0:
                            break

                        exit_data = future_data[future_data.index <= exit_target]
                        if len(exit_data) == 0:
                            break

                        exit_day = exit_data.iloc[-1]
                        exit_date = exit_day.name
                        exit_price = exit_day['close']
                        pct_return = (exit_price - entry_price) / entry_price * 100

                        all_trades.append({
                            'symbol': symbol,
                            'entry_date': entry_date,
                            'entry_price': entry_price,
                            'exit_date': exit_date,
                            'exit_price': exit_price,
                            'days_held': (exit_date - entry_date).days,
                            'pct_return': pct_return,
                            'win': pct_return > 0,
                            'atr_pct': prev_month['atr_pct'],
                            'rsi': prev_month['rsi'],
                            'body_pct': prev_month['body_pct'],
                            'range_pct': prev_month['range_pct'],
                        })
                    break

    return pd.DataFrame(all_trades)


def calculate_metrics(trades_df):
    """Calculate performance metrics."""
    if len(trades_df) == 0:
        return {}

    wins = trades_df[trades_df['win']]
    losses = trades_df[~trades_df['win']]

    win_rate = len(wins) / len(trades_df) * 100
    total_ret = trades_df['pct_return'].sum()
    avg_ret = trades_df['pct_return'].mean()

    gross_profit = wins['pct_return'].sum() if len(wins) > 0 else 0
    gross_loss = abs(losses['pct_return'].sum()) if len(losses) > 0 else 0.001
    profit_factor = gross_profit / gross_loss

    return {
        'trades': len(trades_df),
        'wins': len(wins),
        'losses': len(losses),
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'total_return': total_ret,
        'avg_return': avg_ret,
        'max_win': trades_df['pct_return'].max(),
        'max_loss': trades_df['pct_return'].min(),
        'avg_days': trades_df['days_held'].mean(),
    }


def print_results(metrics, trades_df):
    """Print backtest results."""
    print('\n' + '=' * 70)
    print('MONTHLY BREAKOUT - 6 MONTH HOLD STRATEGY')
    print('=' * 70)
    print(f'\nFilters: ATR% > {ATR_THRESHOLD} + RSI > {RSI_THRESHOLD}')
    print(f'Hold Period: {HOLD_MONTHS} months')
    print('-' * 70)
    print(f"Total Trades:    {metrics['trades']}")
    print(f"Wins/Losses:     {metrics['wins']} / {metrics['losses']}")
    print(f"Win Rate:        {metrics['win_rate']:.1f}%")
    print(f"Profit Factor:   {metrics['profit_factor']:.2f}")
    print(f"Total Return:    {metrics['total_return']:.1f}%")
    print(f"Avg Return:      {metrics['avg_return']:.2f}%")
    print(f"Max Win:         {metrics['max_win']:.1f}%")
    print(f"Max Loss:        {metrics['max_loss']:.1f}%")
    print(f"Avg Days Held:   {metrics['avg_days']:.0f}")

    # Monthly breakdown
    print('\n' + '-' * 70)
    print('MONTHLY BREAKDOWN')
    print('-' * 70)
    trades_df['month'] = trades_df['entry_date'].dt.to_period('M')
    monthly_stats = trades_df.groupby('month').agg({
        'pct_return': ['count', 'sum', 'mean'],
        'win': 'mean'
    })
    monthly_stats.columns = ['trades', 'total_ret', 'avg_ret', 'win_rate']
    monthly_stats['win_rate'] = monthly_stats['win_rate'] * 100
    print(monthly_stats.to_string())


def generate_training_data(all_data):
    """Generate training data for ML model."""
    all_signals = []

    for symbol, daily_df in all_data.items():
        monthly, daily_with_atr = calculate_indicators(daily_df)
        if monthly is None or len(monthly) < 6:
            continue

        # Additional features for ML
        monthly['ret_1m'] = monthly['close'].pct_change() * 100
        monthly['ret_2m'] = monthly['close'].pct_change(2) * 100
        monthly['vol_ratio'] = monthly['volume'] / monthly['volume'].rolling(3).mean()

        for i in range(1, len(monthly)):
            prev_month = monthly.iloc[i - 1]
            curr_month_end = monthly.index[i]

            if not check_filters(prev_month):
                continue

            prev_month_high = prev_month['high']
            prev_month_end = prev_month.name

            month_daily = daily_df[(daily_df.index > prev_month_end) &
                                   (daily_df.index <= curr_month_end)]

            if len(month_daily) < 2:
                continue

            for j in range(len(month_daily) - 1):
                day = month_daily.iloc[j]

                if day['close'] > prev_month_high:
                    next_day_idx = j + 1
                    if next_day_idx >= len(month_daily):
                        break

                    entry_day = month_daily.iloc[next_day_idx]
                    entry_price = entry_day['open']
                    entry_date = entry_day.name

                    # Calculate 2-month return
                    exit_target = entry_date + pd.DateOffset(months=HOLD_MONTHS)
                    future_data = daily_df[daily_df.index > entry_date]

                    if len(future_data) == 0:
                        break

                    exit_data = future_data[future_data.index <= exit_target]
                    if len(exit_data) == 0:
                        break

                    exit_price = exit_data.iloc[-1]['close']
                    pct_return = (exit_price - entry_price) / entry_price * 100

                    all_signals.append({
                        'symbol': symbol,
                        'entry_date': entry_date,
                        'pct_return': pct_return,
                        'win': int(pct_return > 0),
                        'year': entry_date.year,
                        'month': entry_date.month,
                        'atr_pct': prev_month['atr_pct'] if pd.notna(prev_month['atr_pct']) else 3,
                        'rsi': prev_month['rsi'] if pd.notna(prev_month['rsi']) else 55,
                        'body_pct': prev_month['body_pct'] if pd.notna(prev_month['body_pct']) else 0,
                        'range_pct': prev_month['range_pct'] if pd.notna(prev_month['range_pct']) else 5,
                        'ret_1m': prev_month['ret_1m'] if pd.notna(prev_month['ret_1m']) else 0,
                        'ret_2m': prev_month['ret_2m'] if pd.notna(prev_month['ret_2m']) else 0,
                        'vol_ratio': prev_month['vol_ratio'] if pd.notna(prev_month['vol_ratio']) else 1,
                    })
                    break

    return pd.DataFrame(all_signals)


def train_ml_model(training_df):
    """Train CatBoost model on historical data."""
    feature_cols = ['atr_pct', 'rsi', 'body_pct', 'range_pct', 'ret_1m', 'ret_2m', 'vol_ratio', 'month']

    X = training_df[feature_cols].fillna(0)
    y = training_df['win']

    model = CatBoostClassifier(
        iterations=500,
        learning_rate=0.05,
        depth=4,
        loss_function='Logloss',
        verbose=False,
        random_state=42
    )
    model.fit(X, y)

    return model, feature_cols


def run_screener(all_data, ml_model=None, feature_cols=None):
    """Screen for stocks that JUST closed above previous month high."""
    signals = []

    for symbol, daily_df in all_data.items():
        monthly, daily_with_atr = calculate_indicators(daily_df)
        if monthly is None or len(monthly) < 2:
            continue

        # Get last complete month
        last_month = monthly.iloc[-1]

        # Check filters
        if not check_filters(last_month):
            continue

        # Get current month's daily data
        last_month_end = last_month.name
        current_daily = daily_df[daily_df.index > last_month_end]

        if len(current_daily) == 0:
            continue

        last_day = current_daily.iloc[-1]
        prev_month_high = last_month['high']

        # Check if TODAY's close broke above prev month high
        # (and it wasn't already broken before)
        today_broke = last_day['close'] > prev_month_high

        # Check if any PREVIOUS day already broke
        if len(current_daily) > 1:
            previous_days = current_daily.iloc[:-1]
            already_broke = any(previous_days['close'] > prev_month_high)
        else:
            already_broke = False

        # Only show if JUST triggered (today broke, but not before)
        just_triggered = today_broke and not already_broke

        if just_triggered:
            # Additional features for ML
            monthly['ret_1m'] = monthly['close'].pct_change() * 100
            monthly['ret_2m'] = monthly['close'].pct_change(2) * 100
            monthly['vol_ratio'] = monthly['volume'] / monthly['volume'].rolling(3).mean()

            # Calculate ML probability
            ml_prob = None
            if ml_model is not None and feature_cols is not None:
                features = {
                    'atr_pct': last_month['atr_pct'] if pd.notna(last_month['atr_pct']) else 3,
                    'rsi': last_month['rsi'] if pd.notna(last_month['rsi']) else 55,
                    'body_pct': last_month['body_pct'] if pd.notna(last_month['body_pct']) else 0,
                    'range_pct': last_month['range_pct'] if pd.notna(last_month['range_pct']) else 5,
                    'ret_1m': monthly.iloc[-1]['ret_1m'] if pd.notna(monthly.iloc[-1]['ret_1m']) else 0,
                    'ret_2m': monthly.iloc[-1]['ret_2m'] if pd.notna(monthly.iloc[-1]['ret_2m']) else 0,
                    'vol_ratio': monthly.iloc[-1]['vol_ratio'] if pd.notna(monthly.iloc[-1]['vol_ratio']) else 1,
                    'month': last_day.name.month,
                }
                X = pd.DataFrame([features])[feature_cols].fillna(0)
                ml_prob = ml_model.predict_proba(X)[0, 1]

            signals.append({
                'symbol': symbol,
                'trigger_date': last_day.name,
                'close': last_day['close'],
                'prev_month_high': prev_month_high,
                'breakout_pct': (last_day['close'] - prev_month_high) / prev_month_high * 100,
                'atr_pct': last_month['atr_pct'],
                'rsi': last_month['rsi'],
                'ml_prob': ml_prob,
            })

    return pd.DataFrame(signals)


def print_screener(signals_df):
    """Print screener results."""
    print('\n' + '=' * 95)
    print('SCREENER - Stocks That JUST Closed Above Previous Month High')
    print('Filter: ATR% > 3 + RSI > 55')
    print('=' * 95)

    if len(signals_df) == 0:
        print('\nNo new breakout signals today.')
        print('(Stocks must close above previous month high TODAY for the first time)')
        return

    # Sort by ML probability
    signals_df = signals_df.sort_values('ml_prob', ascending=False)

    print(f'\n{"Symbol":<6} {"Date":<12} {"Close":>9} {"Target":>9} {"Break%":>8} {"ATR%":>6} {"RSI":>5} {"ML Prob":>8}')
    print('-' * 75)

    for _, s in signals_df.iterrows():
        ml_str = f"{s['ml_prob']*100:>6.1f}%" if pd.notna(s['ml_prob']) else "   N/A"
        print(f"{s['symbol']:<6} {s['trigger_date'].strftime('%Y-%m-%d'):<12} "
              f"${s['close']:>8.2f} ${s['prev_month_high']:>8.2f} "
              f"{s['breakout_pct']:>+7.2f}% {s['atr_pct']:>5.1f}% {s['rsi']:>5.0f} {ml_str}")

    print(f'\nTotal: {len(signals_df)} new signal(s)')
    print('\nAction: BUY at tomorrow\'s OPEN, hold for 6 months')


if __name__ == '__main__':
    print('Loading data...')
    all_data = load_data()
    print(f'Loaded {len(all_data)} symbols')

    # Run backtest
    print('\nRunning backtest...')
    trades_df = run_backtest(all_data, year=2025)

    if len(trades_df) > 0:
        metrics = calculate_metrics(trades_df)
        print_results(metrics, trades_df)

    # Train ML model on historical data
    print('\nTraining ML model...')
    training_df = generate_training_data(all_data)
    print(f'Training samples: {len(training_df)}')

    if len(training_df) > 50:
        ml_model, feature_cols = train_ml_model(training_df)
        train_acc = (ml_model.predict(training_df[feature_cols].fillna(0)) == training_df['win']).mean()
        print(f'Training accuracy: {train_acc:.1%}')
    else:
        ml_model, feature_cols = None, None
        print('Not enough training data for ML model')

    # Run screener
    print('\nRunning screener...')
    signals_df = run_screener(all_data, ml_model, feature_cols)
    print_screener(signals_df)

    # Save trades to CSV
    if len(trades_df) > 0:
        trades_df.to_csv('trades_2m_hold.csv', index=False)
        print(f'\nTrades saved to trades_2m_hold.csv')

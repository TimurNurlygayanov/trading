#!/usr/bin/env python3
"""
Monthly Breakout & Mean Reversion Screener

STRATEGY 1 - BREAKOUT (Momentum):
- Previous month GREEN + ATR% > 3 + RSI > 55
- Entry: Daily close breaks ABOVE previous month HIGH
- Buy next day at OPEN, hold 6 months
- 2025 Results: 72% win rate, +2209% return, PF 9.61

STRATEGY 2 - MEAN REVERSION (Oversold Bounce):
- RSI < 30 + ATR% > 3
- Entry: Daily close breaks BELOW previous month LOW
- Buy next day at OPEN, hold 6 months
- 2025 Results: 92% win rate, +227% return

Usage:
  python monthly_screener.py
"""

import pandas as pd
import numpy as np
import pandas_ta as ta
from pathlib import Path
from datetime import datetime, timedelta
from tqdm import tqdm
from catboost import CatBoostClassifier
import requests
import time
import warnings
warnings.filterwarnings('ignore')

CACHE_DIR = Path('data_cache_weekly')


def get_polygon_api_key():
    """Get Polygon API key from various sources."""
    import os
    # Try environment variable
    key = os.environ.get('POLYGON_API_KEY', '')
    if key:
        return key
    # Try file locations
    for filepath in ['~/.polygon_api_key', '~/api_key2.txt', '~/Desktop/api_key.txt']:
        path = Path(filepath).expanduser()
        if path.exists():
            return path.read_text().strip()
    raise ValueError("Polygon API key not found")


def download_polygon_data(symbol, start_date, end_date, api_key):
    """Download daily data from Polygon API."""
    url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/day/{start_date}/{end_date}"
    params = {'adjusted': 'true', 'sort': 'asc', 'limit': 50000, 'apiKey': api_key}

    try:
        resp = requests.get(url, params=params, timeout=30)
        data = resp.json()

        if data.get('status') in ['OK', 'DELAYED'] and 'results' in data:
            df = pd.DataFrame(data['results'])
            df['timestamp'] = pd.to_datetime(df['t'], unit='ms')
            df = df.rename(columns={'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 'v': 'volume'})
            df = df.set_index('timestamp')
            df = df[['open', 'high', 'low', 'close', 'volume']]
            return df
    except Exception as e:
        print(f"Error downloading {symbol}: {e}")
    return None


def download_fresh_data(symbols, start_date='2024-01-01'):
    """Download fresh data for all symbols from Polygon."""
    api_key = get_polygon_api_key()
    end_date = datetime.now().strftime('%Y-%m-%d')

    print(f"\nDownloading fresh data from Polygon ({start_date} to {end_date})...")

    all_data = {}
    CACHE_DIR.mkdir(exist_ok=True)

    for symbol in tqdm(symbols, desc='Downloading'):
        df = download_polygon_data(symbol, start_date, end_date, api_key)
        if df is not None and len(df) > 20:
            all_data[symbol] = df
            # Update cache
            cache_file = CACHE_DIR / f'polygon_{symbol}_2024-01-01_2025-12-31.pkl'
            df.to_pickle(cache_file)
        time.sleep(0.15)  # Rate limiting

    print(f'Downloaded {len(all_data)} symbols')
    return all_data


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
        return None

    monthly['is_green'] = monthly['close'] > monthly['open']
    monthly['is_red'] = monthly['close'] < monthly['open']
    monthly['body_pct'] = (monthly['close'] - monthly['open']) / monthly['open'] * 100

    rsi = ta.rsi(monthly['close'], length=14)
    monthly['rsi'] = rsi if rsi is not None else 50

    daily_atr = ta.atr(daily_df['high'], daily_df['low'], daily_df['close'], length=14)
    if daily_atr is None:
        return None

    daily_df_copy = daily_df.copy()
    daily_df_copy['atr'] = daily_atr
    monthly_atr = daily_df_copy['atr'].resample('ME').mean()
    monthly['atr_pct'] = monthly_atr / monthly['close'] * 100

    return monthly


def generate_training_data(all_data):
    """Generate training data for ML model."""
    all_signals = []

    for symbol, daily_df in all_data.items():
        monthly = calculate_indicators(daily_df)
        if monthly is None or len(monthly) < 6:
            continue

        monthly['ret_1m'] = monthly['close'].pct_change() * 100
        monthly['ret_2m'] = monthly['close'].pct_change(2) * 100
        monthly['vol_ratio'] = monthly['volume'] / monthly['volume'].rolling(3).mean()

        for i in range(1, len(monthly)):
            prev_month = monthly.iloc[i - 1]
            curr_month_end = monthly.index[i]
            prev_month_end = prev_month.name

            atr_pct = prev_month['atr_pct'] if pd.notna(prev_month['atr_pct']) else 0
            rsi = prev_month['rsi'] if pd.notna(prev_month['rsi']) else 50

            # BREAKOUT filter
            if not (prev_month['is_green'] and atr_pct > 3 and rsi > 55):
                continue

            prev_month_high = prev_month['high']
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

                    exit_target = entry_date + pd.DateOffset(months=6)
                    future_data = daily_df[daily_df.index > entry_date]
                    future_data = future_data[future_data.index <= exit_target]

                    if len(future_data) == 0:
                        break

                    exit_price = future_data.iloc[-1]['close']
                    pct_return = (exit_price - entry_price) / entry_price * 100

                    all_signals.append({
                        'win': int(pct_return > 0),
                        'month': entry_date.month,
                        'atr_pct': atr_pct,
                        'rsi': rsi,
                        'body_pct': prev_month['body_pct'] if pd.notna(prev_month['body_pct']) else 0,
                        'ret_1m': prev_month['ret_1m'] if pd.notna(prev_month['ret_1m']) else 0,
                        'ret_2m': prev_month['ret_2m'] if pd.notna(prev_month['ret_2m']) else 0,
                        'vol_ratio': prev_month['vol_ratio'] if pd.notna(prev_month['vol_ratio']) else 1,
                    })
                    break

    return pd.DataFrame(all_signals)


def train_ml_model(training_df):
    """Train CatBoost model on historical data."""
    feature_cols = ['atr_pct', 'rsi', 'body_pct', 'ret_1m', 'ret_2m', 'vol_ratio', 'month']

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
    """Screen for signals from both strategies."""
    breakout_signals = []
    mean_reversion_signals = []

    for symbol, daily_df in all_data.items():
        monthly = calculate_indicators(daily_df)
        if monthly is None or len(monthly) < 2:
            continue

        last_month = monthly.iloc[-1]
        last_month_end = last_month.name
        current_daily = daily_df[daily_df.index > last_month_end]

        if len(current_daily) == 0:
            continue

        last_day = current_daily.iloc[-1]

        monthly['ret_1m'] = monthly['close'].pct_change() * 100
        monthly['ret_2m'] = monthly['close'].pct_change(2) * 100
        monthly['vol_ratio'] = monthly['volume'] / monthly['volume'].rolling(3).mean()

        atr_pct = last_month['atr_pct'] if pd.notna(last_month['atr_pct']) else 0
        rsi = last_month['rsi'] if pd.notna(last_month['rsi']) else 50

        # =====================
        # STRATEGY 1: BREAKOUT
        # =====================
        if last_month['is_green'] and atr_pct > 3 and rsi > 55:
            prev_month_high = last_month['high']
            today_broke = last_day['close'] > prev_month_high

            if len(current_daily) > 1:
                previous_days = current_daily.iloc[:-1]
                already_broke = any(previous_days['close'] > prev_month_high)
            else:
                already_broke = False

            just_triggered = today_broke and not already_broke

            if just_triggered:
                ml_prob = None
                if ml_model is not None and feature_cols is not None:
                    features = {
                        'atr_pct': atr_pct,
                        'rsi': rsi,
                        'body_pct': last_month['body_pct'] if pd.notna(last_month['body_pct']) else 0,
                        'ret_1m': monthly.iloc[-1]['ret_1m'] if pd.notna(monthly.iloc[-1]['ret_1m']) else 0,
                        'ret_2m': monthly.iloc[-1]['ret_2m'] if pd.notna(monthly.iloc[-1]['ret_2m']) else 0,
                        'vol_ratio': monthly.iloc[-1]['vol_ratio'] if pd.notna(monthly.iloc[-1]['vol_ratio']) else 1,
                        'month': last_day.name.month,
                    }
                    X = pd.DataFrame([features])[feature_cols].fillna(0)
                    ml_prob = ml_model.predict_proba(X)[0, 1]

                breakout_signals.append({
                    'symbol': symbol,
                    'trigger_date': last_day.name,
                    'close': last_day['close'],
                    'target': prev_month_high,
                    'break_pct': (last_day['close'] - prev_month_high) / prev_month_high * 100,
                    'atr_pct': atr_pct,
                    'rsi': rsi,
                    'ml_prob': ml_prob,
                })

        # ==========================
        # STRATEGY 2: MEAN REVERSION
        # ==========================
        if rsi < 30 and atr_pct > 3:
            prev_month_low = last_month['low']
            today_broke = last_day['close'] < prev_month_low

            if len(current_daily) > 1:
                previous_days = current_daily.iloc[:-1]
                already_broke = any(previous_days['close'] < prev_month_low)
            else:
                already_broke = False

            just_triggered = today_broke and not already_broke

            if just_triggered:
                mean_reversion_signals.append({
                    'symbol': symbol,
                    'trigger_date': last_day.name,
                    'close': last_day['close'],
                    'target': prev_month_low,
                    'break_pct': (prev_month_low - last_day['close']) / prev_month_low * 100,
                    'atr_pct': atr_pct,
                    'rsi': rsi,
                })

    return pd.DataFrame(breakout_signals), pd.DataFrame(mean_reversion_signals)


def print_screener(breakout_df, mean_reversion_df):
    """Print screener results."""
    print('\n' + '=' * 100)
    print('MONTHLY SCREENER - Signals That JUST Triggered Today')
    print('=' * 100)

    # BREAKOUT SIGNALS
    print('\n' + '-' * 100)
    print('STRATEGY 1: BREAKOUT (Momentum)')
    print('Filter: GREEN month + ATR% > 3 + RSI > 55 + Close > Prev Month HIGH')
    print('Action: BUY tomorrow OPEN, hold 6 months | Historical: 72% WR, +923% return')
    print('-' * 100)

    if len(breakout_df) == 0:
        print('\nNo new BREAKOUT signals today.')
    else:
        breakout_df = breakout_df.sort_values('ml_prob', ascending=False)
        print(f'\n{"Symbol":<6} {"Date":<12} {"Close":>9} {"Target":>9} {"Break%":>8} {"ATR%":>6} {"RSI":>5} {"ML Prob":>8}')
        print('-' * 75)

        for _, s in breakout_df.iterrows():
            ml_str = f"{s['ml_prob']*100:>6.1f}%" if pd.notna(s['ml_prob']) else "   N/A"
            print(f"{s['symbol']:<6} {s['trigger_date'].strftime('%Y-%m-%d'):<12} "
                  f"${s['close']:>8.2f} ${s['target']:>8.2f} "
                  f"{s['break_pct']:>+7.2f}% {s['atr_pct']:>5.1f}% {s['rsi']:>5.0f} {ml_str}")

        print(f'\nTotal: {len(breakout_df)} BREAKOUT signal(s)')

    # MEAN REVERSION SIGNALS
    print('\n' + '-' * 100)
    print('STRATEGY 2: MEAN REVERSION (Oversold Bounce)')
    print('Filter: RSI < 30 + ATR% > 3 + Close < Prev Month LOW')
    print('Action: BUY tomorrow OPEN, hold 6 months | Historical: 92% WR, +227% return')
    print('-' * 100)

    if len(mean_reversion_df) == 0:
        print('\nNo new MEAN REVERSION signals today.')
    else:
        mean_reversion_df = mean_reversion_df.sort_values('rsi')
        print(f'\n{"Symbol":<6} {"Date":<12} {"Close":>9} {"Target":>9} {"Break%":>8} {"ATR%":>6} {"RSI":>5}')
        print('-' * 65)

        for _, s in mean_reversion_df.iterrows():
            print(f"{s['symbol']:<6} {s['trigger_date'].strftime('%Y-%m-%d'):<12} "
                  f"${s['close']:>8.2f} ${s['target']:>8.2f} "
                  f"{s['break_pct']:>+7.2f}% {s['atr_pct']:>5.1f}% {s['rsi']:>5.0f}")

        print(f'\nTotal: {len(mean_reversion_df)} MEAN REVERSION signal(s)')

    total_signals = len(breakout_df) + len(mean_reversion_df)
    print('\n' + '=' * 100)
    print(f'TOTAL: {total_signals} signal(s) today')
    if total_signals > 0:
        print('\nAction: BUY at tomorrow\'s OPEN, hold for 2 months')
    print('=' * 100)


def run_backtest(all_data, year=2025):
    """Run backtest for both strategies."""
    breakout_trades = []
    mean_reversion_trades = []

    for symbol, daily_df in tqdm(all_data.items(), desc='Backtesting'):
        monthly = calculate_indicators(daily_df)
        if monthly is None:
            continue

        for i in range(1, len(monthly)):
            prev_month = monthly.iloc[i - 1]
            curr_month_end = monthly.index[i]
            prev_month_end = prev_month.name

            atr_pct = prev_month['atr_pct'] if pd.notna(prev_month['atr_pct']) else 0
            rsi = prev_month['rsi'] if pd.notna(prev_month['rsi']) else 50

            month_daily = daily_df[(daily_df.index > prev_month_end) &
                                   (daily_df.index <= curr_month_end)]

            if len(month_daily) < 2:
                continue

            # BREAKOUT
            if prev_month['is_green'] and atr_pct > 3 and rsi > 55:
                prev_month_high = prev_month['high']

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
                            exit_target = entry_date + pd.DateOffset(months=6)
                            future_data = daily_df[daily_df.index > entry_date]
                            future_data = future_data[future_data.index <= exit_target]

                            if len(future_data) > 0:
                                exit_price = future_data.iloc[-1]['close']
                                pct_return = (exit_price - entry_price) / entry_price * 100
                                breakout_trades.append({'pct_return': pct_return})
                        break

            # MEAN REVERSION
            if rsi < 30 and atr_pct > 3:
                prev_month_low = prev_month['low']

                for j in range(len(month_daily) - 1):
                    day = month_daily.iloc[j]
                    if day['close'] < prev_month_low:
                        next_day_idx = j + 1
                        if next_day_idx >= len(month_daily):
                            break

                        entry_day = month_daily.iloc[next_day_idx]
                        entry_price = entry_day['open']
                        entry_date = entry_day.name

                        if entry_date.year == year:
                            exit_target = entry_date + pd.DateOffset(months=6)
                            future_data = daily_df[daily_df.index > entry_date]
                            future_data = future_data[future_data.index <= exit_target]

                            if len(future_data) > 0:
                                exit_price = future_data.iloc[-1]['close']
                                pct_return = (exit_price - entry_price) / entry_price * 100
                                mean_reversion_trades.append({'pct_return': pct_return})
                        break

    return pd.DataFrame(breakout_trades), pd.DataFrame(mean_reversion_trades)


def calc_metrics(df):
    if len(df) == 0:
        return {'trades': 0, 'win_rate': 0, 'pf': 0, 'total_ret': 0}
    wins = df[df['pct_return'] > 0]
    losses = df[df['pct_return'] <= 0]
    win_rate = len(wins) / len(df) * 100
    total_ret = df['pct_return'].sum()
    gross_profit = wins['pct_return'].sum() if len(wins) > 0 else 0
    gross_loss = abs(losses['pct_return'].sum()) if len(losses) > 0 else 0.001
    pf = gross_profit / gross_loss
    return {'trades': len(df), 'win_rate': win_rate, 'pf': pf, 'total_ret': total_ret}


if __name__ == '__main__':
    import sys
    fresh_data = '--fresh' in sys.argv or '-f' in sys.argv

    print('=' * 100)
    print('MONTHLY BREAKOUT & MEAN REVERSION SCREENER')
    print('=' * 100)

    if fresh_data:
        print('\n>>> FRESH DATA MODE <<<')
        all_data = download_fresh_data(get_sp500_tickers())
    else:
        print('\nLoading cached data...')
        all_data = load_data()
        print(f'Loaded {len(all_data)} symbols')

    # Backtest
    print('\nRunning backtest (2025)...')
    breakout_trades, mr_trades = run_backtest(all_data, year=2025)

    print('\n' + '-' * 100)
    print('2025 BACKTEST RESULTS')
    print('-' * 100)

    m1 = calc_metrics(breakout_trades)
    print(f"\nBREAKOUT:        {m1['trades']:>3} trades | {m1['win_rate']:>5.1f}% WR | {m1['pf']:>5.2f} PF | {m1['total_ret']:>+7.1f}% return")

    m2 = calc_metrics(mr_trades)
    print(f"MEAN REVERSION:  {m2['trades']:>3} trades | {m2['win_rate']:>5.1f}% WR | {m2['pf']:>5.2f} PF | {m2['total_ret']:>+7.1f}% return")

    combined = pd.concat([breakout_trades, mr_trades], ignore_index=True)
    m3 = calc_metrics(combined)
    print(f"COMBINED:        {m3['trades']:>3} trades | {m3['win_rate']:>5.1f}% WR | {m3['pf']:>5.2f} PF | {m3['total_ret']:>+7.1f}% return")

    # Train ML
    print('\nTraining ML model...')
    training_df = generate_training_data(all_data)
    if len(training_df) > 50:
        ml_model, feature_cols = train_ml_model(training_df)
        print(f'ML model trained on {len(training_df)} samples')
    else:
        ml_model, feature_cols = None, None

    # Screener
    print('\nRunning screener...')
    breakout_signals, mr_signals = run_screener(all_data, ml_model, feature_cols)
    print_screener(breakout_signals, mr_signals)

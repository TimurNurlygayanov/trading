#!/usr/bin/env python3
"""
Analyze trades from the strategy to find patterns in bad deals.
"""

import sys
from pathlib import Path
from datetime import datetime
import warnings

import pandas as pd
import numpy as np

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
warnings.filterwarnings('ignore')

from backtest import (
    PolygonClient, get_sp500_tickers, filter_us_exchange_stocks,
    calculate_signals, POLYGON_API_KEY
)


def get_sector_info():
    """Approximate sector mapping for S&P 500 stocks."""
    return {
        # Technology
        'AAPL': 'Tech', 'MSFT': 'Tech', 'NVDA': 'Tech', 'GOOGL': 'Tech', 'GOOG': 'Tech',
        'META': 'Tech', 'AVGO': 'Tech', 'ORCL': 'Tech', 'AMD': 'Tech', 'CRM': 'Tech',
        'ADBE': 'Tech', 'CSCO': 'Tech', 'ACN': 'Tech', 'IBM': 'Tech', 'INTC': 'Tech',
        'QCOM': 'Tech', 'TXN': 'Tech', 'INTU': 'Tech', 'NOW': 'Tech', 'AMAT': 'Tech',
        'ADI': 'Tech', 'LRCX': 'Tech', 'MU': 'Tech', 'KLAC': 'Tech', 'SNPS': 'Tech',
        'CDNS': 'Tech', 'MRVL': 'Tech', 'NXPI': 'Tech', 'MPWR': 'Tech', 'ON': 'Tech',
        'ANET': 'Tech', 'CRWD': 'Tech', 'PANW': 'Tech', 'FTNT': 'Tech',
        # Semiconductors (subset of tech)
        # Consumer Discretionary
        'AMZN': 'Consumer', 'TSLA': 'Consumer', 'HD': 'Consumer', 'MCD': 'Consumer',
        'NKE': 'Consumer', 'SBUX': 'Consumer', 'LOW': 'Consumer', 'TJX': 'Consumer',
        'BKNG': 'Consumer', 'CMG': 'Consumer', 'ORLY': 'Consumer', 'AZO': 'Consumer',
        'ROST': 'Consumer', 'DHI': 'Consumer', 'LEN': 'Consumer', 'GM': 'Consumer',
        'F': 'Consumer', 'ABNB': 'Consumer', 'MAR': 'Consumer', 'HLT': 'Consumer',
        'CCL': 'Consumer', 'RCL': 'Consumer', 'NCLH': 'Consumer', 'LVS': 'Consumer',
        'MGM': 'Consumer', 'WYNN': 'Consumer',
        # Healthcare
        'LLY': 'Health', 'UNH': 'Health', 'JNJ': 'Health', 'ABBV': 'Health', 'MRK': 'Health',
        'PFE': 'Health', 'TMO': 'Health', 'ABT': 'Health', 'DHR': 'Health', 'BMY': 'Health',
        'AMGN': 'Health', 'GILD': 'Health', 'VRTX': 'Health', 'REGN': 'Health', 'ISRG': 'Health',
        'BSX': 'Health', 'MDT': 'Health', 'SYK': 'Health', 'ZTS': 'Health', 'CI': 'Health',
        'ELV': 'Health', 'HUM': 'Health', 'CVS': 'Health', 'MCK': 'Health', 'MRNA': 'Health',
        'BIIB': 'Health', 'DXCM': 'Health', 'IDXX': 'Health', 'ALGN': 'Health',
        # Financials
        'BRK-B': 'Finance', 'JPM': 'Finance', 'V': 'Finance', 'MA': 'Finance', 'BAC': 'Finance',
        'WFC': 'Finance', 'GS': 'Finance', 'MS': 'Finance', 'SPGI': 'Finance', 'BLK': 'Finance',
        'C': 'Finance', 'AXP': 'Finance', 'SCHW': 'Finance', 'CB': 'Finance', 'MMC': 'Finance',
        'PNC': 'Finance', 'USB': 'Finance', 'TFC': 'Finance', 'AIG': 'Finance', 'MET': 'Finance',
        'PRU': 'Finance', 'ALL': 'Finance', 'AFL': 'Finance', 'COF': 'Finance', 'BX': 'Finance',
        'KKR': 'Finance', 'CME': 'Finance', 'ICE': 'Finance', 'PYPL': 'Finance',
        # Energy
        'XOM': 'Energy', 'CVX': 'Energy', 'COP': 'Energy', 'SLB': 'Energy', 'EOG': 'Energy',
        'MPC': 'Energy', 'PSX': 'Energy', 'VLO': 'Energy', 'OXY': 'Energy', 'KMI': 'Energy',
        'WMB': 'Energy', 'HES': 'Energy', 'DVN': 'Energy', 'HAL': 'Energy', 'BKR': 'Energy',
        'FANG': 'Energy', 'OKE': 'Energy', 'TRGP': 'Energy',
        # Industrials
        'GE': 'Industrial', 'CAT': 'Industrial', 'RTX': 'Industrial', 'HON': 'Industrial',
        'UNP': 'Industrial', 'UPS': 'Industrial', 'BA': 'Industrial', 'DE': 'Industrial',
        'LMT': 'Industrial', 'GD': 'Industrial', 'NOC': 'Industrial', 'MMM': 'Industrial',
        'ETN': 'Industrial', 'ITW': 'Industrial', 'EMR': 'Industrial', 'FDX': 'Industrial',
        'CSX': 'Industrial', 'NSC': 'Industrial', 'WM': 'Industrial', 'RSG': 'Industrial',
        'DAL': 'Industrial', 'UAL': 'Industrial', 'LUV': 'Industrial', 'AAL': 'Industrial',
        # Materials
        'LIN': 'Materials', 'APD': 'Materials', 'SHW': 'Materials', 'ECL': 'Materials',
        'NEM': 'Materials', 'FCX': 'Materials', 'NUE': 'Materials', 'DOW': 'Materials',
        'DD': 'Materials', 'PPG': 'Materials', 'VMC': 'Materials', 'MLM': 'Materials',
        'ALB': 'Materials', 'CF': 'Materials', 'MOS': 'Materials',
        # Utilities
        'NEE': 'Utilities', 'DUK': 'Utilities', 'SO': 'Utilities', 'D': 'Utilities',
        'AEP': 'Utilities', 'EXC': 'Utilities', 'SRE': 'Utilities', 'XEL': 'Utilities',
        'PEG': 'Utilities', 'ED': 'Utilities', 'WEC': 'Utilities', 'ES': 'Utilities',
        'AWK': 'Utilities', 'DTE': 'Utilities', 'ETR': 'Utilities', 'PPL': 'Utilities',
        'ATO': 'Utilities', 'FE': 'Utilities', 'CEG': 'Utilities', 'PCG': 'Utilities',
        # Real Estate
        'AMT': 'RealEstate', 'PLD': 'RealEstate', 'CCI': 'RealEstate', 'EQIX': 'RealEstate',
        'PSA': 'RealEstate', 'SPG': 'RealEstate', 'O': 'RealEstate', 'WELL': 'RealEstate',
        'DLR': 'RealEstate', 'AVB': 'RealEstate', 'EQR': 'RealEstate', 'VTR': 'RealEstate',
        'SBAC': 'RealEstate', 'ARE': 'RealEstate', 'MAA': 'RealEstate', 'UDR': 'RealEstate',
        'ESS': 'RealEstate', 'KIM': 'RealEstate', 'REG': 'RealEstate', 'HST': 'RealEstate',
        # Consumer Staples
        'PG': 'Staples', 'KO': 'Staples', 'PEP': 'Staples', 'WMT': 'Staples', 'COST': 'Staples',
        'PM': 'Staples', 'MO': 'Staples', 'MDLZ': 'Staples', 'CL': 'Staples', 'KMB': 'Staples',
        'GIS': 'Staples', 'K': 'Staples', 'HSY': 'Staples', 'SYY': 'Staples', 'KR': 'Staples',
        'STZ': 'Staples', 'MKC': 'Staples', 'CHD': 'Staples', 'CLX': 'Staples', 'KHC': 'Staples',
        'CAG': 'Staples', 'CPB': 'Staples', 'SJM': 'Staples', 'HRL': 'Staples', 'TAP': 'Staples',
        # Communication
        'NFLX': 'Comm', 'DIS': 'Comm', 'CMCSA': 'Comm', 'T': 'Comm', 'VZ': 'Comm',
        'TMUS': 'Comm', 'CHTR': 'Comm', 'EA': 'Comm', 'TTWO': 'Comm', 'WBD': 'Comm',
        'PARA': 'Comm', 'FOX': 'Comm', 'FOXA': 'Comm', 'NWS': 'Comm', 'NWSA': 'Comm',
        'LYV': 'Comm', 'OMC': 'Comm', 'IPG': 'Comm', 'MTCH': 'Comm',
    }


def analyze_trades():
    print("\n" + "=" * 80)
    print("TRADE ANALYSIS - FINDING PATTERNS IN BAD DEALS")
    print("=" * 80)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

    polygon_client = PolygonClient(POLYGON_API_KEY)
    sectors = get_sector_info()

    # Configuration
    MIN_PRICE = 10
    MAX_PRICE = 400
    HOLD_DAYS = 252

    # Get data
    print("\n[1/3] Loading data...")
    universe = get_sp500_tickers()
    universe = filter_us_exchange_stocks(universe, POLYGON_API_KEY)
    print(f"  {len(universe)} stocks")

    end_date = datetime.now().strftime('%Y-%m-%d')
    prices = polygon_client.fetch_prices_parallel(universe, '2010-01-01', end_date)
    prices['SPY'] = polygon_client.get_prices('SPY', '2010-01-01', end_date)
    print(f"  Got prices for {len(prices)} stocks")

    # Collect all trades with detailed info
    print("\n[2/3] Collecting trade details...")

    train_start = pd.to_datetime('2010-01-01')
    train_end = pd.to_datetime('2024-12-31')
    test_start = pd.to_datetime('2025-01-01')

    all_trades = []

    for symbol, df in prices.items():
        if symbol == 'SPY' or len(df) < 500:
            continue

        df = calculate_signals(df)
        df['fwd_return'] = df['adjusted_close'].shift(-HOLD_DAYS) / df['adjusted_close'] - 1

        # Get SPY data for market context
        spy_df = prices.get('SPY')

        # Process both training and test signals
        for period_name, start_dt, end_dt in [('train', train_start, train_end), ('test', test_start, None)]:
            if end_dt:
                period_df = df[(df.index >= start_dt) & (df.index <= end_dt)]
            else:
                period_df = df[df.index >= start_dt]

            period_df = period_df[(period_df['adjusted_close'] >= MIN_PRICE) &
                                   (period_df['adjusted_close'] <= MAX_PRICE)]

            for signal_type in ['optimal_signal', 'dd_rsi_combo']:
                signal_df = period_df[period_df[signal_type] == True]

                for date in signal_df.index:
                    entry_price = df.loc[date, 'adjusted_close']

                    # Calculate exit
                    future_dates = df.index[df.index > date]
                    if len(future_dates) >= HOLD_DAYS:
                        exit_date = future_dates[HOLD_DAYS - 1]
                        exit_price = df.loc[exit_date, 'adjusted_close']
                        ret = exit_price / entry_price - 1
                        is_complete = True
                    elif len(future_dates) > 0:
                        exit_date = future_dates[-1]
                        exit_price = df.loc[exit_date, 'adjusted_close']
                        ret = exit_price / entry_price - 1
                        is_complete = False
                    else:
                        continue

                    # Get additional features at entry
                    row = df.loc[date]

                    # Market context (SPY)
                    spy_return_20d = None
                    spy_return_60d = None
                    if spy_df is not None and date in spy_df.index:
                        spy_idx = spy_df.index.get_loc(date)
                        if spy_idx >= 20:
                            spy_return_20d = spy_df['adjusted_close'].iloc[spy_idx] / spy_df['adjusted_close'].iloc[spy_idx - 20] - 1
                        if spy_idx >= 60:
                            spy_return_60d = spy_df['adjusted_close'].iloc[spy_idx] / spy_df['adjusted_close'].iloc[spy_idx - 60] - 1

                    # Day of week, month
                    entry_dow = date.dayofweek
                    entry_month = date.month

                    # Recent price action
                    idx = df.index.get_loc(date)
                    if idx >= 5:
                        price_change_5d = df['adjusted_close'].iloc[idx] / df['adjusted_close'].iloc[idx - 5] - 1
                    else:
                        price_change_5d = None

                    if idx >= 20:
                        price_change_20d = df['adjusted_close'].iloc[idx] / df['adjusted_close'].iloc[idx - 20] - 1
                    else:
                        price_change_20d = None

                    all_trades.append({
                        'symbol': symbol,
                        'sector': sectors.get(symbol, 'Other'),
                        'signal': signal_type,
                        'period': period_name,
                        'entry_date': date,
                        'entry_price': entry_price,
                        'exit_date': exit_date,
                        'exit_price': exit_price,
                        'return': ret,
                        'is_complete': is_complete,
                        'is_winner': ret > 0,
                        # Entry conditions
                        'rsi': row.get('rsi'),
                        'dist_from_high': row.get('dist_from_high'),
                        'dist_from_ema200': row.get('dist_from_ema200'),
                        'atr_14': row.get('atr_14'),
                        'atr_pct': row.get('atr_14') / entry_price * 100 if row.get('atr_14') else None,
                        # Price action
                        'price_change_5d': price_change_5d,
                        'price_change_20d': price_change_20d,
                        # Market context
                        'spy_return_20d': spy_return_20d,
                        'spy_return_60d': spy_return_60d,
                        # Timing
                        'entry_dow': entry_dow,
                        'entry_month': entry_month,
                        'entry_year': date.year,
                    })

    trades_df = pd.DataFrame(all_trades)
    print(f"  Total trades collected: {len(trades_df)}")

    # Analyze patterns
    print("\n[3/3] Analyzing patterns...")

    # Focus on optimal signal for cleaner analysis
    optimal_df = trades_df[trades_df['signal'] == 'optimal_signal'].copy()
    losers = optimal_df[optimal_df['return'] < 0]
    winners = optimal_df[optimal_df['return'] > 0]

    print(f"\n{'='*80}")
    print("OPTIMAL SIGNAL ANALYSIS")
    print(f"{'='*80}")
    print(f"Total trades: {len(optimal_df)}")
    print(f"Winners: {len(winners)} ({len(winners)/len(optimal_df)*100:.1f}%)")
    print(f"Losers: {len(losers)} ({len(losers)/len(optimal_df)*100:.1f}%)")
    print(f"Avg winner return: {winners['return'].mean()*100:+.1f}%")
    print(f"Avg loser return: {losers['return'].mean()*100:+.1f}%")

    # ========== SECTOR ANALYSIS ==========
    print(f"\n{'='*80}")
    print("SECTOR ANALYSIS")
    print(f"{'='*80}")
    sector_stats = optimal_df.groupby('sector').agg({
        'return': ['count', 'mean', lambda x: (x > 0).mean()],
        'symbol': 'nunique'
    }).round(3)
    sector_stats.columns = ['trades', 'avg_return', 'win_rate', 'stocks']
    sector_stats = sector_stats.sort_values('avg_return', ascending=False)
    print(f"\n{'Sector':<12} {'Trades':>8} {'Stocks':>8} {'Win Rate':>10} {'Avg Return':>12}")
    print("-" * 55)
    for sector, row in sector_stats.iterrows():
        print(f"{sector:<12} {int(row['trades']):>8} {int(row['stocks']):>8} {row['win_rate']*100:>9.1f}% {row['avg_return']*100:>+11.1f}%")

    # ========== RSI ANALYSIS ==========
    print(f"\n{'='*80}")
    print("RSI AT ENTRY ANALYSIS")
    print(f"{'='*80}")
    optimal_df['rsi_bucket'] = pd.cut(optimal_df['rsi'], bins=[0, 15, 20, 25, 30, 35], labels=['0-15', '15-20', '20-25', '25-30', '30-35'])
    rsi_stats = optimal_df.groupby('rsi_bucket').agg({
        'return': ['count', 'mean', lambda x: (x > 0).mean()]
    })
    rsi_stats.columns = ['trades', 'avg_return', 'win_rate']
    print(f"\n{'RSI Range':<12} {'Trades':>8} {'Win Rate':>10} {'Avg Return':>12}")
    print("-" * 45)
    for bucket, row in rsi_stats.iterrows():
        if pd.notna(bucket):
            print(f"{bucket:<12} {int(row['trades']):>8} {row['win_rate']*100:>9.1f}% {row['avg_return']*100:>+11.1f}%")

    # ========== DRAWDOWN DEPTH ANALYSIS ==========
    print(f"\n{'='*80}")
    print("DRAWDOWN DEPTH ANALYSIS (Distance from 52-week high)")
    print(f"{'='*80}")
    optimal_df['dd_bucket'] = pd.cut(optimal_df['dist_from_high'], bins=[0.2, 0.3, 0.4, 0.5, 0.6, 1.0], labels=['20-30%', '30-40%', '40-50%', '50-60%', '60%+'])
    dd_stats = optimal_df.groupby('dd_bucket').agg({
        'return': ['count', 'mean', lambda x: (x > 0).mean()]
    })
    dd_stats.columns = ['trades', 'avg_return', 'win_rate']
    print(f"\n{'Drawdown':<12} {'Trades':>8} {'Win Rate':>10} {'Avg Return':>12}")
    print("-" * 45)
    for bucket, row in dd_stats.iterrows():
        if pd.notna(bucket):
            print(f"{bucket:<12} {int(row['trades']):>8} {row['win_rate']*100:>9.1f}% {row['avg_return']*100:>+11.1f}%")

    # ========== EMA200 DISTANCE ANALYSIS ==========
    print(f"\n{'='*80}")
    print("EMA200 DISTANCE ANALYSIS")
    print(f"{'='*80}")
    optimal_df['ema_bucket'] = pd.cut(optimal_df['dist_from_ema200'], bins=[-100, -50, -40, -30, -20, 0], labels=['-50%+', '-40 to -50%', '-30 to -40%', '-20 to -30%', '> -20%'])
    ema_stats = optimal_df.groupby('ema_bucket').agg({
        'return': ['count', 'mean', lambda x: (x > 0).mean()]
    })
    ema_stats.columns = ['trades', 'avg_return', 'win_rate']
    print(f"\n{'EMA200 Dist':<15} {'Trades':>8} {'Win Rate':>10} {'Avg Return':>12}")
    print("-" * 50)
    for bucket, row in ema_stats.iterrows():
        if pd.notna(bucket):
            print(f"{bucket:<15} {int(row['trades']):>8} {row['win_rate']*100:>9.1f}% {row['avg_return']*100:>+11.1f}%")

    # ========== MARKET CONTEXT ANALYSIS ==========
    print(f"\n{'='*80}")
    print("MARKET CONTEXT ANALYSIS (SPY 20-day return at entry)")
    print(f"{'='*80}")
    optimal_df['spy_bucket'] = pd.cut(optimal_df['spy_return_20d'], bins=[-1, -0.10, -0.05, 0, 0.05, 0.10, 1], labels=['< -10%', '-10 to -5%', '-5 to 0%', '0 to 5%', '5 to 10%', '> 10%'])
    spy_stats = optimal_df.groupby('spy_bucket').agg({
        'return': ['count', 'mean', lambda x: (x > 0).mean()]
    })
    spy_stats.columns = ['trades', 'avg_return', 'win_rate']
    print(f"\n{'SPY 20d Ret':<15} {'Trades':>8} {'Win Rate':>10} {'Avg Return':>12}")
    print("-" * 50)
    for bucket, row in spy_stats.iterrows():
        if pd.notna(bucket) and row['trades'] > 0:
            print(f"{bucket:<15} {int(row['trades']):>8} {row['win_rate']*100:>9.1f}% {row['avg_return']*100:>+11.1f}%")

    # ========== VOLATILITY ANALYSIS ==========
    print(f"\n{'='*80}")
    print("VOLATILITY ANALYSIS (ATR % of price)")
    print(f"{'='*80}")
    optimal_df['atr_bucket'] = pd.cut(optimal_df['atr_pct'], bins=[0, 2, 4, 6, 10, 100], labels=['0-2%', '2-4%', '4-6%', '6-10%', '10%+'])
    atr_stats = optimal_df.groupby('atr_bucket').agg({
        'return': ['count', 'mean', lambda x: (x > 0).mean()]
    })
    atr_stats.columns = ['trades', 'avg_return', 'win_rate']
    print(f"\n{'ATR %':<12} {'Trades':>8} {'Win Rate':>10} {'Avg Return':>12}")
    print("-" * 45)
    for bucket, row in atr_stats.iterrows():
        if pd.notna(bucket) and row['trades'] > 0:
            print(f"{bucket:<12} {int(row['trades']):>8} {row['win_rate']*100:>9.1f}% {row['avg_return']*100:>+11.1f}%")

    # ========== RECENT MOMENTUM ANALYSIS ==========
    print(f"\n{'='*80}")
    print("RECENT MOMENTUM ANALYSIS (5-day price change at entry)")
    print(f"{'='*80}")
    optimal_df['mom5_bucket'] = pd.cut(optimal_df['price_change_5d'], bins=[-1, -0.15, -0.10, -0.05, 0, 0.05, 1], labels=['< -15%', '-15 to -10%', '-10 to -5%', '-5 to 0%', '0 to 5%', '> 5%'])
    mom_stats = optimal_df.groupby('mom5_bucket').agg({
        'return': ['count', 'mean', lambda x: (x > 0).mean()]
    })
    mom_stats.columns = ['trades', 'avg_return', 'win_rate']
    print(f"\n{'5d Change':<15} {'Trades':>8} {'Win Rate':>10} {'Avg Return':>12}")
    print("-" * 50)
    for bucket, row in mom_stats.iterrows():
        if pd.notna(bucket) and row['trades'] > 0:
            print(f"{bucket:<15} {int(row['trades']):>8} {row['win_rate']*100:>9.1f}% {row['avg_return']*100:>+11.1f}%")

    # ========== MONTH ANALYSIS ==========
    print(f"\n{'='*80}")
    print("ENTRY MONTH ANALYSIS")
    print(f"{'='*80}")
    month_names = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
                   7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}
    month_stats = optimal_df.groupby('entry_month').agg({
        'return': ['count', 'mean', lambda x: (x > 0).mean()]
    })
    month_stats.columns = ['trades', 'avg_return', 'win_rate']
    print(f"\n{'Month':<8} {'Trades':>8} {'Win Rate':>10} {'Avg Return':>12}")
    print("-" * 40)
    for month, row in month_stats.iterrows():
        print(f"{month_names[month]:<8} {int(row['trades']):>8} {row['win_rate']*100:>9.1f}% {row['avg_return']*100:>+11.1f}%")

    # ========== LOSING TRADES DETAIL ==========
    print(f"\n{'='*80}")
    print("ALL LOSING TRADES (OPTIMAL SIGNAL)")
    print(f"{'='*80}")
    losers_sorted = losers.sort_values('return')
    print(f"\n{'Symbol':<8} {'Sector':<10} {'Entry':>12} {'Return':>10} {'RSI':>6} {'DD%':>8} {'EMA200':>10}")
    print("-" * 75)
    for _, t in losers_sorted.iterrows():
        print(f"{t['symbol']:<8} {t['sector']:<10} {t['entry_date'].strftime('%Y-%m-%d'):>12} "
              f"{t['return']*100:>+9.1f}% {t['rsi']:>5.1f} {t['dist_from_high']*100:>7.1f}% {t['dist_from_ema200']:>+9.1f}%")

    # ========== PATTERN SUMMARY ==========
    print(f"\n{'='*80}")
    print("PATTERN SUMMARY - POTENTIAL FILTERS")
    print(f"{'='*80}")

    # Find worst performing categories
    print("\nWORST PERFORMERS TO POTENTIALLY FILTER:")
    print("-" * 50)

    # Sectors
    worst_sectors = sector_stats[sector_stats['win_rate'] < 0.8].index.tolist()
    if worst_sectors:
        print(f"  Sectors with <80% win rate: {', '.join(worst_sectors)}")

    # RSI ranges
    worst_rsi = rsi_stats[rsi_stats['win_rate'] < optimal_df['is_winner'].mean()].index.tolist()
    if worst_rsi:
        print(f"  RSI ranges below average: {', '.join([str(x) for x in worst_rsi if pd.notna(x)])}")

    print(f"\n{'='*80}")


if __name__ == '__main__':
    analyze_trades()

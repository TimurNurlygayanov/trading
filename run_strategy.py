#!/usr/bin/env python3
"""
Momentum + Fundamentals Strategy - EODHD ONLY

This version uses only EODHD API for both price and fundamental data.
No Polygon/Massive API required.

Features:
- Quarterly rebalancing
- Top 20 stocks by CatBoost prediction
- Transaction costs included
- Can be used for backtesting and live trading signals
"""

import os
import sys
from pathlib import Path
from datetime import datetime, timedelta
import warnings
import json
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import numpy as np
import requests
from catboost import CatBoostClassifier

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

warnings.filterwarnings('ignore')

# Load API keys
from api_config import get_eodhd_key, get_polygon_key
EODHD_API_KEY = get_eodhd_key()
POLYGON_API_KEY = get_polygon_key()

CACHE_DIR = project_root / "data_cache_eodhd"
CACHE_DIR.mkdir(exist_ok=True)

ROUND_TRIP_COST = 0.0024  # 0.24% transaction cost


class PolygonClient:
    """Polygon API client for price data. No rate limits."""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.polygon.io"

    def get_prices(self, symbol: str, start_date: str, end_date: str, force_refresh: bool = False) -> pd.DataFrame:
        """Get daily prices for a symbol.

        Args:
            force_refresh: If True, skip cache and fetch fresh data from API
        """
        cache_file = CACHE_DIR / f"polygon_{symbol}_{start_date}_{end_date}.pkl"

        if cache_file.exists() and not force_refresh:
            try:
                return pd.read_pickle(cache_file)
            except:
                pass

        url = f"{self.base_url}/v2/aggs/ticker/{symbol}/range/1/day/{start_date}/{end_date}"
        params = {'apiKey': self.api_key, 'adjusted': 'true', 'limit': 50000}

        try:
            resp = requests.get(url, params=params, timeout=30)
            if resp.status_code == 429:
                # Rate limited - wait and retry
                import time
                time.sleep(1)
                resp = requests.get(url, params=params, timeout=30)

            resp.raise_for_status()
            data = resp.json()

            if data.get('results'):
                df = pd.DataFrame(data['results'])
                df['date'] = pd.to_datetime(df['t'], unit='ms')
                df.set_index('date', inplace=True)
                df = df.rename(columns={
                    'o': 'open', 'h': 'high', 'l': 'low',
                    'c': 'close', 'v': 'volume'
                })
                df['adjusted_close'] = df['close']  # Polygon returns adjusted by default
                df.to_pickle(cache_file)
                return df
        except Exception as e:
            print(f"    Error fetching {symbol}: {e}")

        return pd.DataFrame()

    def fetch_prices_parallel(self, symbols: list, start_date: str, end_date: str, force_refresh: bool = False) -> dict:
        """Fetch prices for multiple symbols in parallel.

        Args:
            force_refresh: If True, skip cache and fetch fresh data from API
        """
        results = {}
        to_fetch = []

        for symbol in symbols:
            if force_refresh:
                to_fetch.append(symbol)
            else:
                cache_file = CACHE_DIR / f"polygon_{symbol}_{start_date}_{end_date}.pkl"
                if cache_file.exists():
                    try:
                        df = pd.read_pickle(cache_file)
                        if len(df) > 100:
                            results[symbol] = df
                            continue
                    except:
                        pass
                to_fetch.append(symbol)

        if force_refresh:
            print(f"  Fetching fresh prices for {len(to_fetch)} symbols...")
        else:
            print(f"  Prices: {len(results)} cached, {len(to_fetch)} to fetch")

        if to_fetch:
            def fetch_one(sym):
                return sym, self.get_prices(sym, start_date, end_date, force_refresh=force_refresh)

            with ThreadPoolExecutor(max_workers=20) as executor:
                futures = {executor.submit(fetch_one, s): s for s in to_fetch}

                for i, future in enumerate(as_completed(futures)):
                    if (i + 1) % 50 == 0:
                        print(f"    Progress: {i + 1}/{len(to_fetch)}")

                    symbol, df = future.result()
                    if df is not None and len(df) > 100:
                        results[symbol] = df

        return results


class EODHDClient:
    """EODHD API client for price and fundamental data."""

    def __init__(self, api_key: str, max_workers: int = 8):
        self.api_key = api_key
        self.base_url = "https://eodhd.com/api"
        self.max_workers = max_workers
        self.api_calls = 0
        self._last_request_time = 0
        self._min_request_interval = 0.07  # ~14 req/sec to stay under 1k/min

    def _rate_limit(self):
        """Respect rate limit of ~1000 requests per minute."""
        import time
        elapsed = time.time() - self._last_request_time
        if elapsed < self._min_request_interval:
            time.sleep(self._min_request_interval - elapsed)
        self._last_request_time = time.time()

    def _request(self, endpoint: str, params: dict = None) -> dict:
        """Make API request with rate limiting."""
        self._rate_limit()
        self.api_calls += 1

        params = params or {}
        params['api_token'] = self.api_key
        params['fmt'] = 'json'

        url = f"{self.base_url}/{endpoint}"
        try:
            resp = requests.get(url, params=params, timeout=30)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            return None

    def get_us_stocks(self) -> list:
        """Get all US common stocks from NYSE and NASDAQ."""
        cache_file = CACHE_DIR / "us_stocks_nyse_nasdaq.json"

        if cache_file.exists():
            with open(cache_file) as f:
                return json.load(f)

        print("  Fetching NYSE stocks...")
        nyse = self._request("exchange-symbol-list/NYSE") or []
        print(f"    NYSE: {len(nyse)} symbols")

        print("  Fetching NASDAQ stocks...")
        nasdaq = self._request("exchange-symbol-list/NASDAQ") or []
        print(f"    NASDAQ: {len(nasdaq)} symbols")

        # Filter to common stocks only
        stocks = [s for s in nyse + nasdaq if s.get('Type') == 'Common Stock']
        print(f"  Total common stocks: {len(stocks)}")

        with open(cache_file, 'w') as f:
            json.dump(stocks, f)
        return stocks

    def get_eod_price(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Get end-of-day price data for a symbol."""
        cache_file = CACHE_DIR / f"eod_{symbol}_{start_date}_{end_date}.pkl"

        if cache_file.exists():
            try:
                return pd.read_pickle(cache_file)
            except:
                pass

        data = self._request(f"eod/{symbol}.US", {
            'from': start_date,
            'to': end_date
        })

        if data and isinstance(data, list) and len(data) > 0:
            df = pd.DataFrame(data)
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            df.to_pickle(cache_file)
            return df

        return pd.DataFrame()

    def get_fundamentals(self, symbol: str, force_refresh: bool = False) -> dict:
        """Get fundamental data for a symbol.

        Smart caching: use cached data if fetched in current month.
        This saves API calls while keeping data reasonably fresh.
        """
        import os
        cache_file = CACHE_DIR / f"fund_{symbol}.json"

        if cache_file.exists() and not force_refresh:
            try:
                # Check if cache is from current month
                cache_mtime = datetime.fromtimestamp(os.path.getmtime(cache_file))
                now = datetime.now()
                if cache_mtime.year == now.year and cache_mtime.month == now.month:
                    # Fresh cache - use it
                    with open(cache_file) as f:
                        return json.load(f)
                # Cache is from previous month - will refetch
            except:
                pass

        data = self._request(f"fundamentals/{symbol}.US")

        if data and 'Highlights' in data:
            with open(cache_file, 'w') as f:
                json.dump(data, f)
            return data

        return None

    def fetch_prices_parallel(self, symbols: list, start_date: str, end_date: str) -> dict:
        """Fetch prices for multiple symbols in parallel."""
        results = {}
        to_fetch = []

        # Check cache first
        for symbol in symbols:
            cache_file = CACHE_DIR / f"eod_{symbol}_{start_date}_{end_date}.pkl"
            if cache_file.exists():
                try:
                    df = pd.read_pickle(cache_file)
                    if len(df) > 100:
                        results[symbol] = df
                        continue
                except:
                    pass
            to_fetch.append(symbol)

        print(f"  Prices: {len(results)} cached, {len(to_fetch)} to fetch")

        if to_fetch:
            def fetch_one(sym):
                return sym, self.get_eod_price(sym, start_date, end_date)

            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {executor.submit(fetch_one, s): s for s in to_fetch}

                for i, future in enumerate(as_completed(futures)):
                    if (i + 1) % 50 == 0:
                        print(f"    Progress: {i + 1}/{len(to_fetch)}")

                    symbol, df = future.result()
                    if df is not None and len(df) > 100:
                        results[symbol] = df

        return results

    def fetch_fundamentals_parallel(self, symbols: list) -> dict:
        """Fetch fundamentals for multiple symbols with rate limiting.

        Uses smart caching: only refetch if cache is from a previous month.
        """
        import os
        results = {}
        to_fetch = []
        now = datetime.now()

        for symbol in symbols:
            cache_file = CACHE_DIR / f"fund_{symbol}.json"
            if cache_file.exists():
                try:
                    # Check if cache is from current month
                    cache_mtime = datetime.fromtimestamp(os.path.getmtime(cache_file))
                    if cache_mtime.year == now.year and cache_mtime.month == now.month:
                        with open(cache_file) as f:
                            results[symbol] = json.load(f)
                            continue
                    # Cache is stale - need to refetch
                except:
                    pass
            to_fetch.append(symbol)

        print(f"  Fundamentals: {len(results)} fresh (this month), {len(to_fetch)} to fetch/refresh")

        if to_fetch:
            # Estimate time: ~14 req/sec = ~71 per minute
            est_minutes = len(to_fetch) / 800
            if len(to_fetch) > 100:
                print(f"  Estimated time: ~{est_minutes:.1f} minutes (rate limited to ~800/min)")

            # Sequential with rate limiting to respect API limits
            for i, symbol in enumerate(to_fetch):
                data = self.get_fundamentals(symbol)
                if data:
                    results[symbol] = data

                if (i + 1) % 100 == 0:
                    print(f"    Progress: {i + 1}/{len(to_fetch)} ({self.api_calls} API calls)")

        return results


def get_feature_columns():
    """Feature columns for the ML model."""
    return [
        'return_21d', 'return_63d', 'return_126d', 'return_252d',
        'volatility_21d', 'volatility_63d',
        'dist_from_high', 'dist_from_low',
        'price_to_sma_20', 'price_to_sma_50', 'price_to_sma_200',
        'rsi_14', 'volume_ratio', 'trend_strength',
        'eps', 'eps_estimate_current', 'eps_estimate_next', 'eps_growth_expected',
        'pe_ratio', 'peg_ratio', 'forward_pe',
        'profit_margin', 'roe', 'roa',
        'revenue_growth', 'earnings_growth',
        'price_to_book', 'price_to_sales',
        'dividend_yield',
        'last_earnings_surprise', 'avg_earnings_surprise',
        'days_since_earnings', 'log_market_cap',
    ]


def calculate_momentum_features(prices: pd.DataFrame) -> pd.DataFrame:
    """Calculate momentum features. NO FUTURE LEAKS."""
    df = prices.copy()

    for lookback in [21, 63, 126, 252]:
        df[f'return_{lookback}d'] = df['adjusted_close'].pct_change(lookback).shift(5)

    df['volatility_21d'] = df['adjusted_close'].pct_change().rolling(21).std()
    df['volatility_63d'] = df['adjusted_close'].pct_change().rolling(63).std()

    df['high_252d'] = df['high'].rolling(252).max()
    df['low_252d'] = df['low'].rolling(252).min()
    df['dist_from_high'] = (df['high_252d'] - df['adjusted_close']) / df['high_252d']
    df['dist_from_low'] = (df['adjusted_close'] - df['low_252d']) / df['low_252d']

    for period in [20, 50, 200]:
        df[f'sma_{period}'] = df['adjusted_close'].rolling(period).mean()
        df[f'price_to_sma_{period}'] = df['adjusted_close'] / df[f'sma_{period}'] - 1

    delta = df['adjusted_close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(span=14, adjust=False).mean()
    avg_loss = loss.ewm(span=14, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    df['rsi_14'] = 100 - (100 / (1 + rs))

    df['volume_sma_20'] = df['volume'].rolling(20).mean()
    df['volume_ratio'] = df['volume'] / (df['volume_sma_20'] + 1)

    df['trend_strength'] = abs(df['return_63d']) / (df['volatility_63d'] + 1e-10)

    return df


def extract_fundamentals(fund_data: dict, as_of_date: datetime = None) -> dict:
    """Extract fundamental features from EODHD data."""
    features = {}
    as_of_date = as_of_date or datetime.now()

    highlights = fund_data.get('Highlights', {})
    features['eps'] = highlights.get('EarningsShare')
    features['eps_estimate_current'] = highlights.get('EPSEstimateCurrentYear')
    features['eps_estimate_next'] = highlights.get('EPSEstimateNextYear')
    features['pe_ratio'] = highlights.get('PERatio')
    features['peg_ratio'] = highlights.get('PEGRatio')
    features['profit_margin'] = highlights.get('ProfitMargin')
    features['roe'] = highlights.get('ReturnOnEquityTTM')
    features['roa'] = highlights.get('ReturnOnAssetsTTM')
    features['revenue_growth'] = highlights.get('QuarterlyRevenueGrowthYOY')
    features['earnings_growth'] = highlights.get('QuarterlyEarningsGrowthYOY')
    features['market_cap'] = highlights.get('MarketCapitalization')
    features['dividend_yield'] = highlights.get('DividendYield')

    if features['eps'] and features['eps_estimate_next']:
        try:
            features['eps_growth_expected'] = (
                features['eps_estimate_next'] - features['eps']
            ) / abs(features['eps'] + 1e-10)
        except:
            features['eps_growth_expected'] = None
    else:
        features['eps_growth_expected'] = None

    valuation = fund_data.get('Valuation', {})
    features['forward_pe'] = valuation.get('ForwardPE')
    features['price_to_book'] = valuation.get('PriceBookMRQ')
    features['price_to_sales'] = valuation.get('PriceSalesTTM')

    # Earnings history
    earnings = fund_data.get('Earnings', {})
    if 'History' in earnings:
        history = earnings['History']
        recent = []
        for date_str, data in sorted(history.items(), reverse=True)[:4]:
            if data.get('epsActual') is not None:
                recent.append((date_str, data))

        if recent:
            latest_date, latest = recent[0]
            features['last_earnings_surprise'] = latest.get('surprisePercent', 0) or 0
            surprises = [e[1].get('surprisePercent', 0) or 0 for e in recent]
            features['avg_earnings_surprise'] = np.mean(surprises)

            try:
                report_date = pd.to_datetime(latest.get('reportDate') or latest_date)
                features['days_since_earnings'] = (as_of_date - report_date).days
            except:
                features['days_since_earnings'] = 999
        else:
            features['last_earnings_surprise'] = 0
            features['avg_earnings_surprise'] = 0
            features['days_since_earnings'] = 999
    else:
        features['last_earnings_surprise'] = 0
        features['avg_earnings_surprise'] = 0
        features['days_since_earnings'] = 999

    return features


def run_backtest(
    test_df: pd.DataFrame,
    prices: dict,
    test_dates: list,
    test_end: datetime,
    include_costs: bool = True
) -> dict:
    """Run quarterly backtest."""

    # Quarterly dates
    quarterly_dates = [test_dates[i] for i in range(0, len(test_dates), 3)]

    portfolio_values = [100.0]
    period_returns = []
    holdings_history = []

    prev_holdings = set()

    for i, date in enumerate(quarterly_dates):
        date_df = test_df[test_df['date'] == date].copy()

        if len(date_df) == 0:
            continue

        # Filters
        filtered = date_df[date_df['eps'] > 0].copy()
        filtered = filtered[filtered['return_252d'] > 0]

        if len(filtered) == 0:
            portfolio_values.append(portfolio_values[-1])
            period_returns.append(0)
            continue

        # Top 20 by prediction
        selected = filtered.sort_values('pred_prob', ascending=False).head(20)
        current_holdings = set(selected['symbol'].tolist())

        holdings_history.append({
            'date': date,
            'holdings': list(current_holdings),
            'predictions': selected[['symbol', 'pred_prob']].to_dict('records')
        })

        # Calculate turnover
        if prev_holdings:
            sells = prev_holdings - current_holdings
            buys = current_holdings - prev_holdings
            turnover = (len(sells) + len(buys)) / max(len(current_holdings), 1)
        else:
            turnover = 1.0

        # Next date
        if i < len(quarterly_dates) - 1:
            next_date = quarterly_dates[i + 1]
        else:
            next_date = test_end

        # Calculate return
        portfolio_return = 0
        n_stocks = 0

        for _, row in selected.iterrows():
            symbol = row['symbol']
            if symbol not in prices:
                continue

            price_df = prices[symbol]
            current_prices = price_df[price_df.index <= date]
            future_prices = price_df[(price_df.index > date) & (price_df.index <= next_date)]

            if len(current_prices) > 0 and len(future_prices) > 0:
                buy = current_prices['adjusted_close'].iloc[-1]
                sell = future_prices['adjusted_close'].iloc[-1]
                portfolio_return += (sell - buy) / buy
                n_stocks += 1

        if n_stocks > 0:
            portfolio_return /= n_stocks

        if include_costs:
            portfolio_return -= turnover * ROUND_TRIP_COST

        period_returns.append(portfolio_return)
        new_value = portfolio_values[-1] * (1 + portfolio_return)
        portfolio_values.append(new_value)

        prev_holdings = current_holdings

    # Metrics
    total_return = (portfolio_values[-1] - 100) / 100
    returns_series = pd.Series(period_returns)

    sharpe = returns_series.mean() / returns_series.std() * np.sqrt(4) if returns_series.std() > 0 else 0

    portfolio_series = pd.Series(portfolio_values)
    running_max = portfolio_series.cummax()
    drawdown = (portfolio_series - running_max) / running_max
    max_drawdown = drawdown.min()

    win_rate = (returns_series > 0).mean() if len(returns_series) > 0 else 0

    return {
        'total_return': total_return,
        'sharpe_ratio': sharpe,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'holdings_history': holdings_history,
        'period_returns': list(zip(quarterly_dates[:len(period_returns)], period_returns)),
    }


def get_next_trades(eodhd_client: EODHDClient, polygon_client: PolygonClient, model, feature_cols: list, top_n: int = 20) -> list:
    """Get trading signals for next quarter - for LIVE TRADING."""
    print("\n" + "=" * 50)
    print("GENERATING LIVE TRADING SIGNALS")
    print("=" * 50)

    today = datetime.now()
    start_date = (today - timedelta(days=400)).strftime('%Y-%m-%d')
    end_date = today.strftime('%Y-%m-%d')

    # Get universe
    stocks = eodhd_client.get_us_stocks()
    symbols = [s['Code'] for s in stocks][:500]  # Top 500

    # Fetch data
    print("\nFetching latest data...")
    fundamentals = eodhd_client.fetch_fundamentals_parallel(symbols)

    # Filter by market cap
    valid_symbols = []
    for symbol, fund in fundamentals.items():
        highlights = fund.get('Highlights', {})
        market_cap = highlights.get('MarketCapitalization', 0) or 0
        eps = highlights.get('EarningsShare')
        if market_cap >= 1e9 and eps is not None and eps > 0:
            valid_symbols.append(symbol)

    print(f"  Valid symbols: {len(valid_symbols)}")

    prices = polygon_client.fetch_prices_parallel(valid_symbols, start_date, end_date)
    print(f"  Got prices for: {len(prices)} symbols")

    # Build features for today
    rows = []
    for symbol in prices.keys():
        if symbol not in fundamentals:
            continue

        price_df = prices[symbol]
        fund_data = fundamentals[symbol]

        if len(price_df) < 260:
            continue

        try:
            momentum_df = calculate_momentum_features(price_df)
            latest = momentum_df.iloc[-1]

            fund_features = extract_fundamentals(fund_data, today)
            if fund_features.get('eps') is None or fund_features['eps'] <= 0:
                continue

            row = {
                'symbol': symbol,
                'date': today,
                'close': latest.get('adjusted_close', latest.get('close')),
            }

            for col in ['return_21d', 'return_63d', 'return_126d', 'return_252d',
                       'volatility_21d', 'volatility_63d', 'dist_from_high', 'dist_from_low',
                       'price_to_sma_20', 'price_to_sma_50', 'price_to_sma_200',
                       'rsi_14', 'volume_ratio', 'trend_strength']:
                row[col] = latest.get(col, 0)

            for key, value in fund_features.items():
                row[key] = value if value is not None else 0

            row['log_market_cap'] = np.log(fund_features.get('market_cap', 1e9) + 1)

            # Check momentum filter
            if row['return_252d'] <= 0:
                continue

            rows.append(row)
        except Exception as e:
            continue

    if not rows:
        print("ERROR: No valid stocks found")
        return []

    df = pd.DataFrame(rows)

    # Predict
    X = df[feature_cols].fillna(0).replace([np.inf, -np.inf], 0).values
    df['pred_prob'] = model.predict_proba(X)[:, 1]

    # Select top N
    selected = df.sort_values('pred_prob', ascending=False).head(top_n)

    print(f"\n{'='*50}")
    print(f"TOP {top_n} STOCKS TO BUY (as of {today.strftime('%Y-%m-%d')})")
    print(f"{'='*50}")
    print(f"\n{'Rank':<6} {'Symbol':<8} {'Prob':>8} {'12M Ret':>10} {'EPS':>8} {'Price':>10}")
    print("-" * 56)

    trades = []
    for i, (_, row) in enumerate(selected.iterrows(), 1):
        print(f"{i:<6} {row['symbol']:<8} {row['pred_prob']:>7.1%} {row['return_252d']:>9.1%} {row['eps']:>8.2f} ${row['close']:>9.2f}")
        trades.append({
            'rank': i,
            'symbol': row['symbol'],
            'probability': row['pred_prob'],
            'momentum_12m': row['return_252d'],
            'eps': row['eps'],
            'price': row['close'],
            'weight': 1.0 / top_n  # Equal weight
        })

    print(f"\nTotal investment per stock: {100/top_n:.1f}% of portfolio")
    print(f"Next rebalance: ~3 months from now")

    return trades


def main():
    print("\n" + "=" * 70)
    print("MOMENTUM + FUNDAMENTALS STRATEGY")
    print("=" * 70)
    print(f"\nDate: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("Data sources: Polygon (prices) + EODHD (fundamentals)")

    eodhd_client = EODHDClient(EODHD_API_KEY, max_workers=10)
    polygon_client = PolygonClient(POLYGON_API_KEY)

    # Step 1: Get universe
    print("\n[1/6] Getting US stock universe...")
    stocks = eodhd_client.get_us_stocks()
    print(f"  Total common stocks: {len(stocks)}")

    symbols = [s['Code'] for s in stocks]

    # Step 2: Fetch fundamentals for ALL stocks
    # EODHD has no limit - we fetch all to ensure we capture all large caps
    print("\n[2/6] Fetching fundamentals for ALL stocks...")
    print(f"  This may take a while for {len(symbols)} symbols...")

    fundamentals = eodhd_client.fetch_fundamentals_parallel(symbols)
    print(f"  Got fundamentals for {len(fundamentals)} stocks")

    # Step 3: Filter by market cap and EPS
    print("\n[3/6] Filtering by market cap >= $1B and valid EPS...")
    valid_symbols = []
    for symbol, fund in fundamentals.items():
        highlights = fund.get('Highlights', {})
        market_cap = highlights.get('MarketCapitalization', 0) or 0
        eps = highlights.get('EarningsShare')
        if market_cap >= 1e9 and eps is not None:
            valid_symbols.append((symbol, market_cap))

    # Sort by market cap and take top 500
    valid_symbols.sort(key=lambda x: x[1], reverse=True)
    valid_symbols = [s[0] for s in valid_symbols[:500]]
    print(f"  Stocks with market cap >= $1B & EPS data: {len(valid_symbols)}")

    # Step 4: Fetch prices from Polygon (no rate limits!)
    print("\n[4/6] Fetching price data from Polygon...")
    prices = polygon_client.fetch_prices_parallel(valid_symbols, '2015-01-01', '2025-12-31')
    prices['SPY'] = polygon_client.get_prices('SPY', '2015-01-01', '2025-12-31')
    print(f"  Got prices for {len(prices)} stocks")

    # Step 4b: Filter by volume (avg daily dollar volume > $300k)
    print("\n  Filtering by volume (avg dollar volume > $300k)...")
    min_dollar_volume = 300_000  # $300k daily
    liquid_symbols = []
    for symbol, df in prices.items():
        if symbol == 'SPY':
            continue
        if len(df) < 100:
            continue
        # Calculate average dollar volume (last 60 days)
        recent = df.tail(60)
        avg_dollar_vol = (recent['close'] * recent['volume']).mean()
        if avg_dollar_vol >= min_dollar_volume:
            liquid_symbols.append(symbol)

    # Keep only liquid stocks
    prices = {s: prices[s] for s in liquid_symbols if s in prices}
    prices['SPY'] = polygon_client.get_prices('SPY', '2015-01-01', '2025-12-31')
    print(f"  Liquid stocks (avg $vol > $300k): {len(prices) - 1}")

    # Step 5: Build features
    print("\n[5/6] Building features and training...")

    train_end = datetime(2024, 12, 31)
    test_start = datetime(2025, 1, 1)
    test_end = datetime(2025, 12, 19)

    # Monthly dates (start from 2015 for 10 years of data)
    all_dates = []
    current = datetime(2015, 1, 1).replace(day=1)
    while current <= test_end:
        rebal = current.replace(day=3)
        all_dates.append(rebal)
        current = (current + timedelta(days=32)).replace(day=1)

    train_dates = [d for d in all_dates if d <= train_end]
    test_dates = [d for d in all_dates if d > train_end]
    sampled_train_dates = train_dates[::2]

    print(f"  Train periods: {len(sampled_train_dates)}, Test periods: {len(test_dates)}")

    # Build dataset
    all_rows = []
    for date in sampled_train_dates + test_dates:
        for symbol in prices.keys():
            if symbol not in fundamentals or symbol == 'SPY':
                continue

            price_df = prices[symbol]
            fund_data = fundamentals[symbol]

            price_filtered = price_df[price_df.index <= date]
            if len(price_filtered) < 260:
                continue

            try:
                momentum_df = calculate_momentum_features(price_filtered)
                latest = momentum_df.iloc[-1]

                fund_features = extract_fundamentals(fund_data, date)
                if fund_features.get('eps') is None:
                    continue

                row = {
                    'symbol': symbol,
                    'date': date,
                    'close': latest.get('adjusted_close', latest.get('close')),
                }

                for col in ['return_21d', 'return_63d', 'return_126d', 'return_252d',
                           'volatility_21d', 'volatility_63d', 'dist_from_high', 'dist_from_low',
                           'price_to_sma_20', 'price_to_sma_50', 'price_to_sma_200',
                           'rsi_14', 'volume_ratio', 'trend_strength']:
                    row[col] = latest.get(col, 0)

                for key, value in fund_features.items():
                    row[key] = value if value is not None else 0

                row['log_market_cap'] = np.log(fund_features.get('market_cap', 1e9) + 1)

                all_rows.append(row)
            except:
                continue

    feature_df = pd.DataFrame(all_rows)
    print(f"  Total samples: {len(feature_df)}")

    # Forward returns
    forward_returns = []
    for _, row in feature_df.iterrows():
        symbol = row['symbol']
        current_date = row['date']
        current_price = row['close']

        if symbol not in prices:
            forward_returns.append(np.nan)
            continue

        price_df = prices[symbol]
        future_date = current_date + timedelta(days=21)
        future_prices = price_df[(price_df.index > current_date) & (price_df.index <= future_date)]

        if len(future_prices) > 0:
            future_price = future_prices['adjusted_close'].iloc[-1]
            forward_returns.append((future_price - current_price) / current_price)
        else:
            forward_returns.append(np.nan)

    feature_df['forward_return'] = forward_returns

    # Split and train
    train_df = feature_df[feature_df['date'] <= train_end].dropna(subset=['forward_return'])
    test_df_base = feature_df[feature_df['date'] > train_end]

    market_returns = train_df.groupby('date')['forward_return'].mean()
    train_df = train_df.copy()
    train_df['market_return'] = train_df['date'].map(market_returns)
    train_df['label'] = (train_df['forward_return'] > train_df['market_return']).astype(int)

    feature_cols = get_feature_columns()
    X_train = train_df[feature_cols].fillna(0).replace([np.inf, -np.inf], 0).values
    y_train = train_df['label'].values

    # Use validation for early stopping
    val_size = int(len(X_train) * 0.1)
    X_tr, X_val = X_train[:-val_size], X_train[-val_size:]
    y_tr, y_val = y_train[:-val_size], y_train[-val_size:]

    print(f"\n  Training CatBoost (d=10, lr=0.0005)...")
    model = CatBoostClassifier(
        iterations=30000,
        depth=10,
        learning_rate=0.0005,
        auto_class_weights='Balanced',
        random_state=42,
        verbose=False,
        early_stopping_rounds=500,
        use_best_model=True,
    )
    model.fit(X_tr, y_tr, eval_set=(X_val, y_val))
    print(f"  Model trained (used {model.best_iteration_} iterations)")

    # Predict on test
    X_test = test_df_base[feature_cols].fillna(0).replace([np.inf, -np.inf], 0).values
    test_df = test_df_base.copy()
    test_df['pred_prob'] = model.predict_proba(X_test)[:, 1]

    # Step 6: Backtest
    print("\n[6/6] Backtesting on 2025...")

    result = run_backtest(test_df, prices, test_dates, test_end)

    # SPY benchmark
    spy = prices.get('SPY')
    if spy is not None:
        spy_2025 = spy[(spy.index >= test_start) & (spy.index <= test_end)]
        if len(spy_2025) > 0:
            spy_return = (spy_2025['adjusted_close'].iloc[-1] - spy_2025['adjusted_close'].iloc[0]) / spy_2025['adjusted_close'].iloc[0]
        else:
            spy_return = 0
    else:
        spy_return = 0

    # Results
    print("\n" + "=" * 70)
    print("2025 BACKTEST RESULTS (EODHD DATA ONLY)")
    print("=" * 70)

    print(f"""
    Strategy Return:  {result['total_return']:>8.1%}
    S&P 500 Return:   {spy_return:>8.1%}
    Alpha:            {result['total_return'] - spy_return:>+7.1%}

    Sharpe Ratio:     {result['sharpe_ratio']:>8.2f}
    Max Drawdown:     {result['max_drawdown']:>8.1%}
    Win Rate:         {result['win_rate']:>8.1%}
    """)

    print("Quarterly Returns:")
    for date, ret in result['period_returns']:
        print(f"  {date.strftime('%Y-%m')}: {ret:>+7.1%}")

    print("\nQuarterly Holdings:")
    for h in result['holdings_history']:
        stocks = ', '.join(h['holdings'][:8])
        print(f"  {h['date'].strftime('%Y-%m-%d')}: {stocks}...")

    # Generate next trades
    print("\n" + "=" * 70)
    next_trades = get_next_trades(eodhd_client, polygon_client, model, feature_cols)

    # Save model for live trading
    model_path = CACHE_DIR / "momentum_model.cbm"
    model.save_model(str(model_path))
    print(f"\nModel saved to: {model_path}")

    # Save next trades
    trades_path = CACHE_DIR / f"trades_{datetime.now().strftime('%Y%m%d')}.json"
    with open(trades_path, 'w') as f:
        json.dump(next_trades, f, indent=2)
    print(f"Trade signals saved to: {trades_path}")

    print("\n" + "=" * 70)
    print(f"COMPLETE - Strategy uses Polygon (prices) + EODHD (fundamentals)")
    print(f"Total EODHD API calls: {eodhd_client.api_calls}")
    print("=" * 70)

    return {
        'backtest': result,
        'next_trades': next_trades,
        'spy_return': spy_return,
    }


if __name__ == '__main__':
    main()

"""
News Data Downloader - Fetches financial news from Polygon.io (Massive.com) API.

Supports:
- Historical news data for backtesting (2+ years history)
- Real-time news for live trading
- Sentiment analysis data
- Multiple ticker support

API Documentation: https://massive.com/docs/rest/stocks/news
"""
import os
import time
import requests
import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
import logging
from pathlib import Path
import json

logger = logging.getLogger(__name__)


@dataclass
class NewsConfig:
    """Configuration for news downloader."""
    api_key: str = ""  # Set via environment variable POLYGON_API_KEY
    base_url: str = "https://api.polygon.io"
    rate_limit_delay: float = 0.25  # Seconds between requests (free tier: 5 req/min)
    max_retries: int = 3
    cache_dir: str = "data/news_cache"


class PolygonNewsDownloader:
    """
    Downloads financial news from Polygon.io (Massive.com) API.

    Features:
    - Historical news retrieval with date filtering
    - Sentiment analysis extraction
    - Caching to reduce API calls
    - Rate limiting support
    """

    def __init__(self, config: Optional[NewsConfig] = None):
        """
        Initialize news downloader.

        Args:
            config: Downloader configuration
        """
        self.config = config or NewsConfig()

        # Get API key from environment if not provided
        if not self.config.api_key:
            self.config.api_key = os.environ.get('POLYGON_API_KEY', '')

        if not self.config.api_key:
            logger.warning(
                "POLYGON_API_KEY not set. Set it with: "
                "export POLYGON_API_KEY='your_key_here'"
            )

        # Create cache directory
        self.cache_dir = Path(self.config.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def download_news(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        limit: int = 1000
    ) -> pd.DataFrame:
        """
        Download news for a specific ticker and date range.

        Args:
            ticker: Stock ticker (e.g., 'AAPL', 'EUR/USD' -> 'EURUSD')
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            limit: Max articles per request

        Returns:
            DataFrame with news articles and sentiment
        """
        if not self.config.api_key:
            logger.error("API key not configured")
            return pd.DataFrame()

        # Check cache first
        cache_key = f"{ticker}_{start_date}_{end_date}"
        cached_data = self._load_from_cache(cache_key)
        if cached_data is not None:
            logger.info(f"Loaded {len(cached_data)} articles from cache")
            return cached_data

        # Normalize ticker (remove slashes for forex)
        ticker_clean = ticker.replace('/', '').replace('-', '').upper()

        all_articles = []
        current_date = datetime.strptime(end_date, '%Y-%m-%d')
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')

        while current_date >= start_dt:
            try:
                articles = self._fetch_news_batch(
                    ticker_clean,
                    current_date.strftime('%Y-%m-%d'),
                    limit
                )

                if articles:
                    all_articles.extend(articles)
                    # Get oldest article date for pagination
                    oldest = min(a.get('published_utc', '') for a in articles)
                    if oldest:
                        current_date = datetime.fromisoformat(oldest.replace('Z', '')) - timedelta(days=1)
                    else:
                        break
                else:
                    # Move back by a week if no articles found
                    current_date -= timedelta(days=7)

                time.sleep(self.config.rate_limit_delay)

            except Exception as e:
                logger.error(f"Error fetching news: {e}")
                current_date -= timedelta(days=1)

        if not all_articles:
            logger.warning(f"No news found for {ticker} from {start_date} to {end_date}")
            return pd.DataFrame()

        # Convert to DataFrame
        df = self._process_articles(all_articles)

        # Filter by date range
        df = df[(df['published_utc'] >= start_date) & (df['published_utc'] <= end_date)]

        # Save to cache
        self._save_to_cache(cache_key, df)

        logger.info(f"Downloaded {len(df)} news articles for {ticker}")
        return df

    def _fetch_news_batch(
        self,
        ticker: str,
        published_utc_lte: str,
        limit: int
    ) -> List[Dict]:
        """Fetch a batch of news articles."""
        url = f"{self.config.base_url}/v2/reference/news"

        params = {
            'ticker': ticker,
            'published_utc.lte': published_utc_lte,
            'order': 'desc',
            'limit': limit,
            'apiKey': self.config.api_key
        }

        for attempt in range(self.config.max_retries):
            try:
                response = requests.get(url, params=params, timeout=30)

                if response.status_code == 200:
                    data = response.json()
                    return data.get('results', [])
                elif response.status_code == 429:
                    # Rate limited
                    wait_time = (attempt + 1) * 60
                    logger.warning(f"Rate limited. Waiting {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"API error: {response.status_code} - {response.text}")
                    return []

            except Exception as e:
                logger.error(f"Request failed (attempt {attempt + 1}): {e}")
                time.sleep(5)

        return []

    def _process_articles(self, articles: List[Dict]) -> pd.DataFrame:
        """Process raw articles into DataFrame with sentiment."""
        processed = []

        for article in articles:
            record = {
                'id': article.get('id'),
                'title': article.get('title'),
                'author': article.get('author'),
                'published_utc': article.get('published_utc', '')[:10],  # Date only
                'published_datetime': article.get('published_utc'),
                'article_url': article.get('article_url'),
                'tickers': ','.join(article.get('tickers', [])),
                'description': article.get('description', '')[:500],  # Truncate
                'keywords': ','.join(article.get('keywords', [])),
                'publisher_name': article.get('publisher', {}).get('name', ''),
            }

            # Extract sentiment from insights
            insights = article.get('insights', [])
            if insights:
                # Find insight for our ticker
                for insight in insights:
                    record['sentiment'] = insight.get('sentiment', '')
                    record['sentiment_reasoning'] = insight.get('sentiment_reasoning', '')[:200]
                    break
            else:
                record['sentiment'] = ''
                record['sentiment_reasoning'] = ''

            processed.append(record)

        df = pd.DataFrame(processed)

        # Convert sentiment to numeric score
        df['sentiment_score'] = df['sentiment'].map({
            'positive': 1.0,
            'bullish': 1.0,
            'neutral': 0.0,
            'negative': -1.0,
            'bearish': -1.0,
            '': 0.0
        }).fillna(0.0)

        # Sort by date
        df = df.sort_values('published_datetime').reset_index(drop=True)

        return df

    def _load_from_cache(self, cache_key: str) -> Optional[pd.DataFrame]:
        """Load data from cache if available."""
        cache_file = self.cache_dir / f"{cache_key}.parquet"
        if cache_file.exists():
            return pd.read_parquet(cache_file)
        return None

    def _save_to_cache(self, cache_key: str, df: pd.DataFrame) -> None:
        """Save data to cache."""
        cache_file = self.cache_dir / f"{cache_key}.parquet"
        df.to_parquet(cache_file, index=False)

    def get_sentiment_timeseries(
        self,
        news_df: pd.DataFrame,
        freq: str = '5min',
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.Series:
        """
        Convert news sentiment to time series aligned with price data.

        Uses forward-fill to propagate sentiment between news articles.

        Args:
            news_df: News DataFrame from download_news()
            freq: Target frequency ('5min', '1H', '1D')
            start_date: Start date for time series
            end_date: End date for time series

        Returns:
            Series with sentiment scores at specified frequency
        """
        if news_df.empty:
            return pd.Series(dtype=float)

        # Convert to datetime
        news_df['datetime'] = pd.to_datetime(news_df['published_datetime'])

        # Aggregate sentiment per datetime
        sentiment_agg = news_df.groupby('datetime')['sentiment_score'].mean()

        # Create full datetime index
        if start_date and end_date:
            full_index = pd.date_range(start=start_date, end=end_date, freq=freq)
        else:
            full_index = pd.date_range(
                start=sentiment_agg.index.min(),
                end=sentiment_agg.index.max(),
                freq=freq
            )

        # Reindex and forward-fill
        sentiment_ts = sentiment_agg.reindex(full_index).ffill().fillna(0)

        # Apply decay to old sentiment (exponential decay over 24 hours)
        # This gives more weight to recent news
        decay_periods = 288 if freq == '5min' else (24 if freq == '1H' else 1)  # 24 hours
        sentiment_ts = sentiment_ts.ewm(span=decay_periods).mean()

        return sentiment_ts

    def get_news_features(
        self,
        news_df: pd.DataFrame,
        price_index: pd.DatetimeIndex
    ) -> pd.DataFrame:
        """
        Generate news-based features aligned with price data.

        Features:
        - sentiment_current: Current sentiment score
        - sentiment_ma_4h: 4-hour moving average of sentiment
        - sentiment_ma_24h: 24-hour moving average
        - news_count_4h: Number of news articles in last 4 hours
        - sentiment_volatility: Volatility of sentiment

        Args:
            news_df: News DataFrame
            price_index: DatetimeIndex from price data

        Returns:
            DataFrame with news features
        """
        if news_df.empty:
            return pd.DataFrame(index=price_index)

        features = pd.DataFrame(index=price_index)

        # Get sentiment time series
        sentiment_ts = self.get_sentiment_timeseries(
            news_df,
            freq='5min',
            start_date=str(price_index.min()),
            end_date=str(price_index.max())
        )

        # Align with price index
        sentiment_aligned = sentiment_ts.reindex(price_index, method='ffill').fillna(0)

        features['news_sentiment'] = sentiment_aligned
        features['news_sentiment_ma_4h'] = sentiment_aligned.rolling(48).mean()  # 48 * 5min = 4h
        features['news_sentiment_ma_24h'] = sentiment_aligned.rolling(288).mean()  # 288 * 5min = 24h
        features['news_sentiment_std'] = sentiment_aligned.rolling(48).std()

        # News count features
        news_df['datetime'] = pd.to_datetime(news_df['published_datetime'])
        news_count = news_df.set_index('datetime').resample('5min').size()
        news_count = news_count.reindex(price_index, fill_value=0)

        features['news_count_4h'] = news_count.rolling(48).sum()
        features['news_count_24h'] = news_count.rolling(288).sum()

        # Sentiment change
        features['news_sentiment_change'] = features['news_sentiment'].diff(periods=12)  # 1 hour

        # Extreme sentiment indicators
        features['news_extreme_positive'] = (features['news_sentiment'] > 0.5).astype(int)
        features['news_extreme_negative'] = (features['news_sentiment'] < -0.5).astype(int)

        return features


class ForexNewsDownloader:
    """
    Forex-specific news downloader.

    NOTE: Polygon.io/Massive.com news API does NOT support forex pairs.
    Use these alternatives instead:

    Supported providers:
    - 'eodhd': EODHD APIs (has EURUSD.FOREX with free demo)
    - 'forexnewsapi': ForexNewsAPI.com (dedicated forex news)
    - 'fmp': Financial Modeling Prep Forex News
    - 'alphavantage': Alpha Vantage (general news with forex mentions)
    """

    def __init__(self, provider: str = 'eodhd'):
        """
        Initialize forex news downloader.

        Args:
            provider: 'eodhd', 'forexnewsapi', 'fmp', or 'alphavantage'
        """
        self.provider = provider
        self.api_keys = {
            'eodhd': os.environ.get('EODHD_API_KEY', 'demo'),  # 'demo' works for EURUSD
            'forexnewsapi': os.environ.get('FOREXNEWSAPI_KEY', ''),
            'fmp': os.environ.get('FMP_API_KEY', ''),
            'alphavantage': os.environ.get('ALPHAVANTAGE_API_KEY', '')
        }
        self.cache_dir = Path('data/forex_news_cache')
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def download_news(
        self,
        pair: str,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """
        Download forex news for a currency pair.

        Args:
            pair: Currency pair (e.g., 'EURUSD', 'EUR/USD')
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            DataFrame with news articles and sentiment
        """
        # Normalize pair format
        pair = pair.replace('/', '').replace('-', '').upper()

        if self.provider == 'eodhd':
            return self._download_eodhd(pair, start_date, end_date)
        elif self.provider == 'forexnewsapi':
            return self._download_forexnewsapi(pair, start_date, end_date)
        elif self.provider == 'fmp':
            return self._download_fmp(pair, start_date, end_date)
        elif self.provider == 'alphavantage':
            return self._download_alphavantage_forex(pair)
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

    def _download_eodhd(
        self,
        pair: str,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """
        Download news from EODHD APIs.

        EODHD format: EURUSD.FOREX
        Free demo key works for EURUSD.FOREX
        """
        api_key = self.api_keys['eodhd']

        # EODHD ticker format
        ticker = f"{pair}.FOREX"

        url = "https://eodhd.com/api/news"
        params = {
            's': ticker,
            'from': start_date,
            'to': end_date,
            'limit': 1000,
            'api_token': api_key,
            'fmt': 'json'
        }

        try:
            response = requests.get(url, params=params, timeout=30)
            if response.status_code == 200:
                articles = response.json()
                if articles and isinstance(articles, list) and len(articles) > 0:
                    return self._process_eodhd_articles(articles)
                else:
                    logger.warning(f"EODHD returned empty or invalid response for {ticker}")
            else:
                logger.error(f"EODHD API error: {response.status_code}")
        except Exception as e:
            logger.error(f"EODHD request failed: {e}")

        return pd.DataFrame()

    def _process_eodhd_articles(self, articles: List[Dict]) -> pd.DataFrame:
        """Process EODHD articles."""
        processed = []
        for article in articles:
            # Handle sentiment - can be dict or None
            sentiment_data = article.get('sentiment')
            if isinstance(sentiment_data, dict):
                sentiment_score = sentiment_data.get('polarity', 0)
            else:
                sentiment_score = 0

            record = {
                'title': article.get('title', ''),
                'published_datetime': article.get('date', ''),
                'published_utc': str(article.get('date', ''))[:10],
                'article_url': article.get('link', ''),
                'description': str(article.get('content', ''))[:500] if article.get('content') else '',
                'source': article.get('source', ''),
                'symbols': ','.join(article.get('symbols', []) or []),
                'tags': ','.join(article.get('tags', []) or []),
                'sentiment_score': float(sentiment_score) if sentiment_score else 0.0
            }
            processed.append(record)

        return pd.DataFrame(processed)

    def _download_forexnewsapi(
        self,
        pair: str,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """
        Download from ForexNewsAPI.com.

        Dedicated forex news API with sentiment.
        History back to May 2021.
        """
        api_key = self.api_keys['forexnewsapi']
        if not api_key:
            logger.warning("FOREXNEWSAPI_KEY not set")
            return pd.DataFrame()

        # ForexNewsAPI uses currency codes
        base = pair[:3]
        quote = pair[3:]

        url = "https://forexnewsapi.com/api/v1"
        params = {
            'currencypair': f"{base}/{quote}",
            'date': f"{start_date},{end_date}",
            'token': api_key,
            'items': 500
        }

        try:
            response = requests.get(url, params=params, timeout=30)
            if response.status_code == 200:
                data = response.json()
                if 'data' in data:
                    return self._process_forexnewsapi_articles(data['data'])
            else:
                logger.error(f"ForexNewsAPI error: {response.status_code}")
        except Exception as e:
            logger.error(f"ForexNewsAPI request failed: {e}")

        return pd.DataFrame()

    def _process_forexnewsapi_articles(self, articles: List[Dict]) -> pd.DataFrame:
        """Process ForexNewsAPI articles."""
        processed = []
        for article in articles:
            sentiment_map = {'positive': 1.0, 'negative': -1.0, 'neutral': 0.0}
            record = {
                'title': article.get('title'),
                'published_datetime': article.get('date'),
                'published_utc': article.get('date', '')[:10],
                'article_url': article.get('news_url'),
                'description': article.get('text', '')[:500],
                'source': article.get('source_name', ''),
                'sentiment': article.get('sentiment', 'neutral'),
                'sentiment_score': sentiment_map.get(article.get('sentiment', 'neutral'), 0.0),
                'currencies': ','.join(article.get('currencies', []))
            }
            processed.append(record)

        return pd.DataFrame(processed)

    def _download_fmp(
        self,
        pair: str,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """
        Download from Financial Modeling Prep Forex News API.
        """
        api_key = self.api_keys['fmp']
        if not api_key:
            logger.warning("FMP_API_KEY not set")
            return pd.DataFrame()

        url = f"https://financialmodelingprep.com/api/v4/forex_news"
        params = {
            'symbol': pair,
            'from': start_date,
            'to': end_date,
            'apikey': api_key
        }

        try:
            response = requests.get(url, params=params, timeout=30)
            if response.status_code == 200:
                articles = response.json()
                if articles:
                    return self._process_fmp_articles(articles)
            else:
                logger.error(f"FMP API error: {response.status_code}")
        except Exception as e:
            logger.error(f"FMP request failed: {e}")

        return pd.DataFrame()

    def _process_fmp_articles(self, articles: List[Dict]) -> pd.DataFrame:
        """Process FMP articles."""
        processed = []
        for article in articles:
            record = {
                'title': article.get('title'),
                'published_datetime': article.get('publishedDate'),
                'published_utc': article.get('publishedDate', '')[:10],
                'article_url': article.get('url'),
                'description': article.get('text', '')[:500],
                'source': article.get('site', ''),
                'symbol': article.get('symbol', ''),
                'sentiment_score': 0.0  # FMP doesn't provide sentiment in forex news
            }
            processed.append(record)

        return pd.DataFrame(processed)

    def _download_alphavantage_forex(self, pair: str) -> pd.DataFrame:
        """
        Download forex-related news from Alpha Vantage.

        Uses general news with forex topic filter.
        """
        api_key = self.api_keys['alphavantage']
        if not api_key:
            logger.warning("ALPHAVANTAGE_API_KEY not set")
            return pd.DataFrame()

        url = "https://www.alphavantage.co/query"
        params = {
            'function': 'NEWS_SENTIMENT',
            'topics': 'forex',
            'apikey': api_key
        }

        try:
            response = requests.get(url, params=params, timeout=30)
            if response.status_code == 200:
                data = response.json()
                if 'feed' in data:
                    return self._process_alphavantage_forex(data['feed'], pair)
            else:
                logger.error(f"Alpha Vantage error: {response.status_code}")
        except Exception as e:
            logger.error(f"Alpha Vantage request failed: {e}")

        return pd.DataFrame()

    def _process_alphavantage_forex(
        self,
        articles: List[Dict],
        pair: str
    ) -> pd.DataFrame:
        """Process Alpha Vantage forex news."""
        processed = []
        base = pair[:3].upper()
        quote = pair[3:].upper()

        for article in articles:
            # Check if article mentions our currency pair
            summary = article.get('summary', '').upper()
            title = article.get('title', '').upper()

            if base in summary or quote in summary or base in title or quote in title:
                record = {
                    'title': article.get('title'),
                    'published_datetime': article.get('time_published'),
                    'published_utc': article.get('time_published', '')[:8],
                    'article_url': article.get('url'),
                    'description': article.get('summary', '')[:500],
                    'source': article.get('source', ''),
                    'sentiment_score': float(article.get('overall_sentiment_score', 0))
                }
                processed.append(record)

        return pd.DataFrame(processed)

    def get_sentiment_timeseries(
        self,
        news_df: pd.DataFrame,
        freq: str = '5min',
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.Series:
        """Convert news sentiment to time series."""
        if news_df.empty:
            return pd.Series(dtype=float)

        news_df['datetime'] = pd.to_datetime(news_df['published_datetime'])
        sentiment_agg = news_df.groupby('datetime')['sentiment_score'].mean()

        if start_date and end_date:
            full_index = pd.date_range(start=start_date, end=end_date, freq=freq)
        else:
            full_index = pd.date_range(
                start=sentiment_agg.index.min(),
                end=sentiment_agg.index.max(),
                freq=freq
            )

        sentiment_ts = sentiment_agg.reindex(full_index).ffill().fillna(0)
        decay_periods = 288 if freq == '5min' else 24
        sentiment_ts = sentiment_ts.ewm(span=decay_periods).mean()

        return sentiment_ts


class AlternativeNewsDownloader:
    """
    Alternative news sources when Polygon.io is not available.

    NOTE: For FOREX news, use ForexNewsDownloader instead.
    This class is for stock news from alternative providers.

    Supports:
    - Tiingo News API
    - Finnhub News API
    - Alpha Vantage News Sentiment
    """

    def __init__(self, provider: str = 'tiingo'):
        """
        Initialize alternative news downloader.

        Args:
            provider: 'tiingo', 'finnhub', or 'alphavantage'
        """
        self.provider = provider
        self.api_keys = {
            'tiingo': os.environ.get('TIINGO_API_KEY', ''),
            'finnhub': os.environ.get('FINNHUB_API_KEY', ''),
            'alphavantage': os.environ.get('ALPHAVANTAGE_API_KEY', '')
        }

    def download_news(
        self,
        ticker: str,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """Download news from alternative provider."""
        if self.provider == 'tiingo':
            return self._download_tiingo(ticker, start_date, end_date)
        elif self.provider == 'finnhub':
            return self._download_finnhub(ticker, start_date, end_date)
        elif self.provider == 'alphavantage':
            return self._download_alphavantage(ticker)
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

    def _download_tiingo(
        self,
        ticker: str,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """Download news from Tiingo."""
        api_key = self.api_keys['tiingo']
        if not api_key:
            logger.warning("TIINGO_API_KEY not set")
            return pd.DataFrame()

        url = f"https://api.tiingo.com/tiingo/news"
        headers = {'Content-Type': 'application/json'}
        params = {
            'tickers': ticker.replace('/', ''),
            'startDate': start_date,
            'endDate': end_date,
            'token': api_key
        }

        try:
            response = requests.get(url, headers=headers, params=params, timeout=30)
            if response.status_code == 200:
                articles = response.json()
                return self._process_tiingo_articles(articles)
            else:
                logger.error(f"Tiingo API error: {response.status_code}")
                return pd.DataFrame()
        except Exception as e:
            logger.error(f"Tiingo request failed: {e}")
            return pd.DataFrame()

    def _process_tiingo_articles(self, articles: List[Dict]) -> pd.DataFrame:
        """Process Tiingo articles."""
        if not articles:
            return pd.DataFrame()

        processed = []
        for article in articles:
            record = {
                'id': article.get('id'),
                'title': article.get('title'),
                'published_utc': article.get('publishedDate', '')[:10],
                'published_datetime': article.get('publishedDate'),
                'article_url': article.get('url'),
                'description': article.get('description', '')[:500],
                'tickers': ','.join(article.get('tickers', [])),
                'source': article.get('source', '')
            }
            processed.append(record)

        df = pd.DataFrame(processed)
        df['sentiment_score'] = 0.0  # Tiingo doesn't provide sentiment by default
        return df

    def _download_finnhub(
        self,
        ticker: str,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """Download news from Finnhub."""
        api_key = self.api_keys['finnhub']
        if not api_key:
            logger.warning("FINNHUB_API_KEY not set")
            return pd.DataFrame()

        url = "https://finnhub.io/api/v1/company-news"
        params = {
            'symbol': ticker.replace('/', ''),
            'from': start_date,
            'to': end_date,
            'token': api_key
        }

        try:
            response = requests.get(url, params=params, timeout=30)
            if response.status_code == 200:
                articles = response.json()
                return self._process_finnhub_articles(articles)
            else:
                logger.error(f"Finnhub API error: {response.status_code}")
                return pd.DataFrame()
        except Exception as e:
            logger.error(f"Finnhub request failed: {e}")
            return pd.DataFrame()

    def _process_finnhub_articles(self, articles: List[Dict]) -> pd.DataFrame:
        """Process Finnhub articles."""
        if not articles:
            return pd.DataFrame()

        processed = []
        for article in articles:
            dt = datetime.fromtimestamp(article.get('datetime', 0))
            record = {
                'id': article.get('id'),
                'title': article.get('headline'),
                'published_utc': dt.strftime('%Y-%m-%d'),
                'published_datetime': dt.isoformat(),
                'article_url': article.get('url'),
                'description': article.get('summary', '')[:500],
                'source': article.get('source', '')
            }
            processed.append(record)

        df = pd.DataFrame(processed)
        df['sentiment_score'] = 0.0
        return df

    def _download_alphavantage(self, ticker: str) -> pd.DataFrame:
        """Download news sentiment from Alpha Vantage."""
        api_key = self.api_keys['alphavantage']
        if not api_key:
            logger.warning("ALPHAVANTAGE_API_KEY not set")
            return pd.DataFrame()

        url = "https://www.alphavantage.co/query"
        params = {
            'function': 'NEWS_SENTIMENT',
            'tickers': ticker.replace('/', ''),
            'apikey': api_key
        }

        try:
            response = requests.get(url, params=params, timeout=30)
            if response.status_code == 200:
                data = response.json()
                return self._process_alphavantage_articles(data.get('feed', []))
            else:
                logger.error(f"Alpha Vantage API error: {response.status_code}")
                return pd.DataFrame()
        except Exception as e:
            logger.error(f"Alpha Vantage request failed: {e}")
            return pd.DataFrame()

    def _process_alphavantage_articles(self, articles: List[Dict]) -> pd.DataFrame:
        """Process Alpha Vantage articles with sentiment."""
        if not articles:
            return pd.DataFrame()

        processed = []
        for article in articles:
            record = {
                'title': article.get('title'),
                'published_datetime': article.get('time_published'),
                'published_utc': article.get('time_published', '')[:8],  # YYYYMMDD
                'article_url': article.get('url'),
                'description': article.get('summary', '')[:500],
                'source': article.get('source', ''),
                'overall_sentiment_score': article.get('overall_sentiment_score', 0),
                'overall_sentiment_label': article.get('overall_sentiment_label', '')
            }

            # Get ticker-specific sentiment
            ticker_sentiments = article.get('ticker_sentiment', [])
            if ticker_sentiments:
                record['sentiment_score'] = float(ticker_sentiments[0].get('ticker_sentiment_score', 0))
            else:
                record['sentiment_score'] = float(article.get('overall_sentiment_score', 0))

            processed.append(record)

        return pd.DataFrame(processed)

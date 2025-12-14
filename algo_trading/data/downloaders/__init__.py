"""
Data downloaders for various sources.
"""
from .base import DataDownloader
from .forex_downloader import ForexDownloader

try:
    from .news_downloader import PolygonNewsDownloader, AlternativeNewsDownloader
except ImportError:
    pass  # Optional dependencies

__all__ = [
    'DataDownloader',
    'ForexDownloader',
    'PolygonNewsDownloader',
    'AlternativeNewsDownloader'
]

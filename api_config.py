"""
Configuration module for API keys and settings.

API keys are loaded from:
1. Environment variables (EODHD_API_KEY, POLYGON_API_KEY)
2. Files in home directory (~/.eodhd_api_key, ~/.polygon_api_key)
3. Files specified in the project (~/api_key.txt for EODHD, ~/api_key2.txt for Polygon)

Never commit API keys to version control.
"""

import os
from pathlib import Path


def _load_key_from_file(filepath: str) -> str:
    """Load API key from a file."""
    path = Path(filepath).expanduser()
    if path.exists():
        return path.read_text().strip()
    return ""


def get_eodhd_api_key() -> str:
    """Get EODHD API key from environment or file."""
    # Try environment variable first
    key = os.environ.get('EODHD_API_KEY', '')
    if key:
        return key

    # Try various file locations
    for filepath in [
        '~/.eodhd_api_key',
        '~/api_key.txt',
        '~/.config/eodhd/api_key',
    ]:
        key = _load_key_from_file(filepath)
        if key:
            return key

    raise ValueError(
        "EODHD API key not found. Set EODHD_API_KEY environment variable "
        "or create ~/.eodhd_api_key file with your API key."
    )


def get_polygon_api_key() -> str:
    """Get Polygon API key from environment or file."""
    # Try environment variable first
    key = os.environ.get('POLYGON_API_KEY', '')
    if key:
        return key

    # Try various file locations
    for filepath in [
        '~/.polygon_api_key',
        '~/api_key2.txt',
        '~/.config/polygon/api_key',
        '~/Desktop/api_key.txt',
    ]:
        key = _load_key_from_file(filepath)
        if key:
            return key

    raise ValueError(
        "Polygon API key not found. Set POLYGON_API_KEY environment variable "
        "or create ~/.polygon_api_key file with your API key."
    )


# Lazy-loaded keys (loaded on first access)
_eodhd_key = None
_polygon_key = None


def get_eodhd_key() -> str:
    """Get cached EODHD API key."""
    global _eodhd_key
    if _eodhd_key is None:
        _eodhd_key = get_eodhd_api_key()
    return _eodhd_key


def get_polygon_key() -> str:
    """Get cached Polygon API key."""
    global _polygon_key
    if _polygon_key is None:
        _polygon_key = get_polygon_api_key()
    return _polygon_key


# For backwards compatibility - these will be loaded on import
# but will raise an error if keys are not configured
try:
    EODHD_API_KEY = get_eodhd_key()
except ValueError:
    EODHD_API_KEY = None

try:
    POLYGON_API_KEY = get_polygon_key()
except ValueError:
    POLYGON_API_KEY = None

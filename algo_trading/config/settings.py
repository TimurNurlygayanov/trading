"""
Global settings and configuration for the algo trading system.
"""
import os
from dataclasses import dataclass, field
from typing import Dict, Any, Optional
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "storage" / "historical"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"

# Ensure directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class TradingConfig:
    """Trading configuration parameters."""

    # Default symbol and timeframe
    symbol: str = "EURUSD"
    timeframe: str = "5min"

    # ===== ACCOUNT CONFIGURATION =====
    # $100,000 account with 30x leverage
    initial_capital: float = 100_000.0
    leverage: float = 30.0

    # ===== RISK MANAGEMENT =====
    # $5,000 max daily drawdown (5% of account) - CRITICAL
    # If breached, account is considered lost
    max_daily_drawdown: float = 5_000.0
    max_daily_drawdown_pct: float = 0.05  # 5% of initial capital

    # Position sizing
    max_position_size_pct: float = 0.02  # 2% risk per trade
    max_open_positions: int = 3

    # Total drawdown limit before stopping (10% = $10,000)
    max_total_drawdown_pct: float = 0.10

    # ===== TRANSACTION COSTS =====
    commission: float = 0.00007  # 0.7 pips spread for EURUSD
    slippage: float = 0.00003   # 0.3 pips slippage

    # ===== POSITION SIZING FOR LEVERAGE =====
    # With 30x leverage on $100k, max notional = $3,000,000
    # But we limit risk per trade to 2% = $2,000 max loss
    risk_per_trade: float = 2_000.0  # $2,000 max loss per trade

    # ===== OTHER SETTINGS =====
    risk_free_rate: float = 0.02  # 2% annual
    trading_days: int = 252

    @property
    def max_notional_value(self) -> float:
        """Maximum notional position value with leverage."""
        return self.initial_capital * self.leverage

    @property
    def pip_value_standard_lot(self) -> float:
        """Value of 1 pip for 1 standard lot (100,000 units) in USD."""
        return 10.0  # For EURUSD: 1 pip = $10 per standard lot


@dataclass
class TradingHours:
    """Optimal trading hours configuration (GMT)."""

    # London-NY overlap (most stable for EURUSD)
    stable_start: int = 12  # 12:00 GMT
    stable_end: int = 16    # 16:00 GMT

    # London session
    london_start: int = 7   # 07:00 GMT
    london_end: int = 16    # 16:00 GMT

    # New York session
    ny_start: int = 13      # 13:00 GMT
    ny_end: int = 22        # 22:00 GMT

    # Asian session
    asian_start: int = 0    # 00:00 GMT
    asian_end: int = 9      # 09:00 GMT


@dataclass
class RLConfig:
    """Reinforcement Learning configuration."""

    # Environment
    window_size: int = 60  # 5 hours of 5-min bars
    max_position: float = 1.0

    # PPO hyperparameters
    learning_rate: float = 3e-4
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    clip_range: float = 0.2

    # Training
    total_timesteps: int = 100000
    eval_freq: int = 5000
    n_eval_episodes: int = 5

    # Early stopping
    max_no_improvement_evals: int = 10
    min_evals: int = 20


@dataclass
class BrokerConfig:
    """Broker API configuration."""

    # OANDA
    oanda_api_key: str = field(default_factory=lambda: os.getenv("OANDA_API_KEY", ""))
    oanda_account_id: str = field(default_factory=lambda: os.getenv("OANDA_ACCOUNT_ID", ""))
    oanda_environment: str = "practice"  # "practice" or "live"

    # API endpoints
    oanda_practice_url: str = "https://api-fxpractice.oanda.com"
    oanda_live_url: str = "https://api-fxtrade.oanda.com"

    @property
    def oanda_url(self) -> str:
        if self.oanda_environment == "live":
            return self.oanda_live_url
        return self.oanda_practice_url


@dataclass
class LoggingConfig:
    """Logging configuration."""

    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_path: Optional[str] = None


# Default configurations
DEFAULT_TRADING_CONFIG = TradingConfig()
DEFAULT_TRADING_HOURS = TradingHours()
DEFAULT_RL_CONFIG = RLConfig()
DEFAULT_BROKER_CONFIG = BrokerConfig()
DEFAULT_LOGGING_CONFIG = LoggingConfig()


def get_config() -> Dict[str, Any]:
    """Get all configuration as a dictionary."""
    return {
        "trading": DEFAULT_TRADING_CONFIG,
        "trading_hours": DEFAULT_TRADING_HOURS,
        "rl": DEFAULT_RL_CONFIG,
        "broker": DEFAULT_BROKER_CONFIG,
        "logging": DEFAULT_LOGGING_CONFIG,
        "paths": {
            "project_root": str(PROJECT_ROOT),
            "data_dir": str(DATA_DIR),
            "models_dir": str(MODELS_DIR),
            "logs_dir": str(LOGS_DIR),
        }
    }

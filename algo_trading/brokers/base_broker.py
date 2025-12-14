"""
Abstract base class for broker integrations.

Defines the interface for order execution, position management,
and account operations.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum
from datetime import datetime


class OrderType(Enum):
    """Order types."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderSide(Enum):
    """Order side."""
    BUY = "buy"
    SELL = "sell"


class OrderStatus(Enum):
    """Order status."""
    PENDING = "pending"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


@dataclass
class Order:
    """Represents a trading order."""
    id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    filled_price: float = 0.0
    created_at: datetime = None
    filled_at: datetime = None
    commission: float = 0.0

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()


@dataclass
class Position:
    """Represents an open position."""
    symbol: str
    quantity: float
    entry_price: float
    current_price: float
    unrealized_pnl: float
    realized_pnl: float = 0.0
    side: str = "long"  # "long" or "short"

    @property
    def market_value(self) -> float:
        return self.quantity * self.current_price


@dataclass
class AccountInfo:
    """Account information."""
    balance: float
    equity: float
    margin_used: float
    margin_available: float
    unrealized_pnl: float
    realized_pnl: float


class Broker(ABC):
    """
    Abstract base class for all broker integrations.

    Brokers handle:
    - Order submission and management
    - Position tracking
    - Account information
    - Market data (optional)
    """

    def __init__(self, **kwargs):
        """Initialize broker connection."""
        self.connected = False

    @abstractmethod
    def connect(self) -> bool:
        """
        Establish connection to broker.

        Returns:
            True if connection successful
        """
        pass

    @abstractmethod
    def disconnect(self) -> None:
        """Close broker connection."""
        pass

    @abstractmethod
    def submit_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        order_type: OrderType = OrderType.MARKET,
        price: Optional[float] = None,
        stop_price: Optional[float] = None,
        **kwargs
    ) -> Order:
        """
        Submit a new order.

        Args:
            symbol: Trading symbol
            side: Buy or sell
            quantity: Order quantity
            order_type: Type of order
            price: Limit price (for limit orders)
            stop_price: Stop price (for stop orders)

        Returns:
            Order object with status
        """
        pass

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an open order.

        Args:
            order_id: Order ID to cancel

        Returns:
            True if cancellation successful
        """
        pass

    @abstractmethod
    def get_order(self, order_id: str) -> Optional[Order]:
        """
        Get order by ID.

        Args:
            order_id: Order ID

        Returns:
            Order object or None if not found
        """
        pass

    @abstractmethod
    def get_open_orders(self, symbol: Optional[str] = None) -> List[Order]:
        """
        Get all open orders.

        Args:
            symbol: Filter by symbol (optional)

        Returns:
            List of open orders
        """
        pass

    @abstractmethod
    def get_position(self, symbol: str) -> Optional[Position]:
        """
        Get current position for symbol.

        Args:
            symbol: Trading symbol

        Returns:
            Position object or None if no position
        """
        pass

    @abstractmethod
    def get_all_positions(self) -> List[Position]:
        """
        Get all open positions.

        Returns:
            List of positions
        """
        pass

    @abstractmethod
    def close_position(self, symbol: str) -> Optional[Order]:
        """
        Close position for symbol.

        Args:
            symbol: Trading symbol

        Returns:
            Close order or None if no position
        """
        pass

    @abstractmethod
    def get_account_info(self) -> AccountInfo:
        """
        Get account information.

        Returns:
            AccountInfo object
        """
        pass

    @abstractmethod
    def get_current_price(self, symbol: str) -> Dict[str, float]:
        """
        Get current bid/ask prices.

        Args:
            symbol: Trading symbol

        Returns:
            Dict with 'bid' and 'ask' prices
        """
        pass

    def is_connected(self) -> bool:
        """Check if broker is connected."""
        return self.connected

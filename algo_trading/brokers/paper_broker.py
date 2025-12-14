"""
Paper trading broker for testing strategies without real money.

Simulates order execution with realistic slippage and commission.
"""
import uuid
from datetime import datetime
from typing import Dict, Optional, List, Callable
import logging

from .base_broker import (
    Broker, Order, Position, AccountInfo,
    OrderType, OrderSide, OrderStatus
)

logger = logging.getLogger(__name__)


class PaperBroker(Broker):
    """
    Paper trading broker for backtesting and forward testing.

    Features:
    - Simulated order execution
    - Configurable commission and slippage
    - Position tracking
    - PnL calculation
    """

    def __init__(
        self,
        initial_balance: float = 10000.0,
        commission: float = 0.0001,  # 1 pip
        slippage: float = 0.0001,    # 1 pip
        leverage: float = 1.0,
        price_feed: Optional[Callable[[str], Dict[str, float]]] = None
    ):
        """
        Initialize paper broker.

        Args:
            initial_balance: Starting account balance
            commission: Commission rate (fraction of trade value)
            slippage: Slippage rate (fraction of price)
            leverage: Account leverage
            price_feed: Optional callback to get current prices
        """
        super().__init__()

        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.commission_rate = commission
        self.slippage_rate = slippage
        self.leverage = leverage
        self.price_feed = price_feed

        self.orders: Dict[str, Order] = {}
        self.positions: Dict[str, Position] = {}
        self.realized_pnl = 0.0

        # Simulated prices
        self.current_prices: Dict[str, Dict[str, float]] = {}

    def connect(self) -> bool:
        """Paper broker is always connected."""
        self.connected = True
        logger.info("Paper broker connected")
        return True

    def disconnect(self) -> None:
        """Disconnect paper broker."""
        self.connected = False
        logger.info("Paper broker disconnected")

    def set_price(self, symbol: str, bid: float, ask: float) -> None:
        """
        Set current price for a symbol.

        Args:
            symbol: Trading symbol
            bid: Bid price
            ask: Ask price
        """
        self.current_prices[symbol] = {'bid': bid, 'ask': ask}

        # Update position PnL
        if symbol in self.positions:
            self._update_position_pnl(symbol)

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
        Submit and execute a paper order.

        Market orders are executed immediately.
        Limit/stop orders are stored for later execution.
        """
        order_id = str(uuid.uuid4())[:8]

        order = Order(
            id=order_id,
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            price=price,
            stop_price=stop_price
        )

        self.orders[order_id] = order

        # Execute market orders immediately
        if order_type == OrderType.MARKET:
            self._execute_order(order)

        logger.info(f"Order submitted: {order}")
        return order

    def _execute_order(self, order: Order) -> None:
        """Execute an order."""
        prices = self.get_current_price(order.symbol)

        if not prices:
            order.status = OrderStatus.REJECTED
            logger.warning(f"Order rejected - no price for {order.symbol}")
            return

        # Determine execution price with slippage
        if order.side == OrderSide.BUY:
            base_price = prices['ask']
            exec_price = base_price * (1 + self.slippage_rate)
        else:
            base_price = prices['bid']
            exec_price = base_price * (1 - self.slippage_rate)

        # Calculate commission
        trade_value = order.quantity * exec_price
        commission = trade_value * self.commission_rate

        # Check if we have enough balance
        required_margin = trade_value / self.leverage
        if required_margin + commission > self.balance:
            order.status = OrderStatus.REJECTED
            logger.warning(f"Order rejected - insufficient balance")
            return

        # Update order
        order.filled_quantity = order.quantity
        order.filled_price = exec_price
        order.commission = commission
        order.status = OrderStatus.FILLED
        order.filled_at = datetime.now()

        # Update balance
        self.balance -= commission

        # Update position
        self._update_position(order)

        logger.info(f"Order executed: {order}")

    def _update_position(self, order: Order) -> None:
        """Update position after order execution."""
        symbol = order.symbol

        if symbol in self.positions:
            pos = self.positions[symbol]

            # Same direction - add to position
            if (order.side == OrderSide.BUY and pos.side == "long") or \
               (order.side == OrderSide.SELL and pos.side == "short"):
                # Average entry price
                total_qty = pos.quantity + order.filled_quantity
                pos.entry_price = (
                    (pos.entry_price * pos.quantity + order.filled_price * order.filled_quantity)
                    / total_qty
                )
                pos.quantity = total_qty

            # Opposite direction - reduce or reverse position
            else:
                if order.filled_quantity >= pos.quantity:
                    # Close position and possibly open reverse
                    close_qty = pos.quantity
                    remaining_qty = order.filled_quantity - close_qty

                    # Calculate realized PnL
                    if pos.side == "long":
                        pnl = (order.filled_price - pos.entry_price) * close_qty
                    else:
                        pnl = (pos.entry_price - order.filled_price) * close_qty

                    self.realized_pnl += pnl
                    self.balance += pnl
                    pos.realized_pnl = pnl

                    # Remove old position
                    del self.positions[symbol]

                    # Open reverse position if there's remaining quantity
                    if remaining_qty > 0:
                        new_side = "long" if order.side == OrderSide.BUY else "short"
                        self.positions[symbol] = Position(
                            symbol=symbol,
                            quantity=remaining_qty,
                            entry_price=order.filled_price,
                            current_price=order.filled_price,
                            unrealized_pnl=0.0,
                            side=new_side
                        )
                else:
                    # Reduce position
                    if pos.side == "long":
                        pnl = (order.filled_price - pos.entry_price) * order.filled_quantity
                    else:
                        pnl = (pos.entry_price - order.filled_price) * order.filled_quantity

                    self.realized_pnl += pnl
                    self.balance += pnl
                    pos.quantity -= order.filled_quantity
                    pos.realized_pnl += pnl

        else:
            # Open new position
            side = "long" if order.side == OrderSide.BUY else "short"
            self.positions[symbol] = Position(
                symbol=symbol,
                quantity=order.filled_quantity,
                entry_price=order.filled_price,
                current_price=order.filled_price,
                unrealized_pnl=0.0,
                side=side
            )

    def _update_position_pnl(self, symbol: str) -> None:
        """Update position unrealized PnL."""
        if symbol not in self.positions:
            return

        pos = self.positions[symbol]
        prices = self.get_current_price(symbol)

        if not prices:
            return

        current = (prices['bid'] + prices['ask']) / 2
        pos.current_price = current

        if pos.side == "long":
            pos.unrealized_pnl = (current - pos.entry_price) * pos.quantity
        else:
            pos.unrealized_pnl = (pos.entry_price - current) * pos.quantity

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an open order."""
        if order_id not in self.orders:
            return False

        order = self.orders[order_id]
        if order.status == OrderStatus.PENDING:
            order.status = OrderStatus.CANCELLED
            logger.info(f"Order cancelled: {order_id}")
            return True

        return False

    def get_order(self, order_id: str) -> Optional[Order]:
        """Get order by ID."""
        return self.orders.get(order_id)

    def get_open_orders(self, symbol: Optional[str] = None) -> List[Order]:
        """Get all open orders."""
        orders = [
            o for o in self.orders.values()
            if o.status == OrderStatus.PENDING
        ]

        if symbol:
            orders = [o for o in orders if o.symbol == symbol]

        return orders

    def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for symbol."""
        return self.positions.get(symbol)

    def get_all_positions(self) -> List[Position]:
        """Get all positions."""
        return list(self.positions.values())

    def close_position(self, symbol: str) -> Optional[Order]:
        """Close position for symbol."""
        pos = self.positions.get(symbol)
        if not pos:
            return None

        side = OrderSide.SELL if pos.side == "long" else OrderSide.BUY
        return self.submit_order(symbol, side, pos.quantity)

    def get_account_info(self) -> AccountInfo:
        """Get account information."""
        unrealized = sum(p.unrealized_pnl for p in self.positions.values())
        margin_used = sum(
            p.quantity * p.current_price / self.leverage
            for p in self.positions.values()
        )

        return AccountInfo(
            balance=self.balance,
            equity=self.balance + unrealized,
            margin_used=margin_used,
            margin_available=self.balance - margin_used,
            unrealized_pnl=unrealized,
            realized_pnl=self.realized_pnl
        )

    def get_current_price(self, symbol: str) -> Dict[str, float]:
        """Get current bid/ask prices."""
        # Try price feed first
        if self.price_feed:
            return self.price_feed(symbol)

        return self.current_prices.get(symbol, {})

    def reset(self) -> None:
        """Reset broker to initial state."""
        self.balance = self.initial_balance
        self.orders.clear()
        self.positions.clear()
        self.realized_pnl = 0.0
        logger.info("Paper broker reset")

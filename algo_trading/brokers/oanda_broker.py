"""
OANDA broker integration for live forex trading.

Requires:
- OANDA API key
- OANDA account ID
"""
import logging
from typing import Dict, Optional, List
from datetime import datetime

from .base_broker import (
    Broker, Order, Position, AccountInfo,
    OrderType, OrderSide, OrderStatus
)

logger = logging.getLogger(__name__)


class OANDABroker(Broker):
    """
    OANDA broker integration for live trading.

    Uses OANDA v20 REST API for:
    - Order execution
    - Position management
    - Account information
    - Real-time prices
    """

    def __init__(
        self,
        api_key: str,
        account_id: str,
        environment: str = "practice"  # "practice" or "live"
    ):
        """
        Initialize OANDA broker.

        Args:
            api_key: OANDA API key
            account_id: OANDA account ID
            environment: "practice" for demo, "live" for real trading
        """
        super().__init__()

        self.api_key = api_key
        self.account_id = account_id
        self.environment = environment

        # API URLs
        if environment == "live":
            self.api_url = "https://api-fxtrade.oanda.com"
            self.stream_url = "https://stream-fxtrade.oanda.com"
        else:
            self.api_url = "https://api-fxpractice.oanda.com"
            self.stream_url = "https://stream-fxpractice.oanda.com"

        self.api = None

    def connect(self) -> bool:
        """Connect to OANDA API."""
        try:
            import oandapyV20
            self.api = oandapyV20.API(
                access_token=self.api_key,
                environment=self.environment
            )
            self.connected = True
            logger.info(f"Connected to OANDA ({self.environment})")
            return True
        except ImportError:
            logger.error("oandapyV20 not installed. Run: pip install oandapyV20")
            return False
        except Exception as e:
            logger.error(f"Failed to connect to OANDA: {e}")
            return False

    def disconnect(self) -> None:
        """Disconnect from OANDA."""
        self.api = None
        self.connected = False
        logger.info("Disconnected from OANDA")

    def submit_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        order_type: OrderType = OrderType.MARKET,
        price: Optional[float] = None,
        stop_price: Optional[float] = None,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        **kwargs
    ) -> Order:
        """Submit order to OANDA."""
        if not self.api:
            raise ConnectionError("Not connected to OANDA")

        try:
            import oandapyV20.endpoints.orders as orders
        except ImportError:
            raise ImportError("oandapyV20 not installed")

        # Convert symbol format (EURUSD -> EUR_USD)
        oanda_symbol = symbol[:3] + "_" + symbol[3:]

        # Build order data
        units = int(quantity * (1 if side == OrderSide.BUY else -1))

        order_data = {
            "order": {
                "instrument": oanda_symbol,
                "units": str(units),
                "type": self._convert_order_type(order_type),
                "timeInForce": "FOK" if order_type == OrderType.MARKET else "GTC",
            }
        }

        if order_type in [OrderType.LIMIT, OrderType.STOP_LIMIT] and price:
            order_data["order"]["price"] = str(price)

        if order_type in [OrderType.STOP, OrderType.STOP_LIMIT] and stop_price:
            order_data["order"]["priceBound"] = str(stop_price)

        if stop_loss:
            order_data["order"]["stopLossOnFill"] = {"price": str(stop_loss)}

        if take_profit:
            order_data["order"]["takeProfitOnFill"] = {"price": str(take_profit)}

        try:
            r = orders.OrderCreate(self.account_id, data=order_data)
            response = self.api.request(r)

            # Parse response
            if "orderFillTransaction" in response:
                fill = response["orderFillTransaction"]
                return Order(
                    id=fill["id"],
                    symbol=symbol,
                    side=side,
                    order_type=order_type,
                    quantity=abs(int(fill["units"])),
                    status=OrderStatus.FILLED,
                    filled_quantity=abs(int(fill["units"])),
                    filled_price=float(fill["price"]),
                    filled_at=datetime.fromisoformat(fill["time"].replace("Z", "+00:00"))
                )
            elif "orderCreateTransaction" in response:
                create = response["orderCreateTransaction"]
                return Order(
                    id=create["id"],
                    symbol=symbol,
                    side=side,
                    order_type=order_type,
                    quantity=quantity,
                    price=price,
                    status=OrderStatus.PENDING
                )
            else:
                logger.warning(f"Unexpected order response: {response}")
                return Order(
                    id="unknown",
                    symbol=symbol,
                    side=side,
                    order_type=order_type,
                    quantity=quantity,
                    status=OrderStatus.REJECTED
                )

        except Exception as e:
            logger.error(f"Order submission failed: {e}")
            return Order(
                id="error",
                symbol=symbol,
                side=side,
                order_type=order_type,
                quantity=quantity,
                status=OrderStatus.REJECTED
            )

    def _convert_order_type(self, order_type: OrderType) -> str:
        """Convert order type to OANDA format."""
        mapping = {
            OrderType.MARKET: "MARKET",
            OrderType.LIMIT: "LIMIT",
            OrderType.STOP: "STOP",
            OrderType.STOP_LIMIT: "STOP",
        }
        return mapping.get(order_type, "MARKET")

    def cancel_order(self, order_id: str) -> bool:
        """Cancel order on OANDA."""
        if not self.api:
            return False

        try:
            import oandapyV20.endpoints.orders as orders

            r = orders.OrderCancel(self.account_id, orderID=order_id)
            self.api.request(r)
            logger.info(f"Order {order_id} cancelled")
            return True

        except Exception as e:
            logger.error(f"Failed to cancel order {order_id}: {e}")
            return False

    def get_order(self, order_id: str) -> Optional[Order]:
        """Get order details from OANDA."""
        if not self.api:
            return None

        try:
            import oandapyV20.endpoints.orders as orders

            r = orders.OrderDetails(self.account_id, orderID=order_id)
            response = self.api.request(r)

            order_data = response.get("order", {})
            return self._parse_order(order_data)

        except Exception as e:
            logger.error(f"Failed to get order {order_id}: {e}")
            return None

    def _parse_order(self, order_data: dict) -> Order:
        """Parse OANDA order data to Order object."""
        symbol = order_data.get("instrument", "").replace("_", "")
        units = int(order_data.get("units", 0))

        return Order(
            id=order_data.get("id", ""),
            symbol=symbol,
            side=OrderSide.BUY if units > 0 else OrderSide.SELL,
            order_type=OrderType.MARKET,  # Simplified
            quantity=abs(units),
            status=self._parse_order_status(order_data.get("state", ""))
        )

    def _parse_order_status(self, state: str) -> OrderStatus:
        """Parse OANDA order state to OrderStatus."""
        mapping = {
            "PENDING": OrderStatus.PENDING,
            "FILLED": OrderStatus.FILLED,
            "TRIGGERED": OrderStatus.FILLED,
            "CANCELLED": OrderStatus.CANCELLED,
        }
        return mapping.get(state, OrderStatus.PENDING)

    def get_open_orders(self, symbol: Optional[str] = None) -> List[Order]:
        """Get pending orders from OANDA."""
        if not self.api:
            return []

        try:
            import oandapyV20.endpoints.orders as orders

            r = orders.OrdersPending(self.account_id)
            response = self.api.request(r)

            open_orders = []
            for order_data in response.get("orders", []):
                order = self._parse_order(order_data)
                if symbol is None or order.symbol == symbol:
                    open_orders.append(order)

            return open_orders

        except Exception as e:
            logger.error(f"Failed to get open orders: {e}")
            return []

    def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for symbol from OANDA."""
        if not self.api:
            return None

        try:
            import oandapyV20.endpoints.positions as positions

            oanda_symbol = symbol[:3] + "_" + symbol[3:]
            r = positions.PositionDetails(self.account_id, instrument=oanda_symbol)
            response = self.api.request(r)

            pos_data = response.get("position", {})

            # Long position
            long_units = int(pos_data.get("long", {}).get("units", 0))
            short_units = int(pos_data.get("short", {}).get("units", 0))

            if long_units > 0:
                return Position(
                    symbol=symbol,
                    quantity=long_units,
                    entry_price=float(pos_data["long"].get("averagePrice", 0)),
                    current_price=float(pos_data["long"].get("averagePrice", 0)),
                    unrealized_pnl=float(pos_data["long"].get("unrealizedPL", 0)),
                    side="long"
                )
            elif short_units < 0:
                return Position(
                    symbol=symbol,
                    quantity=abs(short_units),
                    entry_price=float(pos_data["short"].get("averagePrice", 0)),
                    current_price=float(pos_data["short"].get("averagePrice", 0)),
                    unrealized_pnl=float(pos_data["short"].get("unrealizedPL", 0)),
                    side="short"
                )

            return None

        except Exception as e:
            logger.error(f"Failed to get position for {symbol}: {e}")
            return None

    def get_all_positions(self) -> List[Position]:
        """Get all positions from OANDA."""
        if not self.api:
            return []

        try:
            import oandapyV20.endpoints.positions as positions

            r = positions.OpenPositions(self.account_id)
            response = self.api.request(r)

            pos_list = []
            for pos_data in response.get("positions", []):
                symbol = pos_data.get("instrument", "").replace("_", "")

                long_units = int(pos_data.get("long", {}).get("units", 0))
                short_units = int(pos_data.get("short", {}).get("units", 0))

                if long_units > 0:
                    pos_list.append(Position(
                        symbol=symbol,
                        quantity=long_units,
                        entry_price=float(pos_data["long"].get("averagePrice", 0)),
                        current_price=float(pos_data["long"].get("averagePrice", 0)),
                        unrealized_pnl=float(pos_data["long"].get("unrealizedPL", 0)),
                        side="long"
                    ))
                elif short_units < 0:
                    pos_list.append(Position(
                        symbol=symbol,
                        quantity=abs(short_units),
                        entry_price=float(pos_data["short"].get("averagePrice", 0)),
                        current_price=float(pos_data["short"].get("averagePrice", 0)),
                        unrealized_pnl=float(pos_data["short"].get("unrealizedPL", 0)),
                        side="short"
                    ))

            return pos_list

        except Exception as e:
            logger.error(f"Failed to get positions: {e}")
            return []

    def close_position(self, symbol: str) -> Optional[Order]:
        """Close position on OANDA."""
        if not self.api:
            return None

        try:
            import oandapyV20.endpoints.positions as positions

            oanda_symbol = symbol[:3] + "_" + symbol[3:]
            data = {"longUnits": "ALL", "shortUnits": "ALL"}

            r = positions.PositionClose(self.account_id, instrument=oanda_symbol, data=data)
            response = self.api.request(r)

            # Parse close response
            if "longOrderFillTransaction" in response:
                fill = response["longOrderFillTransaction"]
                return Order(
                    id=fill["id"],
                    symbol=symbol,
                    side=OrderSide.SELL,
                    order_type=OrderType.MARKET,
                    quantity=abs(int(fill["units"])),
                    status=OrderStatus.FILLED,
                    filled_quantity=abs(int(fill["units"])),
                    filled_price=float(fill["price"])
                )

            return None

        except Exception as e:
            logger.error(f"Failed to close position {symbol}: {e}")
            return None

    def get_account_info(self) -> AccountInfo:
        """Get account info from OANDA."""
        if not self.api:
            return AccountInfo(0, 0, 0, 0, 0, 0)

        try:
            import oandapyV20.endpoints.accounts as accounts

            r = accounts.AccountSummary(self.account_id)
            response = self.api.request(r)

            acc = response.get("account", {})

            return AccountInfo(
                balance=float(acc.get("balance", 0)),
                equity=float(acc.get("NAV", 0)),
                margin_used=float(acc.get("marginUsed", 0)),
                margin_available=float(acc.get("marginAvailable", 0)),
                unrealized_pnl=float(acc.get("unrealizedPL", 0)),
                realized_pnl=float(acc.get("pl", 0))
            )

        except Exception as e:
            logger.error(f"Failed to get account info: {e}")
            return AccountInfo(0, 0, 0, 0, 0, 0)

    def get_current_price(self, symbol: str) -> Dict[str, float]:
        """Get current prices from OANDA."""
        if not self.api:
            return {}

        try:
            import oandapyV20.endpoints.pricing as pricing

            oanda_symbol = symbol[:3] + "_" + symbol[3:]
            params = {"instruments": oanda_symbol}

            r = pricing.PricingInfo(self.account_id, params=params)
            response = self.api.request(r)

            for price in response.get("prices", []):
                if price.get("instrument") == oanda_symbol:
                    return {
                        "bid": float(price["bids"][0]["price"]),
                        "ask": float(price["asks"][0]["price"])
                    }

            return {}

        except Exception as e:
            logger.error(f"Failed to get price for {symbol}: {e}")
            return {}

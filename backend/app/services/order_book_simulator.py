"""
Order Book Simulator
Simulates realistic market microstructure with:
- Bid/ask depth and spread dynamics
- Price impact from large orders (market impact model)
- Slippage modeling
- Liquidity replenishment by market makers
- Cascade effects (stop-loss waterfalls, liquidation cascades)
"""

import math
import random
import bisect
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field

from ..utils.logger import get_logger

logger = get_logger('mirofish.orderbook')


@dataclass
class Fill:
    """Result of a market order execution."""
    agent_id: int
    side: str  # "buy" or "sell"
    requested_qty: float
    filled_qty: float
    avg_fill_price: float
    slippage_pct: float
    market_impact_pct: float
    fees: float
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class LimitOrder:
    """A resting limit order on the book."""
    order_id: str
    agent_id: int
    side: str  # "buy" or "sell"
    price: float
    quantity: float
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class Trade:
    """A completed trade."""
    price: float
    quantity: float
    side: str  # aggressor side
    agent_id: int
    timestamp: str
    is_liquidation: bool = False


class OrderBookSimulator:
    """
    Simulates a realistic order book with market microstructure dynamics.

    Key features:
    - Configurable initial depth and spread
    - Kyle's lambda market impact model
    - Liquidity replenishment after large trades
    - Stop-loss cascade simulation
    - Trade history and VWAP calculation
    """

    def __init__(
        self,
        initial_price: float,
        tick_size: float = 0.01,
        initial_depth_per_level: float = 10.0,
        num_levels: int = 50,
        spread_bps: float = 10.0,
        fee_rate: float = 0.001,
        impact_coefficient: float = 0.1,
    ):
        """
        Args:
            initial_price: Starting mid price
            tick_size: Minimum price increment
            initial_depth_per_level: Base quantity at each price level
            num_levels: Number of price levels per side
            spread_bps: Initial spread in basis points
            fee_rate: Trading fee rate (0.001 = 0.1%)
            impact_coefficient: Kyle's lambda — controls price sensitivity to order flow
        """
        self.tick_size = tick_size
        self.fee_rate = fee_rate
        self.impact_coefficient = impact_coefficient
        self.mid_price = initial_price

        # Order book: price → total quantity
        self.bids: Dict[float, float] = {}  # price → qty
        self.asks: Dict[float, float] = {}  # price → qty

        # Resting limit orders for tracking
        self._limit_orders: Dict[str, LimitOrder] = {}
        self._next_order_id = 1

        # Trade history
        self.trades: List[Trade] = []

        # Stop losses: price → [(agent_id, qty, side)]
        self._stop_losses: Dict[float, List[Tuple[int, float, str]]] = {}

        # Liquidation levels: price → [(agent_id, qty)]
        self._liquidations: Dict[float, List[Tuple[int, float]]] = {}

        # Initialize the book with synthetic depth
        self._initialize_book(initial_price, spread_bps, initial_depth_per_level, num_levels)

    def _initialize_book(
        self, price: float, spread_bps: float, depth: float, levels: int
    ):
        """Seed the order book with initial liquidity."""
        half_spread = price * spread_bps / 10000 / 2

        for i in range(levels):
            # Depth decreases further from mid (realistic shape)
            depth_multiplier = max(0.3, 1.0 - (i * 0.015))
            level_depth = depth * depth_multiplier * random.uniform(0.7, 1.3)

            bid_price = round(price - half_spread - (i * self.tick_size), 8)
            ask_price = round(price + half_spread + (i * self.tick_size), 8)

            if bid_price > 0:
                self.bids[bid_price] = round(level_depth, 4)
            self.asks[ask_price] = round(level_depth, 4)

    @property
    def best_bid(self) -> float:
        return max(self.bids.keys()) if self.bids else 0.0

    @property
    def best_ask(self) -> float:
        return min(self.asks.keys()) if self.asks else float('inf')

    @property
    def spread(self) -> float:
        return round(self.best_ask - self.best_bid, 8)

    @property
    def spread_bps(self) -> float:
        if self.mid_price <= 0:
            return 0
        return round(self.spread / self.mid_price * 10000, 2)

    def market_buy(self, quantity: float, agent_id: int) -> Fill:
        """
        Execute a market buy order — walks up the ask side consuming liquidity.

        Returns Fill with average price, slippage, and market impact.
        """
        if not self.asks:
            return Fill(
                agent_id=agent_id, side="buy", requested_qty=quantity,
                filled_qty=0, avg_fill_price=0, slippage_pct=0,
                market_impact_pct=0, fees=0,
            )

        pre_mid = self.mid_price
        filled_qty = 0.0
        total_cost = 0.0
        levels_consumed = 0

        # Sort asks ascending
        sorted_asks = sorted(self.asks.keys())

        for price in sorted_asks:
            if filled_qty >= quantity:
                break

            available = self.asks[price]
            fill_at_level = min(available, quantity - filled_qty)

            total_cost += fill_at_level * price
            filled_qty += fill_at_level

            self.asks[price] -= fill_at_level
            if self.asks[price] <= 1e-10:
                del self.asks[price]
                levels_consumed += 1

        if filled_qty <= 0:
            return Fill(
                agent_id=agent_id, side="buy", requested_qty=quantity,
                filled_qty=0, avg_fill_price=0, slippage_pct=0,
                market_impact_pct=0, fees=0,
            )

        avg_price = total_cost / filled_qty
        fees = total_cost * self.fee_rate

        # Update mid price with market impact
        impact = self.impact_coefficient * math.sqrt(filled_qty) * self.tick_size
        self.mid_price = round(self.mid_price + impact, 8)

        slippage_pct = ((avg_price - pre_mid) / pre_mid) * 100 if pre_mid > 0 else 0
        impact_pct = ((self.mid_price - pre_mid) / pre_mid) * 100 if pre_mid > 0 else 0

        # Record trade
        trade = Trade(
            price=round(avg_price, 8), quantity=round(filled_qty, 8),
            side="buy", agent_id=agent_id,
            timestamp=datetime.utcnow().isoformat(),
        )
        self.trades.append(trade)

        # Trigger stop losses and liquidations
        self._check_cascades(direction="up")

        # Replenish liquidity (market makers re-enter)
        if levels_consumed > 0:
            self._replenish_liquidity(side="ask", levels=levels_consumed)

        return Fill(
            agent_id=agent_id, side="buy", requested_qty=quantity,
            filled_qty=round(filled_qty, 8), avg_fill_price=round(avg_price, 8),
            slippage_pct=round(slippage_pct, 4), market_impact_pct=round(impact_pct, 4),
            fees=round(fees, 8),
        )

    def market_sell(self, quantity: float, agent_id: int) -> Fill:
        """
        Execute a market sell order — walks down the bid side consuming liquidity.
        """
        if not self.bids:
            return Fill(
                agent_id=agent_id, side="sell", requested_qty=quantity,
                filled_qty=0, avg_fill_price=0, slippage_pct=0,
                market_impact_pct=0, fees=0,
            )

        pre_mid = self.mid_price
        filled_qty = 0.0
        total_revenue = 0.0
        levels_consumed = 0

        # Sort bids descending (best bid first)
        sorted_bids = sorted(self.bids.keys(), reverse=True)

        for price in sorted_bids:
            if filled_qty >= quantity:
                break

            available = self.bids[price]
            fill_at_level = min(available, quantity - filled_qty)

            total_revenue += fill_at_level * price
            filled_qty += fill_at_level

            self.bids[price] -= fill_at_level
            if self.bids[price] <= 1e-10:
                del self.bids[price]
                levels_consumed += 1

        if filled_qty <= 0:
            return Fill(
                agent_id=agent_id, side="sell", requested_qty=quantity,
                filled_qty=0, avg_fill_price=0, slippage_pct=0,
                market_impact_pct=0, fees=0,
            )

        avg_price = total_revenue / filled_qty
        fees = total_revenue * self.fee_rate

        # Update mid price
        impact = self.impact_coefficient * math.sqrt(filled_qty) * self.tick_size
        self.mid_price = round(self.mid_price - impact, 8)

        slippage_pct = ((pre_mid - avg_price) / pre_mid) * 100 if pre_mid > 0 else 0
        impact_pct = ((pre_mid - self.mid_price) / pre_mid) * 100 if pre_mid > 0 else 0

        trade = Trade(
            price=round(avg_price, 8), quantity=round(filled_qty, 8),
            side="sell", agent_id=agent_id,
            timestamp=datetime.utcnow().isoformat(),
        )
        self.trades.append(trade)

        self._check_cascades(direction="down")

        if levels_consumed > 0:
            self._replenish_liquidity(side="bid", levels=levels_consumed)

        return Fill(
            agent_id=agent_id, side="sell", requested_qty=quantity,
            filled_qty=round(filled_qty, 8), avg_fill_price=round(avg_price, 8),
            slippage_pct=round(slippage_pct, 4), market_impact_pct=round(impact_pct, 4),
            fees=round(fees, 8),
        )

    def limit_order(
        self, side: str, price: float, quantity: float, agent_id: int
    ) -> str:
        """
        Place a limit order. Returns order_id.
        If the order crosses the spread, it fills immediately as a market order.
        """
        # Check for immediate fill (crossing order)
        if side == "buy" and price >= self.best_ask:
            self.market_buy(quantity, agent_id)
            return f"filled_immediately_{self._next_order_id}"
        if side == "sell" and price <= self.best_bid:
            self.market_sell(quantity, agent_id)
            return f"filled_immediately_{self._next_order_id}"

        order_id = f"order_{self._next_order_id}"
        self._next_order_id += 1

        order = LimitOrder(
            order_id=order_id, agent_id=agent_id,
            side=side, price=round(price, 8), quantity=quantity,
        )
        self._limit_orders[order_id] = order

        # Add to book
        if side == "buy":
            self.bids[price] = self.bids.get(price, 0) + quantity
        else:
            self.asks[price] = self.asks.get(price, 0) + quantity

        return order_id

    def cancel_order(self, order_id: str) -> bool:
        """Cancel a resting limit order."""
        order = self._limit_orders.pop(order_id, None)
        if not order:
            return False

        if order.side == "buy":
            if order.price in self.bids:
                self.bids[order.price] -= order.quantity
                if self.bids[order.price] <= 1e-10:
                    del self.bids[order.price]
        else:
            if order.price in self.asks:
                self.asks[order.price] -= order.quantity
                if self.asks[order.price] <= 1e-10:
                    del self.asks[order.price]

        return True

    def add_stop_loss(self, price: float, agent_id: int, quantity: float, side: str):
        """Register a stop-loss that triggers a market order when price crosses."""
        if price not in self._stop_losses:
            self._stop_losses[price] = []
        self._stop_losses[price].append((agent_id, quantity, side))

    def add_liquidation_level(self, price: float, agent_id: int, quantity: float):
        """Register a liquidation level (forced sell)."""
        if price not in self._liquidations:
            self._liquidations[price] = []
        self._liquidations[price].append((agent_id, quantity))

    def _check_cascades(self, direction: str):
        """Check and trigger stop-losses and liquidations."""
        triggered_stops = []
        triggered_liqs = []

        current_price = self.mid_price

        for price, entries in list(self._stop_losses.items()):
            if direction == "down" and current_price <= price:
                triggered_stops.extend(entries)
                del self._stop_losses[price]
            elif direction == "up" and current_price >= price:
                triggered_stops.extend(entries)
                del self._stop_losses[price]

        for price, entries in list(self._liquidations.items()):
            if direction == "down" and current_price <= price:
                triggered_liqs.extend(entries)
                del self._liquidations[price]

        # Execute cascades (this can cause further cascades — capped at 5 rounds)
        cascade_round = 0
        while (triggered_stops or triggered_liqs) and cascade_round < 5:
            cascade_round += 1

            for agent_id, qty, side in triggered_stops:
                if side == "sell":
                    self.market_sell(qty, agent_id)
                else:
                    self.market_buy(qty, agent_id)

            for agent_id, qty in triggered_liqs:
                fill = self.market_sell(qty, agent_id)
                if fill.filled_qty > 0:
                    self.trades[-1].is_liquidation = True

            triggered_stops = []
            triggered_liqs = []

            # Check for new cascades
            for price, entries in list(self._stop_losses.items()):
                if self.mid_price <= price:
                    triggered_stops.extend(entries)
                    del self._stop_losses[price]
            for price, entries in list(self._liquidations.items()):
                if self.mid_price <= price:
                    triggered_liqs.extend(entries)
                    del self._liquidations[price]

    def _replenish_liquidity(self, side: str, levels: int):
        """
        Market makers re-enter after liquidity is consumed.
        Replenishes at slightly worse prices with reduced depth.
        """
        replenish_depth = 3.0  # Reduced depth after big move

        if side == "ask":
            if not self.asks:
                base = self.mid_price + self.spread / 2
            else:
                base = max(self.asks.keys())
            for i in range(levels):
                price = round(base + (i + 1) * self.tick_size, 8)
                depth = replenish_depth * random.uniform(0.5, 1.5)
                self.asks[price] = self.asks.get(price, 0) + round(depth, 4)
        else:
            if not self.bids:
                base = self.mid_price - self.spread / 2
            else:
                base = min(self.bids.keys())
            for i in range(levels):
                price = round(base - (i + 1) * self.tick_size, 8)
                if price > 0:
                    depth = replenish_depth * random.uniform(0.5, 1.5)
                    self.bids[price] = self.bids.get(price, 0) + round(depth, 4)

    def get_depth(self, levels: int = 10) -> Dict[str, Any]:
        """Get order book depth snapshot."""
        sorted_bids = sorted(self.bids.items(), key=lambda x: -x[0])[:levels]
        sorted_asks = sorted(self.asks.items(), key=lambda x: x[0])[:levels]

        return {
            "mid_price": round(self.mid_price, 8),
            "best_bid": self.best_bid,
            "best_ask": self.best_ask,
            "spread": self.spread,
            "spread_bps": self.spread_bps,
            "bids": [[round(p, 8), round(q, 4)] for p, q in sorted_bids],
            "asks": [[round(p, 8), round(q, 4)] for p, q in sorted_asks],
            "total_bid_depth": round(sum(self.bids.values()), 4),
            "total_ask_depth": round(sum(self.asks.values()), 4),
            "bid_ask_imbalance": round(
                (sum(self.bids.values()) - sum(self.asks.values())) /
                max(sum(self.bids.values()) + sum(self.asks.values()), 1), 4
            ),
        }

    def get_vwap(self, n_trades: int = 50) -> float:
        """Volume-weighted average price over last N trades."""
        recent = self.trades[-n_trades:] if self.trades else []
        if not recent:
            return self.mid_price

        total_qty = sum(t.quantity for t in recent)
        if total_qty <= 0:
            return self.mid_price

        vwap = sum(t.price * t.quantity for t in recent) / total_qty
        return round(vwap, 8)

    def get_trade_flow(self, n_trades: int = 100) -> Dict[str, Any]:
        """Analyze recent trade flow — buy vs sell pressure."""
        recent = self.trades[-n_trades:] if self.trades else []

        buy_vol = sum(t.quantity for t in recent if t.side == "buy")
        sell_vol = sum(t.quantity for t in recent if t.side == "sell")
        total = buy_vol + sell_vol

        return {
            "n_trades": len(recent),
            "buy_volume": round(buy_vol, 4),
            "sell_volume": round(sell_vol, 4),
            "net_flow": round(buy_vol - sell_vol, 4),
            "buy_pct": round(buy_vol / total * 100, 2) if total > 0 else 50.0,
            "sell_pct": round(sell_vol / total * 100, 2) if total > 0 else 50.0,
            "liquidation_volume": round(
                sum(t.quantity for t in recent if t.is_liquidation), 4
            ),
            "vwap": self.get_vwap(n_trades),
        }

    def get_price_history(self) -> List[Dict[str, Any]]:
        """Return trade-by-trade price history."""
        return [
            {
                "price": t.price,
                "quantity": t.quantity,
                "side": t.side,
                "agent_id": t.agent_id,
                "timestamp": t.timestamp,
                "is_liquidation": t.is_liquidation,
            }
            for t in self.trades
        ]

    def snapshot(self) -> Dict[str, Any]:
        """Full order book state snapshot."""
        return {
            "depth": self.get_depth(20),
            "trade_flow": self.get_trade_flow(),
            "total_trades": len(self.trades),
            "pending_stop_losses": sum(len(v) for v in self._stop_losses.values()),
            "pending_liquidations": sum(len(v) for v in self._liquidations.values()),
            "resting_orders": len(self._limit_orders),
        }

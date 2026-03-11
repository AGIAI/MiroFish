"""
External Ingestion Service
Receives data FROM external systems (OpenClaw, Meridian, or any execution layer):
- Execution fills (actual entry price, slippage, fees)
- Portfolio state updates (positions, PnL, drawdown)
- External signals (signals from other systems to blend)
- Performance feedback (closed trades, realized returns)
- Market microstructure data (real order book snapshots, trade flow)

This closes the feedback loop: MiroFish signals → OpenClaw executes → OpenClaw reports back.
"""

import threading
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

from ..models.signal import PortfolioState, Position, SignalDirection
from ..utils.logger import get_logger

logger = get_logger('mirofish.external_ingestion')


@dataclass
class ExecutionFill:
    """An execution fill reported by OpenClaw/Meridian."""
    signal_id: str
    asset: str
    direction: str  # LONG, SHORT
    requested_price: float
    actual_price: float
    slippage_pct: float
    quantity: float
    fees: float
    exchange: str
    fill_time: str
    latency_ms: int = 0  # Signal-to-fill latency

    def to_dict(self) -> Dict[str, Any]:
        return {
            "signal_id": self.signal_id,
            "asset": self.asset,
            "direction": self.direction,
            "requested_price": self.requested_price,
            "actual_price": self.actual_price,
            "slippage_pct": self.slippage_pct,
            "quantity": self.quantity,
            "fees": self.fees,
            "exchange": self.exchange,
            "fill_time": self.fill_time,
            "latency_ms": self.latency_ms,
        }


@dataclass
class ClosedTrade:
    """A completed trade reported by OpenClaw/Meridian."""
    signal_id: str
    asset: str
    direction: str
    entry_price: float
    exit_price: float
    return_pct: float
    holding_hours: float
    fees_total: float
    exit_reason: str  # "take_profit", "stop_loss", "manual", "expired", "liquidated"
    closed_at: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "signal_id": self.signal_id,
            "asset": self.asset,
            "direction": self.direction,
            "entry_price": self.entry_price,
            "exit_price": self.exit_price,
            "return_pct": self.return_pct,
            "holding_hours": self.holding_hours,
            "fees_total": self.fees_total,
            "exit_reason": self.exit_reason,
            "closed_at": self.closed_at,
        }


@dataclass
class ExternalSignal:
    """A signal from an external system (OpenClaw's own models, etc.)."""
    source: str  # "openclaw", "meridian", "custom"
    signal_id: str
    asset: str
    direction: str
    conviction: float
    entry_price: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    received_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "signal_id": self.signal_id,
            "asset": self.asset,
            "direction": self.direction,
            "conviction": self.conviction,
            "entry_price": self.entry_price,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "metadata": self.metadata,
            "received_at": self.received_at,
        }


@dataclass
class OrderBookSnapshot:
    """Real order book snapshot from an execution venue."""
    asset: str
    exchange: str
    bids: List[List[float]]  # [[price, qty], ...]
    asks: List[List[float]]
    best_bid: float
    best_ask: float
    mid_price: float
    spread_bps: float
    total_bid_depth: float
    total_ask_depth: float
    timestamp: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "asset": self.asset,
            "exchange": self.exchange,
            "bids": self.bids[:10],  # Top 10 for serialization
            "asks": self.asks[:10],
            "best_bid": self.best_bid,
            "best_ask": self.best_ask,
            "mid_price": self.mid_price,
            "spread_bps": self.spread_bps,
            "total_bid_depth": self.total_bid_depth,
            "total_ask_depth": self.total_ask_depth,
            "timestamp": self.timestamp,
        }


class ExternalIngestionService:
    """
    Receives and manages data from external execution systems.

    This is the RETURN PATH — OpenClaw/Meridian push data here so MiroFish
    can close the feedback loop and improve signals.

    Usage:
        svc = ExternalIngestionService()

        # OpenClaw reports a fill
        svc.ingest_fill({...})

        # OpenClaw reports a closed trade
        svc.ingest_closed_trade({...})

        # OpenClaw pushes portfolio state
        svc.update_portfolio_state({...})

        # OpenClaw sends its own signal for blending
        svc.ingest_external_signal({...})

        # OpenClaw pushes real order book
        svc.ingest_order_book_snapshot({...})

        # MiroFish reads the latest state for signal enrichment
        state = svc.get_current_state()
    """

    def __init__(self):
        self._lock = threading.Lock()

        # Execution data
        self._fills: List[ExecutionFill] = []
        self._closed_trades: List[ClosedTrade] = []

        # Portfolio state (latest from OpenClaw)
        self._portfolio_state: Optional[PortfolioState] = None

        # External signals for blending
        self._external_signals: List[ExternalSignal] = []

        # Real order book snapshots (latest per asset)
        self._order_book_snapshots: Dict[str, OrderBookSnapshot] = {}

        # Callbacks — other services can subscribe to events
        self._on_fill_callbacks: List = []
        self._on_trade_close_callbacks: List = []
        self._on_external_signal_callbacks: List = []

        logger.info("ExternalIngestionService initialized")

    # ========================================================================
    # Ingestion methods (called by API when OpenClaw pushes data)
    # ========================================================================

    def ingest_fill(self, data: Dict[str, Any]) -> ExecutionFill:
        """
        Record an execution fill from OpenClaw.

        Expected data:
        {
            "signal_id": "sig_xxx",
            "asset": "BTC/USDT",
            "direction": "LONG",
            "requested_price": 67500,
            "actual_price": 67520,
            "slippage_pct": 0.03,
            "quantity": 0.5,
            "fees": 3.38,
            "exchange": "binance",
            "fill_time": "2026-03-11T10:30:00Z",
            "latency_ms": 450
        }
        """
        fill = ExecutionFill(
            signal_id=data["signal_id"],
            asset=data["asset"],
            direction=data["direction"],
            requested_price=data["requested_price"],
            actual_price=data["actual_price"],
            slippage_pct=data.get("slippage_pct", 0),
            quantity=data["quantity"],
            fees=data.get("fees", 0),
            exchange=data.get("exchange", "unknown"),
            fill_time=data.get("fill_time", datetime.utcnow().isoformat()),
            latency_ms=data.get("latency_ms", 0),
        )

        with self._lock:
            self._fills.append(fill)

        logger.info(
            f"Fill ingested: {fill.signal_id} {fill.direction} {fill.asset} "
            f"@ {fill.actual_price} (slippage={fill.slippage_pct:.3f}%)"
        )

        # Notify subscribers
        for cb in self._on_fill_callbacks:
            try:
                cb(fill)
            except Exception as e:
                logger.warning(f"Fill callback failed: {e}")

        return fill

    def ingest_closed_trade(self, data: Dict[str, Any]) -> ClosedTrade:
        """
        Record a closed trade from OpenClaw.
        Automatically feeds calibrator with outcome data.

        Expected data:
        {
            "signal_id": "sig_xxx",
            "asset": "BTC/USDT",
            "direction": "LONG",
            "entry_price": 67500,
            "exit_price": 68800,
            "return_pct": 1.93,
            "holding_hours": 18.5,
            "fees_total": 6.75,
            "exit_reason": "take_profit",
            "closed_at": "2026-03-12T05:00:00Z"
        }
        """
        trade = ClosedTrade(
            signal_id=data["signal_id"],
            asset=data["asset"],
            direction=data["direction"],
            entry_price=data["entry_price"],
            exit_price=data["exit_price"],
            return_pct=data["return_pct"],
            holding_hours=data.get("holding_hours", 0),
            fees_total=data.get("fees_total", 0),
            exit_reason=data.get("exit_reason", "unknown"),
            closed_at=data.get("closed_at", datetime.utcnow().isoformat()),
        )

        with self._lock:
            self._closed_trades.append(trade)

        logger.info(
            f"Trade closed: {trade.signal_id} {trade.direction} {trade.asset} "
            f"return={trade.return_pct:+.2f}% ({trade.exit_reason})"
        )

        for cb in self._on_trade_close_callbacks:
            try:
                cb(trade)
            except Exception as e:
                logger.warning(f"Trade close callback failed: {e}")

        return trade

    def update_portfolio_state(self, data: Dict[str, Any]):
        """
        Receive portfolio state update from OpenClaw.

        Expected data:
        {
            "total_equity": 105000,
            "cash_pct": 65.0,
            "positions": [
                {"asset": "BTC/USDT", "direction": "LONG", "entry_price": 67500,
                 "current_price": 68200, "size_pct": 15, "unrealized_pnl_pct": 1.04,
                 "signal_id": "sig_xxx", "entry_time": "..."}
            ],
            "daily_pnl_pct": 0.85,
            "weekly_pnl_pct": 2.3,
            "total_pnl_pct": 5.0,
            "max_drawdown_pct": -3.2,
            "consecutive_losses": 0,
            "active_signals": 3
        }
        """
        positions = []
        for pos_data in data.get("positions", []):
            positions.append(Position(
                asset=pos_data["asset"],
                direction=SignalDirection(pos_data["direction"]),
                entry_price=pos_data["entry_price"],
                current_price=pos_data.get("current_price", 0),
                size_pct=pos_data.get("size_pct", 0),
                unrealized_pnl_pct=pos_data.get("unrealized_pnl_pct", 0),
                signal_id=pos_data.get("signal_id", ""),
                entry_time=pos_data.get("entry_time", ""),
            ))

        state = PortfolioState(
            total_equity=data.get("total_equity", 100000),
            cash_pct=data.get("cash_pct", 100),
            positions=positions,
            daily_pnl_pct=data.get("daily_pnl_pct", 0),
            weekly_pnl_pct=data.get("weekly_pnl_pct", 0),
            total_pnl_pct=data.get("total_pnl_pct", 0),
            max_drawdown_pct=data.get("max_drawdown_pct", 0),
            consecutive_losses=data.get("consecutive_losses", 0),
            active_signals=data.get("active_signals", 0),
        )

        with self._lock:
            self._portfolio_state = state

        logger.info(
            f"Portfolio state updated: equity={state.total_equity} "
            f"positions={len(positions)} daily_pnl={state.daily_pnl_pct:+.2f}%"
        )

    def ingest_external_signal(self, data: Dict[str, Any]) -> ExternalSignal:
        """
        Receive an external signal from OpenClaw or another system.
        These can be blended with MiroFish's own signals.

        Expected data:
        {
            "source": "openclaw",
            "signal_id": "oc_sig_xxx",
            "asset": "BTC/USDT",
            "direction": "LONG",
            "conviction": 0.68,
            "entry_price": 67500,
            "stop_loss": 66200,
            "take_profit": 69000,
            "metadata": {"model": "xgboost_v3", "features": ["momentum", "vol"]}
        }
        """
        signal = ExternalSignal(
            source=data.get("source", "external"),
            signal_id=data.get("signal_id", f"ext_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"),
            asset=data["asset"],
            direction=data["direction"],
            conviction=data.get("conviction", 0.5),
            entry_price=data.get("entry_price", 0),
            stop_loss=data.get("stop_loss"),
            take_profit=data.get("take_profit"),
            metadata=data.get("metadata", {}),
        )

        with self._lock:
            self._external_signals.append(signal)
            # Keep only last 100 external signals
            if len(self._external_signals) > 100:
                self._external_signals = self._external_signals[-100:]

        logger.info(
            f"External signal ingested: {signal.source} {signal.direction} "
            f"{signal.asset} conviction={signal.conviction:.2f}"
        )

        for cb in self._on_external_signal_callbacks:
            try:
                cb(signal)
            except Exception as e:
                logger.warning(f"External signal callback failed: {e}")

        return signal

    def ingest_order_book_snapshot(self, data: Dict[str, Any]) -> OrderBookSnapshot:
        """
        Receive a real order book snapshot from the execution venue.
        Used to calibrate the order book simulator with real-world depth.

        Expected data:
        {
            "asset": "BTC/USDT",
            "exchange": "binance",
            "bids": [[67400, 2.5], [67390, 1.8], ...],
            "asks": [[67420, 3.1], [67430, 2.2], ...],
            "timestamp": "2026-03-11T10:30:00Z"
        }
        """
        bids = data.get("bids", [])
        asks = data.get("asks", [])

        best_bid = bids[0][0] if bids else 0
        best_ask = asks[0][0] if asks else 0
        mid = (best_bid + best_ask) / 2 if best_bid and best_ask else 0
        spread = best_ask - best_bid if best_bid and best_ask else 0
        spread_bps = (spread / mid * 10000) if mid > 0 else 0

        snapshot = OrderBookSnapshot(
            asset=data["asset"],
            exchange=data.get("exchange", "unknown"),
            bids=bids,
            asks=asks,
            best_bid=best_bid,
            best_ask=best_ask,
            mid_price=round(mid, 4),
            spread_bps=round(spread_bps, 2),
            total_bid_depth=round(sum(q for _, q in bids), 4),
            total_ask_depth=round(sum(q for _, q in asks), 4),
            timestamp=data.get("timestamp", datetime.utcnow().isoformat()),
        )

        with self._lock:
            self._order_book_snapshots[data["asset"]] = snapshot

        return snapshot

    # ========================================================================
    # Read methods (used by signal enrichment layer)
    # ========================================================================

    def get_portfolio_state(self) -> Optional[PortfolioState]:
        """Get the latest portfolio state from OpenClaw."""
        with self._lock:
            return self._portfolio_state

    def get_positions_for_asset(self, asset: str) -> List[Position]:
        """Get current positions for a specific asset."""
        with self._lock:
            if not self._portfolio_state:
                return []
            return [p for p in self._portfolio_state.positions if p.asset == asset]

    def get_external_signals_for_asset(
        self, asset: str, max_age_minutes: int = 60
    ) -> List[ExternalSignal]:
        """Get recent external signals for an asset."""
        cutoff = datetime.utcnow().isoformat()
        with self._lock:
            return [
                s for s in self._external_signals
                if s.asset == asset
            ][-10:]  # Last 10

    def get_order_book_snapshot(self, asset: str) -> Optional[OrderBookSnapshot]:
        """Get the latest real order book snapshot for an asset."""
        with self._lock:
            return self._order_book_snapshots.get(asset)

    def get_execution_stats(self, lookback_trades: int = 50) -> Dict[str, Any]:
        """Aggregate execution quality stats from recent fills."""
        with self._lock:
            recent_fills = self._fills[-lookback_trades:]
            recent_trades = self._closed_trades[-lookback_trades:]

        if not recent_fills and not recent_trades:
            return {"status": "no_data"}

        stats = {}

        if recent_fills:
            slippages = [f.slippage_pct for f in recent_fills]
            latencies = [f.latency_ms for f in recent_fills]
            stats["fills"] = {
                "count": len(recent_fills),
                "avg_slippage_pct": round(sum(slippages) / len(slippages), 4),
                "max_slippage_pct": round(max(slippages), 4),
                "avg_latency_ms": round(sum(latencies) / len(latencies), 1),
            }

        if recent_trades:
            returns = [t.return_pct for t in recent_trades]
            wins = [t for t in recent_trades if t.return_pct > 0]
            losses = [t for t in recent_trades if t.return_pct <= 0]

            stats["trades"] = {
                "count": len(recent_trades),
                "win_rate": round(len(wins) / len(recent_trades), 3),
                "avg_return_pct": round(sum(returns) / len(returns), 3),
                "avg_win_pct": round(
                    sum(t.return_pct for t in wins) / len(wins), 3
                ) if wins else 0,
                "avg_loss_pct": round(
                    sum(t.return_pct for t in losses) / len(losses), 3
                ) if losses else 0,
                "profit_factor": round(
                    abs(sum(t.return_pct for t in wins)) /
                    abs(sum(t.return_pct for t in losses)), 2
                ) if losses and sum(t.return_pct for t in losses) != 0 else float('inf'),
                "exit_reasons": {},
            }

            for t in recent_trades:
                reason = t.exit_reason
                stats["trades"]["exit_reasons"][reason] = (
                    stats["trades"]["exit_reasons"].get(reason, 0) + 1
                )

        return stats

    def get_current_state(self) -> Dict[str, Any]:
        """
        Full state snapshot — used by signal enrichment layer.
        Returns everything the enrichment layer needs to enhance signals.
        """
        with self._lock:
            return {
                "portfolio": self._portfolio_state.model_dump() if self._portfolio_state else None,
                "recent_fills": [f.to_dict() for f in self._fills[-20:]],
                "recent_closed_trades": [t.to_dict() for t in self._closed_trades[-20:]],
                "external_signals": [s.to_dict() for s in self._external_signals[-20:]],
                "order_book_snapshots": {
                    k: v.to_dict() for k, v in self._order_book_snapshots.items()
                },
                "execution_stats": self.get_execution_stats(),
            }

    # ========================================================================
    # Event subscription (for other services to react)
    # ========================================================================

    def on_fill(self, callback):
        """Subscribe to execution fill events."""
        self._on_fill_callbacks.append(callback)

    def on_trade_close(self, callback):
        """Subscribe to trade close events."""
        self._on_trade_close_callbacks.append(callback)

    def on_external_signal(self, callback):
        """Subscribe to external signal events."""
        self._on_external_signal_callbacks.append(callback)

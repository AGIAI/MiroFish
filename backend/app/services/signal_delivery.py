"""
Signal Delivery Service
Pushes trading signals to external systems:
- OpenClaw/Meridian webhook integration
- Generic webhook delivery with retry
- Signal formatting for different consumers (Freqtrade, CCXT, 3Commas, Cornix)
"""

import json
import time
import threading
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable

from ..models.signal import TradingSignal, SignalDirection
from ..utils.logger import get_logger

logger = get_logger('mirofish.signal_delivery')


class WebhookTarget:
    """Configuration for a webhook delivery target."""

    def __init__(
        self,
        name: str,
        url: str,
        format: str = "mirofish",  # mirofish, freqtrade, ccxt, threecomas, cornix
        headers: Optional[Dict[str, str]] = None,
        secret: Optional[str] = None,
        enabled: bool = True,
    ):
        self.name = name
        self.url = url
        self.format = format
        self.headers = headers or {"Content-Type": "application/json"}
        self.secret = secret
        self.enabled = enabled
        self.last_delivery: Optional[str] = None
        self.delivery_count: int = 0
        self.error_count: int = 0
        self.last_error: Optional[str] = None


class SignalDeliveryService:
    """
    Manages signal delivery to external systems.
    Supports multiple webhook targets with different formats.
    """

    def __init__(self):
        self._targets: Dict[str, WebhookTarget] = {}
        self._delivery_log: List[Dict[str, Any]] = []
        self._signal_store: List[TradingSignal] = []  # In-memory signal archive

    # ========================================================================
    # Target management
    # ========================================================================

    def register_target(
        self,
        name: str,
        url: str,
        format: str = "mirofish",
        headers: Optional[Dict[str, str]] = None,
        secret: Optional[str] = None,
    ) -> WebhookTarget:
        """Register a new webhook delivery target."""
        target = WebhookTarget(
            name=name, url=url, format=format,
            headers=headers, secret=secret,
        )
        self._targets[name] = target
        logger.info(f"Registered webhook target: {name} → {url} (format={format})")
        return target

    def unregister_target(self, name: str) -> bool:
        """Remove a webhook target."""
        if name in self._targets:
            del self._targets[name]
            return True
        return False

    def list_targets(self) -> List[Dict[str, Any]]:
        """List all registered targets."""
        return [
            {
                "name": t.name,
                "url": t.url,
                "format": t.format,
                "enabled": t.enabled,
                "delivery_count": t.delivery_count,
                "error_count": t.error_count,
                "last_delivery": t.last_delivery,
                "last_error": t.last_error,
            }
            for t in self._targets.values()
        ]

    # ========================================================================
    # Signal delivery
    # ========================================================================

    def deliver(
        self,
        signal: TradingSignal,
        target_names: Optional[List[str]] = None,
        async_delivery: bool = True,
    ) -> Dict[str, Any]:
        """
        Deliver a signal to registered webhook targets.

        Args:
            signal: The trading signal to deliver
            target_names: Specific targets to deliver to (None = all enabled)
            async_delivery: Whether to deliver asynchronously

        Returns:
            Delivery status dict
        """
        # Store signal
        self._signal_store.append(signal)

        # Determine targets
        if target_names:
            targets = [self._targets[n] for n in target_names if n in self._targets]
        else:
            targets = [t for t in self._targets.values() if t.enabled]

        if not targets:
            logger.warning("No delivery targets available")
            return {"delivered_to": 0, "signal_id": signal.signal_id}

        results = {}
        for target in targets:
            if async_delivery:
                thread = threading.Thread(
                    target=self._deliver_to_target,
                    args=(signal, target),
                    daemon=True,
                )
                thread.start()
                results[target.name] = "dispatched"
            else:
                success = self._deliver_to_target(signal, target)
                results[target.name] = "delivered" if success else "failed"

        return {
            "signal_id": signal.signal_id,
            "delivered_to": len(targets),
            "targets": results,
        }

    def _deliver_to_target(
        self, signal: TradingSignal, target: WebhookTarget, max_retries: int = 3,
    ) -> bool:
        """Deliver signal to a single target with retry."""
        import requests

        payload = self._format_signal(signal, target.format)

        headers = dict(target.headers)
        if target.secret:
            import hashlib
            import hmac
            body = json.dumps(payload, separators=(',', ':'))
            signature = hmac.new(
                target.secret.encode(), body.encode(), hashlib.sha256
            ).hexdigest()
            headers["X-MiroFish-Signature"] = signature

        for attempt in range(max_retries):
            try:
                resp = requests.post(
                    target.url,
                    json=payload,
                    headers=headers,
                    timeout=10,
                )

                if resp.status_code < 300:
                    target.delivery_count += 1
                    target.last_delivery = datetime.utcnow().isoformat()
                    self._log_delivery(signal, target, "success", resp.status_code)
                    logger.info(f"Signal {signal.signal_id} delivered to {target.name}")
                    return True
                else:
                    target.last_error = f"HTTP {resp.status_code}: {resp.text[:200]}"
                    logger.warning(f"Delivery to {target.name} failed: {target.last_error}")

            except Exception as e:
                target.last_error = str(e)
                logger.warning(f"Delivery to {target.name} attempt {attempt + 1} failed: {e}")

            # Exponential backoff
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)

        target.error_count += 1
        self._log_delivery(signal, target, "failed", None)
        return False

    def _log_delivery(
        self, signal: TradingSignal, target: WebhookTarget,
        status: str, http_code: Optional[int],
    ):
        """Log delivery attempt."""
        self._delivery_log.append({
            "signal_id": signal.signal_id,
            "target": target.name,
            "status": status,
            "http_code": http_code,
            "timestamp": datetime.utcnow().isoformat(),
        })

    # ========================================================================
    # Signal formatting for different consumers
    # ========================================================================

    def _format_signal(self, signal: TradingSignal, format: str) -> Dict[str, Any]:
        """Format signal for specific consumer system."""
        formatters = {
            "mirofish": self._format_mirofish,
            "freqtrade": self._format_freqtrade,
            "ccxt": self._format_ccxt,
            "threecomas": self._format_threecomas,
            "cornix": self._format_cornix,
            "openclaw": self._format_openclaw,
            "meridian": self._format_openclaw,  # Same format for now
        }
        formatter = formatters.get(format, self._format_mirofish)
        return formatter(signal)

    def _format_mirofish(self, signal: TradingSignal) -> Dict[str, Any]:
        """Native MiroFish format — full signal data."""
        return signal.model_dump()

    def _format_openclaw(self, signal: TradingSignal) -> Dict[str, Any]:
        """OpenClaw/Meridian format — predictive signal with confidence."""
        return {
            "source": "mirofish",
            "version": signal.version,
            "signal_id": signal.signal_id,
            "timestamp": signal.timestamp_utc,

            # Core prediction
            "asset": signal.asset,
            "exchange": signal.exchange,
            "direction": signal.direction.value,
            "conviction": signal.conviction,
            "time_horizon_hours": signal.time_horizon_hours,

            # Trade levels
            "entry_price": signal.entry.price,
            "entry_type": signal.entry.type.value,
            "stop_loss_price": signal.stop_loss.price,
            "stop_loss_risk_pct": signal.stop_loss.risk_pct,
            "take_profit_levels": [
                {"price": tp.price, "close_pct": tp.close_pct}
                for tp in signal.take_profit
            ],
            "position_size_pct": signal.position_size_pct,

            # Market context
            "regime": signal.regime.value,
            "dominant_narrative": signal.dominant_narrative,
            "key_risk": signal.key_risk,

            # Signal decomposition (for transparency)
            "components": {
                "agent_consensus": signal.components.agent_consensus,
                "conviction_strength": signal.components.agent_conviction_strength,
                "narrative_momentum": signal.components.narrative_momentum,
                "flow_imbalance": signal.components.flow_imbalance,
                "crowding_risk": signal.components.crowding_risk,
                "information_edge": signal.components.information_edge_score,
            },

            # Simulation metadata
            "simulation_id": signal.simulation_id,
            "simulation_rounds": signal.simulation_rounds,
            "agents_participated": signal.agents_participated,
        }

    def _format_freqtrade(self, signal: TradingSignal) -> Dict[str, Any]:
        """Freqtrade webhook format."""
        side = "buy" if signal.direction == SignalDirection.LONG else "sell"
        return {
            "type": side,
            "pair": signal.asset.replace("/", "_"),
            "price": signal.entry.price,
            "amount": signal.position_size_pct / 100,
            "stoploss": signal.stop_loss.price,
            "profit_targets": [tp.price for tp in signal.take_profit],
            "tag": f"mirofish_{signal.regime.value}",
            "signal_id": signal.signal_id,
        }

    def _format_ccxt(self, signal: TradingSignal) -> Dict[str, Any]:
        """CCXT-compatible order parameters."""
        side = "buy" if signal.direction == SignalDirection.LONG else "sell"
        return {
            "symbol": signal.asset,
            "type": signal.entry.type.value.lower(),
            "side": side,
            "price": signal.entry.price,
            "amount": None,  # Caller must compute from position_size_pct
            "params": {
                "stopLoss": {"triggerPrice": signal.stop_loss.price},
                "takeProfit": [
                    {"triggerPrice": tp.price, "quantity_pct": tp.close_pct}
                    for tp in signal.take_profit
                ],
                "signal_id": signal.signal_id,
                "source": "mirofish",
            },
        }

    def _format_threecomas(self, signal: TradingSignal) -> Dict[str, Any]:
        """3Commas webhook format."""
        action = "buy" if signal.direction == SignalDirection.LONG else "sell"
        return {
            "message_type": "bot",
            "bot_id": "{{bot_id}}",
            "email_token": "{{email_token}}",
            "delay_seconds": 0,
            "pair": signal.asset.replace("/", "_"),
            "action": action,
            "signal_id": signal.signal_id,
            "close_at": signal.take_profit[0].price if signal.take_profit else None,
        }

    def _format_cornix(self, signal: TradingSignal) -> Dict[str, Any]:
        """Cornix-compatible signal format."""
        direction = "Long" if signal.direction == SignalDirection.LONG else "Short"
        return {
            "pair": signal.asset,
            "direction": direction,
            "exchange": signal.exchange or "Binance",
            "leverage": 1,
            "entry": [signal.entry.price],
            "targets": [tp.price for tp in signal.take_profit],
            "stop": signal.stop_loss.price,
            "signal_id": signal.signal_id,
        }

    # ========================================================================
    # Signal retrieval
    # ========================================================================

    def get_latest_signal(self, asset: Optional[str] = None) -> Optional[TradingSignal]:
        """Get the most recent signal, optionally filtered by asset."""
        for signal in reversed(self._signal_store):
            if asset is None or signal.asset == asset:
                return signal
        return None

    def get_active_signals(self) -> List[TradingSignal]:
        """Get all signals with status='active'."""
        return [s for s in self._signal_store if s.status == "active"]

    def get_signal_history(
        self, asset: Optional[str] = None, limit: int = 50
    ) -> List[TradingSignal]:
        """Get historical signals."""
        signals = self._signal_store
        if asset:
            signals = [s for s in signals if s.asset == asset]
        return list(reversed(signals[-limit:]))

    def get_delivery_log(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent delivery log entries."""
        return list(reversed(self._delivery_log[-limit:]))

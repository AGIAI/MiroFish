"""
Signal Extraction Engine
Converts simulation outcomes + market data into actionable trading signals
with calibrated confidence scores.

This is the core output product — the signal that gets pushed to OpenClaw/Meridian.
"""

import math
import statistics
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any

from ..models.signal import (
    TradingSignal, SignalDirection, MarketRegime,
    SignalEntry, SignalStopLoss, SignalTakeProfit,
    SignalComponents, EntryType, StopLossType,
)
from ..utils.logger import get_logger

logger = get_logger('mirofish.signal_extractor')


class SignalExtractor:
    """
    Converts raw simulation outcomes into calibrated TradingSignal objects.

    Input: simulation results (agent positions, order book state, market data)
    Output: TradingSignal with entry/exit levels, conviction, components
    """

    def extract_from_simulation(
        self,
        agent_positions: List[Dict[str, Any]],
        order_book_state: Dict[str, Any],
        market_data: Dict[str, Any],
        regime: Dict[str, Any],
        asset: str = "BTC/USDT",
        simulation_id: Optional[str] = None,
    ) -> TradingSignal:
        """
        Extract a trading signal from simulation results.

        Args:
            agent_positions: List of {agent_id, archetype, direction, size, conviction}
            order_book_state: Order book snapshot from OrderBookSimulator
            market_data: Current market data {price, atr, volume, etc.}
            regime: Regime detection result
            asset: Trading pair
            simulation_id: Source simulation ID

        Returns:
            A calibrated TradingSignal
        """
        # 1. Compute signal components
        components = self._compute_components(agent_positions, order_book_state, market_data)

        # 2. Determine direction from agent consensus
        direction = self._determine_direction(components)

        # 3. Calculate conviction score
        conviction = self._calculate_conviction(components, regime)

        # 4. Compute entry/exit levels
        current_price = market_data.get("price", 0)
        atr = market_data.get("atr", current_price * 0.02)  # Default 2% ATR
        regime_type = MarketRegime(regime.get("regime", "ranging"))

        entry = self._compute_entry(current_price, direction, atr)
        stop_loss = self._compute_stop_loss(current_price, direction, atr, regime_type)
        take_profits = self._compute_take_profits(current_price, direction, atr, regime_type)

        # 5. Position sizing suggestion
        risk_pct = abs(current_price - stop_loss.price) / current_price * 100 if current_price > 0 else 2
        position_size = self._suggest_position_size(conviction, risk_pct, components.crowding_risk)

        # 6. Time horizon based on regime
        time_horizon = self._estimate_time_horizon(regime_type, market_data)

        # 7. Determine dominant narrative and key risk
        dominant_narrative = self._extract_narrative(agent_positions)
        key_risk = self._identify_key_risk(components, regime)

        signal = TradingSignal(
            asset=asset,
            direction=direction,
            conviction=round(conviction, 3),
            entry=entry,
            stop_loss=stop_loss,
            take_profit=take_profits,
            position_size_pct=round(position_size, 1),
            time_horizon_hours=time_horizon,
            regime=regime_type,
            components=components,
            simulation_id=simulation_id,
            simulation_rounds=market_data.get("simulation_rounds", 0),
            agents_participated=len(agent_positions),
            dominant_narrative=dominant_narrative,
            key_risk=key_risk,
        )

        logger.info(
            f"Signal extracted: {asset} {direction.value} "
            f"conviction={conviction:.2f} entry={entry.price} "
            f"SL={stop_loss.price} regime={regime_type.value}"
        )

        return signal

    def extract_from_market_data_only(
        self,
        ohlcv: List[Dict],
        regime: Dict[str, Any],
        asset: str = "BTC/USDT",
    ) -> TradingSignal:
        """
        Extract a signal from market data alone (no simulation).
        Uses technical indicators as proxy for agent behavior.
        Useful for fast signal generation without running full simulation.
        """
        if len(ohlcv) < 20:
            raise ValueError("Need at least 20 bars for signal extraction")

        closes = [bar["close"] for bar in ohlcv]
        volumes = [bar.get("volume", 0) for bar in ohlcv]
        current_price = closes[-1]

        # Compute ATR
        atr = self._compute_atr(ohlcv, period=14)

        # RSI as proxy for agent consensus
        rsi = self._compute_rsi(closes, period=14)

        # Volume trend as proxy for flow imbalance
        avg_vol = sum(volumes) / len(volumes) if volumes else 1
        recent_vol = sum(volumes[-5:]) / 5 if len(volumes) >= 5 else avg_vol
        vol_ratio = recent_vol / avg_vol if avg_vol > 0 else 1

        # Momentum
        returns_5d = (closes[-1] / closes[-5] - 1) if len(closes) >= 5 else 0
        returns_20d = (closes[-1] / closes[-20] - 1) if len(closes) >= 20 else 0

        # Build synthetic components
        if rsi > 50:
            consensus = 0.5 + (rsi - 50) / 100
            flow = min(1, vol_ratio * returns_5d * 10) if returns_5d > 0 else max(-1, vol_ratio * returns_5d * 10)
        else:
            consensus = 0.5 - (50 - rsi) / 100
            flow = max(-1, vol_ratio * returns_5d * 10) if returns_5d < 0 else min(1, vol_ratio * returns_5d * 10)

        components = SignalComponents(
            agent_consensus=max(0, min(1, consensus)),
            agent_conviction_strength=min(1, abs(rsi - 50) / 30),
            narrative_momentum=max(-1, min(1, returns_20d * 5)),
            flow_imbalance=max(-1, min(1, flow)),
            crowding_risk=max(0, min(1, abs(consensus - 0.5) * 2)),
            information_edge_score=0.3,  # Lower for non-simulation signals
        )

        direction = self._determine_direction(components)
        regime_type = MarketRegime(regime.get("regime", "ranging"))
        conviction = self._calculate_conviction(components, regime)

        market_data = {"price": current_price, "atr": atr, "simulation_rounds": 0}
        entry = self._compute_entry(current_price, direction, atr)
        stop_loss = self._compute_stop_loss(current_price, direction, atr, regime_type)
        take_profits = self._compute_take_profits(current_price, direction, atr, regime_type)

        risk_pct = abs(current_price - stop_loss.price) / current_price * 100 if current_price > 0 else 2
        position_size = self._suggest_position_size(conviction, risk_pct, components.crowding_risk)

        return TradingSignal(
            asset=asset,
            direction=direction,
            conviction=round(conviction, 3),
            entry=entry,
            stop_loss=stop_loss,
            take_profit=take_profits,
            position_size_pct=round(position_size, 1),
            time_horizon_hours=self._estimate_time_horizon(regime_type, market_data),
            regime=regime_type,
            components=components,
            agents_participated=0,
            dominant_narrative=f"Technical signal (RSI={rsi:.0f}, momentum={returns_5d:.1%})",
        )

    def extract_enriched(self, ctx: Any) -> TradingSignal:
        """
        Extract a signal from a fully enriched context (EnrichmentContext).
        This is the PREFERRED path — blends market data, news, on-chain,
        external signals, and portfolio state into the signal.

        Args:
            ctx: EnrichmentContext from SignalEnrichmentService.enrich()

        Returns:
            Fully integrated TradingSignal
        """
        if len(ctx.ohlcv) < 20:
            raise ValueError("Need at least 20 bars in enrichment context")

        closes = [bar["close"] for bar in ctx.ohlcv]
        volumes = [bar.get("volume", 0) for bar in ctx.ohlcv]
        current_price = ctx.current_price or closes[-1]
        atr = ctx.atr or self._compute_atr(ctx.ohlcv)

        # RSI
        rsi = self._compute_rsi(closes, period=14)

        # Volume analysis
        avg_vol = sum(volumes) / len(volumes) if volumes else 1
        recent_vol = sum(volumes[-5:]) / 5 if len(volumes) >= 5 else avg_vol
        vol_ratio = recent_vol / avg_vol if avg_vol > 0 else 1

        # Momentum
        returns_5d = (closes[-1] / closes[-5] - 1) if len(closes) >= 5 else 0
        returns_20d = (closes[-1] / closes[-20] - 1) if len(closes) >= 20 else 0

        # --- Build components with ALL data sources ---

        # 1. Agent consensus: blend RSI with news sentiment + external signals
        tech_consensus = 0.5 + (rsi - 50) / 100
        news_weight = min(0.3, ctx.news_volume / 50)  # More news = more weight, cap at 30%
        consensus = tech_consensus * (1 - news_weight) + (0.5 + ctx.news_sentiment_score * 0.5) * news_weight

        # Blend with external signals if available
        if ctx.external_consensus is not None:
            consensus = consensus * 0.7 + ctx.external_consensus * 0.3

        # 2. Flow imbalance: blend volume ratio with real order book depth
        tech_flow = max(-1, min(1, vol_ratio * returns_5d * 10))
        if ctx.real_bid_depth > 0 and ctx.real_ask_depth > 0:
            total_depth = ctx.real_bid_depth + ctx.real_ask_depth
            real_flow = (ctx.real_bid_depth - ctx.real_ask_depth) / total_depth
            flow = tech_flow * 0.4 + real_flow * 0.6  # Real depth weighted more
        else:
            flow = tech_flow

        # 3. Narrative momentum: blend price momentum with news sentiment
        tech_narrative = max(-1, min(1, returns_20d * 5))
        if ctx.news_sentiment_score != 0:
            narrative = tech_narrative * 0.6 + ctx.news_sentiment_score * 0.4
        else:
            narrative = tech_narrative

        # 4. Information edge: higher when more data sources are available
        data_sources = 1  # Always have OHLCV
        if ctx.news_volume > 0:
            data_sources += 1
        if ctx.real_bid_depth > 0:
            data_sources += 1
        if ctx.external_consensus is not None:
            data_sources += 1
        if ctx.whale_net_flow != 0:
            data_sources += 1
        info_edge = min(1.0, data_sources / 5 * 0.8 + abs(rsi - 50) / 50 * 0.2)

        components = SignalComponents(
            agent_consensus=round(max(0, min(1, consensus)), 3),
            agent_conviction_strength=min(1, abs(rsi - 50) / 30),
            narrative_momentum=round(max(-1, min(1, narrative)), 3),
            flow_imbalance=round(max(-1, min(1, flow)), 3),
            crowding_risk=round(max(0, min(1, abs(consensus - 0.5) * 2)), 3),
            information_edge_score=round(info_edge, 3),
        )

        direction = self._determine_direction(components)
        regime_type = MarketRegime(ctx.regime.get("regime", "ranging"))
        conviction = self._calculate_conviction(components, ctx.regime)

        # --- Portfolio-aware adjustments ---

        # Circuit breaker: reduce conviction when in drawdown
        if ctx.portfolio_drawdown_pct < -5:
            conviction *= 0.7
        elif ctx.portfolio_drawdown_pct < -3:
            conviction *= 0.85

        # Reduce conviction after consecutive losses
        if ctx.consecutive_losses >= 5:
            conviction *= 0.5
        elif ctx.consecutive_losses >= 3:
            conviction *= 0.75

        # Penalize if already positioned in same direction (don't double up blindly)
        if ctx.has_existing_position and ctx.existing_position_direction == direction.value:
            conviction *= 0.8

        # Boost if external signals agree
        if ctx.external_consensus is not None:
            external_agrees = (
                (ctx.external_consensus > 0.6 and direction == SignalDirection.LONG) or
                (ctx.external_consensus < 0.4 and direction == SignalDirection.SHORT)
            )
            if external_agrees:
                conviction = min(0.95, conviction * 1.15)

        # On-chain adjustments
        if ctx.whale_net_flow > 0.3 and direction == SignalDirection.LONG:
            conviction = min(0.95, conviction * 1.1)  # Whales agree with long
        elif ctx.whale_net_flow < -0.3 and direction == SignalDirection.SHORT:
            conviction = min(0.95, conviction * 1.1)  # Whales agree with short

        # Adjust slippage into entry price
        entry = self._compute_entry(current_price, direction, atr)
        if ctx.avg_slippage_pct > 0:
            # Adjust entry to account for expected slippage
            slippage_adj = current_price * ctx.avg_slippage_pct / 100
            if direction == SignalDirection.LONG:
                entry.price = round(entry.price - slippage_adj, 4)
            else:
                entry.price = round(entry.price + slippage_adj, 4)

        stop_loss = self._compute_stop_loss(current_price, direction, atr, regime_type)
        take_profits = self._compute_take_profits(current_price, direction, atr, regime_type)

        risk_pct = abs(current_price - stop_loss.price) / current_price * 100 if current_price > 0 else 2
        position_size = self._suggest_position_size(conviction, risk_pct, components.crowding_risk)

        # Reduce position size for correlated exposure
        if ctx.correlated_exposure_pct > 30:
            position_size *= 0.5
        elif ctx.correlated_exposure_pct > 15:
            position_size *= 0.75

        market_data = {"price": current_price, "atr": atr, "simulation_rounds": 0}

        # Build narrative description
        narrative_parts = [f"RSI={rsi:.0f}"]
        if ctx.news_sentiment_score != 0:
            narrative_parts.append(f"news_sentiment={ctx.news_sentiment_score:+.2f}")
        if ctx.whale_net_flow != 0:
            narrative_parts.append(f"whale_flow={ctx.whale_net_flow:+.2f}")
        if ctx.external_consensus is not None:
            narrative_parts.append(f"external_consensus={ctx.external_consensus:.0%}")

        key_risk = self._identify_key_risk(components, ctx.regime)
        if ctx.has_existing_position:
            key_risk += f"; Existing {ctx.existing_position_direction} position ({ctx.existing_position_pnl_pct:+.1f}%)"
        if ctx.consecutive_losses >= 3:
            key_risk += f"; {ctx.consecutive_losses} consecutive losses"

        signal = TradingSignal(
            asset=ctx.asset,
            direction=direction,
            conviction=round(max(0.05, min(0.95, conviction)), 3),
            entry=entry,
            stop_loss=stop_loss,
            take_profit=take_profits,
            position_size_pct=round(max(1.0, min(25.0, position_size)), 1),
            time_horizon_hours=self._estimate_time_horizon(regime_type, market_data),
            regime=regime_type,
            components=components,
            agents_participated=0,
            dominant_narrative=f"Enriched signal ({', '.join(narrative_parts)})",
            key_risk=key_risk,
        )

        logger.info(
            f"Enriched signal: {ctx.asset} {direction.value} "
            f"conviction={signal.conviction:.2f} "
            f"sources={data_sources} "
            f"news={ctx.news_sentiment_score:+.2f} "
            f"whale={ctx.whale_net_flow:+.2f}"
        )

        return signal

    # ========================================================================
    # Component computation
    # ========================================================================

    def _compute_components(
        self,
        agent_positions: List[Dict],
        order_book: Dict,
        market_data: Dict,
    ) -> SignalComponents:
        """Compute all signal components from simulation data."""

        # Agent consensus: % of agents that are bullish
        total = len(agent_positions) or 1
        longs = sum(1 for a in agent_positions if a.get("direction") == "long")
        shorts = sum(1 for a in agent_positions if a.get("direction") == "short")
        consensus = longs / total

        # Conviction strength: average conviction of majority side
        majority_side = "long" if longs >= shorts else "short"
        majority_convictions = [
            a.get("conviction", 0.5) for a in agent_positions
            if a.get("direction") == majority_side
        ]
        conviction_strength = (
            sum(majority_convictions) / len(majority_convictions)
            if majority_convictions else 0.5
        )

        # Narrative momentum: rate of narrative agents' activity
        narrative_agents = [a for a in agent_positions if a.get("archetype") == "narrative"]
        if narrative_agents:
            narrative_convictions = [a.get("conviction", 0.5) for a in narrative_agents]
            narrative_direction = sum(1 for a in narrative_agents if a.get("direction") == "long") / len(narrative_agents)
            narrative_momentum = (narrative_direction - 0.5) * 2 * (sum(narrative_convictions) / len(narrative_convictions))
        else:
            narrative_momentum = 0.0

        # Flow imbalance from order book
        depth = order_book.get("depth", order_book)
        bid_depth = depth.get("total_bid_depth", 0)
        ask_depth = depth.get("total_ask_depth", 0)
        total_depth = bid_depth + ask_depth
        flow_imbalance = (bid_depth - ask_depth) / total_depth if total_depth > 0 else 0

        # Crowding risk: how one-sided the positioning is
        crowding = abs(consensus - 0.5) * 2  # 0 = perfectly balanced, 1 = all same side

        # Information edge: based on diversity of agent types
        archetypes = set(a.get("archetype", "") for a in agent_positions)
        type_diversity = len(archetypes) / 8  # 8 archetypes total
        info_edge = min(1.0, type_diversity * conviction_strength)

        return SignalComponents(
            agent_consensus=round(max(0, min(1, consensus)), 3),
            agent_conviction_strength=round(max(0, min(1, conviction_strength)), 3),
            narrative_momentum=round(max(-1, min(1, narrative_momentum)), 3),
            flow_imbalance=round(max(-1, min(1, flow_imbalance)), 3),
            crowding_risk=round(max(0, min(1, crowding)), 3),
            information_edge_score=round(max(0, min(1, info_edge)), 3),
        )

    def _determine_direction(self, components: SignalComponents) -> SignalDirection:
        """Determine signal direction from components."""
        # Weighted vote
        score = (
            (components.agent_consensus - 0.5) * 2 * 0.35 +  # Agent consensus (35%)
            components.flow_imbalance * 0.25 +                # Flow imbalance (25%)
            components.narrative_momentum * 0.20 +            # Narrative (20%)
            (components.agent_conviction_strength - 0.5) * 2 * 0.20  # Conviction (20%)
        )

        if score > 0.1:
            return SignalDirection.LONG
        elif score < -0.1:
            return SignalDirection.SHORT
        return SignalDirection.FLAT

    def _calculate_conviction(
        self, components: SignalComponents, regime: Dict
    ) -> float:
        """Calculate overall conviction score, adjusted for regime and crowding."""
        base_conviction = (
            components.agent_consensus * 0.25 +
            components.agent_conviction_strength * 0.25 +
            abs(components.flow_imbalance) * 0.2 +
            abs(components.narrative_momentum) * 0.15 +
            components.information_edge_score * 0.15
        )

        # Penalize crowded trades
        crowding_penalty = components.crowding_risk * 0.2
        base_conviction -= crowding_penalty

        # Regime adjustment
        regime_confidence = regime.get("confidence", 0.5)
        regime_type = regime.get("regime", "ranging")

        # Higher conviction in trending regimes, lower in choppy
        if regime_type in ["trending_bull", "trending_bear"]:
            base_conviction *= (1 + regime_confidence * 0.15)
        elif regime_type in ["high_vol", "crisis"]:
            base_conviction *= (1 - regime_confidence * 0.2)

        return max(0.0, min(1.0, base_conviction))

    # ========================================================================
    # Entry/Exit computation
    # ========================================================================

    def _compute_entry(
        self, price: float, direction: SignalDirection, atr: float
    ) -> SignalEntry:
        """Compute entry price — slight improvement from current price."""
        if direction == SignalDirection.LONG:
            # Limit buy slightly below market
            entry_price = price - atr * 0.1
        elif direction == SignalDirection.SHORT:
            entry_price = price + atr * 0.1
        else:
            entry_price = price

        valid_until = (datetime.utcnow() + timedelta(hours=2)).isoformat()

        return SignalEntry(
            type=EntryType.LIMIT,
            price=round(entry_price, 4),
            valid_until=valid_until,
        )

    def _compute_stop_loss(
        self, price: float, direction: SignalDirection,
        atr: float, regime: MarketRegime,
    ) -> SignalStopLoss:
        """ATR-based stop loss, adjusted for regime."""
        multipliers = {
            MarketRegime.TRENDING_BULL: 2.5,
            MarketRegime.TRENDING_BEAR: 2.0,
            MarketRegime.RANGING: 1.5,
            MarketRegime.HIGH_VOL: 3.0,
            MarketRegime.CRISIS: 4.0,
            MarketRegime.RECOVERY: 2.0,
        }
        mult = multipliers.get(regime, 2.0)
        stop_distance = atr * mult

        if direction == SignalDirection.LONG:
            stop_price = price - stop_distance
        elif direction == SignalDirection.SHORT:
            stop_price = price + stop_distance
        else:
            stop_price = price - stop_distance  # Default to long-side stop

        risk_pct = abs(price - stop_price) / price * 100 if price > 0 else 2

        return SignalStopLoss(
            price=round(stop_price, 4),
            type=StopLossType.ATR,
            risk_pct=round(risk_pct, 2),
        )

    def _compute_take_profits(
        self, price: float, direction: SignalDirection,
        atr: float, regime: MarketRegime,
    ) -> List[SignalTakeProfit]:
        """Scaled take-profit levels (R:R based)."""
        multipliers = {
            MarketRegime.TRENDING_BULL: [2.0, 3.5],
            MarketRegime.TRENDING_BEAR: [1.5, 2.5],
            MarketRegime.RANGING: [1.0, 1.5],
            MarketRegime.HIGH_VOL: [1.5, 2.5],
            MarketRegime.CRISIS: [1.0, 2.0],
            MarketRegime.RECOVERY: [1.5, 3.0],
        }
        mults = multipliers.get(regime, [1.5, 2.5])

        tps = []
        close_pcts = [60, 40]  # Close 60% at TP1, remaining 40% at TP2

        for i, mult in enumerate(mults):
            tp_distance = atr * mult
            if direction == SignalDirection.LONG:
                tp_price = price + tp_distance
            elif direction == SignalDirection.SHORT:
                tp_price = price - tp_distance
            else:
                tp_price = price + tp_distance

            tps.append(SignalTakeProfit(
                price=round(tp_price, 4),
                close_pct=close_pcts[i] if i < len(close_pcts) else 100,
            ))

        return tps

    def _suggest_position_size(
        self, conviction: float, risk_pct: float, crowding: float
    ) -> float:
        """
        Half-Kelly position sizing.
        f* = (p * b - q) / b, then halved for safety.
        """
        p = 0.5 + conviction * 0.15  # Estimated win probability
        q = 1 - p
        b = 2.0  # Assumed payoff ratio (2:1 R:R)

        if b <= 0:
            return 5.0

        kelly = (p * b - q) / b
        half_kelly = max(0, kelly / 2)

        # Scale to percentage, cap at 25%
        size = half_kelly * 100

        # Reduce for crowded trades
        size *= (1 - crowding * 0.3)

        # Never risk more than 2% of portfolio
        if risk_pct > 0:
            max_size_by_risk = (2.0 / risk_pct) * 100
            size = min(size, max_size_by_risk)

        return max(1.0, min(25.0, size))

    def _estimate_time_horizon(
        self, regime: MarketRegime, market_data: Dict
    ) -> int:
        """Estimate holding period in hours based on regime."""
        horizons = {
            MarketRegime.TRENDING_BULL: 48,
            MarketRegime.TRENDING_BEAR: 24,
            MarketRegime.RANGING: 12,
            MarketRegime.HIGH_VOL: 8,
            MarketRegime.CRISIS: 4,
            MarketRegime.RECOVERY: 36,
        }
        return horizons.get(regime, 24)

    def _extract_narrative(self, agent_positions: List[Dict]) -> Optional[str]:
        """Extract the dominant narrative from agent reasoning."""
        narratives = [a.get("narrative", "") for a in agent_positions if a.get("narrative")]
        if not narratives:
            return None
        # Most common narrative
        from collections import Counter
        counter = Counter(narratives)
        return counter.most_common(1)[0][0]

    def _identify_key_risk(self, components: SignalComponents, regime: Dict) -> str:
        """Identify the primary risk to this signal."""
        risks = []

        if components.crowding_risk > 0.7:
            risks.append("Extreme crowding — reversal risk high")
        if components.information_edge_score < 0.3:
            risks.append("Low information edge — signal may be noise")
        if regime.get("regime") in ["high_vol", "crisis"]:
            risks.append(f"Adverse regime: {regime.get('regime')}")
        if abs(components.flow_imbalance) < 0.1:
            risks.append("Weak flow signal — low conviction from order book")
        if regime.get("confidence", 0) < 0.4:
            risks.append("Regime uncertainty — conditions may shift")

        return "; ".join(risks) if risks else "No elevated risks identified"

    # ========================================================================
    # Technical indicator helpers (no external deps)
    # ========================================================================

    def _compute_atr(self, ohlcv: List[Dict], period: int = 14) -> float:
        """Average True Range."""
        if len(ohlcv) < 2:
            return ohlcv[0]["high"] - ohlcv[0]["low"] if ohlcv else 0

        trs = [ohlcv[0]["high"] - ohlcv[0]["low"]]
        for i in range(1, len(ohlcv)):
            tr = max(
                ohlcv[i]["high"] - ohlcv[i]["low"],
                abs(ohlcv[i]["high"] - ohlcv[i - 1]["close"]),
                abs(ohlcv[i]["low"] - ohlcv[i - 1]["close"]),
            )
            trs.append(tr)

        return sum(trs[-period:]) / min(period, len(trs))

    def _compute_rsi(self, closes: List[float], period: int = 14) -> float:
        """Relative Strength Index."""
        if len(closes) < period + 1:
            return 50.0

        gains = []
        losses = []
        for i in range(1, len(closes)):
            delta = closes[i] - closes[i - 1]
            gains.append(max(delta, 0))
            losses.append(max(-delta, 0))

        avg_gain = sum(gains[-period:]) / period
        avg_loss = sum(losses[-period:]) / period

        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

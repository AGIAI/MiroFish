"""
Signal Extraction Engine
Converts data analysis into conviction signals.

MiroFish's core job:
1. What was found? (the opportunity/pattern)
2. What's the trade? (asset, direction, timeframe)
3. What's the confidence? (conviction 0-1)

Execution (entry/SL/TP/sizing) is Meridian's responsibility.
"""

import math
from datetime import datetime
from typing import Dict, List, Optional, Any

from ..models.signal import (
    TradingSignal, SignalDirection, MarketRegime,
    SignalComponents, Finding,
)
from ..utils.logger import get_logger

logger = get_logger('mirofish.signal_extractor')


class SignalExtractor:
    """
    Converts raw data (simulation results, market data, enrichment context)
    into conviction signals with supporting evidence.
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
        Extract a conviction signal from simulation results.

        Args:
            agent_positions: List of {agent_id, archetype, direction, size, conviction}
            order_book_state: Order book snapshot from OrderBookSimulator
            market_data: Current market data {price, atr, volume, etc.}
            regime: Regime detection result
            asset: Trading pair
            simulation_id: Source simulation ID
        """
        components = self._compute_components(agent_positions, order_book_state, market_data)
        direction = self._determine_direction(components)
        conviction = self._calculate_conviction(components, regime)
        regime_type = MarketRegime(regime.get("regime", "ranging"))

        # Build findings from simulation evidence
        findings = self._findings_from_simulation(agent_positions, order_book_state, market_data, components)

        # Build thesis
        thesis = self._build_thesis(asset, direction, conviction, components, regime_type)

        # Identify risks
        key_risks = self._identify_risks(components, regime)

        signal = TradingSignal(
            asset=asset,
            direction=direction,
            conviction=round(conviction, 3),
            time_horizon_hours=self._estimate_time_horizon(regime_type),
            thesis=thesis,
            findings=findings,
            regime=regime_type,
            components=components,
            key_risks=key_risks,
            simulation_id=simulation_id,
            simulation_rounds=market_data.get("simulation_rounds", 0),
            agents_participated=len(agent_positions),
            dominant_narrative=self._extract_narrative(agent_positions),
        )

        logger.info(
            f"Signal extracted: {asset} {direction.value} "
            f"conviction={conviction:.2f} regime={regime_type.value}"
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
        """
        if len(ohlcv) < 20:
            raise ValueError("Need at least 20 bars for signal extraction")

        closes = [bar["close"] for bar in ohlcv]
        volumes = [bar.get("volume", 0) for bar in ohlcv]
        current_price = closes[-1]

        atr = self._compute_atr(ohlcv, period=14)
        rsi = self._compute_rsi(closes, period=14)

        # Volume trend
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

        # Build findings from technical analysis
        findings = []
        findings.append(Finding(
            source="technical",
            detail=f"RSI at {rsi:.0f} — {'overbought territory' if rsi > 70 else 'oversold territory' if rsi < 30 else 'neutral zone'}",
            direction="LONG" if rsi < 40 else "SHORT" if rsi > 60 else None,
            strength=abs(rsi - 50) / 50,
        ))
        findings.append(Finding(
            source="technical",
            detail=f"5-bar momentum {returns_5d:+.1%}, 20-bar momentum {returns_20d:+.1%}",
            direction="LONG" if returns_5d > 0.01 else "SHORT" if returns_5d < -0.01 else None,
            strength=min(1, abs(returns_5d) * 10),
        ))
        if vol_ratio > 1.5:
            findings.append(Finding(
                source="technical",
                detail=f"Volume surge — {vol_ratio:.1f}x average volume",
                strength=min(1, (vol_ratio - 1) / 2),
            ))
        elif vol_ratio < 0.5:
            findings.append(Finding(
                source="technical",
                detail=f"Low volume — {vol_ratio:.1f}x average, weak conviction",
                strength=0.3,
            ))

        thesis = self._build_thesis(asset, direction, conviction, components, regime_type)
        key_risks = self._identify_risks(components, regime)

        return TradingSignal(
            asset=asset,
            direction=direction,
            conviction=round(conviction, 3),
            time_horizon_hours=self._estimate_time_horizon(regime_type),
            thesis=thesis,
            findings=findings,
            regime=regime_type,
            components=components,
            key_risks=key_risks,
            agents_participated=0,
            dominant_narrative=f"Technical signal (RSI={rsi:.0f}, momentum={returns_5d:.1%})",
        )

    def extract_enriched(self, ctx: Any) -> TradingSignal:
        """
        Extract a signal from a fully enriched context (EnrichmentContext).
        This is the PREFERRED path — blends market data, news, on-chain,
        and external signals into conviction + findings.

        Args:
            ctx: EnrichmentContext from SignalEnrichmentService.enrich()
        """
        if len(ctx.ohlcv) < 20:
            raise ValueError("Need at least 20 bars in enrichment context")

        closes = [bar["close"] for bar in ctx.ohlcv]
        volumes = [bar.get("volume", 0) for bar in ctx.ohlcv]
        current_price = ctx.current_price or closes[-1]
        atr = ctx.atr or self._compute_atr(ctx.ohlcv)

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
        news_weight = min(0.3, ctx.news_volume / 50)
        consensus = tech_consensus * (1 - news_weight) + (0.5 + ctx.news_sentiment_score * 0.5) * news_weight

        if ctx.external_consensus is not None:
            consensus = consensus * 0.7 + ctx.external_consensus * 0.3

        # 2. Flow imbalance: blend volume ratio with real order book depth
        tech_flow = max(-1, min(1, vol_ratio * returns_5d * 10))
        if ctx.real_bid_depth > 0 and ctx.real_ask_depth > 0:
            total_depth = ctx.real_bid_depth + ctx.real_ask_depth
            real_flow = (ctx.real_bid_depth - ctx.real_ask_depth) / total_depth
            flow = tech_flow * 0.4 + real_flow * 0.6
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

        # Boost if external signals agree
        if ctx.external_consensus is not None:
            external_agrees = (
                (ctx.external_consensus > 0.6 and direction == SignalDirection.LONG) or
                (ctx.external_consensus < 0.4 and direction == SignalDirection.SHORT)
            )
            if external_agrees:
                conviction = min(0.95, conviction * 1.15)

        # On-chain confirmation
        if ctx.whale_net_flow > 0.3 and direction == SignalDirection.LONG:
            conviction = min(0.95, conviction * 1.1)
        elif ctx.whale_net_flow < -0.3 and direction == SignalDirection.SHORT:
            conviction = min(0.95, conviction * 1.1)

        # --- Build findings from ALL sources ---
        findings = []

        # Technical findings
        findings.append(Finding(
            source="technical",
            detail=f"RSI at {rsi:.0f}, 5-bar momentum {returns_5d:+.1%}, 20-bar momentum {returns_20d:+.1%}",
            direction="LONG" if rsi < 40 else "SHORT" if rsi > 60 else None,
            strength=abs(rsi - 50) / 50,
        ))
        if vol_ratio > 1.3 or vol_ratio < 0.7:
            findings.append(Finding(
                source="technical",
                detail=f"Volume at {vol_ratio:.1f}x average — {'surge confirms move' if vol_ratio > 1.3 else 'low volume, weak conviction'}",
                strength=min(1, abs(vol_ratio - 1)),
            ))

        # News findings
        if ctx.news_volume > 0:
            sentiment_label = "bullish" if ctx.news_sentiment_score > 0.1 else "bearish" if ctx.news_sentiment_score < -0.1 else "neutral"
            findings.append(Finding(
                source="news",
                detail=f"{sentiment_label.capitalize()} sentiment ({ctx.news_sentiment_score:+.2f}) from {ctx.news_volume} articles",
                direction="LONG" if ctx.news_sentiment_score > 0.1 else "SHORT" if ctx.news_sentiment_score < -0.1 else None,
                strength=min(1, abs(ctx.news_sentiment_score)),
            ))

        # Order flow findings
        if ctx.real_bid_depth > 0 and ctx.real_ask_depth > 0:
            total_depth = ctx.real_bid_depth + ctx.real_ask_depth
            bid_pct = ctx.real_bid_depth / total_depth * 100
            findings.append(Finding(
                source="order_flow",
                detail=f"Order book {bid_pct:.0f}% bid-side depth, spread {ctx.real_spread_bps:.1f}bps",
                direction="LONG" if bid_pct > 55 else "SHORT" if bid_pct < 45 else None,
                strength=abs(bid_pct - 50) / 50,
            ))

        # On-chain findings
        if ctx.whale_net_flow != 0:
            flow_label = "inflow (accumulation)" if ctx.whale_net_flow > 0 else "outflow (distribution)"
            findings.append(Finding(
                source="on_chain",
                detail=f"Whale net {flow_label}: {ctx.whale_net_flow:+.2f}",
                direction="LONG" if ctx.whale_net_flow > 0.1 else "SHORT" if ctx.whale_net_flow < -0.1 else None,
                strength=min(1, abs(ctx.whale_net_flow)),
            ))

        if ctx.gas_regime != "normal":
            findings.append(Finding(
                source="on_chain",
                detail=f"Gas regime: {ctx.gas_regime}",
                strength=0.3,
            ))

        # External signal findings
        if ctx.external_consensus is not None:
            ext_dir = "bullish" if ctx.external_consensus > 0.6 else "bearish" if ctx.external_consensus < 0.4 else "mixed"
            findings.append(Finding(
                source="external",
                detail=f"External signals {ext_dir} ({ctx.external_consensus:.0%} long) from {len(ctx.external_signals)} sources",
                direction="LONG" if ctx.external_consensus > 0.6 else "SHORT" if ctx.external_consensus < 0.4 else None,
                strength=abs(ctx.external_consensus - 0.5) * 2,
            ))

        # Build thesis and risks
        thesis = self._build_thesis(ctx.asset, direction, conviction, components, regime_type)
        key_risks = self._identify_risks(components, ctx.regime)

        # Build narrative parts for dominant_narrative
        narrative_parts = [f"RSI={rsi:.0f}"]
        if ctx.news_sentiment_score != 0:
            narrative_parts.append(f"news={ctx.news_sentiment_score:+.2f}")
        if ctx.whale_net_flow != 0:
            narrative_parts.append(f"whale_flow={ctx.whale_net_flow:+.2f}")
        if ctx.external_consensus is not None:
            narrative_parts.append(f"external={ctx.external_consensus:.0%}")

        signal = TradingSignal(
            asset=ctx.asset,
            direction=direction,
            conviction=round(max(0.05, min(0.95, conviction)), 3),
            time_horizon_hours=self._estimate_time_horizon(regime_type),
            thesis=thesis,
            findings=findings,
            regime=regime_type,
            components=components,
            key_risks=key_risks,
            agents_participated=0,
            dominant_narrative=f"Enriched ({', '.join(narrative_parts)})",
        )

        logger.info(
            f"Enriched signal: {ctx.asset} {direction.value} "
            f"conviction={signal.conviction:.2f} "
            f"sources={data_sources} findings={len(findings)}"
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
        total = len(agent_positions) or 1
        longs = sum(1 for a in agent_positions if a.get("direction") == "long")
        shorts = sum(1 for a in agent_positions if a.get("direction") == "short")
        consensus = longs / total

        majority_side = "long" if longs >= shorts else "short"
        majority_convictions = [
            a.get("conviction", 0.5) for a in agent_positions
            if a.get("direction") == majority_side
        ]
        conviction_strength = (
            sum(majority_convictions) / len(majority_convictions)
            if majority_convictions else 0.5
        )

        # Narrative momentum
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

        crowding = abs(consensus - 0.5) * 2

        archetypes = set(a.get("archetype", "") for a in agent_positions)
        type_diversity = len(archetypes) / 8
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
        """Determine signal direction from components via weighted vote."""
        score = (
            (components.agent_consensus - 0.5) * 2 * 0.35 +
            components.flow_imbalance * 0.25 +
            components.narrative_momentum * 0.20 +
            (components.agent_conviction_strength - 0.5) * 2 * 0.20
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

        if regime_type in ["trending_bull", "trending_bear"]:
            base_conviction *= (1 + regime_confidence * 0.15)
        elif regime_type in ["high_vol", "crisis"]:
            base_conviction *= (1 - regime_confidence * 0.2)

        return max(0.0, min(1.0, base_conviction))

    # ========================================================================
    # Findings & thesis builders
    # ========================================================================

    def _findings_from_simulation(
        self, agent_positions: List[Dict], order_book: Dict,
        market_data: Dict, components: SignalComponents,
    ) -> List[Finding]:
        """Build findings list from simulation evidence."""
        findings = []

        total = len(agent_positions) or 1
        longs = sum(1 for a in agent_positions if a.get("direction") == "long")
        shorts = total - longs
        majority = "LONG" if longs > shorts else "SHORT"

        findings.append(Finding(
            source="agent_simulation",
            detail=f"{longs}/{total} agents bullish ({components.agent_consensus:.0%} consensus), avg conviction {components.agent_conviction_strength:.0%}",
            direction=majority,
            strength=components.agent_conviction_strength,
        ))

        # Order flow finding
        depth = order_book.get("depth", order_book)
        bid_depth = depth.get("total_bid_depth", 0)
        ask_depth = depth.get("total_ask_depth", 0)
        if bid_depth + ask_depth > 0:
            bid_pct = bid_depth / (bid_depth + ask_depth) * 100
            findings.append(Finding(
                source="order_flow",
                detail=f"Simulated book {bid_pct:.0f}% bid-side, flow imbalance {components.flow_imbalance:+.2f}",
                direction="LONG" if components.flow_imbalance > 0.1 else "SHORT" if components.flow_imbalance < -0.1 else None,
                strength=abs(components.flow_imbalance),
            ))

        # Narrative finding
        if abs(components.narrative_momentum) > 0.1:
            direction = "LONG" if components.narrative_momentum > 0 else "SHORT"
            findings.append(Finding(
                source="agent_simulation",
                detail=f"Narrative agents {'bullish' if components.narrative_momentum > 0 else 'bearish'} (momentum={components.narrative_momentum:+.2f})",
                direction=direction,
                strength=abs(components.narrative_momentum),
            ))

        # Crowding warning
        if components.crowding_risk > 0.6:
            findings.append(Finding(
                source="agent_simulation",
                detail=f"High crowding risk ({components.crowding_risk:.0%}) — reversal risk elevated",
                strength=components.crowding_risk,
            ))

        return findings

    def _build_thesis(
        self, asset: str, direction: SignalDirection, conviction: float,
        components: SignalComponents, regime: MarketRegime,
    ) -> str:
        """Build a one-sentence thesis summarizing the opportunity."""
        dir_word = "bullish" if direction == SignalDirection.LONG else "bearish" if direction == SignalDirection.SHORT else "neutral"
        strength = "high" if conviction > 0.7 else "moderate" if conviction > 0.4 else "low"

        regime_desc = {
            MarketRegime.TRENDING_BULL: "in a bullish trend",
            MarketRegime.TRENDING_BEAR: "in a bearish trend",
            MarketRegime.RANGING: "in a ranging market",
            MarketRegime.HIGH_VOL: "during high volatility",
            MarketRegime.CRISIS: "during crisis conditions",
            MarketRegime.RECOVERY: "in recovery phase",
        }.get(regime, "")

        # Pick the strongest driver
        drivers = {
            "agent consensus": components.agent_consensus,
            "order flow": abs(components.flow_imbalance),
            "narrative momentum": abs(components.narrative_momentum),
            "information edge": components.information_edge_score,
        }
        top_driver = max(drivers, key=drivers.get)

        return f"{asset} looks {dir_word} with {strength} conviction ({conviction:.0%}) {regime_desc}, primarily driven by {top_driver}"

    def _identify_risks(self, components: SignalComponents, regime: Dict) -> List[str]:
        """Identify key risks to this signal."""
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

        return risks

    def _estimate_time_horizon(self, regime: MarketRegime) -> int:
        """Estimate signal relevance window in hours based on regime."""
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
        from collections import Counter
        counter = Counter(narratives)
        return counter.most_common(1)[0][0]

    # ========================================================================
    # Technical indicator helpers
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

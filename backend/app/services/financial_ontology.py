"""
Financial Domain Ontology Generator
Extends the base OntologyGenerator with a hard-coded financial ontology
for market simulation. The LLM can extend these base types with
asset-specific additions.
"""

import json
from typing import Dict, Any, List, Optional
from ..utils.llm_client import LLMClient
from ..utils.logger import get_logger

logger = get_logger('mirofish.financial_ontology')


# ============================================================================
# Hard-coded financial base ontology
# ============================================================================

FINANCIAL_ENTITY_TYPES = [
    {
        "name": "Asset",
        "description": "A tradeable financial instrument (stock, crypto, commodity, forex pair)",
        "attributes": [
            {"name": "ticker", "type": "text", "description": "Trading symbol (e.g. AAPL, BTC/USDT)"},
            {"name": "asset_class", "type": "text", "description": "Class: equity, crypto, commodity, forex, index"},
            {"name": "sector", "type": "text", "description": "Sector or category"},
        ],
        "examples": ["AAPL", "BTC/USDT", "GLD", "EUR/USD"],
    },
    {
        "name": "Trader",
        "description": "A market participant agent with a defined trading strategy and risk profile",
        "attributes": [
            {"name": "archetype", "type": "text", "description": "Trading style: momentum, mean_reversion, fundamental, narrative, market_maker, whale, retail, quant"},
            {"name": "risk_tolerance", "type": "text", "description": "Risk tolerance level: 0.0-1.0"},
            {"name": "time_horizon", "type": "text", "description": "Trading timeframe: scalp, intraday, swing, position"},
        ],
        "examples": ["MomentumAlpha", "ValueHunter", "WhaleTracker"],
    },
    {
        "name": "Institution",
        "description": "A large market participant: hedge fund, bank, market maker, central bank",
        "attributes": [
            {"name": "institution_type", "type": "text", "description": "Type: hedge_fund, market_maker, central_bank, bank, exchange, retail_aggregate"},
            {"name": "aum_tier", "type": "text", "description": "Assets under management tier: small, medium, large, mega"},
            {"name": "known_bias", "type": "text", "description": "Known directional bias or strategy focus"},
        ],
        "examples": ["Citadel", "BlackRock", "Federal Reserve", "Binance"],
    },
    {
        "name": "MarketEvent",
        "description": "A scheduled or breaking event that can move markets",
        "attributes": [
            {"name": "event_type", "type": "text", "description": "Type: earnings, FOMC, CPI, halving, regulatory, geopolitical, listing, airdrop"},
            {"name": "expected_impact", "type": "text", "description": "Expected market impact: low, medium, high, extreme"},
            {"name": "scheduled_time", "type": "text", "description": "When the event occurs (ISO datetime or 'unscheduled')"},
        ],
        "examples": ["FOMC Rate Decision", "AAPL Q1 Earnings", "BTC Halving 2028"],
    },
    {
        "name": "Narrative",
        "description": "A market narrative or theme driving price action and sentiment",
        "attributes": [
            {"name": "theme", "type": "text", "description": "Narrative theme: AI_hype, rate_cut_hope, recession_fear, memecoin_mania, DeFi_summer"},
            {"name": "strength", "type": "text", "description": "Current narrative strength: weak, moderate, strong, dominant"},
            {"name": "velocity", "type": "text", "description": "Rate of change: accelerating, stable, decelerating, collapsing"},
        ],
        "examples": ["AI Infrastructure Boom", "Fed Pivot Trade", "RWA Tokenization"],
    },
    {
        "name": "Indicator",
        "description": "A technical, fundamental, sentiment, or on-chain metric used for signals",
        "attributes": [
            {"name": "indicator_type", "type": "text", "description": "Type: technical, fundamental, sentiment, on_chain, macro"},
            {"name": "signal_direction", "type": "text", "description": "Current signal: bullish, bearish, neutral"},
            {"name": "lookback_period", "type": "text", "description": "Lookback window (e.g. 14d, 50d, 200d)"},
        ],
        "examples": ["RSI_14", "MVRV_Ratio", "Put_Call_Ratio", "Fear_Greed_Index"],
    },
    {
        "name": "Exchange",
        "description": "A trading venue where assets are bought and sold",
        "attributes": [
            {"name": "exchange_type", "type": "text", "description": "Type: centralized, decentralized, traditional"},
            {"name": "primary_assets", "type": "text", "description": "Primary asset classes traded"},
        ],
        "examples": ["Binance", "NYSE", "Uniswap", "CME"],
    },
    {
        "name": "Regime",
        "description": "A market regime or macro environment state that affects all participants",
        "attributes": [
            {"name": "regime_type", "type": "text", "description": "Type: trending_bull, trending_bear, ranging, high_vol, crisis, recovery"},
            {"name": "duration_estimate", "type": "text", "description": "Estimated remaining duration"},
        ],
        "examples": ["Low Vol Bull", "High Vol Choppy", "Liquidity Crisis"],
    },
    # Fallback types (required by Zep)
    {
        "name": "Person",
        "description": "Any individual market participant not fitting specific trader/analyst types",
        "attributes": [
            {"name": "full_name", "type": "text", "description": "Full name of the person"},
            {"name": "role", "type": "text", "description": "Role in markets (analyst, commentator, regulator)"},
        ],
        "examples": ["retail trader", "financial journalist", "SEC commissioner"],
    },
    {
        "name": "Organization",
        "description": "Any organization not fitting specific institution/exchange types",
        "attributes": [
            {"name": "org_name", "type": "text", "description": "Name of the organization"},
            {"name": "org_type", "type": "text", "description": "Type of organization"},
        ],
        "examples": ["rating agency", "industry association", "research firm"],
    },
]

FINANCIAL_EDGE_TYPES = [
    {
        "name": "EXPOSED_TO",
        "description": "Trader/Institution has exposure to an Asset (long or short)",
        "source_targets": [
            {"source": "Trader", "target": "Asset"},
            {"source": "Institution", "target": "Asset"},
        ],
        "attributes": [
            {"name": "direction", "type": "text", "description": "Position direction: long, short, neutral"},
            {"name": "weight", "type": "text", "description": "Position weight 0.0-1.0"},
        ],
    },
    {
        "name": "CATALYZES",
        "description": "An Event drives price movement in an Asset",
        "source_targets": [
            {"source": "MarketEvent", "target": "Asset"},
        ],
        "attributes": [
            {"name": "impact_direction", "type": "text", "description": "Expected direction: bullish, bearish, uncertain"},
            {"name": "magnitude", "type": "text", "description": "Expected magnitude: low, medium, high"},
        ],
    },
    {
        "name": "CORRELATES_WITH",
        "description": "Statistical correlation between two assets",
        "source_targets": [
            {"source": "Asset", "target": "Asset"},
        ],
        "attributes": [
            {"name": "coefficient", "type": "text", "description": "Correlation coefficient: -1.0 to 1.0"},
            {"name": "rolling_window", "type": "text", "description": "Rolling window for calculation"},
        ],
    },
    {
        "name": "DRIVES",
        "description": "A Narrative drives sentiment and flow into an Asset",
        "source_targets": [
            {"source": "Narrative", "target": "Asset"},
            {"source": "Narrative", "target": "Trader"},
        ],
        "attributes": [
            {"name": "sentiment_score", "type": "text", "description": "Sentiment impact: -1.0 to 1.0"},
        ],
    },
    {
        "name": "MONITORS",
        "description": "A Trader watches a specific Indicator for trading signals",
        "source_targets": [
            {"source": "Trader", "target": "Indicator"},
        ],
        "attributes": [
            {"name": "threshold_value", "type": "text", "description": "Trigger threshold value"},
            {"name": "action_on_trigger", "type": "text", "description": "Action when triggered: buy, sell, hedge"},
        ],
    },
    {
        "name": "REACTS_TO",
        "description": "A Trader reacts to a MarketEvent with a specific behavior",
        "source_targets": [
            {"source": "Trader", "target": "MarketEvent"},
            {"source": "Institution", "target": "MarketEvent"},
        ],
        "attributes": [
            {"name": "typical_action", "type": "text", "description": "Expected response: buy_dip, sell_news, hedge, wait"},
            {"name": "lag_seconds", "type": "text", "description": "Typical reaction delay in seconds"},
        ],
    },
    {
        "name": "COMPETES_WITH",
        "description": "Two traders compete for alpha on the same asset",
        "source_targets": [
            {"source": "Trader", "target": "Trader"},
        ],
        "attributes": [
            {"name": "competition_asset", "type": "text", "description": "Asset being competed on"},
        ],
    },
    {
        "name": "TRADES_ON",
        "description": "An Asset is traded on an Exchange",
        "source_targets": [
            {"source": "Asset", "target": "Exchange"},
        ],
        "attributes": [
            {"name": "avg_daily_volume", "type": "text", "description": "Average daily volume on this exchange"},
        ],
    },
    {
        "name": "OPERATES_IN",
        "description": "A participant operates within a specific market Regime",
        "source_targets": [
            {"source": "Trader", "target": "Regime"},
            {"source": "Institution", "target": "Regime"},
        ],
        "attributes": [
            {"name": "performance_bias", "type": "text", "description": "How well this participant performs in this regime"},
        ],
    },
    {
        "name": "SIGNALS",
        "description": "An Indicator generates a signal relevant to an Asset or Regime",
        "source_targets": [
            {"source": "Indicator", "target": "Asset"},
            {"source": "Indicator", "target": "Regime"},
        ],
        "attributes": [
            {"name": "signal_strength", "type": "text", "description": "Signal confidence: weak, moderate, strong"},
        ],
    },
]


FINANCIAL_ONTOLOGY_SYSTEM_PROMPT = """You are a financial markets ontology expert. Given market context (assets, news, conditions), you may EXTEND the provided base financial ontology with asset-specific or situation-specific entity and edge types.

**Important: You must output valid JSON format data. Do not output anything else.**

You will receive a base ontology with 10 entity types and 10 edge types. Your job is to review the market context and OPTIONALLY suggest modifications:

1. You may replace up to 2 of the non-fallback entity types (positions 1-8) with more specific types relevant to the context
2. You may replace up to 2 edge types with more relevant ones
3. You MUST keep the Person and Organization fallback types (positions 9-10)
4. You MUST keep the total at exactly 10 entity types and exactly 10 edge types

Output the COMPLETE ontology (all 10 entity types, all 10 edge types) with your modifications applied.

Output format:
```json
{
    "entity_types": [...],
    "edge_types": [...],
    "analysis_summary": "Brief analysis of the market context and any modifications made",
    "modifications": ["List of changes made to the base ontology, or 'No modifications needed'"]
}
```
"""


class FinancialOntologyGenerator:
    """
    Financial domain ontology generator.
    Provides a hard-coded base ontology for financial markets,
    with optional LLM-driven extension for specific market contexts.
    """

    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.llm_client = llm_client or LLMClient()

    def get_base_ontology(self) -> Dict[str, Any]:
        """Return the hard-coded financial base ontology."""
        return {
            "entity_types": FINANCIAL_ENTITY_TYPES,
            "edge_types": FINANCIAL_EDGE_TYPES,
            "analysis_summary": "Base financial market ontology with 10 entity types and 10 edge types",
            "domain": "financial",
        }

    def generate(
        self,
        market_context: Optional[str] = None,
        assets: Optional[List[str]] = None,
        news_headlines: Optional[List[str]] = None,
        extend_with_llm: bool = False,
    ) -> Dict[str, Any]:
        """
        Generate a financial ontology.

        If extend_with_llm is False, returns the base ontology directly (fast, no API call).
        If True, uses LLM to potentially extend the base types for the specific market context.

        Args:
            market_context: Description of current market conditions
            assets: List of assets to focus on
            news_headlines: Recent news headlines for context
            extend_with_llm: Whether to use LLM to extend the base ontology

        Returns:
            Complete financial ontology definition
        """
        base = self.get_base_ontology()

        if not extend_with_llm:
            logger.info("Using base financial ontology (no LLM extension)")
            return base

        # Build context for LLM
        context_parts = []
        if market_context:
            context_parts.append(f"## Market Context\n{market_context}")
        if assets:
            context_parts.append(f"## Target Assets\n{', '.join(assets)}")
        if news_headlines:
            headlines = "\n".join(f"- {h}" for h in news_headlines[:20])
            context_parts.append(f"## Recent Headlines\n{headlines}")

        if not context_parts:
            logger.info("No context provided, using base financial ontology")
            return base

        user_message = f"""## Base Financial Ontology

```json
{json.dumps(base, indent=2)}
```

## Market Context for Extension

{chr(10).join(context_parts)}

Review the market context above and determine if the base ontology needs modification.
If the base types adequately cover the situation, return them unchanged.
If specific types would be more useful (e.g., DeFi protocols for crypto, specific commodity types), make targeted replacements.

Remember:
- Keep exactly 10 entity types and 10 edge types
- Keep Person and Organization as fallback types (positions 9-10)
- Only modify if genuinely useful for this specific market context
"""

        messages = [
            {"role": "system", "content": FINANCIAL_ONTOLOGY_SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ]

        try:
            result = self.llm_client.chat_json(
                messages=messages,
                temperature=0.2,
                max_tokens=4096,
            )
            result["domain"] = "financial"
            result = self._validate(result)
            logger.info(f"Financial ontology generated with LLM extension: {result.get('modifications', [])}")
            return result
        except Exception as e:
            logger.warning(f"LLM extension failed, falling back to base ontology: {e}")
            return base

    def _validate(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Validate the ontology meets constraints."""
        # Ensure required fields
        if "entity_types" not in result:
            result["entity_types"] = FINANCIAL_ENTITY_TYPES
        if "edge_types" not in result:
            result["edge_types"] = FINANCIAL_EDGE_TYPES

        # Enforce limits
        if len(result["entity_types"]) > 10:
            result["entity_types"] = result["entity_types"][:10]
        if len(result["edge_types"]) > 10:
            result["edge_types"] = result["edge_types"][:10]

        # Ensure fallback types exist
        entity_names = {e["name"] for e in result["entity_types"]}
        if "Person" not in entity_names:
            result["entity_types"][-2] = FINANCIAL_ENTITY_TYPES[-2]
        if "Organization" not in entity_names:
            result["entity_types"][-1] = FINANCIAL_ENTITY_TYPES[-1]

        # Ensure all entities have required fields
        for entity in result["entity_types"]:
            entity.setdefault("attributes", [])
            entity.setdefault("examples", [])
            if len(entity.get("description", "")) > 100:
                entity["description"] = entity["description"][:97] + "..."

        for edge in result["edge_types"]:
            edge.setdefault("source_targets", [])
            edge.setdefault("attributes", [])
            if len(edge.get("description", "")) > 100:
                edge["description"] = edge["description"][:97] + "..."

        return result

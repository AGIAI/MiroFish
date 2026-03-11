"""
Trader Archetype Profile Generator
Generates quantitative trader agent profiles for market simulation.
Each archetype has calibrated behavioral parameters that determine
how they react to market conditions, news, and other agents.
"""

import random
from typing import Dict, List, Optional, Any
from ..utils.llm_client import LLMClient
from ..utils.logger import get_logger
from ..models.signal import TraderArchetype

logger = get_logger('mirofish.trader_profiles')


# ============================================================================
# Archetype Templates — calibrated defaults for each trading style
# ============================================================================

ARCHETYPE_TEMPLATES: Dict[str, Dict[str, Any]] = {
    "momentum": {
        "display_name_prefix": "Momentum",
        "risk_tolerance": 0.7,
        "time_horizon": "4h",
        "conviction_threshold": 0.55,
        "position_sizing_model": "volatility_adjusted",
        "primary_indicators": ["EMA_20_50_cross", "RSI_14", "MACD"],
        "secondary_indicators": ["volume_breakout", "ADX"],
        "entry_rules": "Buy when price breaks above EMA20 with volume confirmation and RSI > 50",
        "exit_rules": "Sell when RSI > 75 or EMA death cross or trailing stop 2x ATR hit",
        "narrative_sensitivity": 0.4,
        "herding_coefficient": 0.7,
        "contrarian_coefficient": 0.2,
        "fomo_susceptibility": 0.6,
        "panic_susceptibility": 0.3,
        "max_position_pct": 0.3,
        "max_drawdown_tolerance": 0.12,
        "stop_loss_method": "ATR",
        "stop_loss_atr_multiplier": 2.0,
        "information_lag_seconds": 15,
        "historical_accuracy": 0.48,
    },
    "mean_reversion": {
        "display_name_prefix": "MeanRev",
        "risk_tolerance": 0.5,
        "time_horizon": "1d",
        "conviction_threshold": 0.65,
        "position_sizing_model": "kelly",
        "primary_indicators": ["Bollinger_Bands", "RSI_14", "VWAP"],
        "secondary_indicators": ["funding_rate", "put_call_ratio"],
        "entry_rules": "Buy when RSI < 30 and price at lower Bollinger Band with decreasing sell volume",
        "exit_rules": "Sell when RSI > 65 or price at VWAP or 1.5x ATR profit target",
        "narrative_sensitivity": 0.3,
        "herding_coefficient": 0.15,
        "contrarian_coefficient": 0.8,
        "fomo_susceptibility": 0.1,
        "panic_susceptibility": 0.15,
        "max_position_pct": 0.2,
        "max_drawdown_tolerance": 0.08,
        "stop_loss_method": "support_level",
        "stop_loss_atr_multiplier": 1.5,
        "information_lag_seconds": 60,
        "historical_accuracy": 0.55,
    },
    "fundamental": {
        "display_name_prefix": "Fundamental",
        "risk_tolerance": 0.4,
        "time_horizon": "1w",
        "conviction_threshold": 0.75,
        "position_sizing_model": "kelly",
        "primary_indicators": ["PE_ratio", "revenue_growth", "FCF_yield"],
        "secondary_indicators": ["insider_buying", "institutional_ownership"],
        "entry_rules": "Buy when intrinsic value > 30% above market price with positive earnings trajectory",
        "exit_rules": "Sell when price reaches fair value or fundamentals deteriorate",
        "narrative_sensitivity": 0.15,
        "herding_coefficient": 0.1,
        "contrarian_coefficient": 0.6,
        "fomo_susceptibility": 0.05,
        "panic_susceptibility": 0.1,
        "max_position_pct": 0.15,
        "max_drawdown_tolerance": 0.2,
        "stop_loss_method": "fixed_pct",
        "stop_loss_atr_multiplier": 3.0,
        "information_lag_seconds": 300,
        "historical_accuracy": 0.58,
    },
    "narrative": {
        "display_name_prefix": "Narrative",
        "risk_tolerance": 0.65,
        "time_horizon": "1h",
        "conviction_threshold": 0.5,
        "position_sizing_model": "fixed_fraction",
        "primary_indicators": ["social_volume", "news_sentiment", "google_trends"],
        "secondary_indicators": ["reddit_mentions", "twitter_volume"],
        "entry_rules": "Buy when narrative momentum accelerates with increasing social volume",
        "exit_rules": "Sell when narrative velocity decelerates or counter-narrative emerges",
        "narrative_sensitivity": 0.95,
        "herding_coefficient": 0.6,
        "contrarian_coefficient": 0.15,
        "fomo_susceptibility": 0.8,
        "panic_susceptibility": 0.6,
        "max_position_pct": 0.2,
        "max_drawdown_tolerance": 0.15,
        "stop_loss_method": "ATR",
        "stop_loss_atr_multiplier": 2.5,
        "information_lag_seconds": 5,
        "historical_accuracy": 0.45,
    },
    "market_maker": {
        "display_name_prefix": "MM",
        "risk_tolerance": 0.3,
        "time_horizon": "1m",
        "conviction_threshold": 0.4,
        "position_sizing_model": "volatility_adjusted",
        "primary_indicators": ["order_flow_imbalance", "spread_width", "inventory_skew"],
        "secondary_indicators": ["volatility_surface", "depth_ratio"],
        "entry_rules": "Quote both sides continuously, skew prices based on inventory",
        "exit_rules": "Flatten inventory when spread compresses or volatility spikes",
        "narrative_sensitivity": 0.1,
        "herding_coefficient": 0.05,
        "contrarian_coefficient": 0.3,
        "fomo_susceptibility": 0.02,
        "panic_susceptibility": 0.05,
        "max_position_pct": 0.5,
        "max_drawdown_tolerance": 0.05,
        "stop_loss_method": "fixed_pct",
        "stop_loss_atr_multiplier": 1.0,
        "information_lag_seconds": 1,
        "historical_accuracy": 0.62,
    },
    "whale": {
        "display_name_prefix": "Whale",
        "risk_tolerance": 0.6,
        "time_horizon": "1d",
        "conviction_threshold": 0.7,
        "position_sizing_model": "kelly",
        "primary_indicators": ["liquidity_depth", "dark_pool_volume", "options_flow"],
        "secondary_indicators": ["institutional_flow", "COT_positioning"],
        "entry_rules": "Accumulate when liquidity is deep enough to absorb position without > 0.5% slippage",
        "exit_rules": "Distribute gradually through TWAP/VWAP algorithms over multiple sessions",
        "narrative_sensitivity": 0.3,
        "herding_coefficient": 0.1,
        "contrarian_coefficient": 0.5,
        "fomo_susceptibility": 0.05,
        "panic_susceptibility": 0.05,
        "max_position_pct": 0.4,
        "max_drawdown_tolerance": 0.15,
        "stop_loss_method": "support_level",
        "stop_loss_atr_multiplier": 3.0,
        "information_lag_seconds": 0,
        "historical_accuracy": 0.6,
    },
    "retail": {
        "display_name_prefix": "Retail",
        "risk_tolerance": 0.8,
        "time_horizon": "1h",
        "conviction_threshold": 0.4,
        "position_sizing_model": "fixed_fraction",
        "primary_indicators": ["price_action", "simple_MA"],
        "secondary_indicators": ["social_media_hype", "youtuber_calls"],
        "entry_rules": "Buy when price is pumping and everyone on social media is bullish",
        "exit_rules": "Panic sell on first red candle or hold through total drawdown",
        "narrative_sensitivity": 0.9,
        "herding_coefficient": 0.9,
        "contrarian_coefficient": 0.05,
        "fomo_susceptibility": 0.95,
        "panic_susceptibility": 0.85,
        "max_position_pct": 0.5,
        "max_drawdown_tolerance": 0.3,
        "stop_loss_method": "fixed_pct",
        "stop_loss_atr_multiplier": 1.0,
        "information_lag_seconds": 120,
        "historical_accuracy": 0.38,
    },
    "quant": {
        "display_name_prefix": "Quant",
        "risk_tolerance": 0.45,
        "time_horizon": "4h",
        "conviction_threshold": 0.65,
        "position_sizing_model": "kelly",
        "primary_indicators": ["statistical_arb_zscore", "pairs_spread", "volatility_regime"],
        "secondary_indicators": ["correlation_matrix", "mean_reversion_halflife"],
        "entry_rules": "Enter when z-score exceeds 2.0 and mean-reversion halflife < lookback window",
        "exit_rules": "Exit when z-score reverts to 0.5 or holding period exceeds 2x halflife",
        "narrative_sensitivity": 0.0,
        "herding_coefficient": 0.0,
        "contrarian_coefficient": 0.0,
        "fomo_susceptibility": 0.0,
        "panic_susceptibility": 0.0,
        "max_position_pct": 0.1,
        "max_drawdown_tolerance": 0.06,
        "stop_loss_method": "ATR",
        "stop_loss_atr_multiplier": 2.0,
        "information_lag_seconds": 5,
        "historical_accuracy": 0.54,
    },
}

# Default simulation composition — how many of each type
DEFAULT_COMPOSITION = {
    "momentum": 25,
    "mean_reversion": 15,
    "narrative": 20,
    "fundamental": 10,
    "retail": 40,
    "whale": 5,
    "market_maker": 10,
    "quant": 10,
}

# Lighter composition for faster simulations
LIGHT_COMPOSITION = {
    "momentum": 5,
    "mean_reversion": 3,
    "narrative": 5,
    "fundamental": 2,
    "retail": 8,
    "whale": 1,
    "market_maker": 3,
    "quant": 3,
}


class TraderProfileGenerator:
    """
    Generates a diverse population of trader agents for market simulation.
    Each agent has calibrated behavioral parameters based on their archetype,
    with random variation to create heterogeneity within types.
    """

    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.llm_client = llm_client

    def generate_population(
        self,
        composition: Optional[Dict[str, int]] = None,
        asset: str = "BTC/USDT",
        variation_pct: float = 0.15,
    ) -> List[TraderArchetype]:
        """
        Generate a full population of trader agents.

        Args:
            composition: Dict mapping archetype name to count.
                         Defaults to DEFAULT_COMPOSITION (135 agents).
            asset: Primary asset the agents will trade
            variation_pct: Random variation applied to numeric params (0-1)

        Returns:
            List of TraderArchetype profiles
        """
        if composition is None:
            composition = DEFAULT_COMPOSITION

        agents = []
        agent_id = 1

        for archetype_name, count in composition.items():
            template = ARCHETYPE_TEMPLATES.get(archetype_name)
            if not template:
                logger.warning(f"Unknown archetype: {archetype_name}, skipping")
                continue

            for i in range(count):
                agent = self._create_agent(
                    agent_id=agent_id,
                    archetype_name=archetype_name,
                    template=template,
                    variation_pct=variation_pct,
                    instance_num=i + 1,
                )
                agents.append(agent)
                agent_id += 1

        logger.info(
            f"Generated {len(agents)} trader agents: "
            + ", ".join(f"{k}={v}" for k, v in composition.items())
        )
        return agents

    def _create_agent(
        self,
        agent_id: int,
        archetype_name: str,
        template: Dict[str, Any],
        variation_pct: float,
        instance_num: int,
    ) -> TraderArchetype:
        """Create a single agent from template with random variation."""

        def vary(val: float, pct: float = variation_pct) -> float:
            """Apply random variation, clamped to [0, 1]."""
            delta = val * pct * random.uniform(-1, 1)
            return max(0.0, min(1.0, round(val + delta, 3)))

        prefix = template["display_name_prefix"]
        display_name = f"{prefix}_{agent_id:03d}"

        return TraderArchetype(
            agent_id=agent_id,
            archetype=archetype_name,
            display_name=display_name,
            risk_tolerance=vary(template["risk_tolerance"]),
            time_horizon=template["time_horizon"],
            conviction_threshold=vary(template["conviction_threshold"]),
            position_sizing_model=template["position_sizing_model"],
            primary_indicators=template["primary_indicators"],
            secondary_indicators=template["secondary_indicators"],
            entry_rules=template["entry_rules"],
            exit_rules=template["exit_rules"],
            narrative_sensitivity=vary(template["narrative_sensitivity"]),
            herding_coefficient=vary(template["herding_coefficient"]),
            contrarian_coefficient=vary(template["contrarian_coefficient"]),
            fomo_susceptibility=vary(template["fomo_susceptibility"]),
            panic_susceptibility=vary(template["panic_susceptibility"]),
            max_position_pct=vary(template["max_position_pct"]),
            max_drawdown_tolerance=vary(template["max_drawdown_tolerance"]),
            stop_loss_method=template["stop_loss_method"],
            stop_loss_atr_multiplier=round(
                template["stop_loss_atr_multiplier"] * random.uniform(0.85, 1.15), 2
            ),
            information_lag_seconds=max(0, int(
                template["information_lag_seconds"] * random.uniform(0.5, 1.5)
            )),
            historical_accuracy=vary(template["historical_accuracy"], pct=0.05),
        )

    def get_archetype_summary(self) -> Dict[str, Dict[str, Any]]:
        """Return a summary of all available archetypes and their key traits."""
        summary = {}
        for name, template in ARCHETYPE_TEMPLATES.items():
            summary[name] = {
                "time_horizon": template["time_horizon"],
                "risk_tolerance": template["risk_tolerance"],
                "key_indicators": template["primary_indicators"],
                "behavioral_traits": {
                    "narrative_sensitivity": template["narrative_sensitivity"],
                    "herding": template["herding_coefficient"],
                    "contrarian": template["contrarian_coefficient"],
                    "fomo": template["fomo_susceptibility"],
                    "panic": template["panic_susceptibility"],
                },
                "historical_accuracy": template["historical_accuracy"],
                "entry_rules": template["entry_rules"],
            }
        return summary

    def to_simulation_config(
        self,
        agents: List[TraderArchetype],
        asset: str = "BTC/USDT",
    ) -> Dict[str, Any]:
        """
        Convert agent profiles to a simulation-ready config format
        compatible with the existing SimulationRunner.

        Args:
            agents: List of TraderArchetype profiles
            asset: Primary asset

        Returns:
            Dict with simulation config
        """
        agent_configs = []
        for agent in agents:
            agent_configs.append({
                "agent_id": agent.agent_id,
                "display_name": agent.display_name,
                "archetype": agent.archetype,
                "persona": self._build_persona_string(agent),
                "parameters": agent.model_dump(),
            })

        return {
            "asset": asset,
            "agent_count": len(agents),
            "agents": agent_configs,
            "archetype_distribution": self._get_distribution(agents),
        }

    def _build_persona_string(self, agent: TraderArchetype) -> str:
        """Build a natural language persona string for LLM-driven agents."""
        return (
            f"You are {agent.display_name}, a {agent.archetype} trader. "
            f"Risk tolerance: {agent.risk_tolerance:.0%}. "
            f"Time horizon: {agent.time_horizon}. "
            f"You primarily watch {', '.join(agent.primary_indicators)}. "
            f"Entry strategy: {agent.entry_rules}. "
            f"Exit strategy: {agent.exit_rules}. "
            f"You {'are highly' if agent.narrative_sensitivity > 0.7 else 'are somewhat' if agent.narrative_sensitivity > 0.3 else 'are NOT'} influenced by news and narratives. "
            f"You {'tend to follow' if agent.herding_coefficient > 0.5 else 'sometimes follow' if agent.herding_coefficient > 0.2 else 'rarely follow'} the crowd. "
            f"You {'are prone to FOMO' if agent.fomo_susceptibility > 0.5 else 'resist FOMO well'}. "
            f"You {'panic easily' if agent.panic_susceptibility > 0.5 else 'stay calm under pressure'}. "
            f"Max position size: {agent.max_position_pct:.0%} of portfolio. "
            f"Stop loss: {agent.stop_loss_method} at {agent.stop_loss_atr_multiplier}x."
        )

    def _get_distribution(self, agents: List[TraderArchetype]) -> Dict[str, int]:
        """Count agents per archetype."""
        dist: Dict[str, int] = {}
        for a in agents:
            dist[a.archetype] = dist.get(a.archetype, 0) + 1
        return dist

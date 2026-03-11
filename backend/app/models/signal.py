"""
Trading Signal Model
Pydantic models for MiroFish conviction signals.

MiroFish's job: look at data, find the opportunity, state the trade, give confidence.
Execution (entry/SL/TP/sizing) belongs in Meridian.
"""

import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any
from enum import Enum
from pydantic import BaseModel, Field


class SignalDirection(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    FLAT = "FLAT"


class MarketRegime(str, Enum):
    TRENDING_BULL = "trending_bull"
    TRENDING_BEAR = "trending_bear"
    RANGING = "ranging"
    HIGH_VOL = "high_vol"
    CRISIS = "crisis"
    RECOVERY = "recovery"


# ============================================================================
# Signal Decomposition
# ============================================================================

class SignalComponents(BaseModel):
    """Decomposed signal components — explains WHY the signal exists."""
    agent_consensus: float = Field(ge=0, le=1, description="% of agents agreeing on direction")
    agent_conviction_strength: float = Field(ge=0, le=1, description="Average conviction of majority side")
    narrative_momentum: float = Field(ge=-1, le=1, description="Narrative strength + direction")
    flow_imbalance: float = Field(ge=-1, le=1, description="Net buy/sell pressure")
    crowding_risk: float = Field(ge=0, le=1, description="How one-sided the trade is (1.0 = max crowding)")
    information_edge_score: float = Field(ge=0, le=1, description="How much 'new' info vs already priced-in")


class Finding(BaseModel):
    """A single piece of evidence supporting (or contradicting) the signal."""
    source: str = Field(description="technical, news, on_chain, order_flow, agent_simulation, external")
    detail: str = Field(description="Human-readable description of what was found")
    direction: Optional[str] = Field(default=None, description="LONG, SHORT, or None if neutral")
    strength: float = Field(ge=0, le=1, default=0.5, description="How strong this finding is")


# ============================================================================
# Core Signal — What MiroFish Outputs
# ============================================================================

class TradingSignal(BaseModel):
    """
    MiroFish conviction signal.

    Answers three questions:
    1. What was found? (thesis + findings)
    2. What's the trade? (asset + direction + time horizon)
    3. What's the confidence? (conviction 0-1)

    Execution details (entry/SL/TP/sizing) are Meridian's responsibility.
    """
    signal_id: str = Field(default_factory=lambda: f"sig_{datetime.utcnow().strftime('%Y%m%d%H%M')}_{uuid.uuid4().hex[:6]}")
    version: str = "2.0"

    # What to trade
    asset: str = Field(description="Trading pair (e.g. BTC/USDT, AAPL)")
    exchange: Optional[str] = Field(default=None, description="Target exchange (e.g. binance, NYSE)")

    # When
    timestamp_utc: str = Field(default_factory=lambda: datetime.utcnow().isoformat())

    # The call
    direction: SignalDirection
    conviction: float = Field(ge=0, le=1, description="Overall signal confidence 0-1")
    time_horizon_hours: int = Field(ge=1, description="Expected relevance window in hours")

    # What was found
    thesis: str = Field(default="", description="One-sentence summary of the opportunity")
    findings: List[Finding] = Field(default_factory=list, description="Evidence from each data source")

    # Market context
    regime: MarketRegime = MarketRegime.RANGING

    # Signal decomposition
    components: SignalComponents

    # Key risks to this thesis
    key_risks: List[str] = Field(default_factory=list)

    # Metadata
    dominant_narrative: Optional[str] = None
    simulation_id: Optional[str] = None
    simulation_rounds: int = 0
    agents_participated: int = 0

    # Status tracking
    status: str = "active"  # active, expired, invalidated
    actual_outcome_pct: Optional[float] = None


# ============================================================================
# Trader Archetype Model (for MiroFish's agent simulation)
# ============================================================================

class TraderArchetype(BaseModel):
    """A trader agent archetype with quantitative decision parameters."""
    agent_id: int
    archetype: str = Field(description="momentum, mean_reversion, fundamental, narrative, market_maker, whale, retail, quant")
    display_name: str

    # Decision parameters
    risk_tolerance: float = Field(ge=0, le=1, default=0.5)
    time_horizon: str = Field(default="4h", description="scalp, 1m, 5m, 1h, 4h, 1d, 1w")
    conviction_threshold: float = Field(ge=0, le=1, default=0.6, description="Min signal strength to act")

    # Strategy
    primary_indicators: List[str] = Field(default_factory=list)
    secondary_indicators: List[str] = Field(default_factory=list)
    entry_rules: str = ""
    exit_rules: str = ""

    # Behavioral parameters
    narrative_sensitivity: float = Field(ge=0, le=1, default=0.5, description="How much news affects decisions")
    herding_coefficient: float = Field(ge=0, le=1, default=0.3, description="Tendency to follow crowd")
    contrarian_coefficient: float = Field(ge=0, le=1, default=0.5, description="Tendency to fade crowd")
    fomo_susceptibility: float = Field(ge=0, le=1, default=0.3, description="Tendency to chase pumps")
    panic_susceptibility: float = Field(ge=0, le=1, default=0.3, description="Tendency to panic sell")

    # Agent risk parameters (how agents think, not execution)
    position_sizing_model: str = Field(default="kelly", description="kelly, fixed_fraction, volatility_adjusted")
    max_position_pct: float = Field(ge=0, le=1, default=0.25)
    max_drawdown_tolerance: float = Field(ge=0, le=1, default=0.15)
    stop_loss_method: str = Field(default="ATR", description="ATR, fixed_pct, support_level")
    stop_loss_atr_multiplier: float = Field(default=2.0, ge=0.5, le=5.0)

    # Meta
    information_lag_seconds: int = Field(default=30, ge=0)
    historical_accuracy: float = Field(ge=0, le=1, default=0.52)

"""
Trading Signal Model
Pydantic models for trading signals, backtest results, and portfolio state.
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


class EntryType(str, Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"


class StopLossType(str, Enum):
    ATR = "ATR"
    FIXED_PCT = "FIXED_PCT"
    SUPPORT_LEVEL = "SUPPORT_LEVEL"


# ============================================================================
# Core Signal Models
# ============================================================================

class SignalEntry(BaseModel):
    type: EntryType = EntryType.LIMIT
    price: float
    valid_until: Optional[str] = None


class SignalStopLoss(BaseModel):
    price: float
    type: StopLossType = StopLossType.ATR
    risk_pct: float = Field(ge=0, le=100, description="Risk as % of portfolio")


class SignalTakeProfit(BaseModel):
    price: float
    close_pct: float = Field(ge=0, le=100, description="% of position to close at this level")


class SignalComponents(BaseModel):
    """Decomposed signal components — explains WHY the signal exists."""
    agent_consensus: float = Field(ge=0, le=1, description="% of agents agreeing on direction")
    agent_conviction_strength: float = Field(ge=0, le=1, description="Average conviction of majority side")
    narrative_momentum: float = Field(ge=-1, le=1, description="Narrative strength + direction")
    flow_imbalance: float = Field(ge=-1, le=1, description="Net buy/sell pressure from simulation")
    crowding_risk: float = Field(ge=0, le=1, description="How one-sided the trade is (1.0 = max crowding)")
    information_edge_score: float = Field(ge=0, le=1, description="How much 'new' info vs already priced-in")


class HistoricalSetup(BaseModel):
    date: str
    outcome: str
    similarity: float = Field(ge=0, le=1)


class BacktestReference(BaseModel):
    similar_setups_count: int = 0
    similar_setups_win_rate: float = 0.0
    expected_sharpe: float = 0.0


class TradingSignal(BaseModel):
    """
    The core output product — a calibrated, actionable trading signal.
    Compatible with Freqtrade, CCXT, 3Commas, Cornix webhook formats.
    """
    signal_id: str = Field(default_factory=lambda: f"sig_{datetime.utcnow().strftime('%Y%m%d%H%M')}_{uuid.uuid4().hex[:6]}")
    version: str = "1.0"

    # What to trade
    asset: str = Field(description="Trading pair (e.g. BTC/USDT, AAPL)")
    exchange: Optional[str] = Field(default=None, description="Target exchange (e.g. binance, NYSE)")

    # When
    timestamp_utc: str = Field(default_factory=lambda: datetime.utcnow().isoformat())

    # Direction & Conviction
    direction: SignalDirection
    conviction: float = Field(ge=0, le=1, description="Overall signal confidence 0-1")

    # Trade plan
    entry: SignalEntry
    stop_loss: SignalStopLoss
    take_profit: List[SignalTakeProfit] = Field(default_factory=list)
    position_size_pct: float = Field(ge=0, le=100, description="Suggested position size as % of portfolio")
    time_horizon_hours: int = Field(ge=1, description="Expected holding period in hours")

    # Market context
    regime: MarketRegime = MarketRegime.RANGING

    # Signal decomposition
    components: SignalComponents

    # Backtesting context
    backtest: Optional[BacktestReference] = None
    similar_historical_setups: List[HistoricalSetup] = Field(default_factory=list)

    # Metadata
    simulation_id: Optional[str] = None
    simulation_rounds: int = 0
    agents_participated: int = 0
    dominant_narrative: Optional[str] = None
    key_risk: Optional[str] = None

    # Status tracking
    status: str = "active"  # active, expired, closed_win, closed_loss, cancelled
    actual_outcome_pct: Optional[float] = None


# ============================================================================
# Backtest Models
# ============================================================================

class BacktestMetrics(BaseModel):
    """Comprehensive backtest performance metrics."""
    total_signals: int = 0
    win_rate: float = 0.0
    avg_win_pct: float = 0.0
    avg_loss_pct: float = 0.0
    profit_factor: float = Field(default=0.0, description="Gross profit / gross loss")
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    max_drawdown_pct: float = 0.0
    max_drawdown_duration_days: int = 0
    avg_holding_period_hours: float = 0.0
    signal_decay_halflife_hours: float = 0.0
    correlation_to_benchmark: float = Field(default=0.0, description="Low = good, means alpha is unique")
    tail_ratio: float = Field(default=1.0, description="Right tail / left tail of returns")
    expectancy: float = Field(default=0.0, description="Expected $ per $ risked")

    # Per-regime breakdown
    metrics_by_regime: Dict[str, Dict[str, float]] = Field(default_factory=dict)


class BacktestTrade(BaseModel):
    """A single trade within a backtest."""
    signal: TradingSignal
    entry_time: str
    exit_time: Optional[str] = None
    entry_price: float
    exit_price: Optional[float] = None
    return_pct: float = 0.0
    holding_hours: float = 0.0
    regime_at_entry: MarketRegime = MarketRegime.RANGING
    is_win: bool = False


class BacktestResult(BaseModel):
    """Complete backtest result."""
    backtest_id: str = Field(default_factory=lambda: f"bt_{uuid.uuid4().hex[:8]}")
    asset: str
    start_date: str
    end_date: str
    config: Dict[str, Any] = Field(default_factory=dict)
    metrics: BacktestMetrics = Field(default_factory=BacktestMetrics)
    trades: List[BacktestTrade] = Field(default_factory=list)
    equity_curve: List[Dict[str, float]] = Field(default_factory=list, description="[{timestamp, equity, drawdown}]")
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


# ============================================================================
# Portfolio Models
# ============================================================================

class Position(BaseModel):
    """An active portfolio position."""
    asset: str
    direction: SignalDirection
    entry_price: float
    current_price: float = 0.0
    size_pct: float = 0.0
    unrealized_pnl_pct: float = 0.0
    signal_id: str = ""
    entry_time: str = ""


class PortfolioState(BaseModel):
    """Current portfolio state for risk management."""
    total_equity: float = 100000.0
    cash_pct: float = 100.0
    positions: List[Position] = Field(default_factory=list)
    daily_pnl_pct: float = 0.0
    weekly_pnl_pct: float = 0.0
    total_pnl_pct: float = 0.0
    max_drawdown_pct: float = 0.0
    consecutive_losses: int = 0
    active_signals: int = 0


# ============================================================================
# Trader Archetype Model
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
    position_sizing_model: str = Field(default="kelly", description="kelly, fixed_fraction, volatility_adjusted")

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

    # Risk management
    max_position_pct: float = Field(ge=0, le=1, default=0.25)
    max_drawdown_tolerance: float = Field(ge=0, le=1, default=0.15)
    stop_loss_method: str = Field(default="ATR", description="ATR, fixed_pct, support_level")
    stop_loss_atr_multiplier: float = Field(default=2.0, ge=0.5, le=5.0)

    # Meta
    information_lag_seconds: int = Field(default=30, ge=0)
    historical_accuracy: float = Field(ge=0, le=1, default=0.52)

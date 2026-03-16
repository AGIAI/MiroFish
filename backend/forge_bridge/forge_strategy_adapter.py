"""
Forge Strategy Adapter

Wraps MiroFish signals as a Forge-compatible strategy for CPCV validation.
Implements the interface that Forge's backtesting pipeline expects.

Key Properties:
    - n_free_parameters = 0 (no curve fitting → PBO uses z-score form per §8.2)
    - signal_class = "informational" (Mode A kill gate per §10.5)
    - complexity_tier = "minimal" (Sharpe gate modifier = -0.05 per §4.6)

Forge Compliance:
    - §2: CPCV-compatible backtest() method
    - §9: Cost model integration
    - §10.5: Supports override_signals for randomised baseline
    - §10.6: Supports signal inversion
    - §15: Kelly-based position sizing
"""

import numpy as np
import pandas as pd
from datetime import datetime, timezone
from typing import Optional

from .db_writer import SignalDBWriter

import logging

logger = logging.getLogger("forge_bridge.adapter")


class MiroFishForgeStrategy:
    """
    Adapts MiroFish signals for Forge backtesting pipeline.

    This strategy has 0 free parameters because:
    - LLM agent behaviour is not fitted to the backtest period
    - Consensus metrics are derived from information theory, not optimised
    - Kelly sizing is the standard formula, not a tuned parameter
    """

    def __init__(
        self,
        strategy_id: str,
        asset: str,
        asset_class: str,
        db: SignalDBWriter,
        cost_bps: Optional[float] = None,
    ):
        self.strategy_id = strategy_id
        self.asset = asset
        self.asset_class = asset_class
        self.db = db

        # Cost defaults from Forge §9.6 table, selected by asset class
        self._cost_defaults = {
            "crypto_perps": {"maker_bps": 2, "taker_bps": 5, "spread_bps": 3, "impact_k": 1.0},
            "crypto_spot": {"maker_bps": 5, "taker_bps": 10, "spread_bps": 10, "impact_k": 1.0},
            "us_large_cap": {"maker_bps": 0, "taker_bps": 0, "spread_bps": 1, "impact_k": 0.5},
            "us_small_cap": {"maker_bps": 0, "taker_bps": 0, "spread_bps": 15, "impact_k": 2.0},
            "fx_majors": {"maker_bps": 0, "taker_bps": 0, "spread_bps": 0.3, "impact_k": 0.3},
            "defi": {"maker_bps": 15, "taker_bps": 15, "spread_bps": 50, "impact_k": 3.0},
        }

        if cost_bps is not None:
            self.cost_per_turn_bps = cost_bps
        else:
            defaults = self._cost_defaults.get(asset_class, self._cost_defaults["crypto_perps"])
            self.cost_per_turn_bps = defaults["taker_bps"] + defaults["spread_bps"] / 2

    @property
    def n_free_parameters(self) -> int:
        return 0

    @property
    def signal_class(self) -> str:
        return "informational"

    @property
    def complexity_tier(self) -> str:
        return "minimal"  # 0 params → Sharpe gate modifier = -0.05

    @property
    def randomised_baseline_mode(self) -> str:
        return "A"  # kill gate for informational signals

    def backtest(
        self,
        train: pd.DataFrame,
        test: pd.DataFrame,
        override_signals: Optional[list] = None,
        cost_multiplier: float = 1.0,
        execution_delay_bars: int = 0,
    ) -> dict:
        """
        Forge-compatible backtest method.

        Args:
            train: Price data for parameter fitting (unused — 0 free params)
            test: Price data for OOS evaluation. Must have DatetimeIndex and 'close' column.
            override_signals: For randomised baseline test (§10.5). List of signal dicts.
            cost_multiplier: For cost sensitivity (§9.3). 1.0 = realistic, 2.0 = adversarial.
            execution_delay_bars: For latency sensitivity (§10.7).

        Returns:
            Dict with all metrics required by Forge §19.
        """
        # Fetch or use overridden signals
        if override_signals is not None:
            signals = override_signals
        else:
            signals = self._fetch_signals(test.index[0], test.index[-1])

        # Convert signals to position series
        positions = self._signals_to_positions(signals, test, execution_delay_bars)

        # Compute returns with costs
        returns = self._compute_returns(positions, test, cost_multiplier)

        # Compute all required metrics
        return self._compute_metrics(returns, positions, test, signals)

    def backtest_inverse(self, train: pd.DataFrame, test: pd.DataFrame) -> dict:
        """
        Run backtest with inverted signals (Forge §10.6).
        For informational strategies: inverse must have negative Sharpe.
        """
        signals = self._fetch_signals(test.index[0], test.index[-1])
        inverted = []
        for s in signals:
            inv = dict(s)
            inv["direction"] = -inv["direction"]
            inverted.append(inv)
        return self.backtest(train, test, override_signals=inverted)

    def _fetch_signals(self, start: datetime, end: datetime) -> list:
        """Fetch signals from the database for the given time range."""
        if isinstance(start, pd.Timestamp):
            start = start.to_pydatetime()
        if isinstance(end, pd.Timestamp):
            end = end.to_pydatetime()

        # Ensure timezone-aware
        if start.tzinfo is None:
            start = start.replace(tzinfo=timezone.utc)
        if end.tzinfo is None:
            end = end.replace(tzinfo=timezone.utc)

        return self.db.fetch_signals(self.strategy_id, start, end)

    def _signals_to_positions(
        self,
        signals: list,
        prices: pd.DataFrame,
        delay_bars: int = 0,
    ) -> pd.Series:
        """
        Convert MiroFish signals to position series.

        Position sizing uses Kelly criterion (Forge §15):
            f* = (p × b - q) / b
        where p = confidence, b = 1 (symmetric payoff), q = 1 - p
        Then: position = direction × kelly × fractional_kelly_scale

        Fractional Kelly = 0.25 (Forge §15.1 default)
        """
        positions = pd.Series(0.0, index=prices.index)

        for sig in signals:
            ts_from = sig.get("ts_valid_from")
            ts_until = sig.get("ts_valid_until")

            if isinstance(ts_from, str):
                ts_from = pd.Timestamp(ts_from)
            if isinstance(ts_until, str):
                ts_until = pd.Timestamp(ts_until)

            # Apply execution delay
            if delay_bars > 0 and len(prices.index) > 1:
                bar_duration = prices.index[1] - prices.index[0]
                ts_from = ts_from + delay_bars * bar_duration

            valid_mask = (prices.index >= ts_from) & (prices.index < ts_until)

            direction = sig["direction"]
            confidence = sig.get("calibrated_confidence") or sig["confidence"]

            # Kelly fraction: f* = 2p - 1 for symmetric payoff (Forge §15.1)
            # Then scaled by 0.25 (fractional Kelly)
            kelly = max(0.0, 2.0 * confidence - 1.0) * 0.25

            # Position = direction × kelly fraction
            positions[valid_mask] = direction * kelly

        return positions

    def _compute_returns(
        self,
        positions: pd.Series,
        prices: pd.DataFrame,
        cost_multiplier: float = 1.0,
    ) -> pd.Series:
        """
        Compute strategy returns from positions and prices.
        Includes transaction costs per Forge §9.
        """
        price_returns = prices["close"].pct_change().fillna(0)
        strategy_returns = positions.shift(1).fillna(0) * price_returns

        # Transaction costs (applied on position changes)
        position_changes = positions.diff().abs().fillna(0)
        cost_per_change = self.cost_per_turn_bps * cost_multiplier / 10000
        costs = position_changes * cost_per_change

        return strategy_returns - costs

    def _compute_metrics(
        self,
        returns: pd.Series,
        positions: pd.Series,
        prices: pd.DataFrame,
        signals: list,
    ) -> dict:
        """Compute all metrics required by Forge §19."""
        # Annualisation factor (derive from data frequency)
        if len(returns) > 1:
            avg_bar_hours = (returns.index[-1] - returns.index[0]).total_seconds() / 3600 / len(returns)
            ann_factor = 8760 / max(avg_bar_hours, 0.01)  # 8760 hours/year
        else:
            ann_factor = 252  # fallback

        ann_factor_sqrt = np.sqrt(ann_factor)

        # Core metrics
        mean_ret = returns.mean()
        std_ret = returns.std()
        sharpe = mean_ret / std_ret * ann_factor_sqrt if std_ret > 0 else 0.0

        # Sortino
        downside = returns[returns < 0]
        downside_std = downside.std() if len(downside) > 0 else std_ret
        sortino = mean_ret / downside_std * ann_factor_sqrt if downside_std > 0 else 0.0

        # Drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.cummax()
        drawdowns = cumulative / running_max - 1
        max_dd = float(drawdowns.min())
        max_dd_pct = abs(max_dd) * 100

        # Drawdown duration
        in_dd = drawdowns < 0
        if in_dd.any():
            dd_groups = (~in_dd).cumsum()
            dd_durations = in_dd.groupby(dd_groups).sum()
            max_dd_duration = int(dd_durations.max())
        else:
            max_dd_duration = 0

        # CAGR
        total_return = float(cumulative.iloc[-1] - 1) if len(cumulative) > 0 else 0.0
        n_years = len(returns) / ann_factor if ann_factor > 0 else 1
        cagr = (1 + total_return) ** (1 / max(n_years, 0.01)) - 1 if total_return > -1 else -1

        # Win rate
        trades = returns[returns != 0]
        n_trades = len(trades)
        win_rate = (trades > 0).mean() if n_trades > 0 else 0.0

        # Profit factor
        gross_profit = trades[trades > 0].sum()
        gross_loss = abs(trades[trades < 0].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        # Trade statistics
        position_changes = positions.diff().fillna(0)
        n_trades_actual = int((position_changes.abs() > 0.001).sum())

        # Average holding period
        if n_trades_actual > 0:
            avg_holding_bars = len(positions) / max(n_trades_actual, 1)
        else:
            avg_holding_bars = len(positions)

        # Calmar
        calmar = cagr / max(max_dd_pct / 100, 0.001)

        # Ulcer Index
        dd_sq = drawdowns**2
        ulcer = float(np.sqrt(dd_sq.mean())) if len(dd_sq) > 0 else 0.0

        # Skewness and kurtosis
        skewness = float(returns.skew()) if len(returns) > 2 else 0.0
        kurt = float(returns.kurtosis()) if len(returns) > 3 else 0.0

        return {
            "sharpe": round(sharpe, 4),
            "sortino": round(sortino, 4),
            "cagr": round(cagr, 4),
            "total_return": round(total_return, 4),
            "max_dd_pct": round(max_dd_pct, 2),
            "max_dd_duration_bars": max_dd_duration,
            "n_trades": n_trades_actual,
            "win_rate": round(win_rate, 4),
            "profit_factor": round(profit_factor, 4) if profit_factor != float("inf") else 999.0,
            "calmar": round(calmar, 4),
            "ulcer_index": round(ulcer, 4),
            "skewness": round(skewness, 4),
            "kurtosis": round(kurt, 4),
            "avg_holding_bars": round(avg_holding_bars, 1),
            "annual_return": round(mean_ret * ann_factor, 4),
            "annual_vol": round(std_ret * ann_factor_sqrt, 4),
            "ann_factor": ann_factor,
        }

    def generate_random_signals(self, n_bars: int, n_signals: int, avg_holding_bars: float) -> list:
        """
        Generate random signals for the randomised baseline test (Forge §10.5).

        Preserves the same number of signals and average holding period
        as the original strategy, but with random entry timing.
        """
        rng = np.random.default_rng()
        signals = []

        for _ in range(n_signals):
            # Random entry point
            entry_bar = rng.integers(0, max(1, n_bars - int(avg_holding_bars)))
            # Random direction
            direction = rng.choice([-1.0, -0.5, 0.5, 1.0])
            # Random confidence (uniform between 0.5 and 0.9)
            confidence = rng.uniform(0.5, 0.9)

            signals.append({
                "ts_valid_from": entry_bar,  # bar index (converted by caller)
                "ts_valid_until": entry_bar + int(avg_holding_bars),
                "direction": direction,
                "confidence": confidence,
                "calibrated_confidence": None,
            })

        return signals

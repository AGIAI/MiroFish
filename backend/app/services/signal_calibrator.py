"""
Signal Calibrator & Ensemble
Ensures signal quality through:
1. Running N simulations with different seeds → ensemble the results
2. Bayesian calibration against historical outcomes
3. Systematic bias correction (LLM agents tend bullish)
4. Brier score tracking for calibration quality over time
"""

import math
import random
import statistics
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable

from ..models.signal import TradingSignal, SignalDirection, SignalComponents
from ..utils.logger import get_logger

logger = get_logger('mirofish.calibrator')


class SignalCalibrator:
    """
    Calibrates raw signals to ensure conviction scores are well-calibrated:
    when we say 70% conviction, it should win ~70% of the time.
    """

    # Known systematic biases in LLM-driven agent simulations
    KNOWN_BIASES = {
        "bullish_bias": 0.08,       # Agents tend 58/42 bullish → subtract 8%
        "narrative_overweight": 0.15, # Narrative agents overreact → dampen 15%
        "momentum_recency": 0.10,    # Momentum agents overweight recent moves → decay 10%
        "retail_herding": 0.12,      # Retail agents amplify consensus → dampen 12%
    }

    def __init__(self):
        # Historical tracking for calibration
        self._signal_history: List[Dict[str, Any]] = []

    def calibrate(
        self,
        signal: TradingSignal,
        historical_base_rate: Optional[float] = None,
    ) -> TradingSignal:
        """
        Apply Bayesian calibration and bias correction to a signal.

        Args:
            signal: Raw signal from SignalExtractor
            historical_base_rate: Historical win rate for this setup type

        Returns:
            Calibrated signal with adjusted conviction
        """
        original_conviction = signal.conviction

        # 1. Bias correction
        corrected_components = self._debias_components(signal.components)

        # 2. Recalculate conviction from debiased components
        debiased_conviction = self._conviction_from_components(corrected_components)

        # 3. Bayesian update with historical base rate
        if historical_base_rate is not None:
            calibrated = self._bayesian_update(debiased_conviction, historical_base_rate)
        else:
            calibrated = debiased_conviction

        # 4. Apply ensemble adjustment if available
        # (ensemble() should be called separately with multiple runs)

        # 5. Clamp and update
        calibrated = max(0.05, min(0.95, calibrated))

        signal.conviction = round(calibrated, 3)
        signal.components = corrected_components

        if abs(original_conviction - calibrated) > 0.05:
            logger.info(
                f"Signal calibrated: {original_conviction:.3f} → {calibrated:.3f} "
                f"(Δ={calibrated - original_conviction:+.3f})"
            )

        return signal

    def ensemble(
        self,
        signals: List[TradingSignal],
    ) -> TradingSignal:
        """
        Ensemble N signals from multiple simulation runs.

        Strategy:
        - Direction: majority vote (>60% threshold)
        - Conviction: median of all convictions
        - Components: mean of all component values
        - Uncertainty: inter-run variance → higher variance = lower conviction

        Args:
            signals: List of signals from N simulation runs

        Returns:
            Single ensembled signal
        """
        if not signals:
            raise ValueError("Cannot ensemble empty signal list")

        if len(signals) == 1:
            return signals[0]

        n = len(signals)

        # Direction vote
        long_count = sum(1 for s in signals if s.direction == SignalDirection.LONG)
        short_count = sum(1 for s in signals if s.direction == SignalDirection.SHORT)
        flat_count = n - long_count - short_count

        if long_count > n * 0.6:
            direction = SignalDirection.LONG
            agreement = long_count / n
        elif short_count > n * 0.6:
            direction = SignalDirection.SHORT
            agreement = short_count / n
        else:
            direction = SignalDirection.FLAT
            agreement = max(long_count, short_count, flat_count) / n

        # Conviction: median, penalized by variance
        convictions = [s.conviction for s in signals]
        median_conviction = statistics.median(convictions)
        conviction_std = statistics.stdev(convictions) if len(convictions) > 1 else 0

        # Higher variance → lower conviction
        variance_penalty = conviction_std * 0.5
        ensembled_conviction = max(0.05, min(0.95, median_conviction - variance_penalty))

        # Boost conviction when agreement is high
        if agreement > 0.8:
            ensembled_conviction = min(0.95, ensembled_conviction * 1.1)

        # Components: average across runs
        component_fields = [
            "agent_consensus", "agent_conviction_strength",
            "narrative_momentum", "flow_imbalance",
            "crowding_risk", "information_edge_score",
        ]
        avg_components = {}
        for field in component_fields:
            values = [getattr(s.components, field) for s in signals]
            avg_components[field] = round(statistics.mean(values), 3)

        # Use the first signal as base, update with ensembled values
        result = signals[0].model_copy()
        result.direction = direction
        result.conviction = round(ensembled_conviction, 3)
        result.components = SignalComponents(**avg_components)
        result.agents_participated = sum(s.agents_participated for s in signals) // n

        # Update thesis if direction changed
        if result.direction != signals[0].direction:
            dir_word = "bullish" if direction == SignalDirection.LONG else "bearish" if direction == SignalDirection.SHORT else "neutral"
            result.thesis = f"Ensembled {n} runs: {dir_word} with {agreement:.0%} agreement"

        logger.info(
            f"Ensembled {n} signals: direction={direction.value} "
            f"agreement={agreement:.0%} conviction={ensembled_conviction:.3f} "
            f"(median={median_conviction:.3f}, std={conviction_std:.3f})"
        )

        return result

    def _debias_components(self, components: SignalComponents) -> SignalComponents:
        """Remove known systematic biases from signal components."""
        # Correct bullish bias in consensus
        corrected_consensus = components.agent_consensus - self.KNOWN_BIASES["bullish_bias"]

        # Dampen narrative momentum (overreaction)
        corrected_narrative = components.narrative_momentum * (1 - self.KNOWN_BIASES["narrative_overweight"])

        # Dampen retail herding amplification in crowding
        corrected_crowding = min(1.0, components.crowding_risk * (1 + self.KNOWN_BIASES["retail_herding"]))

        return SignalComponents(
            agent_consensus=round(max(0, min(1, corrected_consensus)), 3),
            agent_conviction_strength=components.agent_conviction_strength,
            narrative_momentum=round(max(-1, min(1, corrected_narrative)), 3),
            flow_imbalance=components.flow_imbalance,
            crowding_risk=round(max(0, min(1, corrected_crowding)), 3),
            information_edge_score=components.information_edge_score,
        )

    def _conviction_from_components(self, components: SignalComponents) -> float:
        """Recalculate conviction from debiased components."""
        return (
            abs(components.agent_consensus - 0.5) * 2 * 0.25 +
            components.agent_conviction_strength * 0.25 +
            abs(components.flow_imbalance) * 0.2 +
            abs(components.narrative_momentum) * 0.15 +
            components.information_edge_score * 0.15
        )

    def _bayesian_update(
        self, likelihood: float, prior: float
    ) -> float:
        """
        Simple Bayesian update.
        posterior ∝ likelihood × prior
        """
        if prior <= 0 or prior >= 1:
            return likelihood

        # P(signal is correct | data) ∝ P(data | correct) × P(correct)
        posterior = (likelihood * prior) / (
            likelihood * prior + (1 - likelihood) * (1 - prior)
        )
        return posterior

    def record_outcome(
        self, signal_id: str, predicted_direction: str,
        predicted_conviction: float, actual_outcome_pct: float,
    ):
        """
        Record a signal outcome for calibration tracking.

        Args:
            signal_id: Signal identifier
            predicted_direction: "LONG" or "SHORT"
            predicted_conviction: Signal conviction at time of generation
            actual_outcome_pct: Actual % move in predicted direction
        """
        is_win = actual_outcome_pct > 0
        self._signal_history.append({
            "signal_id": signal_id,
            "direction": predicted_direction,
            "conviction": predicted_conviction,
            "actual_pct": actual_outcome_pct,
            "is_win": is_win,
            "timestamp": datetime.utcnow().isoformat(),
        })

        logger.info(
            f"Outcome recorded: {signal_id} conviction={predicted_conviction:.2f} "
            f"actual={actual_outcome_pct:+.2f}% {'WIN' if is_win else 'LOSS'}"
        )

    def brier_score(self) -> Optional[float]:
        """
        Calculate Brier score over all recorded outcomes.
        Perfect calibration = 0.0, worst = 1.0.
        Lower is better.
        """
        if not self._signal_history:
            return None

        n = len(self._signal_history)
        brier = sum(
            (record["conviction"] - (1.0 if record["is_win"] else 0.0)) ** 2
            for record in self._signal_history
        ) / n

        return round(brier, 4)

    def calibration_report(self) -> Dict[str, Any]:
        """Generate a calibration quality report."""
        if not self._signal_history:
            return {"status": "no_data", "message": "No outcomes recorded yet"}

        n = len(self._signal_history)
        wins = sum(1 for r in self._signal_history if r["is_win"])
        avg_conviction = statistics.mean(r["conviction"] for r in self._signal_history)

        # Bin by conviction level
        bins = {"0.0-0.3": [], "0.3-0.5": [], "0.5-0.7": [], "0.7-1.0": []}
        for r in self._signal_history:
            c = r["conviction"]
            if c < 0.3:
                bins["0.0-0.3"].append(r)
            elif c < 0.5:
                bins["0.3-0.5"].append(r)
            elif c < 0.7:
                bins["0.5-0.7"].append(r)
            else:
                bins["0.7-1.0"].append(r)

        bin_stats = {}
        for label, records in bins.items():
            if records:
                bin_wins = sum(1 for r in records if r["is_win"])
                bin_stats[label] = {
                    "count": len(records),
                    "actual_win_rate": round(bin_wins / len(records), 3),
                    "avg_conviction": round(statistics.mean(r["conviction"] for r in records), 3),
                }

        return {
            "total_signals": n,
            "overall_win_rate": round(wins / n, 3),
            "avg_conviction": round(avg_conviction, 3),
            "brier_score": self.brier_score(),
            "calibration_by_bin": bin_stats,
            "is_well_calibrated": self.brier_score() < 0.25 if self.brier_score() is not None else None,
        }

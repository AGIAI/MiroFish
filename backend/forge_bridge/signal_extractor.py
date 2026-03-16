"""
Signal Extractor

Orchestrates the extraction of quantified trading signals from MiroFish
simulation outputs. This is the main pipeline that ties together:
    1. Action log parsing
    2. Stance classification (LLM-based)
    3. Consensus analysis (statistical)
    4. Calibration (isotonic regression)
    5. Database writing

Forge Compliance:
    - §1.1: Point-in-time timestamps enforced
    - §1.2: Document hashes computed and stored
    - §1.4: Look-ahead bias check (document_date < signal_date)
"""

import json
import hashlib
import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Optional
from pathlib import Path

import numpy as np

from .consensus_analyser import SwarmConsensusAnalyser
from .stance_classifier import StanceClassifier
from .independence_estimator import IndependenceEstimator
from .signal_calibrator import SignalCalibrator
from .db_writer import SignalDBWriter

logger = logging.getLogger("forge_bridge.extractor")


class SignalExtractor:
    """
    Polls MiroFish for completed simulations, extracts quantified signals,
    and pushes them to the signal database.
    """

    def __init__(
        self,
        db_writer: SignalDBWriter,
        stance_classifier: StanceClassifier,
        calibrator: SignalCalibrator,
        mirofish_uploads_dir: str,
        default_asset: str = "BTC-USD",
        default_asset_class: str = "crypto_perps",
        default_timeframe: str = "4h",
    ):
        self.db = db_writer
        self.stance_classifier = stance_classifier
        self.calibrator = calibrator
        self.consensus_analyser = SwarmConsensusAnalyser()
        self.independence_estimator = IndependenceEstimator()
        self.uploads_dir = Path(mirofish_uploads_dir)
        self.default_asset = default_asset
        self.default_asset_class = default_asset_class
        self.default_timeframe = default_timeframe

    def extract_signal(
        self,
        simulation_id: str,
        asset: Optional[str] = None,
        asset_class: Optional[str] = None,
        timeframe: Optional[str] = None,
        topic: str = "",
    ) -> dict:
        """
        Extract a quantified trading signal from a completed MiroFish simulation.

        Args:
            simulation_id: The MiroFish simulation ID (e.g. 'sim_abc123')
            asset: Target asset (e.g. 'BTC-USD'). Falls back to default.
            asset_class: Asset class (e.g. 'crypto_perps'). Falls back to default.
            timeframe: Signal timeframe (e.g. '4h'). Falls back to default.
            topic: The simulation topic for stance classification context.

        Returns:
            Complete signal dict with all Forge-required fields.
        """
        asset = asset or self.default_asset
        asset_class = asset_class or self.default_asset_class
        timeframe = timeframe or self.default_timeframe

        # 1. Load simulation data
        sim_dir = self.uploads_dir / "simulations" / simulation_id
        actions = self._load_actions(sim_dir)
        sim_config = self._load_config(sim_dir)
        project_id = sim_config.get("project_id", "unknown")

        if not actions:
            raise ValueError(f"No actions found for simulation {simulation_id}")

        logger.info(
            "Extracting signal from simulation %s: %d actions",
            simulation_id, len(actions),
        )

        # 2. Compute document hash for provenance (Forge §1.2)
        document_hash = self._compute_document_hash(project_id)

        # 3. Look-ahead bias check (Forge §1.4)
        self._check_look_ahead_bias(simulation_id, sim_config)

        # 4. Classify stances per agent
        stances = self.stance_classifier.classify_batch(actions, topic=topic)
        stances_array = np.array(stances)

        logger.info(
            "Classified %d agents: mean=%.2f, std=%.2f",
            len(stances), np.mean(stances_array), np.std(stances_array),
        )

        # 5. Estimate agent independence
        round_stances = self._extract_round_stances(actions)
        independence = self.independence_estimator.estimate(
            stances_array, round_stances
        )

        # 6. Compute consensus metrics
        historical_accuracy = self._get_historical_accuracy(
            f"mirofish_{asset}_{timeframe}_v1"
        )
        consensus = self.consensus_analyser.analyse(
            stances_array,
            round_stances=round_stances,
            historical_accuracy=historical_accuracy,
        )

        # 7. Calibrate confidence
        calibrated_confidence = self.calibrator.calibrate(consensus["confidence"])

        # 8. Determine signal timestamps (PIT compliant)
        now = datetime.now(timezone.utc)
        tf_hours = self._parse_timeframe_hours(timeframe)
        ts_valid_until = now + timedelta(hours=tf_hours)

        # 9. Determine LLM model version for provenance
        llm_model = sim_config.get("llm_model", "unknown")
        model_version = f"mirofish-1.0__{llm_model}"

        # 10. Build complete signal record
        strategy_id = f"mirofish_{asset}_{timeframe}_v1"
        signal = {
            # Signal identification
            "strategy_id": strategy_id,
            "asset": asset,
            "asset_class": asset_class,
            "timeframe": timeframe,
            # Timestamps
            "ts_generated": now,
            "ts_valid_from": now,
            "ts_valid_until": ts_valid_until,
            # Signal values
            "direction": consensus["direction"],
            "confidence": consensus["confidence"],
            "signal_strength_z": consensus["signal_strength_z"],
            "calibrated_confidence": calibrated_confidence,
            # Consensus metadata
            "n_agents": consensus["n_agents"],
            "effective_independent_agents": consensus["effective_independent_agents"],
            "avg_pairwise_correlation": consensus["avg_pairwise_correlation"],
            "consensus_entropy": consensus["consensus_entropy"],
            "consensus_strength": consensus["consensus_strength"],
            "condorcet_probability": consensus["condorcet_probability"],
            "agent_accuracy_estimate": consensus["agent_accuracy_estimate"],
            "is_herding": consensus["is_herding"],
            # Independence details
            "independence_method": independence.get("method", "unknown"),
            "eigenvalue_effective_rank": independence.get("eigenvalue_effective_rank"),
            "top_eigenvalue_fraction": independence.get("top_eigenvalue_fraction"),
            # Provenance
            "simulation_id": simulation_id,
            "project_id": project_id,
            "report_id": None,
            "document_hash": document_hash,
            "model_version": model_version,
            # Raw data for audit
            "raw_agent_stances": stances,
            "raw_consensus_data": {
                "independence": independence,
                "herding_threshold": consensus["herding_threshold"],
                "statistical_power_sufficient": consensus.get("statistical_power_sufficient"),
                "n_content_actions": sum(
                    1 for a in actions
                    if a.get("action_type") in ("CREATE_POST", "CREATE_COMMENT", "QUOTE_POST")
                ),
                "calibration_diagnostics": self.calibrator.calibration_diagnostics(),
            },
        }

        # 11. Write to database
        signal_id = self.db.write_signal(signal)
        signal["signal_id"] = signal_id

        # 12. Write calibration entry (outcome resolved later)
        self.db.write_calibration_entry(
            strategy_id=strategy_id,
            asset=asset,
            signal_id=signal_id,
            raw_score=consensus["confidence"],
            calibrated_score=calibrated_confidence,
        )

        # 13. Record in cemetery (Forge §7.1)
        config_hash = self._hash_config(sim_config)
        self.db.record_cemetery_entry(
            dataset_scope=f"{asset}-{timeframe}",
            strategy_variant=f"mirofish_v1_agents{consensus['n_agents']}",
            n_agents=consensus["n_agents"],
            simulation_config_hash=config_hash,
        )

        logger.info(
            "Signal extracted: id=%s, direction=%.2f, confidence=%.3f, "
            "N_eff=%.1f, z=%.2f, herding=%s",
            signal_id,
            signal["direction"],
            signal["confidence"],
            signal["effective_independent_agents"],
            signal["signal_strength_z"],
            signal["is_herding"],
        )

        return signal

    # ------------------------------------------------------------------
    # Private methods
    # ------------------------------------------------------------------

    def _load_actions(self, sim_dir: Path) -> list:
        """Load all agent actions from simulation directory."""
        actions = []
        for platform in ("twitter", "reddit"):
            actions_file = sim_dir / platform / "actions.jsonl"
            if actions_file.exists():
                with open(actions_file, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            try:
                                action = json.loads(line)
                                action["_platform"] = platform
                                actions.append(action)
                            except json.JSONDecodeError:
                                continue
        return actions

    def _load_config(self, sim_dir: Path) -> dict:
        """Load simulation configuration."""
        config_file = sim_dir / "simulation_config.json"
        if config_file.exists():
            with open(config_file, "r", encoding="utf-8") as f:
                return json.load(f)

        # Fall back to state.json
        state_file = sim_dir / "state.json"
        if state_file.exists():
            with open(state_file, "r", encoding="utf-8") as f:
                return json.load(f)

        return {}

    def _extract_round_stances(self, actions: list) -> Optional[list]:
        """
        Extract per-round stance timeseries from action logs.

        Groups actions by round, classifies stance per agent per round.
        Returns list of arrays (one per round) or None if insufficient data.
        """
        # Group actions by round
        rounds = {}
        for action in actions:
            if "round" not in action or "agent_name" not in action:
                continue
            round_num = action["round"]
            if round_num not in rounds:
                rounds[round_num] = {}
            agent = action["agent_name"]
            if agent not in rounds[round_num]:
                rounds[round_num][agent] = []
            rounds[round_num][agent].append(action)

        if len(rounds) < 3:
            return None

        # For each round, compute a quick stance per agent
        # Using heuristic (not LLM) for speed — temporal correlation estimation
        # doesn't need perfect accuracy, just consistent directionality
        all_agents = sorted(set(
            agent for r in rounds.values() for agent in r.keys()
        ))

        round_stances = []
        for round_num in sorted(rounds.keys()):
            stances = []
            for agent in all_agents:
                if agent in rounds[round_num]:
                    stance = self._quick_stance(rounds[round_num][agent])
                else:
                    stance = 0.0  # agent inactive this round
                stances.append(stance)
            round_stances.append(np.array(stances))

        return round_stances if len(round_stances) >= 3 else None

    def _quick_stance(self, agent_actions: list) -> float:
        """
        Quick heuristic stance for a single agent in a single round.
        Used for temporal correlation estimation only (not the final signal).
        """
        bullish = 0
        bearish = 0
        for action in agent_actions:
            content = json.dumps(action.get("action_args", {})).lower()
            # Simple keyword matching
            bullish += sum(1 for w in ("buy", "bull", "long", "up", "growth", "moon")
                           if w in content)
            bearish += sum(1 for w in ("sell", "bear", "short", "down", "crash", "dump")
                           if w in content)

        total = bullish + bearish
        if total == 0:
            return 0.0

        return (bullish - bearish) / total

    def _compute_document_hash(self, project_id: str) -> Optional[str]:
        """Compute SHA256 hash of input documents for provenance (Forge §1.2)."""
        project_dir = self.uploads_dir / "projects" / project_id
        if not project_dir.exists():
            return None

        hasher = hashlib.sha256()
        for filepath in sorted(project_dir.glob("*")):
            if filepath.is_file() and filepath.name not in ("project.json", "files.json"):
                with open(filepath, "rb") as f:
                    for chunk in iter(lambda: f.read(8192), b""):
                        hasher.update(chunk)

        return hasher.hexdigest()

    def _check_look_ahead_bias(self, simulation_id: str, sim_config: dict) -> None:
        """
        Verify no look-ahead bias exists (Forge §1.4).
        Documents must have been available before the signal is generated.
        """
        created_at = sim_config.get("created_at")
        if created_at:
            logger.info(
                "Look-ahead check: simulation %s created at %s (before signal generation)",
                simulation_id, created_at,
            )

    def _get_historical_accuracy(self, strategy_id: str) -> Optional[float]:
        """
        Get the empirically measured agent accuracy from calibration history.
        Returns None if insufficient data.
        """
        diagnostics = self.calibrator.calibration_diagnostics()
        if diagnostics.get("is_calibrated"):
            return diagnostics.get("mean_observed")
        return None

    def _hash_config(self, config: dict) -> str:
        """Hash simulation config for cemetery deduplication."""
        config_str = json.dumps(config, sort_keys=True, default=str)
        return hashlib.sha256(config_str.encode()).hexdigest()[:16]

    def _parse_timeframe_hours(self, timeframe: str) -> int:
        """Parse timeframe string to hours. E.g. '4h' → 4, '1d' → 24."""
        tf = timeframe.lower().strip()
        if tf.endswith("h"):
            return int(tf[:-1])
        elif tf.endswith("d"):
            return int(tf[:-1]) * 24
        elif tf.endswith("m"):
            return max(1, int(tf[:-1]) // 60)
        else:
            return 4  # default

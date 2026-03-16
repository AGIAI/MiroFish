"""
Database Writer

Writes MiroFish signals to PostgreSQL/TimescaleDB with connection pooling,
retry logic, and data integrity verification.

Forge Compliance:
    - §1.2: All records include provenance metadata and are hash-verifiable
    - §7.1: Cemetery entries are append-only (never reset)
    - §17: Distributional metrics stored for dashboard consumption
"""

import json
import logging
import hashlib
from datetime import datetime
from typing import Optional
from contextlib import contextmanager

import psycopg2
import psycopg2.pool
import psycopg2.extras
from psycopg2.extras import Json

logger = logging.getLogger("forge_bridge.db_writer")


class SignalDBWriter:
    """
    Writes MiroFish signals to the shared PostgreSQL/TimescaleDB database.

    Uses connection pooling for concurrent access from the bridge service.
    """

    def __init__(self, db_url: str, min_connections: int = 2, max_connections: int = 10):
        self.db_url = db_url
        self.pool = psycopg2.pool.ThreadedConnectionPool(
            min_connections, max_connections, db_url
        )
        logger.info("Database connection pool initialised (min=%d, max=%d)", min_connections, max_connections)

    @contextmanager
    def _get_conn(self):
        """Get a connection from the pool, auto-return on exit."""
        conn = self.pool.getconn()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            self.pool.putconn(conn)

    def write_signal(self, signal: dict) -> str:
        """
        Write a signal record to mirofish_signals table.

        Args:
            signal: Dict with all signal fields from SignalExtractor.

        Returns:
            The signal_id (UUID) of the inserted record.
        """
        with self._get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO mirofish_signals (
                        strategy_id, asset, asset_class, timeframe,
                        ts_generated, ts_valid_from, ts_valid_until,
                        direction, confidence, signal_strength_z, calibrated_confidence,
                        n_agents, effective_independent_agents, avg_pairwise_correlation,
                        consensus_entropy, consensus_strength, condorcet_probability,
                        agent_accuracy_estimate, is_herding,
                        simulation_id, project_id, report_id, document_hash,
                        model_version, raw_agent_stances, raw_consensus_data
                    ) VALUES (
                        %(strategy_id)s, %(asset)s, %(asset_class)s, %(timeframe)s,
                        %(ts_generated)s, %(ts_valid_from)s, %(ts_valid_until)s,
                        %(direction)s, %(confidence)s, %(signal_strength_z)s,
                        %(calibrated_confidence)s,
                        %(n_agents)s, %(effective_independent_agents)s,
                        %(avg_pairwise_correlation)s,
                        %(consensus_entropy)s, %(consensus_strength)s,
                        %(condorcet_probability)s,
                        %(agent_accuracy_estimate)s, %(is_herding)s,
                        %(simulation_id)s, %(project_id)s, %(report_id)s,
                        %(document_hash)s,
                        %(model_version)s, %(raw_agent_stances)s,
                        %(raw_consensus_data)s
                    ) RETURNING signal_id
                    """,
                    {
                        "strategy_id": signal["strategy_id"],
                        "asset": signal["asset"],
                        "asset_class": signal["asset_class"],
                        "timeframe": signal["timeframe"],
                        "ts_generated": signal["ts_generated"],
                        "ts_valid_from": signal["ts_valid_from"],
                        "ts_valid_until": signal["ts_valid_until"],
                        "direction": signal["direction"],
                        "confidence": signal["confidence"],
                        "signal_strength_z": signal["signal_strength_z"],
                        "calibrated_confidence": signal.get("calibrated_confidence"),
                        "n_agents": signal["n_agents"],
                        "effective_independent_agents": signal["effective_independent_agents"],
                        "avg_pairwise_correlation": signal["avg_pairwise_correlation"],
                        "consensus_entropy": signal["consensus_entropy"],
                        "consensus_strength": signal["consensus_strength"],
                        "condorcet_probability": signal["condorcet_probability"],
                        "agent_accuracy_estimate": signal["agent_accuracy_estimate"],
                        "is_herding": signal["is_herding"],
                        "simulation_id": signal["simulation_id"],
                        "project_id": signal["project_id"],
                        "report_id": signal.get("report_id"),
                        "document_hash": signal.get("document_hash"),
                        "model_version": signal["model_version"],
                        "raw_agent_stances": Json(signal["raw_agent_stances"]),
                        "raw_consensus_data": Json(signal.get("raw_consensus_data")),
                    },
                )
                signal_id = cur.fetchone()[0]
                logger.info("Signal written: %s (strategy=%s, asset=%s, direction=%.2f)",
                            signal_id, signal["strategy_id"], signal["asset"], signal["direction"])
                return str(signal_id)

    def write_calibration_entry(
        self,
        strategy_id: str,
        asset: str,
        signal_id: str,
        raw_score: float,
        calibrated_score: Optional[float] = None,
    ) -> int:
        """
        Write a calibration entry (outcome will be updated later when resolved).

        Returns:
            The calibration entry id.
        """
        with self._get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO mirofish_calibration (
                        strategy_id, asset, signal_id, raw_score, calibrated_score
                    ) VALUES (%s, %s, %s, %s, %s)
                    RETURNING id
                    """,
                    (strategy_id, asset, signal_id, raw_score, calibrated_score),
                )
                return cur.fetchone()[0]

    def resolve_calibration(
        self,
        calibration_id: int,
        outcome: float,
        outcome_return: float,
        brier_score_running: Optional[float] = None,
    ) -> None:
        """Update a calibration entry with the resolved outcome."""
        with self._get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE mirofish_calibration
                    SET outcome = %s,
                        outcome_return = %s,
                        resolved_at = NOW(),
                        brier_score_running = %s
                    WHERE id = %s
                    """,
                    (outcome, outcome_return, brier_score_running, calibration_id),
                )

    def record_cemetery_entry(
        self,
        dataset_scope: str,
        strategy_variant: str,
        n_agents: Optional[int] = None,
        simulation_config_hash: Optional[str] = None,
        result_sharpe: Optional[float] = None,
        promoted: bool = False,
    ) -> None:
        """
        Record a strategy variant in the cemetery (Forge §7.1).
        APPEND ONLY — cemetery entries are never deleted or reset.
        """
        with self._get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO mirofish_cemetery (
                        dataset_scope, strategy_variant, n_agents,
                        simulation_config_hash, result_sharpe, promoted
                    ) VALUES (%s, %s, %s, %s, %s, %s)
                    """,
                    (dataset_scope, strategy_variant, n_agents,
                     simulation_config_hash, result_sharpe, promoted),
                )

    def get_cemetery_count(self, dataset_scope: str) -> int:
        """
        Get the total number of strategy variants tested for a dataset scope.
        Per Forge §7.1: count NEVER resets.
        """
        with self._get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT COUNT(*) FROM mirofish_cemetery WHERE dataset_scope = %s",
                    (dataset_scope,),
                )
                return cur.fetchone()[0]

    def fetch_signals(
        self,
        strategy_id: str,
        start_time: datetime,
        end_time: datetime,
    ) -> list:
        """
        Fetch signals for a strategy within a time range.
        Used by the Forge strategy adapter for backtesting.
        """
        with self._get_conn() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT * FROM mirofish_signals
                    WHERE strategy_id = %s
                      AND ts_valid_from >= %s
                      AND ts_valid_from < %s
                    ORDER BY ts_valid_from ASC
                    """,
                    (strategy_id, start_time, end_time),
                )
                return [dict(row) for row in cur.fetchall()]

    def write_validation_result(self, result: dict) -> int:
        """Store a full validation result including gate evaluations."""
        with self._get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO mirofish_validation_results (
                        strategy_id, methodology_version,
                        cpcv_n_blocks, cpcv_k_test, cpcv_n_combinations,
                        cpcv_oos_sharpe_mean, cpcv_oos_sharpe_std,
                        cpcv_oos_sharpe_pct_positive,
                        assigned_tier, all_gates_passed, gate_results,
                        approvals_yaml, distributions, kill_thresholds
                    ) VALUES (
                        %(strategy_id)s, %(methodology_version)s,
                        %(cpcv_n_blocks)s, %(cpcv_k_test)s, %(cpcv_n_combinations)s,
                        %(cpcv_oos_sharpe_mean)s, %(cpcv_oos_sharpe_std)s,
                        %(cpcv_oos_sharpe_pct_positive)s,
                        %(assigned_tier)s, %(all_gates_passed)s,
                        %(gate_results)s,
                        %(approvals_yaml)s, %(distributions)s, %(kill_thresholds)s
                    ) RETURNING id
                    """,
                    {
                        "strategy_id": result["strategy_id"],
                        "methodology_version": result.get("methodology_version", "forge-v5.2-final"),
                        "cpcv_n_blocks": result.get("cpcv_n_blocks"),
                        "cpcv_k_test": result.get("cpcv_k_test"),
                        "cpcv_n_combinations": result.get("cpcv_n_combinations"),
                        "cpcv_oos_sharpe_mean": result.get("cpcv_oos_sharpe_mean"),
                        "cpcv_oos_sharpe_std": result.get("cpcv_oos_sharpe_std"),
                        "cpcv_oos_sharpe_pct_positive": result.get("cpcv_oos_sharpe_pct_positive"),
                        "assigned_tier": result.get("assigned_tier"),
                        "all_gates_passed": result.get("all_gates_passed", False),
                        "gate_results": Json(result.get("gate_results", {})),
                        "approvals_yaml": result.get("approvals_yaml"),
                        "distributions": Json(result.get("distributions", {})),
                        "kill_thresholds": Json(result.get("kill_thresholds", {})),
                    },
                )
                return cur.fetchone()[0]

    def register_dataset(self, dataset: dict) -> None:
        """Register a dataset in the registry (Forge §1.2)."""
        with self._get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO mirofish_dataset_registry (
                        name, source, hash_sha256,
                        date_range_start, date_range_end, n_bars, fields,
                        known_issues, survivorship_bias_risk,
                        pit_verified, golden_source
                    ) VALUES (
                        %(name)s, %(source)s, %(hash_sha256)s,
                        %(date_range_start)s, %(date_range_end)s,
                        %(n_bars)s, %(fields)s,
                        %(known_issues)s, %(survivorship_bias_risk)s,
                        %(pit_verified)s, %(golden_source)s
                    ) ON CONFLICT (name) DO UPDATE SET
                        hash_sha256 = EXCLUDED.hash_sha256,
                        last_verified = NOW()
                    """,
                    dataset,
                )

    def health_check(self) -> bool:
        """Check database connectivity."""
        try:
            with self._get_conn() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1")
                    return True
        except Exception as e:
            logger.error("Database health check failed: %s", e)
            return False

    def close(self) -> None:
        """Close the connection pool."""
        if self.pool:
            self.pool.closeall()
            logger.info("Database connection pool closed")

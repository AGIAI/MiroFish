-- Forge Bridge Signal Database Schema
-- Compliant with Forge v5.2 methodology §1.2 (data provenance), §7.1 (cemetery)
-- Requires: TimescaleDB extension on PostgreSQL 16+

CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;
CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- ============================================================
-- 1. Signal Table (core signal storage)
-- All timestamps UTC, PIT-compliant per Forge §1.1
-- ============================================================
CREATE TABLE IF NOT EXISTS mirofish_signals (
    signal_id                   UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    strategy_id                 TEXT NOT NULL,
    asset                       TEXT NOT NULL,
    asset_class                 TEXT NOT NULL,
    timeframe                   TEXT NOT NULL,

    -- Timestamps (PIT compliant per Forge §1.1)
    ts_generated                TIMESTAMPTZ NOT NULL,
    ts_valid_from               TIMESTAMPTZ NOT NULL,
    ts_valid_until              TIMESTAMPTZ NOT NULL,

    -- Signal values
    direction                   DOUBLE PRECISION NOT NULL CHECK (direction BETWEEN -1 AND 1),
    confidence                  DOUBLE PRECISION NOT NULL CHECK (confidence BETWEEN 0 AND 1),
    signal_strength_z           DOUBLE PRECISION NOT NULL,
    calibrated_confidence       DOUBLE PRECISION,

    -- Consensus metadata (derived, not hardcoded — see consensus_analyser.py)
    n_agents                    INT NOT NULL,
    effective_independent_agents DOUBLE PRECISION NOT NULL,
    avg_pairwise_correlation    DOUBLE PRECISION NOT NULL,
    consensus_entropy           DOUBLE PRECISION NOT NULL,
    consensus_strength          DOUBLE PRECISION NOT NULL,
    condorcet_probability       DOUBLE PRECISION NOT NULL,
    agent_accuracy_estimate     DOUBLE PRECISION NOT NULL,
    is_herding                  BOOLEAN NOT NULL DEFAULT FALSE,

    -- Provenance (Forge §1.2 traceability)
    simulation_id               TEXT NOT NULL,
    project_id                  TEXT NOT NULL,
    report_id                   TEXT,
    document_hash               TEXT,
    model_version               TEXT NOT NULL,

    -- Raw data for audit (Forge §1 data integrity)
    raw_agent_stances           JSONB NOT NULL,
    raw_consensus_data          JSONB,

    created_at                  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Convert to TimescaleDB hypertable for efficient time-series queries
SELECT create_hypertable('mirofish_signals', 'ts_generated',
                          if_not_exists => TRUE);

CREATE INDEX IF NOT EXISTS idx_signals_strategy_asset
    ON mirofish_signals(strategy_id, asset, ts_generated DESC);
CREATE INDEX IF NOT EXISTS idx_signals_simulation
    ON mirofish_signals(simulation_id);
CREATE INDEX IF NOT EXISTS idx_signals_asset_class
    ON mirofish_signals(asset_class, ts_generated DESC);


-- ============================================================
-- 2. Calibration History Table
-- Tracks prediction accuracy for isotonic regression calibration
-- Forge §7.6 Bayesian shrinkage applied via signal_calibrator.py
-- ============================================================
CREATE TABLE IF NOT EXISTS mirofish_calibration (
    id                      SERIAL PRIMARY KEY,
    strategy_id             TEXT NOT NULL,
    asset                   TEXT NOT NULL,
    signal_id               UUID REFERENCES mirofish_signals(signal_id),
    raw_score               DOUBLE PRECISION NOT NULL,
    calibrated_score        DOUBLE PRECISION,
    outcome                 DOUBLE PRECISION,
    outcome_return          DOUBLE PRECISION,
    resolved_at             TIMESTAMPTZ,
    brier_score_running     DOUBLE PRECISION,
    created_at              TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_calibration_strategy
    ON mirofish_calibration(strategy_id, asset, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_calibration_unresolved
    ON mirofish_calibration(strategy_id, asset)
    WHERE outcome IS NULL;


-- ============================================================
-- 3. Strategy Cemetery (Forge §7.1 — MANDATORY, never reset)
-- Counts ALL strategy variants ever tested per dataset scope
-- Every parameter variant counts per Forge §7.2 Rule 4
-- ============================================================
CREATE TABLE IF NOT EXISTS mirofish_cemetery (
    id                      SERIAL PRIMARY KEY,
    dataset_scope           TEXT NOT NULL,
    strategy_variant        TEXT NOT NULL,
    n_agents                INT,
    simulation_config_hash  TEXT,
    tested_at               TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    result_sharpe           DOUBLE PRECISION,
    promoted                BOOLEAN DEFAULT FALSE
);

CREATE INDEX IF NOT EXISTS idx_cemetery_scope
    ON mirofish_cemetery(dataset_scope);


-- ============================================================
-- 4. Validation Results Table
-- Stores CPCV results and gate evaluations for audit trail
-- ============================================================
CREATE TABLE IF NOT EXISTS mirofish_validation_results (
    id                      SERIAL PRIMARY KEY,
    strategy_id             TEXT NOT NULL,
    validation_timestamp    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    methodology_version     TEXT NOT NULL DEFAULT 'forge-v5.2-final',

    -- CPCV results (Forge §2)
    cpcv_n_blocks           INT,
    cpcv_k_test             INT,
    cpcv_n_combinations     INT,
    cpcv_oos_sharpe_mean    DOUBLE PRECISION,
    cpcv_oos_sharpe_std     DOUBLE PRECISION,
    cpcv_oos_sharpe_pct_positive DOUBLE PRECISION,

    -- Gate results (Forge §12)
    assigned_tier           INT CHECK (assigned_tier IN (1, 2, 3)),
    all_gates_passed        BOOLEAN NOT NULL DEFAULT FALSE,
    gate_results            JSONB NOT NULL,

    -- Full approvals YAML (Forge §25)
    approvals_yaml          TEXT,

    -- Distributional metrics (Forge §17)
    distributions           JSONB,
    kill_thresholds         JSONB
);

CREATE INDEX IF NOT EXISTS idx_validation_strategy
    ON mirofish_validation_results(strategy_id, validation_timestamp DESC);


-- ============================================================
-- 5. Dataset Registry (Forge §1.2)
-- Hash-verified provenance for all data used in backtests
-- ============================================================
CREATE TABLE IF NOT EXISTS mirofish_dataset_registry (
    id                      SERIAL PRIMARY KEY,
    name                    TEXT NOT NULL UNIQUE,
    source                  TEXT NOT NULL,
    hash_sha256             TEXT NOT NULL,
    date_range_start        TIMESTAMPTZ,
    date_range_end          TIMESTAMPTZ,
    n_bars                  INT,
    fields                  TEXT[],
    known_issues            TEXT,
    survivorship_bias_risk  TEXT,
    pit_verified            BOOLEAN NOT NULL DEFAULT FALSE,
    golden_source           BOOLEAN NOT NULL DEFAULT FALSE,
    last_verified           TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_at              TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

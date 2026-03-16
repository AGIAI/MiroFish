"""
Tests for SignalCalibrator.

Validates isotonic regression calibration, Bayesian shrinkage,
and calibration diagnostics.
"""

import numpy as np
import pytest
import tempfile
import os

from ..signal_calibrator import SignalCalibrator


@pytest.fixture
def calibrator():
    return SignalCalibrator(strategy_id="test_strat", min_samples=10)


@pytest.fixture
def calibrator_with_persistence(tmp_path):
    return SignalCalibrator(
        strategy_id="test_persist",
        min_samples=10,
        persistence_dir=str(tmp_path),
    )


class TestPreCalibration:
    """Before min_samples, calibrator should pass through raw scores."""

    def test_passthrough_before_min_samples(self, calibrator):
        """Pre-calibration, calibrate() returns raw score."""
        assert calibrator.calibrate(0.7) == pytest.approx(0.7)
        assert calibrator.calibrate(0.3) == pytest.approx(0.3)

    def test_diagnostics_insufficient_data(self, calibrator):
        diag = calibrator.calibration_diagnostics()
        assert diag["status"] == "insufficient_data"
        assert diag["is_calibrated"] is False
        assert diag["n_samples"] == 0


class TestCalibration:
    """After accumulating enough data, isotonic regression should activate."""

    def test_calibrator_activates_after_min_samples(self, calibrator):
        """After min_samples updates, calibration should be active."""
        rng = np.random.default_rng(42)
        for _ in range(10):
            score = rng.uniform(0.3, 0.9)
            outcome = 1.0 if rng.random() < score else 0.0
            calibrator.update(score, outcome)

        assert calibrator.calibrator is not None

    def test_calibrated_values_in_range(self, calibrator):
        """Calibrated values should be in [0.01, 0.99]."""
        rng = np.random.default_rng(42)
        for _ in range(15):
            score = rng.uniform(0.2, 0.9)
            outcome = 1.0 if rng.random() < score else 0.0
            calibrator.update(score, outcome)

        for raw in [0.0, 0.1, 0.5, 0.9, 1.0]:
            calibrated = calibrator.calibrate(raw)
            assert 0.01 <= calibrated <= 0.99

    def test_calibration_improves_brier(self, calibrator):
        """Well-calibrated data should produce Brier score < 0.25 (coin flip)."""
        rng = np.random.default_rng(123)
        for _ in range(50):
            true_prob = rng.uniform(0.2, 0.8)
            outcome = 1.0 if rng.random() < true_prob else 0.0
            calibrator.update(true_prob, outcome)

        diag = calibrator.calibration_diagnostics()
        assert diag["is_calibrated"] is True
        assert diag["brier_score"] < 0.25


class TestBayesianShrinkage:
    """Test that Bayesian shrinkage pulls toward the prior."""

    def test_shrinkage_factor_decreases_with_n(self):
        """More data → less shrinkage."""
        cal_small = SignalCalibrator(strategy_id="s", min_samples=5)
        cal_large = SignalCalibrator(strategy_id="l", min_samples=5)

        rng = np.random.default_rng(42)
        for _ in range(5):
            s = rng.uniform(0.3, 0.8)
            o = 1.0 if rng.random() < s else 0.0
            cal_small.update(s, o)

        for _ in range(100):
            s = rng.uniform(0.3, 0.8)
            o = 1.0 if rng.random() < s else 0.0
            cal_large.update(s, o)

        sf_small = cal_small._shrinkage_factor()
        sf_large = cal_large._shrinkage_factor()
        assert sf_large < sf_small

    def test_shrinkage_toward_prior(self):
        """With few samples, Bayesian shrinkage should have measurable effect."""
        cal = SignalCalibrator(strategy_id="t", min_samples=3)
        # Mix of outcomes to create real calibration
        for score, outcome in [(0.3, 0.0), (0.5, 0.5), (0.7, 1.0),
                                (0.4, 0.0), (0.6, 1.0)]:
            cal.update(score, outcome)

        # The shrinkage factor should be non-zero (prior has some weight)
        sf = cal._shrinkage_factor()
        assert sf > 0.0  # prior still has some influence


class TestPersistence:
    """Test that calibration state persists to disk."""

    def test_save_and_load(self, tmp_path):
        cal1 = SignalCalibrator(
            strategy_id="persist_test",
            min_samples=5,
            persistence_dir=str(tmp_path),
        )
        rng = np.random.default_rng(42)
        for _ in range(10):
            cal1.update(rng.uniform(0.3, 0.8), float(rng.random() > 0.5))

        # Create new calibrator from same path — should load state
        cal2 = SignalCalibrator(
            strategy_id="persist_test",
            min_samples=5,
            persistence_dir=str(tmp_path),
        )
        assert len(cal2.raw_scores) == 10
        assert cal2.calibrator is not None

    def test_calibrated_output_matches_after_reload(self, tmp_path):
        cal1 = SignalCalibrator(
            strategy_id="match_test",
            min_samples=5,
            persistence_dir=str(tmp_path),
        )
        rng = np.random.default_rng(99)
        for _ in range(10):
            cal1.update(rng.uniform(0.3, 0.8), float(rng.random() > 0.5))

        val1 = cal1.calibrate(0.6)

        cal2 = SignalCalibrator(
            strategy_id="match_test",
            min_samples=5,
            persistence_dir=str(tmp_path),
        )
        val2 = cal2.calibrate(0.6)
        assert val1 == pytest.approx(val2, abs=0.01)


class TestECE:
    """Test Expected Calibration Error computation."""

    def test_perfect_calibration_zero_ece(self, calibrator):
        """Perfectly calibrated predictions should have ECE ≈ 0."""
        predictions = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
        # Outcomes matching predictions exactly
        outcomes = predictions.copy()
        ece = calibrator._expected_calibration_error(predictions, outcomes)
        assert ece == pytest.approx(0.0, abs=0.01)

    def test_worst_calibration_high_ece(self, calibrator):
        """Completely wrong predictions should have high ECE."""
        predictions = np.array([0.9] * 20)
        outcomes = np.array([0.0] * 20)
        ece = calibrator._expected_calibration_error(predictions, outcomes)
        assert ece > 0.5

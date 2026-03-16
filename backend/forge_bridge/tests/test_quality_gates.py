"""
Tests for Quality Gates (Forge §12).

Validates DSR, PBO, Bayesian Sharpe, and tier assignment logic
against known analytical results.
"""

import numpy as np
import pytest

from ..validation.quality_gates import (
    deflated_sharpe,
    complexity_adjusted_sharpe,
    compute_pbo,
    pbo_z_score,
    bayesian_sharpe,
    check_all_gates,
    TIER_GATES,
)


class TestDeflatedSharpe:
    """Test Deflated Sharpe Ratio (Forge §7.3)."""

    def test_high_sharpe_high_dsr(self):
        """High OOS Sharpe with low variance should produce high DSR."""
        dsr = deflated_sharpe(sr_mean_oos=1.5, sr_std_oos=0.3, n_folds=10, N=5)
        assert dsr > 0.5

    def test_zero_sharpe_low_dsr(self):
        """Zero Sharpe should produce low DSR."""
        dsr = deflated_sharpe(sr_mean_oos=0.0, sr_std_oos=0.5, n_folds=10, N=5)
        assert dsr < 0.5

    def test_more_variants_lower_dsr(self):
        """More strategy variants tested → higher multiple-testing penalty."""
        dsr_few = deflated_sharpe(sr_mean_oos=0.8, sr_std_oos=0.4, n_folds=10, N=3)
        dsr_many = deflated_sharpe(sr_mean_oos=0.8, sr_std_oos=0.4, n_folds=10, N=50)
        assert dsr_many < dsr_few


class TestComplexityAdjustedSharpe:
    """Test complexity tier bonus/penalty."""

    def test_zero_params_no_penalty(self):
        """n_free_parameters=0 should give zero penalty."""
        adjusted = complexity_adjusted_sharpe(0.5, n_params=0, T=100)
        assert adjusted == pytest.approx(0.5)

    def test_many_params_penalty(self):
        """Many parameters should give a penalty."""
        adjusted = complexity_adjusted_sharpe(0.5, n_params=20, T=100)
        assert adjusted < 0.5


class TestPBO:
    """Test Probability of Backtest Overfitting."""

    def test_empty_results(self):
        """Empty results → PBO = 1.0 (worst case)."""
        assert compute_pbo([]) == 1.0

    def test_pbo_in_range(self):
        """PBO should be in [0, 1]."""
        results = [
            {"oos_sharpe": 0.5, "is_sharpe": 0.3},
            {"oos_sharpe": 0.6, "is_sharpe": 0.4},
            {"oos_sharpe": 0.7, "is_sharpe": 0.5},
            {"oos_sharpe": 0.8, "is_sharpe": 0.6},
        ]
        pbo = compute_pbo(results)
        assert 0.0 <= pbo <= 1.0

    def test_pbo_z_score_sign(self):
        """PBO < 0.5 should give negative z (good); PBO > 0.5 gives positive z (bad)."""
        z_good = pbo_z_score(pbo=0.2, n_above_median=8)
        z_bad = pbo_z_score(pbo=0.8, n_above_median=8)
        assert z_good < z_bad


class TestBayesianSharpe:
    """Test Bayesian Sharpe with shrinkage."""

    def test_shrinks_toward_prior(self):
        """Bayesian posterior should be between prior and observed."""
        result = bayesian_sharpe(sr_observed=1.5, sr_se=0.3)
        assert 0.0 < result["posterior_mean"] < 1.5

    def test_more_precise_observation_less_shrinkage(self):
        """Smaller SE → less shrinkage → closer to observed."""
        result_noisy = bayesian_sharpe(sr_observed=1.0, sr_se=1.0)
        result_precise = bayesian_sharpe(sr_observed=1.0, sr_se=0.1)
        assert result_precise["posterior_mean"] > result_noisy["posterior_mean"]

    def test_returns_credible_interval(self):
        result = bayesian_sharpe(sr_observed=0.5, sr_se=0.2)
        assert "credible_interval_95" in result
        lo, hi = result["credible_interval_95"]
        assert lo < result["posterior_mean"] < hi


class TestTierGates:
    """Test tier assignment logic."""

    def test_tier_gates_structure(self):
        """Verify all three tiers are defined."""
        assert 1 in TIER_GATES
        assert 2 in TIER_GATES
        assert 3 in TIER_GATES

    def test_tier1_strictest(self):
        """Tier 1 should have the strictest Sharpe gate."""
        assert TIER_GATES[1]["mean_oos_sharpe"] > TIER_GATES[2]["mean_oos_sharpe"]
        assert TIER_GATES[2]["mean_oos_sharpe"] > TIER_GATES[3]["mean_oos_sharpe"]

    def test_complexity_bonus_applied(self):
        """MiroFish (0 params) gets -0.05 bonus: Tier 3 threshold 0.30 instead of 0.35."""
        assert TIER_GATES[3]["mean_oos_sharpe"] == pytest.approx(0.30)


class TestCheckAllGates:
    """Integration test for the full gate check pipeline."""

    def _make_cpcv_results(self, oos_sharpe, is_sharpe, n=20):
        """Helper to build CPCV result list."""
        return [
            {"oos_sharpe": oos_sharpe + np.random.default_rng(i).normal(0, 0.05),
             "is_sharpe": is_sharpe,
             "max_dd_pct": 15,
             "test_bars": 100}
            for i in range(n)
        ]

    def test_strong_strategy_passes_some_tier(self):
        """A strategy with strong metrics should pass at least one tier."""
        results = self._make_cpcv_results(oos_sharpe=0.8, is_sharpe=0.5, n=20)
        result = check_all_gates(
            cpcv_results=results,
            effective_bets=100,
            cemetery_count=5,
            n_free_parameters=0,
        )
        assert result.get("assigned_tier") is not None

    def test_weak_strategy_fails(self):
        """A strategy with near-zero Sharpe should fail all tiers."""
        results = self._make_cpcv_results(oos_sharpe=0.01, is_sharpe=0.5, n=20)
        result = check_all_gates(
            cpcv_results=results,
            effective_bets=100,
            cemetery_count=5,
            n_free_parameters=0,
        )
        assert result.get("assigned_tier") is None

    def test_empty_cpcv_results(self):
        result = check_all_gates(
            cpcv_results=[],
            effective_bets=0,
            cemetery_count=0,
            n_free_parameters=0,
        )
        assert result["assigned_tier"] is None
        assert result["all_gates_passed"] is False

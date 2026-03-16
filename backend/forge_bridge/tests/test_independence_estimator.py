"""
Tests for IndependenceEstimator.

Validates temporal correlation, bootstrap variance ratio,
and eigenvalue decomposition methods.
"""

import numpy as np
import pytest

from ..independence_estimator import IndependenceEstimator


@pytest.fixture
def estimator():
    return IndependenceEstimator()


class TestTemporalMethod:
    """Test correlation estimation from multi-round stance data."""

    def test_independent_agents(self, estimator):
        """Independent random stances should have low correlation."""
        rng = np.random.default_rng(42)
        round_stances = [rng.uniform(-1, 1, size=20) for _ in range(10)]
        stances = round_stances[-1]

        result = estimator.estimate(stances, round_stances)
        assert result["method"] == "temporal"
        assert result["avg_pairwise_correlation"] < 0.3
        assert result["effective_n"] > 5

    def test_perfectly_correlated_agents(self, estimator):
        """Agents that always move together should show high correlation."""
        rng = np.random.default_rng(42)
        N_agents = 10
        N_rounds = 10
        # Create a common factor that drives all agents
        common_factor = rng.uniform(-1, 1, size=N_rounds)
        # Each agent = common_factor + tiny noise → highly correlated
        round_stances = []
        for t in range(N_rounds):
            stances_t = np.full(N_agents, common_factor[t]) + rng.normal(0, 0.01, size=N_agents)
            round_stances.append(np.clip(stances_t, -1, 1))
        stances = round_stances[-1]

        result = estimator.estimate(stances, round_stances)
        assert result["avg_pairwise_correlation"] > 0.5
        assert result["effective_n"] < 5

    def test_constant_agents_detected(self, estimator):
        """Agents with zero variance should not break the computation."""
        round_stances = [np.array([1.0, 1.0, 1.0, 1.0, 1.0]) for _ in range(5)]
        stances = round_stances[-1]

        result = estimator.estimate(stances, round_stances)
        # All constant → degenerate
        assert result["effective_n"] <= 2


class TestBootstrapMethod:
    """Test bootstrap variance ratio for single-round data."""

    def test_diverse_stances(self, estimator):
        """Diverse stances should indicate lower correlation."""
        rng = np.random.default_rng(42)
        stances = rng.choice([-1.0, -0.5, 0.0, 0.5, 1.0], size=30)

        result = estimator.estimate(stances, round_stances=None)
        assert result["method"] == "bootstrap"
        assert 0.0 <= result["avg_pairwise_correlation"] <= 1.0

    def test_uniform_stances(self, estimator):
        """All identical stances should show very high correlation."""
        stances = np.array([0.5] * 20)

        result = estimator.estimate(stances, round_stances=None)
        assert result["avg_pairwise_correlation"] > 0.5
        assert result["effective_n"] < 5

    def test_small_n(self, estimator):
        """Small N should use the small-N fallback."""
        stances = np.array([0.5, -0.5])
        result = estimator.estimate(stances, round_stances=None)
        assert result["method"] == "bootstrap_small_n"


class TestEigenvalueAnalysis:
    """Test eigenvalue decomposition for effective rank."""

    def test_full_rank(self, estimator):
        """Independent agents should yield effective rank ≈ N."""
        rng = np.random.default_rng(42)
        round_stances = [rng.uniform(-1, 1, size=10) for _ in range(20)]

        result = estimator._eigenvalue_analysis(round_stances, N=10)
        assert result["eigenvalue_effective_rank"] > 3
        assert result["top_eigenvalue_fraction"] < 0.5

    def test_rank_one(self, estimator):
        """All agents driven by same factor should yield low effective rank."""
        rng = np.random.default_rng(42)
        N_agents = 20
        N_rounds = 10
        common_factor = rng.uniform(-1, 1, size=N_rounds)
        round_stances = []
        for t in range(N_rounds):
            stances_t = np.full(N_agents, common_factor[t]) + rng.normal(0, 0.01, size=N_agents)
            round_stances.append(np.clip(stances_t, -1, 1))

        result = estimator._eigenvalue_analysis(round_stances, N=N_agents)
        assert result["eigenvalue_effective_rank"] < 3
        assert result["top_eigenvalue_fraction"] > 0.5


class TestEdgeCases:
    """Edge cases for the independence estimator."""

    def test_single_agent(self, estimator):
        result = estimator.estimate(np.array([0.5]))
        assert result["method"] == "trivial"
        assert result["effective_n"] == 1.0

    def test_output_fields(self, estimator):
        stances = np.array([0.5, -0.5, 0.0, 1.0, -1.0])
        result = estimator.estimate(stances)
        assert "avg_pairwise_correlation" in result
        assert "effective_n" in result
        assert "method" in result
        assert "eigenvalue_effective_rank" in result

    def test_conservative_estimate_chosen(self, estimator):
        """When both temporal and eigenvalue methods are available,
        the more conservative (lower) N_eff should be used."""
        rng = np.random.default_rng(42)
        round_stances = [rng.uniform(-1, 1, size=10) for _ in range(10)]
        stances = round_stances[-1]

        result = estimator.estimate(stances, round_stances)
        # effective_n should be ≤ both individual estimates
        assert result["effective_n"] <= result["eigenvalue_effective_rank"] + 0.5

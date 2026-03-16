"""
Tests for SwarmConsensusAnalyser.

Validates that the core statistical engine produces correct results
against known analytical solutions and edge cases.
"""

import numpy as np
import pytest
from scipy.special import comb

from ..consensus_analyser import SwarmConsensusAnalyser


@pytest.fixture
def analyser():
    return SwarmConsensusAnalyser()


class TestDirection:
    """Test that direction is correctly computed from stances."""

    def test_all_bullish(self, analyser):
        stances = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
        result = analyser.analyse(stances)
        assert result["direction"] == pytest.approx(1.0)

    def test_all_bearish(self, analyser):
        stances = np.array([-1.0, -1.0, -1.0, -1.0, -1.0])
        result = analyser.analyse(stances)
        assert result["direction"] == pytest.approx(-1.0)

    def test_balanced(self, analyser):
        stances = np.array([1.0, -1.0, 0.5, -0.5, 0.0])
        result = analyser.analyse(stances)
        assert result["direction"] == pytest.approx(0.0)

    def test_direction_clipped_to_range(self, analyser):
        stances = np.array([0.5, 0.5, 0.5])
        result = analyser.analyse(stances)
        assert -1.0 <= result["direction"] <= 1.0


class TestEffectiveN:
    """Test the Forge §4.3 ESS formula."""

    def test_independent_agents(self, analyser):
        """With zero correlation, N_eff should equal N."""
        N_eff = analyser._effective_n(10, avg_pairwise_corr=0.0)
        assert N_eff == pytest.approx(10.0)

    def test_perfectly_correlated(self, analyser):
        """With perfect correlation, N_eff should approach 1."""
        N_eff = analyser._effective_n(100, avg_pairwise_corr=0.99)
        assert N_eff == pytest.approx(1.0, abs=0.5)

    def test_moderate_correlation(self, analyser):
        """N_eff should be between 1 and N for moderate correlation."""
        N = 20
        N_eff = analyser._effective_n(N, avg_pairwise_corr=0.3)
        assert 1.0 < N_eff < N

    def test_ess_formula_matches_forge(self, analyser):
        """Verify formula: N_eff = N × (1-ρ) / (1+(N-1)×ρ)"""
        N, rho = 50, 0.2
        expected = N * (1 - rho) / (1 + (N - 1) * rho)
        actual = analyser._effective_n(N, rho)
        assert actual == pytest.approx(expected)


class TestConsensusEntropy:
    """Test Shannon entropy and consensus strength."""

    def test_perfect_consensus(self, analyser):
        """All agents in one bin → entropy=0, strength=1."""
        stances = np.array([1.0] * 20)
        H, strength = analyser._consensus_entropy(stances)
        assert H == pytest.approx(0.0)
        assert strength == pytest.approx(1.0)

    def test_uniform_distribution(self, analyser):
        """Equal distribution across bins → max entropy, strength≈0."""
        stances = np.array([-1.0] * 10 + [-0.5] * 10 + [0.0] * 10 +
                           [0.5] * 10 + [1.0] * 10)
        H, strength = analyser._consensus_entropy(stances)
        H_max = np.log2(5)
        assert H == pytest.approx(H_max, abs=0.01)
        assert strength == pytest.approx(0.0, abs=0.01)

    def test_strength_between_0_and_1(self, analyser):
        stances = np.array([0.5, 0.5, 0.5, 1.0, 0.0])
        _, strength = analyser._consensus_entropy(stances)
        assert 0.0 <= strength <= 1.0


class TestCondorcet:
    """Test Condorcet Jury Theorem probability calculation."""

    def test_coin_flip_accuracy(self, analyser):
        """If p=0.5, Condorcet gives p (no aggregation benefit)."""
        prob = analyser._condorcet_probability(N_eff=10, p=0.5)
        assert prob == pytest.approx(0.5)

    def test_high_accuracy_large_n(self, analyser):
        """If p=0.7 and N_eff=21, majority probability should be very high."""
        prob = analyser._condorcet_probability(N_eff=21, p=0.7)
        # Analytical: sum C(21,k) × 0.7^k × 0.3^(21-k) for k=11..21
        expected = sum(
            comb(21, k, exact=True) * 0.7**k * 0.3**(21 - k)
            for k in range(11, 22)
        )
        assert prob == pytest.approx(expected, abs=0.001)

    def test_condorcet_monotonic_in_p(self, analyser):
        """Higher p should give higher probability."""
        p1 = analyser._condorcet_probability(N_eff=10, p=0.6)
        p2 = analyser._condorcet_probability(N_eff=10, p=0.7)
        assert p2 > p1

    def test_condorcet_monotonic_in_n(self, analyser):
        """More effective agents should give higher probability (for p>0.5)."""
        p1 = analyser._condorcet_probability(N_eff=5, p=0.65)
        p2 = analyser._condorcet_probability(N_eff=21, p=0.65)
        assert p2 > p1


class TestHerding:
    """Test herding detection threshold."""

    def test_herding_detected_when_correlated(self, analyser):
        """Herding should be flagged when correlation exceeds 1/sqrt(N)."""
        stances = np.array([1.0] * 100)
        result = analyser.analyse(stances)
        # Perfect agreement → high correlation → herding
        assert result["is_herding"] is True

    def test_herding_threshold_formula(self, analyser):
        """Threshold should be 1/sqrt(N)."""
        N = 25
        stances = np.zeros(N)
        result = analyser.analyse(stances)
        assert result["herding_threshold"] == pytest.approx(1.0 / np.sqrt(N), abs=0.001)


class TestSignalZ:
    """Test signal strength z-score."""

    def test_neutral_stances_give_zero_z(self, analyser):
        """If stances average to 0, z should be 0."""
        stances = np.array([1.0, -1.0, 0.5, -0.5, 0.0])
        result = analyser.analyse(stances)
        assert result["signal_strength_z"] == pytest.approx(0.0, abs=0.1)

    def test_strong_consensus_gives_high_z(self, analyser):
        """Strong directional consensus should produce high z."""
        stances = np.array([0.8, 0.9, 1.0, 0.7, 0.8, 0.9, 1.0, 0.8, 0.9, 1.0])
        result = analyser.analyse(stances)
        assert abs(result["signal_strength_z"]) > 2.0


class TestEdgeCases:
    """Test edge cases and degenerate inputs."""

    def test_single_agent(self, analyser):
        stances = np.array([0.5])
        result = analyser.analyse(stances)
        assert result["n_agents"] == 1
        assert result["direction"] == pytest.approx(0.5)

    def test_two_agents(self, analyser):
        stances = np.array([1.0, -1.0])
        result = analyser.analyse(stances)
        assert result["direction"] == pytest.approx(0.0)

    def test_empty_raises(self, analyser):
        with pytest.raises(ValueError):
            analyser.analyse(np.array([]))

    def test_all_output_fields_present(self, analyser):
        stances = np.array([0.5, -0.5, 0.0, 1.0, -1.0])
        result = analyser.analyse(stances)
        expected_keys = {
            "direction", "confidence", "signal_strength_z", "n_agents",
            "effective_independent_agents", "avg_pairwise_correlation",
            "consensus_entropy", "consensus_strength", "condorcet_probability",
            "agent_accuracy_estimate", "is_herding", "herding_threshold",
            "statistical_power_sufficient",
        }
        assert expected_keys.issubset(set(result.keys()))

    def test_historical_accuracy_overrides_estimate(self, analyser):
        """When historical accuracy is provided, it should be used instead of consensus-derived."""
        stances = np.array([0.5, 0.5, 0.5, 0.5, 0.5])
        result_default = analyser.analyse(stances)
        result_override = analyser.analyse(stances, historical_accuracy=0.75)
        assert result_override["agent_accuracy_estimate"] == pytest.approx(0.75)
        assert result_override["agent_accuracy_estimate"] != result_default["agent_accuracy_estimate"]

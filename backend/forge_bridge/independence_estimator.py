"""
Agent Independence Estimator

Estimates the effective number of independent agents in a MiroFish swarm,
accounting for the correlation induced by shared LLM training data.

This is the MiroFish analogue of Forge §4.3 (effective independent bets).
Without this correction, raw agent count N would overstate the information
content of the swarm, leading to overconfident signals.

Statistical Methods:
    1. Temporal correlation (multi-round simulations): pairwise correlation matrix
    2. Bootstrap variance ratio (single-round): compares bootstrap variance to theoretical
    3. Eigenvalue decomposition: estimates the effective rank of the agent correlation matrix

Forge Compliance:
    - §4.3: Mirrors effective_independent_bets formula
    - §4.5: Statistical power adjusted for effective N
"""

import numpy as np
from typing import Optional

import logging

logger = logging.getLogger("forge_bridge.independence")


class IndependenceEstimator:
    """
    Estimates agent independence using multiple methods depending on data availability.
    """

    def estimate(
        self,
        agent_stances: np.ndarray,
        round_stances: Optional[list] = None,
    ) -> dict:
        """
        Estimate the effective number of independent agents.

        Args:
            agent_stances: Final-round stances, shape (N_agents,)
            round_stances: Optional list of arrays (one per round), each shape (N_agents,)

        Returns:
            dict with correlation estimate, effective N, and method used.
        """
        N = len(agent_stances)
        if N <= 1:
            return {
                "avg_pairwise_correlation": 0.99,
                "effective_n": 1.0,
                "method": "trivial",
                "eigenvalue_effective_rank": 1.0,
                "top_eigenvalue_fraction": 1.0,
            }

        if round_stances is not None and len(round_stances) >= 3:
            result = self._temporal_method(round_stances, N)
        else:
            result = self._bootstrap_method(agent_stances, N)

        # Supplement with eigenvalue analysis if multi-round data available
        if round_stances is not None and len(round_stances) >= 3:
            eigen_result = self._eigenvalue_analysis(round_stances, N)
            result.update(eigen_result)

            # Use the more conservative (lower) effective N estimate
            result["effective_n"] = min(
                result["effective_n"],
                result.get("eigenvalue_effective_rank", result["effective_n"]),
            )
        else:
            result["eigenvalue_effective_rank"] = result["effective_n"]
            result["top_eigenvalue_fraction"] = 1.0 / N

        return result

    def _temporal_method(self, round_stances: list, N: int) -> dict:
        """
        Estimate correlation from multi-round stance evolution.

        Build (N_agents × N_rounds) matrix, compute pairwise correlation,
        then apply the Forge ESS formula.
        """
        matrix = np.column_stack(round_stances)  # (N_agents, N_rounds)

        # Remove agents with zero variance
        variances = np.var(matrix, axis=1)
        active_mask = variances > 1e-10
        n_active = active_mask.sum()

        if n_active < 2:
            return {
                "avg_pairwise_correlation": 0.99,
                "effective_n": 1.0,
                "method": "temporal_degenerate",
            }

        active_matrix = matrix[active_mask]
        corr_matrix = np.corrcoef(active_matrix)

        # Average off-diagonal correlation
        total_corr = corr_matrix.sum() - n_active
        n_pairs = n_active * (n_active - 1)
        avg_corr = total_corr / n_pairs if n_pairs > 0 else 0.5
        avg_corr = float(np.clip(avg_corr, 0.0, 0.99))

        # Forge ESS formula: N_eff = N × (1 - ρ) / (1 + (N-1) × ρ)
        N_eff = N * (1 - avg_corr) / (1 + (N - 1) * avg_corr)
        N_eff = max(1.0, float(N_eff))

        return {
            "avg_pairwise_correlation": round(avg_corr, 4),
            "effective_n": round(N_eff, 2),
            "method": "temporal",
            "n_active_agents": int(n_active),
            "n_constant_agents": int(N - n_active),
        }

    def _bootstrap_method(
        self,
        agent_stances: np.ndarray,
        N: int,
        n_bootstrap: int = 1000,
    ) -> dict:
        """
        Bootstrap independence test for single-round simulations.

        Compares the bootstrap variance of the sample mean against the
        theoretical variance under independence.

        If agents are correlated:
            - The true variance of the mean is HIGHER than σ²/N
            - But bootstrap (which resamples from the empirical distribution)
              can't reconstruct the correlation structure
            - So bootstrap variance approximates σ²/N regardless
            - The RATIO bootstrap_var / true_var < 1 indicates correlation

        We use the variance of the stances themselves to detect structure.
        """
        if N <= 2:
            return {
                "avg_pairwise_correlation": 0.5,
                "effective_n": max(1.0, float(N) * 0.5),
                "method": "bootstrap_small_n",
            }

        rng = np.random.default_rng(42)
        bootstrap_means = np.array(
            [
                np.mean(rng.choice(agent_stances, size=N, replace=True))
                for _ in range(n_bootstrap)
            ]
        )

        bootstrap_var = np.var(bootstrap_means)
        empirical_var = np.var(agent_stances)
        theoretical_var = empirical_var / N

        if theoretical_var < 1e-12:
            return {
                "avg_pairwise_correlation": 0.99,
                "effective_n": 1.0,
                "method": "bootstrap_zero_var",
            }

        # Estimate correlation from variance ratio
        # Under correlation ρ: Var(mean) = σ²/N × (1 + (N-1)ρ)
        # Bootstrap gives: σ²/N
        # So ratio = 1 / (1 + (N-1)ρ)
        # → ρ = (1/ratio - 1) / (N-1)
        ratio = bootstrap_var / theoretical_var if theoretical_var > 0 else 1.0

        if ratio >= 1.0:
            # No evidence of correlation (or negative correlation)
            avg_corr = 0.0
        else:
            avg_corr = (1.0 / max(ratio, 0.01) - 1.0) / max(N - 1, 1)

        avg_corr = float(np.clip(avg_corr, 0.0, 0.99))

        N_eff = N * (1 - avg_corr) / (1 + (N - 1) * avg_corr)
        N_eff = max(1.0, float(N_eff))

        return {
            "avg_pairwise_correlation": round(avg_corr, 4),
            "effective_n": round(N_eff, 2),
            "method": "bootstrap",
            "bootstrap_variance_ratio": round(float(ratio), 4),
        }

    def _eigenvalue_analysis(self, round_stances: list, N: int) -> dict:
        """
        Eigenvalue decomposition of the agent correlation matrix.

        The effective rank of the correlation matrix tells us how many
        truly independent dimensions of information the swarm provides.

        Uses the Roy (2007) effective rank: exp(Shannon entropy of normalised eigenvalues).
        This is a continuous measure that smoothly interpolates between
        1 (all agents identical) and N (all agents independent).
        """
        matrix = np.column_stack(round_stances)

        # Remove zero-variance agents
        variances = np.var(matrix, axis=1)
        active_mask = variances > 1e-10
        if active_mask.sum() < 2:
            return {
                "eigenvalue_effective_rank": 1.0,
                "top_eigenvalue_fraction": 1.0,
            }

        active_matrix = matrix[active_mask]
        corr_matrix = np.corrcoef(active_matrix)

        eigenvalues = np.linalg.eigvalsh(corr_matrix)
        eigenvalues = np.maximum(eigenvalues, 0)  # numerical stability

        # Normalise eigenvalues to form a probability distribution
        total = eigenvalues.sum()
        if total < 1e-12:
            return {
                "eigenvalue_effective_rank": 1.0,
                "top_eigenvalue_fraction": 1.0,
            }

        p = eigenvalues / total
        p = p[p > 1e-12]  # remove numerical zeros

        # Shannon entropy of eigenvalue distribution
        entropy = -np.sum(p * np.log(p))
        effective_rank = float(np.exp(entropy))

        # Fraction of variance explained by top eigenvalue
        top_fraction = float(eigenvalues.max() / total)

        return {
            "eigenvalue_effective_rank": round(effective_rank, 2),
            "top_eigenvalue_fraction": round(top_fraction, 4),
        }

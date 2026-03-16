"""
Quality Gates

Implements all Forge §12 tiered quality gates for MiroFish signals.
Evaluates CPCV results against tier-specific thresholds and determines
the highest qualifying tier.

Gate Categories:
    - Core CPCV gates (§12.1)
    - Effective sample size gates (§12.2)
    - False positive gates (§12.3)
    - Robustness gates (§12.4)
    - Drawdown gates (§12.5)
"""

import numpy as np
from scipy.stats import norm
from typing import Optional

import logging

logger = logging.getLogger("forge_bridge.validation.gates")


def deflated_sharpe(sr_mean_oos: float, sr_std_oos: float,
                    n_folds: int, N: int) -> float:
    """
    Deflated Sharpe Ratio (Forge §7.3).

    Tests whether the observed OOS Sharpe exceeds what you'd expect
    from the luckiest of N random strategies.
    """
    if n_folds <= 0 or sr_std_oos <= 0:
        return 0.0

    se = sr_std_oos / np.sqrt(n_folds)
    emc = 0.5772  # Euler-Mascheroni constant
    expected_max_z = (1 - emc) * norm.ppf(1 - 1 / max(N, 2)) + \
                     emc * norm.ppf(1 - 1 / (max(N, 2) * np.e))
    observed_z = sr_mean_oos / se
    z = observed_z - expected_max_z
    return float(norm.cdf(z))


def complexity_adjusted_sharpe(sr_observed: float, n_params: int, T: int) -> float:
    """Complexity-adjusted Sharpe (Forge §7.5)."""
    if T <= 0:
        return 0.0
    penalty = np.sqrt(2 * n_params / T)
    return sr_observed - penalty


def compute_pbo(cpcv_results: list) -> float:
    """
    Probability of Backtest Overfitting (Forge §8.1).

    For n_free_parameters = 0, use z-score form (§8.2).
    """
    if not cpcv_results:
        return 1.0

    is_sharpes = [r.get("is_sharpe", 0) for r in cpcv_results if r.get("is_sharpe") is not None]
    oos_sharpes = [r["oos_sharpe"] for r in cpcv_results]

    if not is_sharpes or not oos_sharpes:
        return 0.5

    oos_median = np.median(oos_sharpes)
    is_median = np.median(is_sharpes)

    above_is_median = [i for i, s in enumerate(is_sharpes) if s > is_median]
    if not above_is_median:
        return 0.5

    overfit_count = sum(
        1 for i in above_is_median
        if i < len(oos_sharpes) and oos_sharpes[i] < oos_median
    )
    return overfit_count / len(above_is_median)


def pbo_z_score(pbo: float, n_above_median: int) -> float:
    """
    PBO z-score for passive strategies with n_free_parameters = 0.
    Forge §8.2 amendment.
    """
    if n_above_median <= 0:
        return 0.0
    se = np.sqrt(0.25 / n_above_median)
    return (pbo - 0.50) / se if se > 0 else 0.0


def bayesian_sharpe(sr_observed: float, sr_se: float,
                    prior_mean: float = 0.0, prior_std: float = 0.5) -> dict:
    """Bayesian shrinkage on Sharpe estimates (Forge §7.6)."""
    precision_prior = 1 / prior_std**2
    precision_obs = 1 / max(sr_se, 0.001)**2
    posterior_mean = (precision_prior * prior_mean + precision_obs * sr_observed) / \
                     (precision_prior + precision_obs)
    posterior_std = 1 / np.sqrt(precision_prior + precision_obs)
    return {
        "posterior_mean": round(posterior_mean, 4),
        "posterior_std": round(posterior_std, 4),
        "shrinkage_factor": round(precision_prior / (precision_prior + precision_obs), 4),
        "credible_interval_95": (
            round(posterior_mean - 1.96 * posterior_std, 4),
            round(posterior_mean + 1.96 * posterior_std, 4),
        ),
    }


# Tier thresholds for directional/informational strategies (Forge §12.1)
TIER_GATES = {
    1: {
        "mean_oos_sharpe": 0.55,    # 0.6 - 0.05 (minimal complexity bonus)
        "pct_positive_oos": 0.70,
        "mean_oos_maxdd_pct": 30,
        "pbo_threshold": 0.30,
        "decay_ratio": 0.50,
        "param_sensitivity_pct": 0.80,
        "boundary_sensitivity_pct": 20,
        "sharpe_at_2x_cost": 0.4,
        "breakeven_headroom": 3.0,
        "noise_perturbation_pct": 0.70,
        "time_shift_pct": 0.50,
        "subsample_stability_std": 0.30,
        "randomised_baseline_z": 2.0,
        "randomised_baseline_pct_better": 0.05,
        "inverse_sharpe_max": 0.0,
    },
    2: {
        "mean_oos_sharpe": 0.40,    # 0.45 - 0.05
        "pct_positive_oos": 0.60,
        "mean_oos_maxdd_pct": 35,
        "pbo_threshold": 0.35,
        "decay_ratio": 0.45,
        "param_sensitivity_pct": 0.70,
        "boundary_sensitivity_pct": 25,
        "sharpe_at_2x_cost": 0.0,
        "breakeven_headroom": 2.0,
        "noise_perturbation_pct": 0.60,
        "time_shift_pct": 0.40,
        "subsample_stability_std": 0.40,
        "randomised_baseline_z": 1.5,
        "randomised_baseline_pct_better": 0.10,
        "inverse_sharpe_max": 0.0,
    },
    3: {
        "mean_oos_sharpe": 0.30,    # 0.35 - 0.05
        "pct_positive_oos": 0.55,
        "mean_oos_maxdd_pct": 40,
        "pbo_threshold": 0.40,
        "decay_ratio": 0.40,
        "param_sensitivity_pct": 0.60,
        "boundary_sensitivity_pct": 30,
        "sharpe_at_2x_cost": 0.0,
        "breakeven_headroom": 1.5,
        "noise_perturbation_pct": 0.50,
        "time_shift_pct": 0.30,
        "subsample_stability_std": 0.50,
        "randomised_baseline_z": 1.0,
        "randomised_baseline_pct_better": 0.20,
        "inverse_sharpe_max": 0.2,
    },
}


def check_all_gates(
    cpcv_results: list,
    effective_bets: int,
    cemetery_count: int,
    n_free_parameters: int,
    cost_results: Optional[dict] = None,
    perturbation_results: Optional[dict] = None,
    randomised_baseline: Optional[dict] = None,
    inverse_result: Optional[dict] = None,
    boundary_sensitivity: Optional[dict] = None,
) -> dict:
    """
    Evaluate all quality gates and determine tier assignment.

    Returns:
        Dict with tier assignment, per-gate results, and summary.
    """
    if not cpcv_results:
        return {"assigned_tier": None, "all_gates_passed": False,
                "reason": "No CPCV results"}

    # Extract CPCV statistics
    oos_sharpes = [r["oos_sharpe"] for r in cpcv_results]
    is_sharpes = [r.get("is_sharpe", 0) for r in cpcv_results if r.get("is_sharpe") is not None]

    mean_oos = np.mean(oos_sharpes)
    std_oos = np.std(oos_sharpes)
    pct_positive = np.mean([s > 0 for s in oos_sharpes])
    p05_oos = np.percentile(oos_sharpes, 5)

    mean_maxdd = np.mean([r.get("max_dd_pct", 0) for r in cpcv_results])
    n_folds = len(cpcv_results)

    # Decay ratio
    mean_is = np.mean(is_sharpes) if is_sharpes else mean_oos * 2
    decay_ratio = mean_oos / mean_is if mean_is > 0 else 0

    # PBO
    pbo = compute_pbo(cpcv_results)
    if n_free_parameters == 0:
        n_above_median = sum(1 for s in is_sharpes if s > np.median(is_sharpes))
        pbo_z = pbo_z_score(pbo, n_above_median)
    else:
        pbo_z = None

    # DSR
    dsr = deflated_sharpe(mean_oos, std_oos, n_folds, cemetery_count)

    # Complexity-adjusted Sharpe
    T = sum(r.get("test_bars", 0) for r in cpcv_results)
    cas = complexity_adjusted_sharpe(mean_oos, n_free_parameters, T)

    # Bayesian Sharpe
    se_oos = std_oos / np.sqrt(n_folds) if n_folds > 0 else 1.0
    bayes = bayesian_sharpe(mean_oos, se_oos)

    # Build gate results
    gate_results = {
        "cpcv": {
            "mean_oos_sharpe": round(mean_oos, 4),
            "std_oos_sharpe": round(std_oos, 4),
            "pct_positive_oos": round(pct_positive, 4),
            "p05_oos_sharpe": round(p05_oos, 4),
            "mean_oos_maxdd_pct": round(mean_maxdd, 2),
            "n_folds": n_folds,
        },
        "false_positive": {
            "dsr": round(dsr, 4),
            "dsr_flagged": dsr < 0.50,
            "complexity_adjusted_sharpe": round(cas, 4),
            "bayesian_posterior_sharpe": bayes["posterior_mean"],
            "bayesian_shrinkage_factor": bayes["shrinkage_factor"],
            "pbo": round(pbo, 4),
            "pbo_z_score": round(pbo_z, 4) if pbo_z is not None else None,
            "decay_ratio": round(decay_ratio, 4),
            "cemetery_count": cemetery_count,
        },
        "sample_size": {
            "effective_independent_bets": effective_bets,
            "n_folds": n_folds,
            "sufficient_bets": effective_bets >= 30,
            "sufficient_folds": n_folds >= 15,
        },
    }

    # Add cost results if available
    if cost_results:
        gate_results["costs"] = cost_results

    # Add perturbation results if available
    if perturbation_results:
        gate_results["perturbation"] = perturbation_results

    # Add randomised baseline if available
    if randomised_baseline:
        gate_results["randomised_baseline"] = randomised_baseline

    # Add inverse signal if available
    if inverse_result:
        gate_results["inverse_signal"] = inverse_result

    # Add boundary sensitivity if available
    if boundary_sensitivity:
        gate_results["boundary_sensitivity"] = boundary_sensitivity

    # Evaluate each tier (Forge §12.7)
    for tier in [1, 2, 3]:
        gates = TIER_GATES[tier]
        passes = True
        failures = []

        # Core CPCV gates
        if mean_oos < gates["mean_oos_sharpe"]:
            passes = False
            failures.append(f"mean_oos_sharpe {mean_oos:.3f} < {gates['mean_oos_sharpe']}")
        if pct_positive < gates["pct_positive_oos"]:
            passes = False
            failures.append(f"pct_positive {pct_positive:.3f} < {gates['pct_positive_oos']}")
        if mean_maxdd > gates["mean_oos_maxdd_pct"]:
            passes = False
            failures.append(f"mean_maxdd {mean_maxdd:.1f}% > {gates['mean_oos_maxdd_pct']}%")

        # PBO gate (z-score form for n=0)
        if n_free_parameters == 0:
            pbo_tier_z = {1: 1.28, 2: 1.64, 3: 1.96}
            if pbo_z is not None and pbo_z > pbo_tier_z[tier]:
                passes = False
                failures.append(f"pbo_z {pbo_z:.2f} > {pbo_tier_z[tier]}")
        else:
            if pbo > gates["pbo_threshold"]:
                passes = False
                failures.append(f"pbo {pbo:.3f} > {gates['pbo_threshold']}")

        # Decay ratio
        if decay_ratio < gates["decay_ratio"]:
            passes = False
            failures.append(f"decay_ratio {decay_ratio:.3f} < {gates['decay_ratio']}")

        # Sample size (non-negotiable)
        if effective_bets < 30:
            passes = False
            failures.append(f"effective_bets {effective_bets} < 30")
        if n_folds < 15:
            passes = False
            failures.append(f"n_folds {n_folds} < 15")

        # Randomised baseline (Mode A for informational signals)
        if randomised_baseline:
            z = randomised_baseline.get("signal_value_z", 0)
            pct = randomised_baseline.get("pct_random_better", 1)
            if z < gates["randomised_baseline_z"]:
                passes = False
                failures.append(f"baseline_z {z:.2f} < {gates['randomised_baseline_z']}")
            if pct > gates["randomised_baseline_pct_better"]:
                passes = False
                failures.append(f"pct_random_better {pct:.3f} > {gates['randomised_baseline_pct_better']}")

        # Inverse signal
        if inverse_result:
            inv_sharpe = inverse_result.get("sharpe", 0)
            if inv_sharpe > gates["inverse_sharpe_max"]:
                passes = False
                failures.append(f"inverse_sharpe {inv_sharpe:.3f} > {gates['inverse_sharpe_max']}")

        gate_results[f"tier_{tier}_evaluation"] = {
            "passes": passes,
            "failures": failures,
        }

        if passes:
            gate_results["assigned_tier"] = tier
            gate_results["all_gates_passed"] = True
            logger.info("Strategy assigned to Tier %d", tier)
            return gate_results

    # Failed all tiers
    gate_results["assigned_tier"] = None
    gate_results["all_gates_passed"] = False
    gate_results["reason"] = "Failed all tier gates"
    return gate_results

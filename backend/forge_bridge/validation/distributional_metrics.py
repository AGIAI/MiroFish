"""
Distributional Metrics from CPCV

Implements Forge §17: full percentile tables, equity curve fan data,
and Tukey-fence-derived kill thresholds.

Every metric is a distribution, not a point estimate.
"""

import numpy as np
from typing import Optional

import logging

logger = logging.getLogger("forge_bridge.validation.distributions")


def tukey_upper_fence(values: list) -> float:
    """
    Tukey upper fence: P75 + 1.5 × IQR.
    Used for drawdown/duration metrics (larger = worse).
    Forge §17.3.
    """
    if not values:
        return float("inf")
    q25, q75 = np.percentile(values, [25, 75])
    iqr = q75 - q25
    return float(q75 + 1.5 * iqr)


def tukey_lower_fence(values: list) -> float:
    """
    Tukey lower fence: P25 - 1.5 × IQR.
    Used for return/Sharpe metrics (smaller = worse).
    Forge §17.3.
    """
    if not values:
        return float("-inf")
    q25, q75 = np.percentile(values, [25, 75])
    iqr = q75 - q25
    return float(q25 - 1.5 * iqr)


def compute_percentiles(values: list) -> dict:
    """Compute full percentile table."""
    if not values:
        return {}
    return {
        "p1": round(float(np.percentile(values, 1)), 4),
        "p5": round(float(np.percentile(values, 5)), 4),
        "p10": round(float(np.percentile(values, 10)), 4),
        "p25": round(float(np.percentile(values, 25)), 4),
        "p50": round(float(np.percentile(values, 50)), 4),
        "p75": round(float(np.percentile(values, 75)), 4),
        "p90": round(float(np.percentile(values, 90)), 4),
        "p95": round(float(np.percentile(values, 95)), 4),
        "p99": round(float(np.percentile(values, 99)), 4),
        "mean": round(float(np.mean(values)), 4),
        "std": round(float(np.std(values)), 4),
    }


def distributions_from_cpcv(cpcv_results: list) -> dict:
    """
    Compute all distributional metrics from CPCV fold results.
    Forge §17.2.
    """
    if not cpcv_results:
        return {"distributions_note": "No CPCV results available"}

    oos_sharpes = [r["oos_sharpe"] for r in cpcv_results]
    maxdd_pcts = [r.get("max_dd_pct", 0) for r in cpcv_results]
    maxdd_durations = [r.get("max_dd_duration_bars", 0) for r in cpcv_results]
    cagrs = [r.get("cagr", 0) for r in cpcv_results]
    calmars = [r.get("calmar", 0) for r in cpcv_results]

    return {
        "sharpe": {
            **compute_percentiles(oos_sharpes),
            "prob_positive": round(float(np.mean([s > 0 for s in oos_sharpes])), 4),
        },
        "maxdd_pct": compute_percentiles(maxdd_pcts),
        "maxdd_duration_bars": compute_percentiles(maxdd_durations),
        "annual_return_pct": compute_percentiles([c * 100 for c in cagrs]),
        "calmar": compute_percentiles(calmars),
    }


def compute_kill_thresholds(cpcv_results: list, bar_size_hours: float = 4.0) -> dict:
    """
    Derive kill thresholds from CPCV distributions using Tukey fences.
    Forge §17.3 & §17.4.
    """
    if not cpcv_results:
        return {}

    oos_sharpes = [r["oos_sharpe"] for r in cpcv_results]
    maxdd_pcts = [r.get("max_dd_pct", 0) for r in cpcv_results]
    maxdd_durations = [r.get("max_dd_duration_bars", 0) for r in cpcv_results]

    # Convert duration bars to days for readability
    bars_per_day = 24.0 / bar_size_hours
    maxdd_duration_days = [d / bars_per_day for d in maxdd_durations]

    # Kill thresholds (Forge §17.3)
    maxdd_duration_kill = tukey_upper_fence(maxdd_duration_days)
    maxdd_pct_kill = tukey_upper_fence(maxdd_pcts)
    sharpe_lower_kill = tukey_lower_fence(oos_sharpes)

    # Observation window (Forge §17.4)
    p50_maxdd_dur_days = float(np.percentile(maxdd_duration_days, 50)) if maxdd_duration_days else 1.0
    observation_window_hours = max(
        5.0 * bar_size_hours,
        0.1 * p50_maxdd_dur_days * 24,
    )

    # Daily PnL kill threshold (§17.3 Amendment 6)
    daily_pnls = []
    for r in cpcv_results:
        ann_ret = r.get("annual_return", 0)
        daily_pnls.append(ann_ret / 365.0)  # approximate daily PnL
    daily_pnl_kill = tukey_lower_fence(daily_pnls)

    # Primary kill condition string
    primary_kill = (
        f"MaxDD duration > {maxdd_duration_kill:.1f} days OR "
        f"MaxDD > {maxdd_pct_kill:.1f}% OR "
        f"Sharpe < {sharpe_lower_kill:.2f}"
    )

    return {
        "maxdd_duration_days": round(maxdd_duration_kill, 2),
        "maxdd_pct": round(maxdd_pct_kill, 2),
        "sharpe_lower_fence": round(sharpe_lower_kill, 4),
        "daily_pnl_lower_fence": round(daily_pnl_kill, 6),
        "observation_window_hours": round(observation_window_hours, 1),
        "primary_kill_condition": primary_kill,
        "n_cpcv_folds": len(cpcv_results),
        "bar_size_hours": bar_size_hours,
    }


def equity_curve_fan_data(cpcv_results: list) -> dict:
    """
    Compute P5/P50/P95 equity curve bands across all CPCV paths.
    Forge §17.1.
    """
    if not cpcv_results:
        return {}

    # Collect total returns per fold
    returns = [r.get("total_return", 0) for r in cpcv_results]
    if not returns:
        return {}

    return {
        "p5_total_return": round(float(np.percentile(returns, 5)), 4),
        "p50_total_return": round(float(np.percentile(returns, 50)), 4),
        "p95_total_return": round(float(np.percentile(returns, 95)), 4),
        "mean_total_return": round(float(np.mean(returns)), 4),
        "n_paths": len(returns),
    }

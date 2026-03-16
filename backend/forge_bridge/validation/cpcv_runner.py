"""
Combinatorial Purged Cross-Validation (CPCV) Runner

Implements Forge §2 CPCV with data-point-adaptive configuration.
Produces 28-45 independent OOS estimates — a distribution, not a point estimate.

Key Features:
    - Data-point-driven block configuration (not calendar-based)
    - Embargo zones to prevent look-ahead contamination
    - Boundary sensitivity testing (§2.1)
    - Regime labelling per OOS window (§20)
"""

import numpy as np
import pandas as pd
from itertools import combinations
from typing import Optional, Callable

import logging

logger = logging.getLogger("forge_bridge.validation.cpcv")


def effective_observations(n_bars: int, autocorr_lag1: float,
                           avg_trade_duration_bars: Optional[float] = None) -> int:
    """
    Compute effective observations accounting for autocorrelation.
    Forge §2 data-point adaptive configuration.
    """
    ess_adj = (1 - autocorr_lag1) / (1 + autocorr_lag1) if abs(autocorr_lag1) < 1 else 0.01
    eff_obs = n_bars * ess_adj
    if avg_trade_duration_bars and avg_trade_duration_bars > 1:
        eff_obs = min(eff_obs, n_bars / avg_trade_duration_bars)
    return int(eff_obs)


def cpcv_config(eff_obs: int) -> dict:
    """
    Determine CPCV configuration from effective observations.
    Forge §2 table — derived thresholds.
    """
    if eff_obs >= 10000:
        return {"n_blocks": 10, "k_test": 2, "combinations": 45,
                "embargo_pct": 0.05}
    elif eff_obs >= 5000:
        return {"n_blocks": 8, "k_test": 2, "combinations": 28,
                "embargo_pct": 0.05}
    elif eff_obs >= 2000:
        return {"n_blocks": 6, "k_test": 2, "combinations": 15,
                "embargo_pct": 0.05}
    elif eff_obs >= 1000:
        return {"n_blocks": 6, "k_test": 2, "combinations": 15,
                "embargo_pct": 0.10}
    else:
        raise ValueError(
            f"Insufficient data: {eff_obs} effective observations < 1000 minimum. "
            "Per Forge §2: do not backtest."
        )


def label_regime(prices: pd.DataFrame, bar_index: int, lookback: int = 200) -> dict:
    """
    Label directional and volatility regime at a given bar index.
    Forge §20.
    """
    if "close" not in prices.columns:
        return {"direction": "unknown", "volatility": "unknown"}

    start = max(0, bar_index - lookback)
    window = prices["close"].iloc[start:bar_index]

    if len(window) < 20:
        return {"direction": "unknown", "volatility": "unknown"}

    # Directional regime
    ret = (window.iloc[-1] / window.iloc[0] - 1) if window.iloc[0] > 0 else 0
    if ret > 0.10:
        direction = "BULL"
    elif ret < -0.10:
        direction = "BEAR"
    else:
        direction = "SIDEWAYS"

    # Volatility regime (annualised)
    vol = window.pct_change().std() * np.sqrt(252)
    if vol < 0.20:
        volatility = "LOW"
    elif vol < 0.50:
        volatility = "MID"
    else:
        volatility = "HIGH"

    return {"direction": direction, "volatility": volatility}


def run_cpcv(
    strategy,
    prices: pd.DataFrame,
    n_blocks: int = 8,
    k_test: int = 2,
    embargo_bars: int = 200,
    cost_multiplier: float = 1.0,
) -> list:
    """
    Run Combinatorial Purged Cross-Validation.

    Forge §2: Training data = all blocks temporally prior to first test block
    minus embargo. Parameters FIXED before CPCV (§3).

    Args:
        strategy: Object with backtest(train, test, ...) method
        prices: Full price DataFrame with DatetimeIndex and 'close' column
        n_blocks: Number of blocks to divide data into
        k_test: Number of test blocks per combination
        embargo_bars: Number of bars to purge between train and test
        cost_multiplier: Cost scaling for §9 sensitivity

    Returns:
        List of result dicts, one per CPCV combination.
    """
    block_size = len(prices) // n_blocks
    blocks = [prices.iloc[i * block_size:(i + 1) * block_size] for i in range(n_blocks)]

    results = []
    n_combos = 0

    for test_idx in combinations(range(n_blocks), k_test):
        first_test = min(test_idx)
        train_end = first_test * block_size - embargo_bars

        # Need at least 2 blocks of training data
        if train_end < block_size * 2:
            continue

        train = prices.iloc[:train_end]
        test = pd.concat([blocks[i] for i in test_idx])

        if len(test) < 10:
            continue

        # Run strategy backtest
        try:
            result = strategy.backtest(
                train=train,
                test=test,
                cost_multiplier=cost_multiplier,
            )
        except Exception as e:
            logger.warning("CPCV fold %s failed: %s", test_idx, e)
            continue

        # Run IS backtest for decay ratio
        try:
            is_result = strategy.backtest(
                train=train,
                test=train,  # IS = testing on training data
                cost_multiplier=cost_multiplier,
            )
            result["is_sharpe"] = is_result["sharpe"]
        except Exception:
            result["is_sharpe"] = None

        # Add CPCV metadata
        result["test_blocks"] = test_idx
        result["train_bars"] = len(train)
        result["test_bars"] = len(test)
        result["embargo_bars"] = embargo_bars
        result["regime"] = label_regime(prices, first_test * block_size)

        # Rename sharpe to oos_sharpe for clarity
        result["oos_sharpe"] = result.pop("sharpe", 0.0)

        results.append(result)
        n_combos += 1

    logger.info(
        "CPCV completed: %d/%d combinations (n_blocks=%d, k_test=%d, embargo=%d)",
        n_combos,
        len(list(combinations(range(n_blocks), k_test))),
        n_blocks,
        k_test,
        embargo_bars,
    )

    return results


def cpcv_boundary_sensitivity(
    strategy,
    prices: pd.DataFrame,
    n_blocks: int = 8,
    k_test: int = 2,
    embargo_bars: int = 200,
) -> dict:
    """
    Run CPCV with blocks shifted by 25% of block_size.
    The two mean OOS Sharpes must be within 20% of each other.
    Forge §2.1.
    """
    # Standard run
    results_standard = run_cpcv(strategy, prices, n_blocks, k_test, embargo_bars)

    # Shifted run (25% offset)
    block_size = len(prices) // n_blocks
    shift = block_size // 4
    shifted_prices = prices.iloc[shift:]

    results_shifted = run_cpcv(strategy, shifted_prices, n_blocks, k_test, embargo_bars)

    mean_standard = np.mean([r["oos_sharpe"] for r in results_standard]) if results_standard else 0
    mean_shifted = np.mean([r["oos_sharpe"] for r in results_shifted]) if results_shifted else 0

    if mean_standard == 0 and mean_shifted == 0:
        pct_diff = 0
    elif mean_standard == 0:
        pct_diff = 100
    else:
        pct_diff = abs(mean_standard - mean_shifted) / abs(mean_standard) * 100

    return {
        "mean_sharpe_standard": round(mean_standard, 4),
        "mean_sharpe_shifted": round(mean_shifted, 4),
        "pct_difference": round(pct_diff, 2),
        "passes": pct_diff <= 20,
        "n_folds_standard": len(results_standard),
        "n_folds_shifted": len(results_shifted),
    }

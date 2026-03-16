"""
Perturbation & Robustness Testing

Implements Forge §10 robustness tests for MiroFish signals:
    - §10.1: Data perturbation (noise injection)
    - §10.2: Time shift test
    - §10.3: Subsample stability
    - §10.5: Randomised signal baseline (Mode A kill gate)
    - §10.6: Inverse signal test
    - §10.7: Latency sensitivity analysis
"""

import numpy as np
import pandas as pd
from typing import Optional

import logging

logger = logging.getLogger("forge_bridge.validation.perturbation")


def noise_perturbation_test(
    strategy,
    prices: pd.DataFrame,
    noise_level: float = 0.005,
    n_iterations: int = 100,
) -> dict:
    """
    Forge §10.1: Inject noise into returns and verify Sharpe stability.

    noise_level = 0.005 means 0.5% of return standard deviation.
    Sharpe must remain ≥ 70% of unperturbed.
    """
    # Baseline (unperturbed)
    base_result = strategy.backtest(train=prices, test=prices)
    base_sharpe = base_result["sharpe"]

    rng = np.random.default_rng(42)
    ret_std = prices["close"].pct_change().std()

    perturbed_sharpes = []
    for _ in range(n_iterations):
        perturbed = prices.copy()
        noise = rng.normal(0, noise_level * ret_std, len(perturbed))
        perturbed["close"] = perturbed["close"] * (1 + noise)

        result = strategy.backtest(train=perturbed, test=perturbed)
        perturbed_sharpes.append(result["sharpe"])

    mean_perturbed = np.mean(perturbed_sharpes)
    survival_ratio = mean_perturbed / base_sharpe if base_sharpe != 0 else 0

    return {
        "base_sharpe": round(base_sharpe, 4),
        "mean_perturbed_sharpe": round(mean_perturbed, 4),
        "survival_ratio": round(survival_ratio, 4),
        "noise_level": noise_level,
        "n_iterations": n_iterations,
    }


def time_shift_test(
    strategy,
    prices: pd.DataFrame,
    shift_bars: int = 1,
) -> dict:
    """
    Forge §10.2: Shift all input data forward by shift_bars.
    Sharpe must remain ≥ 50% of original.
    """
    base_result = strategy.backtest(train=prices, test=prices)
    base_sharpe = base_result["sharpe"]

    shifted = prices.copy()
    shifted["close"] = shifted["close"].shift(shift_bars).fillna(method="bfill")

    shifted_result = strategy.backtest(train=shifted, test=shifted)
    shifted_sharpe = shifted_result["sharpe"]

    survival_ratio = shifted_sharpe / base_sharpe if base_sharpe != 0 else 0

    return {
        "base_sharpe": round(base_sharpe, 4),
        "shifted_sharpe": round(shifted_sharpe, 4),
        "shift_bars": shift_bars,
        "survival_ratio": round(survival_ratio, 4),
    }


def subsample_stability_test(
    strategy,
    prices: pd.DataFrame,
    n_subperiods: int = 5,
) -> dict:
    """
    Forge §10.3 (v5.2 normalised gate):
    Divide into 5 equal sub-periods, run independently, check stability.
    """
    sub_size = len(prices) // n_subperiods
    sub_sharpes = []

    for i in range(n_subperiods):
        sub = prices.iloc[i * sub_size:(i + 1) * sub_size]
        if len(sub) < 10:
            continue
        result = strategy.backtest(train=sub, test=sub)
        sub_sharpes.append(result["sharpe"])

    if len(sub_sharpes) < 2:
        return {"error": "Insufficient sub-periods", "n_valid": len(sub_sharpes)}

    mean_sharpe = np.mean(sub_sharpes)
    std_sharpe = np.std(sub_sharpes)

    # Normalised gate (§10.3 v5.2)
    if abs(mean_sharpe) >= 0.75:
        # MRSS and CV
        min_sharpe = min(sub_sharpes)
        mrss = min_sharpe / mean_sharpe if mean_sharpe != 0 else 0
        cv = std_sharpe / abs(mean_sharpe) if mean_sharpe != 0 else float("inf")
        method = "normalised"
    else:
        # Low-Sharpe fallback: absolute std gate
        mrss = None
        cv = None
        method = "absolute"

    return {
        "sub_sharpes": [round(s, 4) for s in sub_sharpes],
        "mean_sharpe": round(mean_sharpe, 4),
        "std_sharpe": round(std_sharpe, 4),
        "mrss": round(mrss, 4) if mrss is not None else None,
        "cv": round(cv, 4) if cv is not None else None,
        "method": method,
        "n_subperiods": len(sub_sharpes),
    }


def randomised_baseline_test(
    strategy,
    prices: pd.DataFrame,
    n_iterations: int = 1000,
) -> dict:
    """
    Forge §10.5 Mode A (kill gate for informational signals):
    Replace entry signals with random entries, keep holding period and sizing.

    The signal must demonstrably beat random entries.
    """
    # Original strategy result
    original_result = strategy.backtest(train=prices, test=prices)
    original_sharpe = original_result["sharpe"]
    n_trades = original_result.get("n_trades", 10)
    avg_holding = original_result.get("avg_holding_bars", 20)

    # Run random baselines
    random_sharpes = []
    for _ in range(n_iterations):
        random_signals = strategy.generate_random_signals(
            n_bars=len(prices),
            n_signals=max(1, n_trades),
            avg_holding_bars=max(1, avg_holding),
        )

        # Convert bar indices to timestamps for the adapter
        idx = prices.index
        for sig in random_signals:
            entry = int(sig["ts_valid_from"])
            exit_bar = int(sig["ts_valid_until"])
            sig["ts_valid_from"] = idx[min(entry, len(idx) - 1)]
            sig["ts_valid_until"] = idx[min(exit_bar, len(idx) - 1)]

        result = strategy.backtest(
            train=prices, test=prices, override_signals=random_signals
        )
        random_sharpes.append(result["sharpe"])

    random_mean = np.mean(random_sharpes)
    random_std = np.std(random_sharpes) if np.std(random_sharpes) > 0 else 0.001

    signal_value_z = (original_sharpe - random_mean) / random_std
    pct_random_better = np.mean([r > original_sharpe for r in random_sharpes])

    signal_contribution_pct = (
        (original_sharpe - random_mean) / original_sharpe * 100
        if original_sharpe > 0
        else 0
    )
    exposure_contribution_pct = (
        random_mean / original_sharpe * 100 if original_sharpe > 0 else 100
    )

    return {
        "strategy_sharpe": round(original_sharpe, 4),
        "random_mean_sharpe": round(random_mean, 4),
        "random_std_sharpe": round(random_std, 4),
        "signal_value_z": round(signal_value_z, 4),
        "pct_random_better": round(pct_random_better, 4),
        "signal_contribution_pct": round(signal_contribution_pct, 2),
        "exposure_contribution_pct": round(exposure_contribution_pct, 2),
        "n_iterations": n_iterations,
        "mode": "A",  # kill gate for informational signals
    }


def inverse_signal_test(
    strategy,
    prices: pd.DataFrame,
) -> dict:
    """
    Forge §10.6: Flip all signals and check Sharpe.
    For informational strategies: inverse must have negative Sharpe.
    """
    original_result = strategy.backtest(train=prices, test=prices)
    inverse_result = strategy.backtest_inverse(train=prices, test=prices)

    original_sharpe = original_result["sharpe"]
    inverse_sharpe = inverse_result["sharpe"]
    differential = original_sharpe - inverse_sharpe

    return {
        "original_sharpe": round(original_sharpe, 4),
        "inverse_sharpe": round(inverse_sharpe, 4),
        "differential": round(differential, 4),
        "inverse_negative": inverse_sharpe < 0,
        "inverse_worse_than_original": inverse_sharpe < original_sharpe,
    }


def latency_sensitivity_test(
    strategy,
    prices: pd.DataFrame,
    delays_bars: Optional[list] = None,
) -> dict:
    """
    Forge §10.7: Test strategy performance with execution delays.
    Sharpe at +1 bar delay must be ≥ 70% of zero-delay Sharpe.
    """
    if delays_bars is None:
        delays_bars = [0, 1, 2, 5, 10]

    results = []
    for delay in delays_bars:
        result = strategy.backtest(
            train=prices, test=prices, execution_delay_bars=delay
        )
        results.append({"delay_bars": delay, "sharpe": result["sharpe"]})

    base_sharpe = results[0]["sharpe"] if results else 0
    half_sharpe = base_sharpe * 0.5

    alpha_halflife_bars = float("inf")
    for r in results:
        if r["sharpe"] <= half_sharpe:
            alpha_halflife_bars = r["delay_bars"]
            break

    # Sharpe at +1 bar delay
    sharpe_at_1 = next((r["sharpe"] for r in results if r["delay_bars"] == 1), base_sharpe)
    survival_at_1 = sharpe_at_1 / base_sharpe if base_sharpe != 0 else 0

    return {
        "results": results,
        "alpha_halflife_bars": alpha_halflife_bars,
        "is_latency_sensitive": alpha_halflife_bars <= 2,
        "sharpe_at_1_bar_delay": round(sharpe_at_1, 4),
        "survival_ratio_1_bar": round(survival_at_1, 4),
    }

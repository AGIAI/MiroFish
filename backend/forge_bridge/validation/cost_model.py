"""
Transaction Cost Model

Implements Forge §9 three-tier cost modelling with asset-class-specific defaults.

Tier levels:
    Baseline: Exchange fees + estimated half-spread
    Realistic: Baseline + slippage + funding
    Adversarial: Realistic × 2.0
"""

import numpy as np
from typing import Optional

import logging

logger = logging.getLogger("forge_bridge.validation.costs")

# Forge §9.6 asset-class-specific cost defaults
COST_DEFAULTS = {
    "crypto_perps":  {"maker_bps": 2,  "taker_bps": 5,  "spread_bps": 3,  "impact_k": 1.0},
    "crypto_spot":   {"maker_bps": 5,  "taker_bps": 10, "spread_bps": 10, "impact_k": 1.0},
    "us_large_cap":  {"maker_bps": 0,  "taker_bps": 0,  "spread_bps": 1,  "impact_k": 0.5},
    "us_small_cap":  {"maker_bps": 0,  "taker_bps": 0,  "spread_bps": 15, "impact_k": 2.0},
    "fx_majors":     {"maker_bps": 0,  "taker_bps": 0,  "spread_bps": 0.3, "impact_k": 0.3},
    "defi":          {"maker_bps": 15, "taker_bps": 15, "spread_bps": 50, "impact_k": 3.0},
}


def get_cost_tiers(asset_class: str) -> dict:
    """
    Compute the three cost tiers for a given asset class.
    Forge §9.1.
    """
    defaults = COST_DEFAULTS.get(asset_class, COST_DEFAULTS["crypto_perps"])

    baseline = defaults["taker_bps"] + defaults["spread_bps"] / 2
    realistic = baseline * 1.5  # includes slippage and funding estimates
    adversarial = realistic * 2.0

    return {
        "baseline_bps": round(baseline, 2),
        "realistic_bps": round(realistic, 2),
        "adversarial_bps": round(adversarial, 2),
        "asset_class": asset_class,
        "defaults": defaults,
    }


def cost_sensitivity_report(
    strategy,
    prices,
    asset_class: str,
) -> dict:
    """
    Forge §9.3: Report Sharpe at 0×, 1×, 2×, 3× costs.
    Report break-even cost (bps).
    """
    tiers = get_cost_tiers(asset_class)

    sharpes = {}
    for multiplier in [0, 1, 2, 3]:
        result = strategy.backtest(
            train=prices, test=prices, cost_multiplier=multiplier
        )
        sharpes[f"{multiplier}x"] = result["sharpe"]

    # Estimate break-even cost via interpolation
    # Break-even = cost level where Sharpe = 0
    if sharpes["0x"] > 0 and sharpes["3x"] < sharpes["0x"]:
        # Linear interpolation between cost levels
        slope = (sharpes["3x"] - sharpes["0x"]) / (3 * tiers["realistic_bps"])
        if slope < 0:
            breakeven_bps = -sharpes["0x"] / slope
        else:
            breakeven_bps = float("inf")
    else:
        breakeven_bps = float("inf") if sharpes["0x"] > 0 else 0

    headroom = breakeven_bps / tiers["realistic_bps"] if tiers["realistic_bps"] > 0 else float("inf")

    return {
        "sharpe_at_0x": round(sharpes["0x"], 4),
        "sharpe_at_1x": round(sharpes["1x"], 4),
        "sharpe_at_2x": round(sharpes["2x"], 4),
        "sharpe_at_3x": round(sharpes["3x"], 4),
        "breakeven_bps": round(breakeven_bps, 2) if breakeven_bps != float("inf") else 9999,
        "breakeven_headroom_x": round(headroom, 2) if headroom != float("inf") else 999,
        "cost_tiers": tiers,
    }


def turnover_budget(annual_turnover: float, cost_per_turn_bps: float,
                    gross_return_pct: float) -> dict:
    """Forge §9.8: Turnover cost budget."""
    annual_cost_pct = annual_turnover * cost_per_turn_bps / 100
    cost_drag_pct = (
        annual_cost_pct / gross_return_pct * 100 if gross_return_pct > 0 else float("inf")
    )
    return {
        "annual_cost_pct": round(annual_cost_pct, 4),
        "cost_as_pct_of_gross": round(cost_drag_pct, 2) if cost_drag_pct != float("inf") else 999,
        "viable": cost_drag_pct < 50,
    }


def capacity_estimation(breakeven_bps: float, impact_k: float,
                        adv_usd: float) -> dict:
    """Forge §9.5: Strategy capacity estimation."""
    if impact_k <= 0 or breakeven_bps <= 0:
        return {"capacity_usd": 0, "is_micro": True}

    capacity = (breakeven_bps / impact_k) ** 2 * adv_usd
    return {
        "capacity_usd": round(capacity, 0),
        "is_micro": capacity < 50000,
    }

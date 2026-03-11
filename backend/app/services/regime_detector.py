"""
Market Regime Detector
Detects the current market regime using an ensemble of statistical methods:
- ADX for trend strength
- ATR percentile for volatility regime
- Hurst exponent for trending vs mean-reverting
- Volume profile for breakout detection
- Autocorrelation for directional persistence

Adjusts simulation parameters based on detected regime.
"""

import math
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime

from ..models.signal import MarketRegime
from ..utils.logger import get_logger

logger = get_logger('mirofish.regime')


class RegimeDetector:
    """
    Detects market regime from OHLCV data using pure math (no external TA libs).
    Each method votes on the regime; the ensemble determines the final call.
    """

    def detect(
        self,
        ohlcv: List[Dict],
        lookback: int = 50,
    ) -> Dict[str, Any]:
        """
        Detect current market regime from OHLCV data.

        Args:
            ohlcv: List of {open, high, low, close, volume} dicts (oldest first)
            lookback: Number of bars to analyze

        Returns:
            Dict with regime, confidence, components, and agent adjustments
        """
        if len(ohlcv) < lookback:
            return {
                "regime": MarketRegime.RANGING.value,
                "confidence": 0.0,
                "reason": f"Insufficient data ({len(ohlcv)} bars, need {lookback})",
                "components": {},
                "agent_adjustments": {},
            }

        data = ohlcv[-lookback:]
        closes = [bar["close"] for bar in data]
        highs = [bar["high"] for bar in data]
        lows = [bar["low"] for bar in data]
        volumes = [bar.get("volume", 0) for bar in data]

        # Component signals
        adx_result = self._calculate_adx(highs, lows, closes, period=14)
        volatility = self._volatility_regime(highs, lows, closes)
        hurst = self._hurst_exponent(closes)
        volume_profile = self._volume_analysis(volumes)
        direction = self._direction_analysis(closes)

        # Ensemble voting
        regime, confidence, reason = self._ensemble_vote(
            adx_result, volatility, hurst, volume_profile, direction
        )

        components = {
            "adx": adx_result,
            "volatility": volatility,
            "hurst_exponent": hurst,
            "volume": volume_profile,
            "direction": direction,
        }

        # Agent behavior adjustments for this regime
        adjustments = REGIME_ADJUSTMENTS.get(regime, {})

        return {
            "regime": regime,
            "confidence": round(confidence, 3),
            "reason": reason,
            "components": components,
            "agent_adjustments": adjustments,
            "timestamp": datetime.utcnow().isoformat(),
        }

    def _calculate_adx(
        self, highs: List[float], lows: List[float], closes: List[float], period: int = 14
    ) -> Dict[str, float]:
        """Average Directional Index — measures trend strength (0-100)."""
        n = len(closes)
        if n < period + 1:
            return {"adx": 0, "plus_di": 0, "minus_di": 0}

        # True Range
        tr_list = []
        plus_dm_list = []
        minus_dm_list = []

        for i in range(1, n):
            high_diff = highs[i] - highs[i - 1]
            low_diff = lows[i - 1] - lows[i]

            plus_dm = max(high_diff, 0) if high_diff > low_diff else 0
            minus_dm = max(low_diff, 0) if low_diff > high_diff else 0

            tr = max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i - 1]),
                abs(lows[i] - closes[i - 1])
            )

            tr_list.append(tr)
            plus_dm_list.append(plus_dm)
            minus_dm_list.append(minus_dm)

        # Smoothed averages (Wilder's smoothing)
        atr = sum(tr_list[:period]) / period
        plus_dm_avg = sum(plus_dm_list[:period]) / period
        minus_dm_avg = sum(minus_dm_list[:period]) / period

        dx_values = []

        for i in range(period, len(tr_list)):
            atr = (atr * (period - 1) + tr_list[i]) / period
            plus_dm_avg = (plus_dm_avg * (period - 1) + plus_dm_list[i]) / period
            minus_dm_avg = (minus_dm_avg * (period - 1) + minus_dm_list[i]) / period

            if atr > 0:
                plus_di = (plus_dm_avg / atr) * 100
                minus_di = (minus_dm_avg / atr) * 100
            else:
                plus_di = minus_di = 0

            di_sum = plus_di + minus_di
            if di_sum > 0:
                dx = abs(plus_di - minus_di) / di_sum * 100
            else:
                dx = 0
            dx_values.append(dx)

        adx = sum(dx_values[-period:]) / period if len(dx_values) >= period else (
            sum(dx_values) / len(dx_values) if dx_values else 0
        )

        return {
            "adx": round(adx, 2),
            "plus_di": round(plus_di, 2) if 'plus_di' in dir() else 0,
            "minus_di": round(minus_di, 2) if 'minus_di' in dir() else 0,
            "trending": adx > 25,
        }

    def _volatility_regime(
        self, highs: List[float], lows: List[float], closes: List[float]
    ) -> Dict[str, float]:
        """Classify volatility using ATR percentile and realized vol."""
        n = len(closes)

        # ATR
        tr_list = [highs[0] - lows[0]]
        for i in range(1, n):
            tr = max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i - 1]),
                abs(lows[i] - closes[i - 1])
            )
            tr_list.append(tr)

        atr_14 = sum(tr_list[-14:]) / 14 if len(tr_list) >= 14 else sum(tr_list) / len(tr_list)
        avg_price = sum(closes) / len(closes)
        atr_pct = (atr_14 / avg_price * 100) if avg_price > 0 else 0

        # Realized volatility (annualized)
        returns = [(closes[i] / closes[i - 1] - 1) for i in range(1, n)]
        if returns:
            mean_ret = sum(returns) / len(returns)
            variance = sum((r - mean_ret) ** 2 for r in returns) / len(returns)
            daily_vol = math.sqrt(variance)
            annualized_vol = daily_vol * math.sqrt(252) * 100
        else:
            daily_vol = 0
            annualized_vol = 0

        # Percentile rank ATR vs recent history
        sorted_trs = sorted(tr_list)
        atr_percentile = sorted_trs.index(min(sorted_trs, key=lambda x: abs(x - atr_14))) / len(sorted_trs) * 100

        regime = "low" if atr_percentile < 30 else "medium" if atr_percentile < 70 else "high"

        return {
            "atr_14": round(atr_14, 4),
            "atr_pct": round(atr_pct, 4),
            "atr_percentile": round(atr_percentile, 1),
            "realized_vol_annualized": round(annualized_vol, 2),
            "regime": regime,
        }

    def _hurst_exponent(self, closes: List[float], max_lag: int = 20) -> Dict[str, Any]:
        """
        Hurst exponent via R/S analysis.
        H < 0.5: mean-reverting
        H = 0.5: random walk
        H > 0.5: trending
        """
        n = len(closes)
        if n < max_lag * 2:
            return {"hurst": 0.5, "interpretation": "insufficient_data"}

        returns = [math.log(closes[i] / closes[i - 1]) for i in range(1, n) if closes[i - 1] > 0]

        if len(returns) < max_lag:
            return {"hurst": 0.5, "interpretation": "insufficient_data"}

        lags = range(2, max_lag + 1)
        rs_values = []

        for lag in lags:
            chunks = [returns[i:i + lag] for i in range(0, len(returns) - lag + 1, lag)]
            if not chunks:
                continue

            rs_list = []
            for chunk in chunks:
                if len(chunk) < 2:
                    continue
                mean = sum(chunk) / len(chunk)
                deviations = [x - mean for x in chunk]
                cumulative = []
                s = 0
                for d in deviations:
                    s += d
                    cumulative.append(s)
                r = max(cumulative) - min(cumulative)
                std = math.sqrt(sum(d ** 2 for d in deviations) / len(deviations))
                if std > 0:
                    rs_list.append(r / std)

            if rs_list:
                rs_values.append((math.log(lag), math.log(sum(rs_list) / len(rs_list))))

        if len(rs_values) < 3:
            return {"hurst": 0.5, "interpretation": "insufficient_data"}

        # Linear regression for slope (Hurst exponent)
        x_vals = [p[0] for p in rs_values]
        y_vals = [p[1] for p in rs_values]
        n_pts = len(x_vals)
        x_mean = sum(x_vals) / n_pts
        y_mean = sum(y_vals) / n_pts

        numerator = sum((x_vals[i] - x_mean) * (y_vals[i] - y_mean) for i in range(n_pts))
        denominator = sum((x_vals[i] - x_mean) ** 2 for i in range(n_pts))

        hurst = numerator / denominator if denominator != 0 else 0.5
        hurst = max(0.0, min(1.0, hurst))

        if hurst < 0.4:
            interp = "mean_reverting"
        elif hurst > 0.6:
            interp = "trending"
        else:
            interp = "random_walk"

        return {"hurst": round(hurst, 3), "interpretation": interp}

    def _volume_analysis(self, volumes: List[float]) -> Dict[str, Any]:
        """Analyze volume for breakout and exhaustion signals."""
        if not volumes or all(v == 0 for v in volumes):
            return {"avg_volume": 0, "volume_trend": "flat", "breakout_signal": False}

        n = len(volumes)
        avg_vol = sum(volumes) / n
        recent_avg = sum(volumes[-5:]) / min(5, n)
        earlier_avg = sum(volumes[:n // 2]) / max(n // 2, 1)

        # Volume ratio
        vol_ratio = recent_avg / avg_vol if avg_vol > 0 else 1

        # Volume trend
        if vol_ratio > 1.5:
            trend = "surging"
        elif vol_ratio > 1.1:
            trend = "increasing"
        elif vol_ratio < 0.7:
            trend = "declining"
        else:
            trend = "stable"

        # Breakout: volume spike > 2x average
        breakout = vol_ratio > 2.0

        return {
            "avg_volume": round(avg_vol, 2),
            "recent_volume_ratio": round(vol_ratio, 3),
            "volume_trend": trend,
            "breakout_signal": breakout,
        }

    def _direction_analysis(self, closes: List[float]) -> Dict[str, Any]:
        """Determine directional bias from price action."""
        n = len(closes)
        if n < 5:
            return {"direction": "neutral", "strength": 0}

        # Simple return over period
        total_return = (closes[-1] / closes[0] - 1) * 100 if closes[0] > 0 else 0

        # Recent momentum (last 20% of bars)
        recent_start = max(0, n - n // 5)
        recent_return = (closes[-1] / closes[recent_start] - 1) * 100 if closes[recent_start] > 0 else 0

        # Count up vs down bars
        up_bars = sum(1 for i in range(1, n) if closes[i] > closes[i - 1])
        down_bars = n - 1 - up_bars
        up_ratio = up_bars / (n - 1) if n > 1 else 0.5

        if total_return > 3 and up_ratio > 0.55:
            direction = "bullish"
        elif total_return < -3 and up_ratio < 0.45:
            direction = "bearish"
        else:
            direction = "neutral"

        strength = min(1.0, abs(total_return) / 10)

        return {
            "direction": direction,
            "total_return_pct": round(total_return, 2),
            "recent_return_pct": round(recent_return, 2),
            "up_bar_ratio": round(up_ratio, 3),
            "strength": round(strength, 3),
        }

    def _ensemble_vote(
        self,
        adx: Dict,
        volatility: Dict,
        hurst: Dict,
        volume: Dict,
        direction: Dict,
    ) -> Tuple[str, float, str]:
        """Ensemble of all components to determine final regime."""

        adx_val = adx.get("adx", 0)
        vol_regime = volatility.get("regime", "medium")
        hurst_val = hurst.get("hurst", 0.5)
        hurst_interp = hurst.get("interpretation", "random_walk")
        vol_breakout = volume.get("breakout_signal", False)
        dir_val = direction.get("direction", "neutral")
        dir_strength = direction.get("strength", 0)

        # Crisis detection: high vol + bearish + declining volume
        if vol_regime == "high" and dir_val == "bearish" and dir_strength > 0.5:
            return MarketRegime.CRISIS.value, 0.8, "High volatility with strong bearish direction"

        # Breakout: volume surge + trending
        if vol_breakout and adx_val > 20:
            if dir_val == "bullish":
                return MarketRegime.TRENDING_BULL.value, 0.75, "Volume breakout with bullish momentum"
            elif dir_val == "bearish":
                return MarketRegime.TRENDING_BEAR.value, 0.75, "Volume breakout with bearish momentum"

        # Strong trend
        if adx_val > 30 and hurst_interp == "trending":
            if dir_val == "bullish":
                return MarketRegime.TRENDING_BULL.value, 0.85, f"Strong trend (ADX={adx_val:.0f}, Hurst={hurst_val:.2f})"
            elif dir_val == "bearish":
                return MarketRegime.TRENDING_BEAR.value, 0.85, f"Strong downtrend (ADX={adx_val:.0f}, Hurst={hurst_val:.2f})"
            else:
                return MarketRegime.TRENDING_BULL.value, 0.6, f"Trending but directionless (ADX={adx_val:.0f})"

        # Mean reverting
        if hurst_interp == "mean_reverting" and adx_val < 25:
            return MarketRegime.RANGING.value, 0.7, f"Mean-reverting range (Hurst={hurst_val:.2f}, ADX={adx_val:.0f})"

        # High volatility chop
        if vol_regime == "high" and adx_val < 25:
            return MarketRegime.HIGH_VOL.value, 0.7, "High volatility without clear trend"

        # Recovery: coming off lows with increasing volume
        if dir_val == "bullish" and volume.get("volume_trend") == "increasing":
            return MarketRegime.RECOVERY.value, 0.55, "Bullish direction with increasing volume"

        # Default: ranging
        return MarketRegime.RANGING.value, 0.5, f"No strong regime signal (ADX={adx_val:.0f})"


# ============================================================================
# Agent behavior adjustments per regime
# ============================================================================

REGIME_ADJUSTMENTS = {
    MarketRegime.TRENDING_BULL.value: {
        "description": "Momentum traders dominate. Increase conviction, wider stops.",
        "momentum": {"conviction_boost": 0.15, "position_size_mult": 1.3},
        "mean_reversion": {"conviction_boost": -0.15, "position_size_mult": 0.5},
        "narrative": {"conviction_boost": 0.1, "position_size_mult": 1.2},
        "retail": {"fomo_boost": 0.2},
        "whale": {"position_size_mult": 1.2},
        "stop_loss_atr_mult": 2.5,
        "take_profit_mult": 1.5,
    },
    MarketRegime.TRENDING_BEAR.value: {
        "description": "Bearish trend. Shorts dominate, panic selling increases.",
        "momentum": {"conviction_boost": 0.1, "position_size_mult": 1.2},
        "mean_reversion": {"conviction_boost": -0.1, "position_size_mult": 0.6},
        "retail": {"panic_boost": 0.3, "fomo_boost": -0.2},
        "whale": {"position_size_mult": 0.8},
        "stop_loss_atr_mult": 2.0,
        "take_profit_mult": 1.3,
    },
    MarketRegime.RANGING.value: {
        "description": "Mean-reversion works. Tight stops, fade extremes.",
        "momentum": {"conviction_boost": -0.15, "position_size_mult": 0.6},
        "mean_reversion": {"conviction_boost": 0.2, "position_size_mult": 1.4},
        "market_maker": {"position_size_mult": 1.3},
        "stop_loss_atr_mult": 1.5,
        "take_profit_mult": 0.8,
    },
    MarketRegime.HIGH_VOL.value: {
        "description": "All agents underperform. Reduce size, tighten stops.",
        "momentum": {"position_size_mult": 0.5},
        "mean_reversion": {"position_size_mult": 0.5},
        "narrative": {"position_size_mult": 0.5},
        "retail": {"panic_boost": 0.2},
        "market_maker": {"position_size_mult": 0.3},
        "stop_loss_atr_mult": 3.0,
        "take_profit_mult": 1.0,
    },
    MarketRegime.CRISIS.value: {
        "description": "Liquidity crisis. Market makers withdraw, spreads widen.",
        "momentum": {"position_size_mult": 0.3},
        "mean_reversion": {"position_size_mult": 0.2},
        "narrative": {"position_size_mult": 0.3},
        "retail": {"panic_boost": 0.5, "fomo_boost": -0.3},
        "market_maker": {"position_size_mult": 0.1},
        "whale": {"position_size_mult": 0.5},
        "stop_loss_atr_mult": 4.0,
        "take_profit_mult": 0.5,
    },
    MarketRegime.RECOVERY.value: {
        "description": "Early recovery. Cautious buying, improving sentiment.",
        "momentum": {"conviction_boost": 0.05, "position_size_mult": 1.0},
        "mean_reversion": {"conviction_boost": 0.1, "position_size_mult": 1.1},
        "fundamental": {"conviction_boost": 0.15, "position_size_mult": 1.3},
        "retail": {"fomo_boost": 0.1},
        "stop_loss_atr_mult": 2.0,
        "take_profit_mult": 1.2,
    },
}

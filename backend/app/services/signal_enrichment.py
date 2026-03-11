"""
Signal Enrichment Layer
The connective tissue between data collection and signal extraction.

Blends ALL available data sources into a unified enrichment context:
- Market data (OHLCV, order book depth)
- News sentiment (CryptoPanic, RSS, Reddit)
- On-chain data (whale flows, gas, mempool)
- External signals (from OpenClaw/Meridian)
- Portfolio state (existing positions, correlation risk)
- Real order book snapshots (from execution venue)

The enrichment context feeds directly into SignalExtractor, replacing
the disconnected market-data-only path with a fully integrated one.
"""

import math
import time
from datetime import datetime
from typing import Dict, List, Optional, Any

from ..utils.logger import get_logger

logger = get_logger('mirofish.enrichment')


class EnrichmentContext:
    """
    Unified data context for signal generation.
    All data sources merge here before signal extraction.
    """

    def __init__(self):
        # Core market data
        self.asset: str = ""
        self.asset_type: str = "crypto"
        self.current_price: float = 0
        self.ohlcv: List[Dict] = []
        self.atr: float = 0

        # Regime
        self.regime: Dict[str, Any] = {}

        # News & sentiment (blended score: -1 to +1)
        self.news_sentiment_score: float = 0.0
        self.news_volume: int = 0
        self.news_sources: List[Dict] = []

        # On-chain data
        self.whale_net_flow: float = 0.0  # Positive = inflow (bullish)
        self.gas_regime: str = "normal"  # low, normal, high, extreme
        self.mempool_pressure: float = 0.0  # 0 = empty, 1 = congested

        # Real order book (from execution venue or ccxt)
        self.real_bid_depth: float = 0
        self.real_ask_depth: float = 0
        self.real_spread_bps: float = 0
        self.real_mid_price: float = 0

        # External signals (from OpenClaw)
        self.external_signals: List[Dict] = []
        self.external_consensus: Optional[float] = None  # 0-1 if external signals exist

        # Portfolio context
        self.has_existing_position: bool = False
        self.existing_position_direction: Optional[str] = None
        self.existing_position_pnl_pct: float = 0.0
        self.portfolio_daily_pnl_pct: float = 0.0
        self.portfolio_drawdown_pct: float = 0.0
        self.consecutive_losses: int = 0
        self.correlated_exposure_pct: float = 0.0  # % of portfolio in correlated assets

        # Execution quality (from feedback loop)
        self.avg_slippage_pct: float = 0.0
        self.recent_win_rate: float = 0.5

        # Timestamps
        self.enriched_at: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "asset": self.asset,
            "asset_type": self.asset_type,
            "current_price": self.current_price,
            "atr": self.atr,
            "regime": self.regime,
            "news_sentiment_score": self.news_sentiment_score,
            "news_volume": self.news_volume,
            "whale_net_flow": self.whale_net_flow,
            "gas_regime": self.gas_regime,
            "mempool_pressure": self.mempool_pressure,
            "real_bid_depth": self.real_bid_depth,
            "real_ask_depth": self.real_ask_depth,
            "real_spread_bps": self.real_spread_bps,
            "external_consensus": self.external_consensus,
            "has_existing_position": self.has_existing_position,
            "existing_position_direction": self.existing_position_direction,
            "portfolio_drawdown_pct": self.portfolio_drawdown_pct,
            "consecutive_losses": self.consecutive_losses,
            "correlated_exposure_pct": self.correlated_exposure_pct,
            "avg_slippage_pct": self.avg_slippage_pct,
            "recent_win_rate": self.recent_win_rate,
            "enriched_at": self.enriched_at,
        }


class SignalEnrichmentService:
    """
    Pulls data from all available sources and builds an EnrichmentContext
    that the SignalExtractor uses for fully-integrated signal generation.

    This replaces the disconnected "market data only" path.
    """

    def __init__(self):
        # Lazy-loaded services
        self._market_data = None
        self._news_feed = None
        self._onchain = None
        self._external = None

    def _get_market_data(self):
        if self._market_data is None:
            from .market_data_service import MarketDataService
            self._market_data = MarketDataService()
        return self._market_data

    def _get_news_feed(self):
        if self._news_feed is None:
            from .news_feed_service import NewsFeedService
            self._news_feed = NewsFeedService()
        return self._news_feed

    def _get_onchain(self):
        if self._onchain is None:
            from .onchain_data_service import OnChainDataService
            self._onchain = OnChainDataService()
        return self._onchain

    def _get_external(self):
        if self._external is None:
            from .external_ingestion import ExternalIngestionService
            self._external = ExternalIngestionService()
        return self._external

    def enrich(
        self,
        asset: str,
        asset_type: str = "crypto",
        exchange: Optional[str] = None,
        timeframe: str = "1h",
        bars: int = 100,
        include_news: bool = True,
        include_onchain: bool = True,
        include_external: bool = True,
        include_portfolio: bool = True,
    ) -> EnrichmentContext:
        """
        Build a fully enriched context for signal generation.

        This is the main entry point. Collects data from ALL sources,
        handles failures gracefully (each source is optional).

        Args:
            asset: Trading pair (e.g. "BTC/USDT" or "AAPL")
            asset_type: "crypto" or "equity"
            exchange: Target exchange (optional)
            timeframe: OHLCV timeframe
            bars: Number of bars
            include_news: Whether to fetch news sentiment
            include_onchain: Whether to fetch on-chain data
            include_external: Whether to include external signals
            include_portfolio: Whether to include portfolio state

        Returns:
            EnrichmentContext with all available data blended
        """
        ctx = EnrichmentContext()
        ctx.asset = asset
        ctx.asset_type = asset_type

        # 1. Core market data (required)
        self._enrich_market_data(ctx, asset, asset_type, exchange, timeframe, bars)

        # 2. Regime detection
        self._enrich_regime(ctx)

        # 3. Real order book depth
        self._enrich_order_book(ctx, asset, asset_type, exchange)

        # 4. News & sentiment
        if include_news:
            self._enrich_news_sentiment(ctx, asset, asset_type)

        # 5. On-chain data
        if include_onchain and asset_type == "crypto":
            self._enrich_onchain(ctx, asset)

        # 6. External signals & portfolio
        if include_external:
            self._enrich_external_signals(ctx, asset)

        if include_portfolio:
            self._enrich_portfolio(ctx, asset)

        # 7. Execution quality feedback
        if include_external:
            self._enrich_execution_quality(ctx)

        ctx.enriched_at = datetime.utcnow().isoformat()

        logger.info(
            f"Enrichment complete for {asset}: "
            f"price={ctx.current_price} regime={ctx.regime.get('regime', '?')} "
            f"news_sentiment={ctx.news_sentiment_score:+.2f} "
            f"whale_flow={ctx.whale_net_flow:+.2f} "
            f"external_signals={len(ctx.external_signals)} "
            f"has_position={ctx.has_existing_position}"
        )

        return ctx

    # ========================================================================
    # Individual enrichment steps (each handles its own errors)
    # ========================================================================

    def _enrich_market_data(
        self, ctx: EnrichmentContext, asset: str, asset_type: str,
        exchange: Optional[str], timeframe: str, bars: int,
    ):
        """Fetch OHLCV and compute ATR."""
        svc = self._get_market_data()

        try:
            if asset_type == "crypto":
                ctx.ohlcv = svc.crypto.get_ohlcv(
                    asset, timeframe=timeframe, limit=bars, exchange_id=exchange
                )
            elif asset_type == "equity":
                ctx.ohlcv = svc.equity.get_ohlcv(asset, period="3mo", interval="1d")
        except Exception as e:
            logger.warning(f"OHLCV fetch failed for {asset}: {e}")
            ctx.ohlcv = []

        if ctx.ohlcv:
            ctx.current_price = ctx.ohlcv[-1]["close"]
            ctx.atr = self._compute_atr(ctx.ohlcv)

    def _enrich_regime(self, ctx: EnrichmentContext):
        """Detect market regime from OHLCV."""
        if len(ctx.ohlcv) < 30:
            ctx.regime = {"regime": "ranging", "confidence": 0}
            return

        try:
            from .regime_detector import RegimeDetector
            detector = RegimeDetector()
            ctx.regime = detector.detect(ctx.ohlcv)
        except Exception as e:
            logger.warning(f"Regime detection failed: {e}")
            ctx.regime = {"regime": "ranging", "confidence": 0}

    def _enrich_order_book(
        self, ctx: EnrichmentContext, asset: str, asset_type: str,
        exchange: Optional[str],
    ):
        """Get real order book depth — first from external ingestion, then from ccxt."""
        # Try real snapshot from OpenClaw first
        try:
            external = self._get_external()
            snapshot = external.get_order_book_snapshot(asset)
            if snapshot:
                ctx.real_bid_depth = snapshot.total_bid_depth
                ctx.real_ask_depth = snapshot.total_ask_depth
                ctx.real_spread_bps = snapshot.spread_bps
                ctx.real_mid_price = snapshot.mid_price
                return
        except Exception:
            pass

        # Fall back to live ccxt fetch
        if asset_type == "crypto":
            try:
                svc = self._get_market_data()
                book = svc.crypto.get_order_book(asset, limit=20, exchange_id=exchange)
                ctx.real_bid_depth = sum(q for _, q in book.get("bids", []))
                ctx.real_ask_depth = sum(q for _, q in book.get("asks", []))
                ctx.real_spread_bps = book.get("spread_pct", 0) * 100
                ctx.real_mid_price = book.get("mid_price", ctx.current_price)
            except Exception as e:
                logger.debug(f"Order book fetch failed for {asset}: {e}")

    def _enrich_news_sentiment(
        self, ctx: EnrichmentContext, asset: str, asset_type: str,
    ):
        """Aggregate news sentiment from all available sources."""
        svc = self._get_news_feed()
        sentiment_scores = []
        total_items = 0

        # CryptoPanic (crypto only)
        if asset_type == "crypto" and svc.cryptopanic.available:
            try:
                # Extract coin code from pair (e.g. "BTC/USDT" → "BTC")
                coin_code = asset.split("/")[0] if "/" in asset else asset
                news = svc.cryptopanic.get_news(currencies=coin_code, limit=20)
                total_items += len(news)

                for item in news:
                    votes = item.get("votes", {})
                    if votes:
                        pos = votes.get("positive", 0) + votes.get("liked", 0)
                        neg = votes.get("negative", 0) + votes.get("toxic", 0)
                        total = pos + neg
                        if total > 0:
                            sentiment_scores.append((pos - neg) / total)
            except Exception as e:
                logger.debug(f"CryptoPanic sentiment failed: {e}")

        # Reddit sentiment
        if svc.reddit.available:
            try:
                if asset_type == "crypto":
                    subs = ["cryptocurrency", "bitcoin"] if "BTC" in asset else ["cryptocurrency"]
                else:
                    subs = ["stocks", "wallstreetbets"]

                for sub in subs:
                    try:
                        posts = svc.reddit.get_hot_posts(sub, limit=10)
                        total_items += len(posts)
                        # Score-weighted sentiment from upvote ratio
                        for post in posts:
                            title_lower = post.get("title", "").lower()
                            ticker = asset.split("/")[0].lower() if "/" in asset else asset.lower()
                            if ticker in title_lower:
                                ratio = post.get("upvote_ratio", 0.5)
                                score = post.get("score", 0)
                                # High upvote ratio + high score = strong sentiment
                                sentiment = (ratio - 0.5) * 2  # Map 0-1 to -1 to +1
                                weight = min(1.0, math.log1p(score) / 10)
                                sentiment_scores.append(sentiment * weight)
                        time.sleep(0.3)
                    except Exception:
                        pass
            except Exception as e:
                logger.debug(f"Reddit sentiment failed: {e}")

        # Aggregate
        if sentiment_scores:
            ctx.news_sentiment_score = round(
                sum(sentiment_scores) / len(sentiment_scores), 3
            )
        ctx.news_volume = total_items
        ctx.news_sources = [{"score": s} for s in sentiment_scores[:10]]

    def _enrich_onchain(self, ctx: EnrichmentContext, asset: str):
        """On-chain metrics for crypto assets."""
        svc = self._get_onchain()

        # Bitcoin-specific
        if "BTC" in asset.upper():
            try:
                mempool = svc.bitcoin.get_mempool_info()
                unconfirmed = mempool.get("unconfirmed_txs", 0)
                # Normalize: 50k+ txs = congested
                ctx.mempool_pressure = min(1.0, unconfirmed / 50000)
            except Exception as e:
                logger.debug(f"BTC mempool failed: {e}")

        # Ethereum-specific
        if "ETH" in asset.upper() and svc.etherscan.available:
            try:
                gas = svc.etherscan.get_gas_oracle()
                avg_gas = gas.get("average_gwei", 0)
                if avg_gas < 10:
                    ctx.gas_regime = "low"
                elif avg_gas < 30:
                    ctx.gas_regime = "normal"
                elif avg_gas < 100:
                    ctx.gas_regime = "high"
                else:
                    ctx.gas_regime = "extreme"
            except Exception as e:
                logger.debug(f"ETH gas failed: {e}")

        # Whale flow tracking would need known whale addresses
        # For now, use bid/ask imbalance as proxy
        if ctx.real_bid_depth > 0 and ctx.real_ask_depth > 0:
            total = ctx.real_bid_depth + ctx.real_ask_depth
            ctx.whale_net_flow = round(
                (ctx.real_bid_depth - ctx.real_ask_depth) / total, 3
            )

    def _enrich_external_signals(self, ctx: EnrichmentContext, asset: str):
        """Get external signals from OpenClaw for blending."""
        try:
            external = self._get_external()
            signals = external.get_external_signals_for_asset(asset)
            ctx.external_signals = [s.to_dict() for s in signals]

            if signals:
                # Calculate consensus from external signals
                long_count = sum(1 for s in signals if s.direction == "LONG")
                ctx.external_consensus = long_count / len(signals)
        except Exception as e:
            logger.debug(f"External signals failed: {e}")

    def _enrich_portfolio(self, ctx: EnrichmentContext, asset: str):
        """Portfolio state for risk-aware signal generation."""
        try:
            external = self._get_external()
            portfolio = external.get_portfolio_state()
            if not portfolio:
                return

            ctx.portfolio_daily_pnl_pct = portfolio.daily_pnl_pct
            ctx.portfolio_drawdown_pct = portfolio.max_drawdown_pct
            ctx.consecutive_losses = portfolio.consecutive_losses

            # Check for existing position in this asset
            positions = external.get_positions_for_asset(asset)
            if positions:
                pos = positions[0]
                ctx.has_existing_position = True
                ctx.existing_position_direction = pos.direction.value
                ctx.existing_position_pnl_pct = pos.unrealized_pnl_pct

            # Calculate correlated exposure
            # Simple: count total position size in same asset class
            total_exposure = sum(p.size_pct for p in portfolio.positions)
            ctx.correlated_exposure_pct = total_exposure

        except Exception as e:
            logger.debug(f"Portfolio enrichment failed: {e}")

    def _enrich_execution_quality(self, ctx: EnrichmentContext):
        """Execution quality metrics from historical fills."""
        try:
            external = self._get_external()
            stats = external.get_execution_stats()

            if "fills" in stats:
                ctx.avg_slippage_pct = stats["fills"].get("avg_slippage_pct", 0)
            if "trades" in stats:
                ctx.recent_win_rate = stats["trades"].get("win_rate", 0.5)
        except Exception as e:
            logger.debug(f"Execution quality enrichment failed: {e}")

    # ========================================================================
    # Helpers
    # ========================================================================

    def _compute_atr(self, ohlcv: List[Dict], period: int = 14) -> float:
        """Compute Average True Range."""
        if len(ohlcv) < 2:
            return ohlcv[0]["high"] - ohlcv[0]["low"] if ohlcv else 0

        trs = [ohlcv[0]["high"] - ohlcv[0]["low"]]
        for i in range(1, len(ohlcv)):
            tr = max(
                ohlcv[i]["high"] - ohlcv[i]["low"],
                abs(ohlcv[i]["high"] - ohlcv[i - 1]["close"]),
                abs(ohlcv[i]["low"] - ohlcv[i - 1]["close"]),
            )
            trs.append(tr)

        return sum(trs[-period:]) / min(period, len(trs))

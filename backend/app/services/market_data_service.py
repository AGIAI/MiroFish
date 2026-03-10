"""
Market Data Service
Provides unified access to free market data sources:
- yfinance: Stock/ETF OHLCV, fundamentals
- ccxt: Crypto OHLCV, order books, tickers (100+ exchanges)
- FRED: Macro economic indicators (CPI, GDP, rates)
- CoinGecko: Crypto market caps, trending, global stats
"""

import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any

from ..config import Config
from ..utils.logger import get_logger

logger = get_logger('mirofish.market_data')


class SimpleCache:
    """TTL-based in-memory cache to avoid hammering free APIs."""

    def __init__(self, ttl: int = 300):
        self._store: Dict[str, Any] = {}
        self._expiry: Dict[str, float] = {}
        self.ttl = ttl

    def get(self, key: str) -> Optional[Any]:
        if key in self._store and time.time() < self._expiry.get(key, 0):
            return self._store[key]
        self._store.pop(key, None)
        self._expiry.pop(key, None)
        return None

    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        self._store[key] = value
        self._expiry[key] = time.time() + (ttl or self.ttl)

    def clear(self):
        self._store.clear()
        self._expiry.clear()


# ---------------------------------------------------------------------------
# Equity adapter (yfinance — free, no key needed)
# ---------------------------------------------------------------------------
class EquityAdapter:
    """Stock/ETF data via yfinance."""

    def __init__(self):
        try:
            import yfinance  # noqa: F401
            self._yf = yfinance
            self.available = True
        except ImportError:
            self._yf = None
            self.available = False
            logger.warning("yfinance not installed — equity data unavailable. pip install yfinance")

    def get_ohlcv(
        self,
        ticker: str,
        period: str = "1mo",
        interval: str = "1d",
    ) -> List[Dict]:
        """
        Fetch OHLCV bars.

        Args:
            ticker: Stock symbol (e.g. "AAPL", "SPY")
            period: Data period — 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max
            interval: Bar interval — 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo

        Returns:
            List of OHLCV dicts
        """
        if not self.available:
            raise RuntimeError("yfinance is not installed")

        t = self._yf.Ticker(ticker)
        df = t.history(period=period, interval=interval)

        if df.empty:
            return []

        bars = []
        for idx, row in df.iterrows():
            bars.append({
                "timestamp": idx.isoformat(),
                "open": round(float(row["Open"]), 4),
                "high": round(float(row["High"]), 4),
                "low": round(float(row["Low"]), 4),
                "close": round(float(row["Close"]), 4),
                "volume": int(row["Volume"]),
            })
        return bars

    def get_quote(self, ticker: str) -> Dict:
        """Current quote / fast info for a ticker."""
        if not self.available:
            raise RuntimeError("yfinance is not installed")

        t = self._yf.Ticker(ticker)
        info = t.fast_info

        return {
            "ticker": ticker,
            "price": round(float(info.last_price), 4) if info.last_price else None,
            "market_cap": int(info.market_cap) if info.market_cap else None,
            "currency": info.currency if hasattr(info, 'currency') else "USD",
            "exchange": info.exchange if hasattr(info, 'exchange') else None,
            "fifty_day_average": round(float(info.fifty_day_average), 4) if info.fifty_day_average else None,
            "two_hundred_day_average": round(float(info.two_hundred_day_average), 4) if info.two_hundred_day_average else None,
        }

    def get_fundamentals(self, ticker: str) -> Dict:
        """Key fundamental data for a ticker."""
        if not self.available:
            raise RuntimeError("yfinance is not installed")

        t = self._yf.Ticker(ticker)
        info = t.info

        keys = [
            "sector", "industry", "trailingPE", "forwardPE", "priceToBook",
            "dividendYield", "profitMargins", "revenueGrowth", "earningsGrowth",
            "debtToEquity", "returnOnEquity", "freeCashflow", "shortRatio",
            "beta", "fiftyTwoWeekHigh", "fiftyTwoWeekLow",
        ]
        return {k: info.get(k) for k in keys if info.get(k) is not None}


# ---------------------------------------------------------------------------
# Crypto adapter (ccxt — free, no key needed for public endpoints)
# ---------------------------------------------------------------------------
class CryptoAdapter:
    """Crypto data via ccxt (Binance, Bybit, OKX, etc.)."""

    def __init__(self, exchange_id: str = "binance"):
        try:
            import ccxt  # noqa: F401
            self._ccxt = ccxt
            self.available = True
            self._exchanges: Dict[str, Any] = {}
            self._default_exchange_id = exchange_id
        except ImportError:
            self._ccxt = None
            self.available = False
            logger.warning("ccxt not installed — crypto data unavailable. pip install ccxt")

    def _get_exchange(self, exchange_id: Optional[str] = None):
        eid = exchange_id or self._default_exchange_id
        if eid not in self._exchanges:
            exchange_class = getattr(self._ccxt, eid, None)
            if exchange_class is None:
                raise ValueError(f"Unknown exchange: {eid}")
            self._exchanges[eid] = exchange_class({"enableRateLimit": True})
        return self._exchanges[eid]

    def get_ohlcv(
        self,
        symbol: str,
        timeframe: str = "1h",
        limit: int = 200,
        exchange_id: Optional[str] = None,
    ) -> List[Dict]:
        """
        Fetch OHLCV candles.

        Args:
            symbol: Trading pair (e.g. "BTC/USDT")
            timeframe: Candle interval — 1m, 5m, 15m, 1h, 4h, 1d, 1w
            limit: Number of candles (max varies by exchange, typically 500-1000)
            exchange_id: Override exchange (default: binance)

        Returns:
            List of OHLCV dicts
        """
        if not self.available:
            raise RuntimeError("ccxt is not installed")

        exchange = self._get_exchange(exchange_id)
        raw = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)

        return [
            {
                "timestamp": datetime.utcfromtimestamp(bar[0] / 1000).isoformat(),
                "open": float(bar[1]),
                "high": float(bar[2]),
                "low": float(bar[3]),
                "close": float(bar[4]),
                "volume": float(bar[5]),
            }
            for bar in raw
        ]

    def get_order_book(
        self,
        symbol: str,
        limit: int = 20,
        exchange_id: Optional[str] = None,
    ) -> Dict:
        """
        Fetch order book depth.

        Args:
            symbol: Trading pair (e.g. "BTC/USDT")
            limit: Number of levels per side
            exchange_id: Override exchange

        Returns:
            Dict with bids, asks, spread, mid_price
        """
        if not self.available:
            raise RuntimeError("ccxt is not installed")

        exchange = self._get_exchange(exchange_id)
        book = exchange.fetch_order_book(symbol, limit=limit)

        bids = [[float(p), float(q)] for p, q in book["bids"][:limit]]
        asks = [[float(p), float(q)] for p, q in book["asks"][:limit]]

        best_bid = bids[0][0] if bids else 0
        best_ask = asks[0][0] if asks else 0
        mid_price = (best_bid + best_ask) / 2 if best_bid and best_ask else 0
        spread = best_ask - best_bid if best_bid and best_ask else 0

        return {
            "symbol": symbol,
            "exchange": exchange_id or self._default_exchange_id,
            "bids": bids,
            "asks": asks,
            "best_bid": best_bid,
            "best_ask": best_ask,
            "mid_price": round(mid_price, 4),
            "spread": round(spread, 4),
            "spread_pct": round(spread / mid_price * 100, 4) if mid_price else 0,
            "timestamp": datetime.utcnow().isoformat(),
        }

    def get_ticker(
        self,
        symbol: str,
        exchange_id: Optional[str] = None,
    ) -> Dict:
        """24h ticker summary for a symbol."""
        if not self.available:
            raise RuntimeError("ccxt is not installed")

        exchange = self._get_exchange(exchange_id)
        t = exchange.fetch_ticker(symbol)

        return {
            "symbol": t.get("symbol"),
            "last": t.get("last"),
            "bid": t.get("bid"),
            "ask": t.get("ask"),
            "high_24h": t.get("high"),
            "low_24h": t.get("low"),
            "volume_24h": t.get("baseVolume"),
            "quote_volume_24h": t.get("quoteVolume"),
            "change_pct_24h": t.get("percentage"),
            "vwap": t.get("vwap"),
            "timestamp": t.get("datetime"),
        }

    def get_funding_rate(
        self,
        symbol: str,
        exchange_id: Optional[str] = None,
    ) -> Optional[Dict]:
        """Fetch current funding rate (derivatives exchanges only)."""
        if not self.available:
            raise RuntimeError("ccxt is not installed")

        exchange = self._get_exchange(exchange_id or "binance")

        if not hasattr(exchange, 'fetch_funding_rate'):
            return None

        try:
            fr = exchange.fetch_funding_rate(symbol)
            return {
                "symbol": symbol,
                "funding_rate": fr.get("fundingRate"),
                "next_funding_time": fr.get("fundingDatetime"),
                "timestamp": fr.get("datetime"),
            }
        except Exception as e:
            logger.debug(f"Funding rate unavailable for {symbol}: {e}")
            return None

    def list_exchanges(self) -> List[str]:
        """List all supported exchanges."""
        if not self.available:
            return []
        return self._ccxt.exchanges


# ---------------------------------------------------------------------------
# Macro adapter (FRED — free with API key)
# ---------------------------------------------------------------------------
class MacroAdapter:
    """US macro economic data via FRED API."""

    FRED_BASE_URL = "https://api.stlouisfed.org/fred"

    # Common series IDs for quant signals
    COMMON_SERIES = {
        "fed_funds_rate": "FEDFUNDS",
        "cpi": "CPIAUCSL",
        "cpi_yoy": "CPIAUCNS",
        "unemployment": "UNRATE",
        "gdp": "GDP",
        "treasury_10y": "DGS10",
        "treasury_2y": "DGS2",
        "yield_spread_10y_2y": "T10Y2Y",
        "vix": "VIXCLS",
        "sp500": "SP500",
        "m2_money_supply": "M2SL",
        "consumer_sentiment": "UMCSENT",
        "initial_claims": "ICSA",
        "industrial_production": "INDPRO",
        "retail_sales": "RSAFS",
        "pce_price_index": "PCEPI",
    }

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or Config.FRED_API_KEY
        self.available = bool(self.api_key)
        if not self.available:
            logger.warning("FRED_API_KEY not set — macro data unavailable. Get a free key at https://fred.stlouisfed.org/docs/api/api_key.html")

    def get_series(
        self,
        series_id: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict]:
        """
        Fetch a FRED data series.

        Args:
            series_id: FRED series ID (e.g. "FEDFUNDS", "CPIAUCSL")
            start_date: Start date YYYY-MM-DD
            end_date: End date YYYY-MM-DD
            limit: Max observations (most recent)

        Returns:
            List of {date, value} dicts
        """
        if not self.available:
            raise RuntimeError("FRED API key not configured")

        import requests

        if not start_date:
            start_date = (datetime.now() - timedelta(days=365 * 2)).strftime("%Y-%m-%d")

        params = {
            "series_id": series_id,
            "api_key": self.api_key,
            "file_type": "json",
            "observation_start": start_date,
            "sort_order": "desc",
            "limit": limit,
        }
        if end_date:
            params["observation_end"] = end_date

        resp = requests.get(f"{self.FRED_BASE_URL}/series/observations", params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        observations = []
        for obs in data.get("observations", []):
            val = obs.get("value", ".")
            if val != ".":
                observations.append({
                    "date": obs["date"],
                    "value": float(val),
                })
        return observations

    def get_common_indicator(self, name: str, **kwargs) -> List[Dict]:
        """
        Fetch a common macro indicator by friendly name.

        Args:
            name: One of the keys in COMMON_SERIES (e.g. "cpi", "fed_funds_rate", "vix")

        Returns:
            List of observations
        """
        series_id = self.COMMON_SERIES.get(name)
        if not series_id:
            raise ValueError(f"Unknown indicator: {name}. Available: {list(self.COMMON_SERIES.keys())}")
        return self.get_series(series_id, **kwargs)

    def get_yield_curve(self) -> Dict:
        """Fetch current yield curve snapshot (2Y, 5Y, 10Y, 30Y)."""
        if not self.available:
            raise RuntimeError("FRED API key not configured")

        series_ids = {
            "2Y": "DGS2",
            "5Y": "DGS5",
            "10Y": "DGS10",
            "30Y": "DGS30",
        }

        curve = {}
        for label, sid in series_ids.items():
            data = self.get_series(sid, limit=1)
            if data:
                curve[label] = data[0]["value"]

        spread = None
        if "10Y" in curve and "2Y" in curve:
            spread = round(curve["10Y"] - curve["2Y"], 4)

        return {
            "curve": curve,
            "spread_10y_2y": spread,
            "inverted": spread < 0 if spread is not None else None,
            "timestamp": datetime.utcnow().isoformat(),
        }


# ---------------------------------------------------------------------------
# CoinGecko adapter (free, no key needed for basic endpoints)
# ---------------------------------------------------------------------------
class CoinGeckoAdapter:
    """Crypto market overview via CoinGecko free API."""

    BASE_URL = "https://api.coingecko.com/api/v3"

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or Config.COINGECKO_API_KEY
        self.available = True  # Works without key, just rate-limited

    def _get(self, endpoint: str, params: Optional[Dict] = None) -> Any:
        import requests

        url = f"{self.BASE_URL}/{endpoint}"
        headers = {}
        if self.api_key:
            headers["x-cg-demo-api-key"] = self.api_key

        resp = requests.get(url, params=params or {}, headers=headers, timeout=10)
        resp.raise_for_status()
        return resp.json()

    def get_market_overview(self) -> Dict:
        """Global crypto market stats."""
        data = self._get("global")["data"]
        return {
            "total_market_cap_usd": data.get("total_market_cap", {}).get("usd"),
            "total_volume_24h_usd": data.get("total_volume", {}).get("usd"),
            "btc_dominance": data.get("market_cap_percentage", {}).get("btc"),
            "eth_dominance": data.get("market_cap_percentage", {}).get("eth"),
            "active_cryptocurrencies": data.get("active_cryptocurrencies"),
            "market_cap_change_pct_24h": data.get("market_cap_change_percentage_24h_usd"),
            "timestamp": datetime.utcnow().isoformat(),
        }

    def get_top_coins(self, limit: int = 20, page: int = 1) -> List[Dict]:
        """Top coins by market cap."""
        data = self._get("coins/markets", {
            "vs_currency": "usd",
            "order": "market_cap_desc",
            "per_page": limit,
            "page": page,
            "sparkline": "false",
        })
        return [
            {
                "id": coin["id"],
                "symbol": coin["symbol"].upper(),
                "name": coin["name"],
                "price": coin.get("current_price"),
                "market_cap": coin.get("market_cap"),
                "market_cap_rank": coin.get("market_cap_rank"),
                "volume_24h": coin.get("total_volume"),
                "change_pct_24h": coin.get("price_change_percentage_24h"),
                "high_24h": coin.get("high_24h"),
                "low_24h": coin.get("low_24h"),
                "ath": coin.get("ath"),
                "ath_change_pct": coin.get("ath_change_percentage"),
            }
            for coin in data
        ]

    def get_trending(self) -> List[Dict]:
        """Trending coins on CoinGecko (search-based)."""
        data = self._get("search/trending")
        return [
            {
                "id": item["item"]["id"],
                "symbol": item["item"]["symbol"],
                "name": item["item"]["name"],
                "market_cap_rank": item["item"].get("market_cap_rank"),
                "score": item["item"].get("score"),
            }
            for item in data.get("coins", [])
        ]

    def get_coin_history(
        self,
        coin_id: str,
        days: int = 30,
    ) -> List[Dict]:
        """Historical price data for a coin."""
        data = self._get(f"coins/{coin_id}/market_chart", {
            "vs_currency": "usd",
            "days": days,
        })
        prices = data.get("prices", [])
        volumes = data.get("total_volumes", [])

        result = []
        for i, (ts, price) in enumerate(prices):
            entry = {
                "timestamp": datetime.utcfromtimestamp(ts / 1000).isoformat(),
                "price": price,
            }
            if i < len(volumes):
                entry["volume"] = volumes[i][1]
            result.append(entry)
        return result


# ---------------------------------------------------------------------------
# Unified MarketDataService
# ---------------------------------------------------------------------------
class MarketDataService:
    """
    Unified entry point for all market data.
    All sources are free-tier. No paid APIs required.

    Usage:
        svc = MarketDataService()

        # Equities
        bars = svc.equity.get_ohlcv("AAPL", period="3mo", interval="1d")
        quote = svc.equity.get_quote("TSLA")

        # Crypto
        bars = svc.crypto.get_ohlcv("BTC/USDT", timeframe="1h", limit=100)
        book = svc.crypto.get_order_book("ETH/USDT", limit=20)
        ticker = svc.crypto.get_ticker("BTC/USDT")

        # Macro
        cpi = svc.macro.get_common_indicator("cpi", limit=24)
        curve = svc.macro.get_yield_curve()

        # CoinGecko
        overview = svc.coingecko.get_market_overview()
        trending = svc.coingecko.get_trending()
    """

    def __init__(self):
        self._cache = SimpleCache(ttl=Config.MARKET_DATA_CACHE_TTL)
        self.equity = EquityAdapter()
        self.crypto = CryptoAdapter()
        self.macro = MacroAdapter()
        self.coingecko = CoinGeckoAdapter()

        logger.info(
            f"MarketDataService initialized — "
            f"equity={'OK' if self.equity.available else 'MISSING yfinance'}, "
            f"crypto={'OK' if self.crypto.available else 'MISSING ccxt'}, "
            f"macro={'OK' if self.macro.available else 'NO FRED KEY'}, "
            f"coingecko=OK"
        )

    def get_status(self) -> Dict:
        """Health check / status of all adapters."""
        return {
            "equity": {"available": self.equity.available, "source": "yfinance"},
            "crypto": {"available": self.crypto.available, "source": "ccxt"},
            "macro": {"available": self.macro.available, "source": "FRED"},
            "coingecko": {"available": self.coingecko.available, "source": "CoinGecko"},
            "cache_ttl": self._cache.ttl,
        }

    def get_multi_asset_snapshot(self, assets: List[Dict]) -> List[Dict]:
        """
        Fetch current price data for multiple assets at once.

        Args:
            assets: List of dicts with "type" and "symbol", e.g.
                [{"type": "equity", "symbol": "AAPL"},
                 {"type": "crypto", "symbol": "BTC/USDT"}]

        Returns:
            List of results with price data
        """
        results = []
        for asset in assets:
            asset_type = asset.get("type", "").lower()
            symbol = asset.get("symbol", "")

            cache_key = f"snapshot:{asset_type}:{symbol}"
            cached = self._cache.get(cache_key)
            if cached:
                results.append(cached)
                continue

            try:
                if asset_type == "equity":
                    data = self.equity.get_quote(symbol)
                elif asset_type == "crypto":
                    data = self.crypto.get_ticker(symbol)
                else:
                    data = {"error": f"Unknown asset type: {asset_type}"}

                data["asset_type"] = asset_type
                data["symbol"] = symbol
                self._cache.set(cache_key, data)
                results.append(data)

            except Exception as e:
                logger.error(f"Failed to fetch {asset_type}:{symbol}: {e}")
                results.append({
                    "asset_type": asset_type,
                    "symbol": symbol,
                    "error": str(e),
                })

        return results

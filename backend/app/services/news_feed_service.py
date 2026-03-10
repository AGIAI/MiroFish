"""
News Feed Service
Aggregates free news and sentiment sources for market context:
- CryptoPanic: Crypto news aggregator with sentiment tags (free tier)
- RSS feeds: Financial news from public RSS sources
- Reddit: Public subreddit posts via old.reddit.com JSON (no auth needed)
"""

import time
from datetime import datetime
from typing import Dict, List, Optional, Any

from ..config import Config
from ..utils.logger import get_logger

logger = get_logger('mirofish.news_feed')


# ---------------------------------------------------------------------------
# CryptoPanic adapter (free tier: 5 req/min, no sentiment filter)
# ---------------------------------------------------------------------------
class CryptoPanicAdapter:
    """
    Crypto news aggregated from 30+ sources.
    Free tier: 5 requests/minute, public posts only.
    With API key: sentiment tags, filters, more requests.
    """

    BASE_URL = "https://cryptopanic.com/api/free/v1"

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or Config.CRYPTOPANIC_API_KEY
        self.available = bool(self.api_key)
        if not self.available:
            logger.warning(
                "CRYPTOPANIC_API_KEY not set — CryptoPanic news unavailable. "
                "Get a free key at https://cryptopanic.com/developers/api/"
            )

    def get_news(
        self,
        currencies: Optional[str] = None,
        kind: str = "news",
        limit: int = 20,
    ) -> List[Dict]:
        """
        Fetch latest crypto news.

        Args:
            currencies: Comma-separated coin codes, e.g. "BTC,ETH,SOL"
            kind: "news" or "media" or "all"
            limit: Number of results (max ~40 on free tier)

        Returns:
            List of news items
        """
        if not self.available:
            raise RuntimeError("CryptoPanic API key not configured")

        import requests

        params = {
            "auth_token": self.api_key,
            "kind": kind,
            "public": "true",
        }
        if currencies:
            params["currencies"] = currencies

        resp = requests.get(f"{self.BASE_URL}/posts/", params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        results = []
        for post in data.get("results", [])[:limit]:
            item = {
                "title": post.get("title"),
                "url": post.get("url"),
                "source": post.get("source", {}).get("title"),
                "published_at": post.get("published_at"),
                "currencies": [c.get("code") for c in post.get("currencies", [])],
                "kind": post.get("kind"),
            }
            # Votes indicate community sentiment
            votes = post.get("votes", {})
            if votes:
                item["votes"] = {
                    "positive": votes.get("positive", 0),
                    "negative": votes.get("negative", 0),
                    "important": votes.get("important", 0),
                    "liked": votes.get("liked", 0),
                    "lol": votes.get("lol", 0),
                    "toxic": votes.get("toxic", 0),
                }
            results.append(item)

        return results


# ---------------------------------------------------------------------------
# RSS adapter (free, no key needed)
# ---------------------------------------------------------------------------
class RSSAdapter:
    """
    Financial news from public RSS feeds.
    No API key needed — just standard RSS/Atom parsing.
    """

    DEFAULT_FEEDS = {
        "reuters_markets": "https://www.reutersagency.com/feed/?taxonomy=best-sectors&post_type=best",
        "coindesk": "https://www.coindesk.com/arc/outboundfeeds/rss/",
        "cointelegraph": "https://cointelegraph.com/rss",
        "investing_com_news": "https://www.investing.com/rss/news.rss",
        "yahoo_finance": "https://finance.yahoo.com/news/rssindex",
        "fed_press": "https://www.federalreserve.gov/feeds/press_all.xml",
    }

    def __init__(self):
        try:
            import feedparser  # noqa: F401
            self._feedparser = feedparser
            self.available = True
        except ImportError:
            self._feedparser = None
            self.available = False
            logger.warning("feedparser not installed — RSS feeds unavailable. pip install feedparser")

    def fetch_feed(self, feed_url: str, limit: int = 15) -> List[Dict]:
        """Parse an RSS/Atom feed and return structured entries."""
        if not self.available:
            raise RuntimeError("feedparser is not installed")

        feed = self._feedparser.parse(feed_url)
        entries = []
        for entry in feed.entries[:limit]:
            published = entry.get("published_parsed") or entry.get("updated_parsed")
            pub_dt = None
            if published:
                pub_dt = datetime(*published[:6]).isoformat()

            entries.append({
                "title": entry.get("title", "").strip(),
                "url": entry.get("link", ""),
                "summary": entry.get("summary", "")[:500],
                "published_at": pub_dt,
                "source": feed.feed.get("title", feed_url),
            })
        return entries

    def fetch_all_default(self, limit_per_feed: int = 10) -> Dict[str, List[Dict]]:
        """Fetch all default feeds."""
        if not self.available:
            raise RuntimeError("feedparser is not installed")

        results = {}
        for name, url in self.DEFAULT_FEEDS.items():
            try:
                results[name] = self.fetch_feed(url, limit=limit_per_feed)
            except Exception as e:
                logger.warning(f"Failed to fetch RSS feed {name}: {e}")
                results[name] = []
        return results


# ---------------------------------------------------------------------------
# Reddit adapter (free, no auth needed — uses public JSON endpoint)
# ---------------------------------------------------------------------------
class RedditAdapter:
    """
    Public Reddit data via old.reddit.com JSON endpoints.
    No API key or OAuth needed. Rate limit: ~30 req/min.
    """

    BASE_URL = "https://old.reddit.com"
    FINANCE_SUBREDDITS = [
        "wallstreetbets", "stocks", "investing", "cryptocurrency",
        "bitcoin", "ethtrader", "options", "forex",
    ]

    def __init__(self):
        self.available = True
        self._headers = {
            "User-Agent": "MiroFish/1.0 (market research bot)"
        }

    def get_hot_posts(
        self,
        subreddit: str,
        limit: int = 15,
    ) -> List[Dict]:
        """
        Fetch hot posts from a subreddit.

        Args:
            subreddit: Subreddit name (without r/)
            limit: Number of posts

        Returns:
            List of post dicts
        """
        import requests

        url = f"{self.BASE_URL}/r/{subreddit}/hot.json"
        params = {"limit": limit, "raw_json": 1}

        resp = requests.get(url, params=params, headers=self._headers, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        posts = []
        for child in data.get("data", {}).get("children", []):
            post = child.get("data", {})
            if post.get("stickied"):
                continue
            posts.append({
                "title": post.get("title", ""),
                "subreddit": subreddit,
                "score": post.get("score", 0),
                "upvote_ratio": post.get("upvote_ratio", 0),
                "num_comments": post.get("num_comments", 0),
                "url": f"https://reddit.com{post.get('permalink', '')}",
                "selftext": (post.get("selftext") or "")[:500],
                "created_utc": datetime.utcfromtimestamp(
                    post.get("created_utc", 0)
                ).isoformat() if post.get("created_utc") else None,
                "flair": post.get("link_flair_text"),
                "author": post.get("author"),
            })
        return posts

    def get_finance_sentiment(self, limit_per_sub: int = 10) -> Dict[str, List[Dict]]:
        """Fetch hot posts from all finance subreddits."""
        results = {}
        for sub in self.FINANCE_SUBREDDITS:
            try:
                results[sub] = self.get_hot_posts(sub, limit=limit_per_sub)
            except Exception as e:
                logger.warning(f"Failed to fetch r/{sub}: {e}")
                results[sub] = []
            # Be polite with rate limits
            time.sleep(0.5)
        return results


# ---------------------------------------------------------------------------
# Unified NewsFeedService
# ---------------------------------------------------------------------------
class NewsFeedService:
    """
    Unified entry point for news and sentiment data.
    All sources are free-tier.

    Usage:
        svc = NewsFeedService()

        # CryptoPanic
        news = svc.cryptopanic.get_news(currencies="BTC,ETH")

        # RSS
        feeds = svc.rss.fetch_all_default()

        # Reddit
        wsb = svc.reddit.get_hot_posts("wallstreetbets")
        all_finance = svc.reddit.get_finance_sentiment()
    """

    def __init__(self):
        self.cryptopanic = CryptoPanicAdapter()
        self.rss = RSSAdapter()
        self.reddit = RedditAdapter()

        logger.info(
            f"NewsFeedService initialized — "
            f"cryptopanic={'OK' if self.cryptopanic.available else 'NO KEY'}, "
            f"rss={'OK' if self.rss.available else 'MISSING feedparser'}, "
            f"reddit=OK"
        )

    def get_status(self) -> Dict:
        """Health check for all news adapters."""
        return {
            "cryptopanic": {"available": self.cryptopanic.available, "source": "CryptoPanic"},
            "rss": {"available": self.rss.available, "source": "RSS feeds"},
            "reddit": {"available": self.reddit.available, "source": "Reddit JSON"},
        }

    def get_aggregated_news(
        self,
        crypto_currencies: Optional[str] = None,
        subreddits: Optional[List[str]] = None,
        limit: int = 10,
    ) -> Dict[str, Any]:
        """
        Aggregate news from all available sources into a single response.

        Args:
            crypto_currencies: Comma-separated crypto codes for CryptoPanic
            subreddits: List of subreddits to scan (defaults to finance subs)
            limit: Items per source

        Returns:
            Dict with news from each available source
        """
        result = {"timestamp": datetime.utcnow().isoformat(), "sources": {}}

        # CryptoPanic
        if self.cryptopanic.available:
            try:
                result["sources"]["cryptopanic"] = self.cryptopanic.get_news(
                    currencies=crypto_currencies, limit=limit
                )
            except Exception as e:
                logger.warning(f"CryptoPanic fetch failed: {e}")
                result["sources"]["cryptopanic"] = {"error": str(e)}

        # RSS
        if self.rss.available:
            try:
                result["sources"]["rss"] = self.rss.fetch_all_default(limit_per_feed=limit)
            except Exception as e:
                logger.warning(f"RSS fetch failed: {e}")
                result["sources"]["rss"] = {"error": str(e)}

        # Reddit
        subs = subreddits or ["wallstreetbets", "cryptocurrency", "stocks"]
        try:
            reddit_data = {}
            for sub in subs:
                try:
                    reddit_data[sub] = self.reddit.get_hot_posts(sub, limit=limit)
                    time.sleep(0.3)
                except Exception as e:
                    reddit_data[sub] = {"error": str(e)}
            result["sources"]["reddit"] = reddit_data
        except Exception as e:
            logger.warning(f"Reddit fetch failed: {e}")
            result["sources"]["reddit"] = {"error": str(e)}

        return result

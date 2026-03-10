"""
Market Data API routes
Provides REST endpoints for market data, news, and on-chain information.
All data sources use free-tier APIs.
"""

from flask import request, jsonify

from . import market_data_bp
from ..services.market_data_service import MarketDataService
from ..services.news_feed_service import NewsFeedService
from ..services.onchain_data_service import OnChainDataService
from ..utils.logger import get_logger

logger = get_logger('mirofish.api.market_data')

# Singleton service instances (initialized on first request)
_market_data_svc = None
_news_feed_svc = None
_onchain_svc = None


def _get_market_data():
    global _market_data_svc
    if _market_data_svc is None:
        _market_data_svc = MarketDataService()
    return _market_data_svc


def _get_news_feed():
    global _news_feed_svc
    if _news_feed_svc is None:
        _news_feed_svc = NewsFeedService()
    return _news_feed_svc


def _get_onchain():
    global _onchain_svc
    if _onchain_svc is None:
        _onchain_svc = OnChainDataService()
    return _onchain_svc


# ============== Health / Status ==============

@market_data_bp.route('/status', methods=['GET'])
def market_data_status():
    """Health check for all market data adapters."""
    try:
        status = {
            "market_data": _get_market_data().get_status(),
            "news_feed": _get_news_feed().get_status(),
            "onchain": _get_onchain().get_status(),
        }
        return jsonify({"success": True, "data": status})
    except Exception as e:
        logger.error(f"Status check failed: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


# ============== Equity Endpoints ==============

@market_data_bp.route('/equity/ohlcv/<ticker>', methods=['GET'])
def equity_ohlcv(ticker: str):
    """
    Fetch stock/ETF OHLCV data.
    Query params: period (default: 1mo), interval (default: 1d)
    """
    try:
        svc = _get_market_data()
        period = request.args.get('period', '1mo')
        interval = request.args.get('interval', '1d')

        bars = svc.equity.get_ohlcv(ticker, period=period, interval=interval)
        return jsonify({"success": True, "ticker": ticker, "count": len(bars), "data": bars})
    except Exception as e:
        logger.error(f"Equity OHLCV failed for {ticker}: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@market_data_bp.route('/equity/quote/<ticker>', methods=['GET'])
def equity_quote(ticker: str):
    """Current quote for a stock/ETF."""
    try:
        svc = _get_market_data()
        quote = svc.equity.get_quote(ticker)
        return jsonify({"success": True, "data": quote})
    except Exception as e:
        logger.error(f"Equity quote failed for {ticker}: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@market_data_bp.route('/equity/fundamentals/<ticker>', methods=['GET'])
def equity_fundamentals(ticker: str):
    """Key fundamentals for a stock."""
    try:
        svc = _get_market_data()
        data = svc.equity.get_fundamentals(ticker)
        return jsonify({"success": True, "ticker": ticker, "data": data})
    except Exception as e:
        logger.error(f"Equity fundamentals failed for {ticker}: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


# ============== Crypto Endpoints ==============

@market_data_bp.route('/crypto/ohlcv', methods=['GET'])
def crypto_ohlcv():
    """
    Fetch crypto OHLCV data.
    Query params: symbol (required, e.g. BTC/USDT), timeframe (default: 1h),
                  limit (default: 200), exchange (default: binance)
    """
    try:
        svc = _get_market_data()
        symbol = request.args.get('symbol')
        if not symbol:
            return jsonify({"success": False, "error": "symbol parameter required"}), 400

        timeframe = request.args.get('timeframe', '1h')
        limit = int(request.args.get('limit', '200'))
        exchange = request.args.get('exchange')

        bars = svc.crypto.get_ohlcv(symbol, timeframe=timeframe, limit=limit, exchange_id=exchange)
        return jsonify({"success": True, "symbol": symbol, "count": len(bars), "data": bars})
    except Exception as e:
        logger.error(f"Crypto OHLCV failed: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@market_data_bp.route('/crypto/orderbook', methods=['GET'])
def crypto_orderbook():
    """
    Fetch crypto order book.
    Query params: symbol (required), limit (default: 20), exchange (optional)
    """
    try:
        svc = _get_market_data()
        symbol = request.args.get('symbol')
        if not symbol:
            return jsonify({"success": False, "error": "symbol parameter required"}), 400

        limit = int(request.args.get('limit', '20'))
        exchange = request.args.get('exchange')

        book = svc.crypto.get_order_book(symbol, limit=limit, exchange_id=exchange)
        return jsonify({"success": True, "data": book})
    except Exception as e:
        logger.error(f"Crypto orderbook failed: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@market_data_bp.route('/crypto/ticker', methods=['GET'])
def crypto_ticker():
    """
    24h ticker for a crypto pair.
    Query params: symbol (required), exchange (optional)
    """
    try:
        svc = _get_market_data()
        symbol = request.args.get('symbol')
        if not symbol:
            return jsonify({"success": False, "error": "symbol parameter required"}), 400

        exchange = request.args.get('exchange')
        data = svc.crypto.get_ticker(symbol, exchange_id=exchange)
        return jsonify({"success": True, "data": data})
    except Exception as e:
        logger.error(f"Crypto ticker failed: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@market_data_bp.route('/crypto/funding-rate', methods=['GET'])
def crypto_funding_rate():
    """
    Current funding rate for a derivative pair.
    Query params: symbol (required), exchange (optional, default: binance)
    """
    try:
        svc = _get_market_data()
        symbol = request.args.get('symbol')
        if not symbol:
            return jsonify({"success": False, "error": "symbol parameter required"}), 400

        exchange = request.args.get('exchange')
        data = svc.crypto.get_funding_rate(symbol, exchange_id=exchange)
        if data is None:
            return jsonify({"success": False, "error": "Funding rate not available for this pair/exchange"}), 404
        return jsonify({"success": True, "data": data})
    except Exception as e:
        logger.error(f"Crypto funding rate failed: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@market_data_bp.route('/crypto/exchanges', methods=['GET'])
def crypto_exchanges():
    """List all supported crypto exchanges."""
    try:
        svc = _get_market_data()
        exchanges = svc.crypto.list_exchanges()
        return jsonify({"success": True, "count": len(exchanges), "data": exchanges})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# ============== CoinGecko Endpoints ==============

@market_data_bp.route('/crypto/market-overview', methods=['GET'])
def crypto_market_overview():
    """Global crypto market stats from CoinGecko."""
    try:
        svc = _get_market_data()
        data = svc.coingecko.get_market_overview()
        return jsonify({"success": True, "data": data})
    except Exception as e:
        logger.error(f"CoinGecko market overview failed: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@market_data_bp.route('/crypto/top-coins', methods=['GET'])
def crypto_top_coins():
    """
    Top coins by market cap.
    Query params: limit (default: 20), page (default: 1)
    """
    try:
        svc = _get_market_data()
        limit = int(request.args.get('limit', '20'))
        page = int(request.args.get('page', '1'))
        data = svc.coingecko.get_top_coins(limit=limit, page=page)
        return jsonify({"success": True, "count": len(data), "data": data})
    except Exception as e:
        logger.error(f"CoinGecko top coins failed: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@market_data_bp.route('/crypto/trending', methods=['GET'])
def crypto_trending():
    """Trending coins on CoinGecko."""
    try:
        svc = _get_market_data()
        data = svc.coingecko.get_trending()
        return jsonify({"success": True, "data": data})
    except Exception as e:
        logger.error(f"CoinGecko trending failed: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


# ============== Macro Endpoints ==============

@market_data_bp.route('/macro/series/<series_id>', methods=['GET'])
def macro_series(series_id: str):
    """
    Fetch a FRED data series.
    Query params: start_date, end_date, limit (default: 100)
    """
    try:
        svc = _get_market_data()
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        limit = int(request.args.get('limit', '100'))

        data = svc.macro.get_series(series_id, start_date=start_date, end_date=end_date, limit=limit)
        return jsonify({"success": True, "series_id": series_id, "count": len(data), "data": data})
    except Exception as e:
        logger.error(f"FRED series failed for {series_id}: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@market_data_bp.route('/macro/indicator/<name>', methods=['GET'])
def macro_indicator(name: str):
    """
    Fetch a common macro indicator by friendly name.
    Available: cpi, fed_funds_rate, unemployment, gdp, treasury_10y, vix, etc.
    Query params: limit (default: 24)
    """
    try:
        svc = _get_market_data()
        limit = int(request.args.get('limit', '24'))
        data = svc.macro.get_common_indicator(name, limit=limit)
        return jsonify({"success": True, "indicator": name, "count": len(data), "data": data})
    except Exception as e:
        logger.error(f"Macro indicator failed for {name}: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@market_data_bp.route('/macro/indicators', methods=['GET'])
def macro_indicators_list():
    """List all available common macro indicators."""
    from ..services.market_data_service import MacroAdapter
    return jsonify({
        "success": True,
        "data": list(MacroAdapter.COMMON_SERIES.keys()),
    })


@market_data_bp.route('/macro/yield-curve', methods=['GET'])
def macro_yield_curve():
    """Current US Treasury yield curve."""
    try:
        svc = _get_market_data()
        data = svc.macro.get_yield_curve()
        return jsonify({"success": True, "data": data})
    except Exception as e:
        logger.error(f"Yield curve failed: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


# ============== Multi-Asset Snapshot ==============

@market_data_bp.route('/snapshot', methods=['POST'])
def multi_asset_snapshot():
    """
    Fetch current price data for multiple assets at once.
    Body: {"assets": [{"type": "equity", "symbol": "AAPL"}, {"type": "crypto", "symbol": "BTC/USDT"}]}
    """
    try:
        svc = _get_market_data()
        body = request.get_json()
        if not body or "assets" not in body:
            return jsonify({"success": False, "error": "Request body must include 'assets' array"}), 400

        data = svc.get_multi_asset_snapshot(body["assets"])
        return jsonify({"success": True, "data": data})
    except Exception as e:
        logger.error(f"Multi-asset snapshot failed: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


# ============== News Endpoints ==============

@market_data_bp.route('/news/crypto', methods=['GET'])
def news_crypto():
    """
    Crypto news from CryptoPanic.
    Query params: currencies (e.g. BTC,ETH), limit (default: 20)
    """
    try:
        svc = _get_news_feed()
        currencies = request.args.get('currencies')
        limit = int(request.args.get('limit', '20'))
        data = svc.cryptopanic.get_news(currencies=currencies, limit=limit)
        return jsonify({"success": True, "count": len(data), "data": data})
    except Exception as e:
        logger.error(f"CryptoPanic news failed: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@market_data_bp.route('/news/rss', methods=['GET'])
def news_rss():
    """
    Financial news from RSS feeds.
    Query params: feed (optional, specific feed name), limit (default: 10)
    """
    try:
        svc = _get_news_feed()
        feed_name = request.args.get('feed')
        limit = int(request.args.get('limit', '10'))

        if feed_name:
            from ..services.news_feed_service import RSSAdapter
            url = RSSAdapter.DEFAULT_FEEDS.get(feed_name)
            if not url:
                return jsonify({
                    "success": False,
                    "error": f"Unknown feed: {feed_name}",
                    "available_feeds": list(RSSAdapter.DEFAULT_FEEDS.keys()),
                }), 400
            data = svc.rss.fetch_feed(url, limit=limit)
            return jsonify({"success": True, "feed": feed_name, "count": len(data), "data": data})
        else:
            data = svc.rss.fetch_all_default(limit_per_feed=limit)
            return jsonify({"success": True, "data": data})
    except Exception as e:
        logger.error(f"RSS news failed: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@market_data_bp.route('/news/reddit/<subreddit>', methods=['GET'])
def news_reddit(subreddit: str):
    """
    Hot posts from a subreddit.
    Query params: limit (default: 15)
    """
    try:
        svc = _get_news_feed()
        limit = int(request.args.get('limit', '15'))
        data = svc.reddit.get_hot_posts(subreddit, limit=limit)
        return jsonify({"success": True, "subreddit": subreddit, "count": len(data), "data": data})
    except Exception as e:
        logger.error(f"Reddit fetch failed for r/{subreddit}: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@market_data_bp.route('/news/reddit', methods=['GET'])
def news_reddit_finance():
    """Hot posts from all finance subreddits."""
    try:
        svc = _get_news_feed()
        limit = int(request.args.get('limit', '10'))
        data = svc.reddit.get_finance_sentiment(limit_per_sub=limit)
        return jsonify({"success": True, "data": data})
    except Exception as e:
        logger.error(f"Reddit finance sentiment failed: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@market_data_bp.route('/news/aggregated', methods=['GET'])
def news_aggregated():
    """
    Aggregated news from all available sources.
    Query params: currencies (for crypto), subreddits (comma-separated), limit (default: 10)
    """
    try:
        svc = _get_news_feed()
        currencies = request.args.get('currencies')
        subreddits = request.args.get('subreddits')
        limit = int(request.args.get('limit', '10'))

        sub_list = subreddits.split(',') if subreddits else None
        data = svc.get_aggregated_news(
            crypto_currencies=currencies,
            subreddits=sub_list,
            limit=limit,
        )
        return jsonify({"success": True, "data": data})
    except Exception as e:
        logger.error(f"Aggregated news failed: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


# ============== On-Chain Endpoints ==============

@market_data_bp.route('/onchain/status', methods=['GET'])
def onchain_status():
    """On-chain data service status."""
    try:
        data = _get_onchain().get_status()
        return jsonify({"success": True, "data": data})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@market_data_bp.route('/onchain/overview', methods=['GET'])
def onchain_overview():
    """Combined overview of major blockchain networks."""
    try:
        data = _get_onchain().get_network_overview()
        return jsonify({"success": True, "data": data})
    except Exception as e:
        logger.error(f"On-chain overview failed: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@market_data_bp.route('/onchain/eth/gas', methods=['GET'])
def onchain_eth_gas():
    """Current Ethereum gas prices."""
    try:
        data = _get_onchain().etherscan.get_gas_oracle()
        return jsonify({"success": True, "data": data})
    except Exception as e:
        logger.error(f"ETH gas oracle failed: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@market_data_bp.route('/onchain/eth/price', methods=['GET'])
def onchain_eth_price():
    """Current ETH price from Etherscan."""
    try:
        data = _get_onchain().etherscan.get_eth_price()
        return jsonify({"success": True, "data": data})
    except Exception as e:
        logger.error(f"ETH price failed: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@market_data_bp.route('/onchain/eth/balance/<address>', methods=['GET'])
def onchain_eth_balance(address: str):
    """ETH balance for a wallet address."""
    try:
        data = _get_onchain().etherscan.get_balance(address)
        return jsonify({"success": True, "data": data})
    except Exception as e:
        logger.error(f"ETH balance failed: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@market_data_bp.route('/onchain/eth/token-transfers/<address>', methods=['GET'])
def onchain_eth_token_transfers(address: str):
    """
    ERC-20 token transfers for an address.
    Query params: contract (optional token contract), page (default: 1), limit (default: 20)
    """
    try:
        svc = _get_onchain()
        contract = request.args.get('contract')
        page = int(request.args.get('page', '1'))
        limit = int(request.args.get('limit', '20'))

        data = svc.etherscan.get_token_transfers(
            address, contract_address=contract, page=page, offset=limit
        )
        return jsonify({"success": True, "address": address, "count": len(data), "data": data})
    except Exception as e:
        logger.error(f"Token transfers failed: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@market_data_bp.route('/onchain/btc/stats', methods=['GET'])
def onchain_btc_stats():
    """Bitcoin network statistics."""
    try:
        data = _get_onchain().bitcoin.get_stats()
        return jsonify({"success": True, "data": data})
    except Exception as e:
        logger.error(f"BTC stats failed: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@market_data_bp.route('/onchain/btc/mempool', methods=['GET'])
def onchain_btc_mempool():
    """Bitcoin mempool info (unconfirmed transactions)."""
    try:
        data = _get_onchain().bitcoin.get_mempool_info()
        return jsonify({"success": True, "data": data})
    except Exception as e:
        logger.error(f"BTC mempool failed: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

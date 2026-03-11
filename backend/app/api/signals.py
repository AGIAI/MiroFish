"""
Signal API routes
Provides REST endpoints for:
- Signal generation (from market data or simulation)
- Signal retrieval (latest, active, history)
- Webhook management (register/unregister targets for OpenClaw/Meridian)
- Regime detection
- Trader archetype info
- Signal delivery status
"""

from flask import request, jsonify

from . import signals_bp
from ..utils.logger import get_logger

logger = get_logger('mirofish.api.signals')

# Lazy-initialized singleton services
_signal_extractor = None
_signal_calibrator = None
_delivery_service = None
_regime_detector = None
_trader_gen = None
_market_data_svc = None


def _get_signal_extractor():
    global _signal_extractor
    if _signal_extractor is None:
        from ..services.signal_extractor import SignalExtractor
        _signal_extractor = SignalExtractor()
    return _signal_extractor


def _get_calibrator():
    global _signal_calibrator
    if _signal_calibrator is None:
        from ..services.signal_calibrator import SignalCalibrator
        _signal_calibrator = SignalCalibrator()
    return _signal_calibrator


def _get_delivery():
    global _delivery_service
    if _delivery_service is None:
        from ..services.signal_delivery import SignalDeliveryService
        _delivery_service = SignalDeliveryService()
    return _delivery_service


def _get_regime_detector():
    global _regime_detector
    if _regime_detector is None:
        from ..services.regime_detector import RegimeDetector
        _regime_detector = RegimeDetector()
    return _regime_detector


def _get_trader_gen():
    global _trader_gen
    if _trader_gen is None:
        from ..services.trader_profile_generator import TraderProfileGenerator
        _trader_gen = TraderProfileGenerator()
    return _trader_gen


def _get_market_data():
    global _market_data_svc
    if _market_data_svc is None:
        from ..services.market_data_service import MarketDataService
        _market_data_svc = MarketDataService()
    return _market_data_svc


# ============== Signal Generation ==============

@signals_bp.route('/generate', methods=['POST'])
def generate_signal():
    """
    Generate a trading signal from current market data.

    Body:
    {
        "asset": "BTC/USDT",
        "asset_type": "crypto",   // "crypto" or "equity"
        "exchange": "binance",    // optional
        "timeframe": "1h",        // OHLCV timeframe
        "bars": 100,              // number of bars
        "calibrate": true,        // apply calibration
        "deliver": true,          // push to registered webhooks
        "targets": ["openclaw"]   // optional specific targets
    }
    """
    try:
        body = request.get_json() or {}
        asset = body.get("asset", "BTC/USDT")
        asset_type = body.get("asset_type", "crypto")
        exchange = body.get("exchange")
        timeframe = body.get("timeframe", "1h")
        bars = int(body.get("bars", 100))
        do_calibrate = body.get("calibrate", True)
        do_deliver = body.get("deliver", False)
        target_names = body.get("targets")

        svc = _get_market_data()
        extractor = _get_signal_extractor()
        detector = _get_regime_detector()

        # 1. Fetch market data
        if asset_type == "crypto":
            ohlcv = svc.crypto.get_ohlcv(asset, timeframe=timeframe, limit=bars, exchange_id=exchange)
        elif asset_type == "equity":
            ohlcv = svc.equity.get_ohlcv(asset, period="3mo", interval="1d")
        else:
            return jsonify({"success": False, "error": f"Unknown asset_type: {asset_type}"}), 400

        if not ohlcv or len(ohlcv) < 20:
            return jsonify({"success": False, "error": "Insufficient market data"}), 400

        # 2. Detect regime
        regime = detector.detect(ohlcv)

        # 3. Extract signal from market data
        signal = extractor.extract_from_market_data_only(ohlcv, regime, asset=asset)
        signal.exchange = exchange

        # 4. Calibrate
        if do_calibrate:
            calibrator = _get_calibrator()
            signal = calibrator.calibrate(signal)

        # 5. Deliver to webhooks
        delivery_result = None
        if do_deliver:
            delivery = _get_delivery()
            delivery_result = delivery.deliver(signal, target_names=target_names)
        else:
            # Still store the signal
            _get_delivery()._signal_store.append(signal)

        return jsonify({
            "success": True,
            "signal": signal.model_dump(),
            "regime": regime,
            "delivery": delivery_result,
        })

    except Exception as e:
        logger.error(f"Signal generation failed: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500


@signals_bp.route('/generate/batch', methods=['POST'])
def generate_batch_signals():
    """
    Generate signals for multiple assets at once.

    Body:
    {
        "assets": [
            {"asset": "BTC/USDT", "asset_type": "crypto"},
            {"asset": "ETH/USDT", "asset_type": "crypto"},
            {"asset": "AAPL", "asset_type": "equity"}
        ],
        "calibrate": true,
        "deliver": false
    }
    """
    try:
        body = request.get_json() or {}
        assets = body.get("assets", [])
        do_calibrate = body.get("calibrate", True)
        do_deliver = body.get("deliver", False)

        if not assets:
            return jsonify({"success": False, "error": "No assets provided"}), 400

        svc = _get_market_data()
        extractor = _get_signal_extractor()
        detector = _get_regime_detector()
        calibrator = _get_calibrator() if do_calibrate else None

        results = []
        for asset_config in assets:
            asset = asset_config.get("asset", "")
            asset_type = asset_config.get("asset_type", "crypto")

            try:
                if asset_type == "crypto":
                    ohlcv = svc.crypto.get_ohlcv(asset, timeframe="1h", limit=100)
                else:
                    ohlcv = svc.equity.get_ohlcv(asset, period="3mo", interval="1d")

                if not ohlcv or len(ohlcv) < 20:
                    results.append({"asset": asset, "error": "Insufficient data"})
                    continue

                regime = detector.detect(ohlcv)
                signal = extractor.extract_from_market_data_only(ohlcv, regime, asset=asset)

                if calibrator:
                    signal = calibrator.calibrate(signal)

                if do_deliver:
                    _get_delivery().deliver(signal)
                else:
                    _get_delivery()._signal_store.append(signal)

                results.append({
                    "asset": asset,
                    "signal": signal.model_dump(),
                    "regime": regime.get("regime"),
                })

            except Exception as e:
                logger.warning(f"Signal generation failed for {asset}: {e}")
                results.append({"asset": asset, "error": str(e)})

        return jsonify({"success": True, "count": len(results), "signals": results})

    except Exception as e:
        logger.error(f"Batch signal generation failed: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500


# ============== Signal Retrieval ==============

@signals_bp.route('/latest', methods=['GET'])
@signals_bp.route('/latest/<asset>', methods=['GET'])
def get_latest_signal(asset: str = None):
    """Get the most recent signal, optionally for a specific asset."""
    try:
        delivery = _get_delivery()
        signal = delivery.get_latest_signal(asset=asset)
        if signal is None:
            return jsonify({"success": False, "error": "No signals found"}), 404
        return jsonify({"success": True, "signal": signal.model_dump()})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@signals_bp.route('/active', methods=['GET'])
def get_active_signals():
    """Get all currently active signals."""
    try:
        delivery = _get_delivery()
        signals = delivery.get_active_signals()
        return jsonify({
            "success": True,
            "count": len(signals),
            "signals": [s.model_dump() for s in signals],
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@signals_bp.route('/history', methods=['GET'])
@signals_bp.route('/history/<asset>', methods=['GET'])
def get_signal_history(asset: str = None):
    """
    Get historical signals.
    Query params: limit (default: 50)
    """
    try:
        delivery = _get_delivery()
        limit = int(request.args.get("limit", "50"))
        signals = delivery.get_signal_history(asset=asset, limit=limit)
        return jsonify({
            "success": True,
            "count": len(signals),
            "signals": [s.model_dump() for s in signals],
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# ============== Regime Detection ==============

@signals_bp.route('/regime', methods=['GET'])
def detect_regime():
    """
    Detect current market regime for an asset.
    Query params: asset (default: BTC/USDT), asset_type (default: crypto),
                  timeframe (default: 1h), bars (default: 100)
    """
    try:
        asset = request.args.get("asset", "BTC/USDT")
        asset_type = request.args.get("asset_type", "crypto")
        timeframe = request.args.get("timeframe", "1h")
        bars = int(request.args.get("bars", "100"))

        svc = _get_market_data()
        detector = _get_regime_detector()

        if asset_type == "crypto":
            ohlcv = svc.crypto.get_ohlcv(asset, timeframe=timeframe, limit=bars)
        else:
            ohlcv = svc.equity.get_ohlcv(asset, period="6mo", interval="1d")

        if not ohlcv:
            return jsonify({"success": False, "error": "No market data available"}), 400

        regime = detector.detect(ohlcv)
        return jsonify({"success": True, "asset": asset, "data": regime})

    except Exception as e:
        logger.error(f"Regime detection failed: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


# ============== Webhook Management ==============

@signals_bp.route('/webhooks', methods=['GET'])
def list_webhooks():
    """List all registered webhook targets."""
    try:
        delivery = _get_delivery()
        targets = delivery.list_targets()
        return jsonify({"success": True, "targets": targets})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@signals_bp.route('/webhooks', methods=['POST'])
def register_webhook():
    """
    Register a new webhook target.
    Body:
    {
        "name": "openclaw",
        "url": "https://api.openclaw.io/signals/webhook",
        "format": "openclaw",  // mirofish, openclaw, freqtrade, ccxt, threecomas, cornix
        "headers": {"Authorization": "Bearer xxx"},  // optional
        "secret": "hmac_secret"  // optional, for signature verification
    }
    """
    try:
        body = request.get_json() or {}
        name = body.get("name")
        url = body.get("url")

        if not name or not url:
            return jsonify({"success": False, "error": "name and url are required"}), 400

        delivery = _get_delivery()
        target = delivery.register_target(
            name=name,
            url=url,
            format=body.get("format", "mirofish"),
            headers=body.get("headers"),
            secret=body.get("secret"),
        )

        return jsonify({
            "success": True,
            "target": {
                "name": target.name,
                "url": target.url,
                "format": target.format,
                "enabled": target.enabled,
            },
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@signals_bp.route('/webhooks/<name>', methods=['DELETE'])
def unregister_webhook(name: str):
    """Remove a webhook target."""
    try:
        delivery = _get_delivery()
        removed = delivery.unregister_target(name)
        if removed:
            return jsonify({"success": True, "message": f"Target '{name}' removed"})
        return jsonify({"success": False, "error": f"Target '{name}' not found"}), 404
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@signals_bp.route('/webhooks/test/<name>', methods=['POST'])
def test_webhook(name: str):
    """Send a test signal to a specific webhook target."""
    try:
        delivery = _get_delivery()

        # Generate a quick test signal
        from ..models.signal import (
            TradingSignal, SignalDirection, MarketRegime,
            SignalEntry, SignalStopLoss, SignalTakeProfit,
            SignalComponents, EntryType, StopLossType,
        )

        test_signal = TradingSignal(
            asset="TEST/USDT",
            direction=SignalDirection.LONG,
            conviction=0.75,
            entry=SignalEntry(type=EntryType.LIMIT, price=100.0),
            stop_loss=SignalStopLoss(price=95.0, type=StopLossType.ATR, risk_pct=5.0),
            take_profit=[
                SignalTakeProfit(price=110.0, close_pct=60),
                SignalTakeProfit(price=115.0, close_pct=40),
            ],
            position_size_pct=10.0,
            time_horizon_hours=24,
            regime=MarketRegime.TRENDING_BULL,
            components=SignalComponents(
                agent_consensus=0.72,
                agent_conviction_strength=0.68,
                narrative_momentum=0.55,
                flow_imbalance=0.3,
                crowding_risk=0.25,
                information_edge_score=0.6,
            ),
            dominant_narrative="Test signal from MiroFish",
        )

        result = delivery.deliver(test_signal, target_names=[name], async_delivery=False)
        return jsonify({"success": True, "result": result})

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# ============== Delivery Log ==============

@signals_bp.route('/delivery-log', methods=['GET'])
def delivery_log():
    """Get recent delivery log entries."""
    try:
        delivery = _get_delivery()
        limit = int(request.args.get("limit", "100"))
        log = delivery.get_delivery_log(limit=limit)
        return jsonify({"success": True, "count": len(log), "log": log})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# ============== Signal Outcome Recording ==============

@signals_bp.route('/outcome', methods=['POST'])
def record_outcome():
    """
    Record actual outcome for calibration tracking.
    Body:
    {
        "signal_id": "sig_xxx",
        "direction": "LONG",
        "conviction": 0.73,
        "actual_outcome_pct": 2.1
    }
    """
    try:
        body = request.get_json() or {}
        signal_id = body.get("signal_id")
        direction = body.get("direction")
        conviction = body.get("conviction")
        actual = body.get("actual_outcome_pct")

        if not all([signal_id, direction, conviction is not None, actual is not None]):
            return jsonify({"success": False, "error": "All fields required"}), 400

        calibrator = _get_calibrator()
        calibrator.record_outcome(signal_id, direction, conviction, actual)

        return jsonify({"success": True, "message": "Outcome recorded"})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@signals_bp.route('/calibration', methods=['GET'])
def calibration_report():
    """Get calibration quality report."""
    try:
        calibrator = _get_calibrator()
        report = calibrator.calibration_report()
        return jsonify({"success": True, "data": report})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# ============== Trader Archetypes ==============

@signals_bp.route('/archetypes', methods=['GET'])
def list_archetypes():
    """List all available trader archetypes and their traits."""
    try:
        gen = _get_trader_gen()
        summary = gen.get_archetype_summary()
        return jsonify({"success": True, "data": summary})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@signals_bp.route('/archetypes/generate', methods=['POST'])
def generate_population():
    """
    Generate a trader agent population.
    Body:
    {
        "composition": {"momentum": 10, "mean_reversion": 5, ...},  // optional
        "asset": "BTC/USDT",
        "variation_pct": 0.15
    }
    """
    try:
        body = request.get_json() or {}
        gen = _get_trader_gen()

        composition = body.get("composition")
        asset = body.get("asset", "BTC/USDT")
        variation = body.get("variation_pct", 0.15)

        agents = gen.generate_population(
            composition=composition,
            asset=asset,
            variation_pct=variation,
        )

        config = gen.to_simulation_config(agents, asset=asset)

        return jsonify({
            "success": True,
            "count": len(agents),
            "config": config,
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# ============== Financial Ontology ==============

@signals_bp.route('/ontology', methods=['GET'])
def get_financial_ontology():
    """Get the base financial ontology."""
    try:
        from ..services.financial_ontology import FinancialOntologyGenerator
        gen = FinancialOntologyGenerator()
        ontology = gen.get_base_ontology()
        return jsonify({"success": True, "data": ontology})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@signals_bp.route('/ontology/extend', methods=['POST'])
def extend_ontology():
    """
    Extend the base financial ontology with LLM for specific context.
    Body:
    {
        "market_context": "Current focus is on AI infrastructure stocks...",
        "assets": ["NVDA", "AMD", "MSFT"],
        "news_headlines": ["NVIDIA reports record data center revenue..."]
    }
    """
    try:
        body = request.get_json() or {}
        from ..services.financial_ontology import FinancialOntologyGenerator
        gen = FinancialOntologyGenerator()

        ontology = gen.generate(
            market_context=body.get("market_context"),
            assets=body.get("assets"),
            news_headlines=body.get("news_headlines"),
            extend_with_llm=True,
        )
        return jsonify({"success": True, "data": ontology})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

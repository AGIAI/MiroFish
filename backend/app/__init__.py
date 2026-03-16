"""
MiroFish Backend - Flask Application Factory
"""

import os
import warnings

# Suppress multiprocessing resource_tracker warnings (from third-party libraries like transformers)
# Must be set before all other imports
warnings.filterwarnings("ignore", message=".*resource_tracker.*")

from flask import Flask, request
from flask_cors import CORS

from .config import Config
from .utils.logger import setup_logger, get_logger, set_correlation_id


def create_app(config_class=Config):
    """Flask application factory function"""
    app = Flask(__name__)
    app.config.from_object(config_class)

    # Set JSON encoding: ensure Chinese characters are displayed directly (instead of \uXXXX format)
    # Flask >= 2.3 uses app.json.ensure_ascii, older versions use JSON_AS_ASCII config
    if hasattr(app, 'json') and hasattr(app.json, 'ensure_ascii'):
        app.json.ensure_ascii = False

    # Set up logging
    logger = setup_logger('mirofish')

    # Only print startup info in the reloader subprocess (avoid printing twice in debug mode)
    is_reloader_process = os.environ.get('WERKZEUG_RUN_MAIN') == 'true'
    debug_mode = app.config.get('DEBUG', False)
    should_log_startup = not debug_mode or is_reloader_process

    if should_log_startup:
        logger.info("=" * 50)
        logger.info("MiroFish Backend starting...")
        logger.info("=" * 50)

    # Enable CORS (configurable via CORS_ORIGINS env var)
    cors_origins = app.config.get('CORS_ORIGINS', '*')
    if cors_origins == '*':
        allowed_origins = '*'
    else:
        allowed_origins = [o.strip() for o in cors_origins.split(',')]
    CORS(app, resources={r"/api/*": {"origins": allowed_origins}})

    # Register simulation process cleanup function (ensure all simulation processes are terminated when the server shuts down)
    from .services.simulation_runner import SimulationRunner
    SimulationRunner.register_cleanup()
    if should_log_startup:
        logger.info("Simulation process cleanup function registered")

    # Request logging middleware with correlation ID
    # Fields to redact from request body logs
    _SENSITIVE_KEYS = {'api_key', 'password', 'secret', 'token', 'authorization'}

    @app.before_request
    def log_request():
        # Set correlation ID from header or generate a new one
        cid = request.headers.get('X-Correlation-ID') or None
        set_correlation_id(cid)
        req_logger = get_logger('mirofish.request')
        req_logger.debug(f"Request: {request.method} {request.path}")
        if request.content_type and 'json' in request.content_type:
            body = request.get_json(silent=True)
            if body and isinstance(body, dict):
                safe_body = {
                    k: ('***' if k.lower() in _SENSITIVE_KEYS else v)
                    for k, v in body.items()
                }
                req_logger.debug(f"Request body: {safe_body}")
            else:
                req_logger.debug(f"Request body: {body}")

    @app.after_request
    def log_response(response):
        logger = get_logger('mirofish.request')
        logger.debug(f"Response: {response.status_code}")
        return response

    # Register blueprints
    from .api import graph_bp, simulation_bp, report_bp
    app.register_blueprint(graph_bp, url_prefix='/api/graph')
    app.register_blueprint(simulation_bp, url_prefix='/api/simulation')
    app.register_blueprint(report_bp, url_prefix='/api/report')

    # Root route (Issue #133)
    @app.route('/')
    def root():
        return {
            'service': 'MiroFish Backend',
            'status': 'running',
            'version': '1.0.0',
            'endpoints': {
                'health': '/health',
                'graph': '/api/graph',
                'simulation': '/api/simulation',
                'report': '/api/report',
            }
        }

    # Track startup time for uptime calculation
    import time as _time
    _startup_time = _time.time()

    # Health check (enriched for quant-platform monitoring)
    @app.route('/health')
    def health():
        uptime_seconds = int(_time.time() - _startup_time)
        health_data = {
            'status': 'ok',
            'service': 'MiroFish Backend',
            'version': '1.0.0',
            'uptime_seconds': uptime_seconds,
            'config': {
                'llm_configured': bool(Config.LLM_API_KEY),
                'zep_configured': bool(Config.ZEP_API_KEY),
                'llm_model': Config.LLM_MODEL_NAME,
                'debug': Config.DEBUG,
            }
        }
        # Report active simulations if available
        try:
            running = SimulationRunner.get_running_simulations()
            health_data['active_simulations'] = len(running)
        except Exception:
            health_data['active_simulations'] = -1
        return health_data

    # Warn if SECRET_KEY is not explicitly configured
    if not os.environ.get('SECRET_KEY'):
        logger.warning("SECRET_KEY not set in environment; using random key (sessions will not persist across restarts)")

    # Error handlers — always return JSON, never HTML
    @app.errorhandler(404)
    def not_found(error):
        return {'success': False, 'error': 'Endpoint not found'}, 404

    @app.errorhandler(405)
    def method_not_allowed(error):
        return {'success': False, 'error': 'Method not allowed'}, 405

    @app.errorhandler(413)
    def request_entity_too_large(error):
        return {'success': False, 'error': 'Request payload too large (max 50MB)'}, 413

    @app.errorhandler(500)
    def internal_server_error(error):
        logger.error(f"Internal server error: {error}")
        return {'success': False, 'error': 'Internal server error'}, 500

    if should_log_startup:
        logger.info("MiroFish Backend startup complete")

    return app

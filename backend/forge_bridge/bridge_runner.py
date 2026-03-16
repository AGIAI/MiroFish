"""
Bridge Runner — Entry Point

Starts the MiroFish → Forge Signal Bridge service.
This runs as a standalone process (or Docker container) that:
    1. Polls MiroFish for completed simulations
    2. Extracts quantified signals
    3. Writes signals to the shared PostgreSQL/TimescaleDB
    4. Serves a health endpoint for Docker health checks

Usage:
    python -m forge_bridge.bridge_runner
"""

import logging
import signal
import sys
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler

from .bridge_config import BridgeConfig
from .db_writer import SignalDBWriter
from .stance_classifier import StanceClassifier
from .signal_calibrator import SignalCalibrator
from .signal_extractor import SignalExtractor
from .scheduler import SimulationScheduler

logger = logging.getLogger("forge_bridge")


class HealthHandler(BaseHTTPRequestHandler):
    """Simple health check HTTP handler for Docker HEALTHCHECK."""

    db_writer = None

    def do_GET(self):
        if self.path == "/health":
            healthy = self.db_writer.health_check() if self.db_writer else False
            status = 200 if healthy else 503
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(
                b'{"status": "healthy"}' if healthy else b'{"status": "unhealthy"}'
            )
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        """Suppress default HTTP logging."""
        pass


def setup_logging(level: str) -> None:
    """Configure structured logging."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )


def main() -> None:
    """Main entry point for the bridge service."""
    # Load configuration
    config = BridgeConfig.from_env()
    setup_logging(config.log_level)

    # Validate
    errors = config.validate()
    if errors:
        for e in errors:
            logger.error("Configuration error: %s", e)
        sys.exit(1)

    logger.info("Starting MiroFish → Forge Signal Bridge")
    logger.info("  MiroFish URL: %s", config.mirofish_url)
    logger.info("  Signal DB: %s", config.signal_db_url.split("@")[-1])  # hide credentials
    logger.info("  Poll interval: %ds", config.poll_interval_seconds)
    logger.info("  Default asset: %s (%s)", config.default_asset, config.default_asset_class)
    logger.info("  Default timeframe: %s", config.default_timeframe)

    # Initialise components
    db_writer = SignalDBWriter(config.signal_db_url)

    stance_classifier = StanceClassifier(
        api_key=config.llm_api_key,
        base_url=config.llm_base_url,
        model=config.llm_model,
    )

    calibrator = SignalCalibrator(
        strategy_id=f"mirofish_{config.default_asset}_{config.default_timeframe}_v1",
        min_samples=config.calibration_min_samples,
        persistence_dir=config.calibration_persistence_dir,
    )

    extractor = SignalExtractor(
        db_writer=db_writer,
        stance_classifier=stance_classifier,
        calibrator=calibrator,
        mirofish_uploads_dir=config.mirofish_uploads_dir,
        default_asset=config.default_asset,
        default_asset_class=config.default_asset_class,
        default_timeframe=config.default_timeframe,
    )

    scheduler = SimulationScheduler(
        mirofish_url=config.mirofish_url,
        uploads_dir=config.mirofish_uploads_dir,
        poll_interval_seconds=config.poll_interval_seconds,
    )

    # Start health check server in background
    HealthHandler.db_writer = db_writer
    health_server = HTTPServer(("0.0.0.0", config.health_port), HealthHandler)
    health_thread = threading.Thread(target=health_server.serve_forever, daemon=True)
    health_thread.start()
    logger.info("Health check server started on port %d", config.health_port)

    # Graceful shutdown handler
    shutdown_event = threading.Event()

    def handle_shutdown(signum, frame):
        logger.info("Shutdown signal received, cleaning up...")
        shutdown_event.set()
        health_server.shutdown()
        db_writer.close()
        sys.exit(0)

    signal.signal(signal.SIGTERM, handle_shutdown)
    signal.signal(signal.SIGINT, handle_shutdown)

    # Define the extraction callback
    def process_simulation(simulation_id: str) -> None:
        """Extract signal from a completed simulation."""
        logger.info("Processing simulation: %s", simulation_id)
        try:
            result = extractor.extract_signal(simulation_id)
            logger.info(
                "Signal extracted: direction=%.2f, confidence=%.3f, z=%.2f",
                result["direction"],
                result["confidence"],
                result["signal_strength_z"],
            )
        except Exception as e:
            logger.error("Signal extraction failed for %s: %s", simulation_id, e)
            raise

    # Run the polling loop
    logger.info("Bridge service ready. Starting poll loop...")
    scheduler.run_loop(process_simulation)


if __name__ == "__main__":
    main()

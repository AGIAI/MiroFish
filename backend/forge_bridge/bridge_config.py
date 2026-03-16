"""
Bridge Configuration

Loads all configuration from environment variables for the Signal Bridge service.
No hardcoded values — everything is configurable or derived.
"""

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class BridgeConfig:
    """Configuration for the MiroFish → Forge Signal Bridge."""

    # MiroFish connection
    mirofish_url: str = "http://localhost:5001"
    mirofish_uploads_dir: str = "/app/backend/uploads"

    # Database
    signal_db_url: str = "postgresql://forge:forge@localhost:5432/forge_signals"

    # LLM for stance classification
    llm_api_key: str = ""
    llm_base_url: str = "https://api.openai.com/v1"
    llm_model: str = "gpt-4o-mini"

    # Polling
    poll_interval_seconds: int = 300  # 5 minutes

    # Defaults (overridable per simulation)
    default_asset: str = "BTC-USD"
    default_asset_class: str = "crypto_perps"
    default_timeframe: str = "4h"

    # Calibration
    calibration_min_samples: int = 30
    calibration_persistence_dir: str = "/app/data/calibration"

    # Bridge health server
    health_port: int = 8080

    # Logging
    log_level: str = "INFO"

    @classmethod
    def from_env(cls) -> "BridgeConfig":
        """Load configuration from environment variables."""
        return cls(
            mirofish_url=os.environ.get("MIROFISH_URL", cls.mirofish_url),
            mirofish_uploads_dir=os.environ.get(
                "MIROFISH_UPLOADS_DIR", cls.mirofish_uploads_dir
            ),
            signal_db_url=os.environ.get("SIGNAL_DB_URL", cls.signal_db_url),
            llm_api_key=os.environ.get("LLM_API_KEY", ""),
            llm_base_url=os.environ.get("LLM_BASE_URL", cls.llm_base_url),
            llm_model=os.environ.get("LLM_MODEL_NAME", cls.llm_model),
            poll_interval_seconds=int(
                os.environ.get("POLL_INTERVAL_SECONDS", str(cls.poll_interval_seconds))
            ),
            default_asset=os.environ.get("FORGE_ASSET_DEFAULT", cls.default_asset),
            default_asset_class=os.environ.get(
                "FORGE_ASSET_CLASS_DEFAULT", cls.default_asset_class
            ),
            default_timeframe=os.environ.get(
                "FORGE_TIMEFRAME_DEFAULT", cls.default_timeframe
            ),
            calibration_min_samples=int(
                os.environ.get("CALIBRATION_MIN_SAMPLES", str(cls.calibration_min_samples))
            ),
            calibration_persistence_dir=os.environ.get(
                "CALIBRATION_PERSISTENCE_DIR", cls.calibration_persistence_dir
            ),
            health_port=int(os.environ.get("BRIDGE_HEALTH_PORT", str(cls.health_port))),
            log_level=os.environ.get("BRIDGE_LOG_LEVEL", cls.log_level),
        )

    def validate(self) -> list:
        """Validate required configuration fields. Returns list of errors."""
        errors = []
        if not self.llm_api_key:
            errors.append("LLM_API_KEY is required for stance classification")
        if not self.signal_db_url:
            errors.append("SIGNAL_DB_URL is required")
        return errors

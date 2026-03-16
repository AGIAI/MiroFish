"""
Logging Configuration Module
Provides unified log management with output to both console and file
"""

import os
import sys
import logging
import threading
import uuid
from datetime import datetime
from logging.handlers import RotatingFileHandler


# Thread-local storage for correlation IDs
_correlation = threading.local()


def set_correlation_id(correlation_id: str = None) -> str:
    """Set a correlation ID for the current thread/request. Returns the ID."""
    cid = correlation_id or uuid.uuid4().hex[:12]
    _correlation.id = cid
    return cid


def get_correlation_id() -> str:
    """Get the correlation ID for the current thread, or '-' if not set."""
    return getattr(_correlation, 'id', '-')


class CorrelationFilter(logging.Filter):
    """Injects correlation_id into every log record."""
    def filter(self, record):
        record.correlation_id = get_correlation_id()
        return True


def _ensure_utf8_stdout():
    """
    Ensure stdout/stderr use UTF-8 encoding.
    Fixes encoding issues for non-ASCII characters in the Windows console.
    """
    if sys.platform == 'win32':
        # Reconfigure standard output to UTF-8 on Windows
        if hasattr(sys.stdout, 'reconfigure'):
            sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        if hasattr(sys.stderr, 'reconfigure'):
            sys.stderr.reconfigure(encoding='utf-8', errors='replace')


# Log directory
LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'logs')


def setup_logger(name: str = 'mirofish', level: int = logging.DEBUG) -> logging.Logger:
    """
    Set up a logger.

    Args:
        name: Logger name
        level: Log level

    Returns:
        Configured logger
    """
    # Ensure log directory exists
    os.makedirs(LOG_DIR, exist_ok=True)

    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Prevent log propagation to root logger to avoid duplicate output
    logger.propagate = False

    # If handlers already exist, don't add duplicates
    if logger.handlers:
        return logger

    # Add correlation ID filter
    logger.addFilter(CorrelationFilter())

    # Log format (includes correlation_id for traceability)
    detailed_formatter = logging.Formatter(
        '[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] [%(correlation_id)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    simple_formatter = logging.Formatter(
        '[%(asctime)s] %(levelname)s [%(correlation_id)s]: %(message)s',
        datefmt='%H:%M:%S'
    )

    # 1. File handler - detailed logs (named by date, with rotation)
    log_filename = datetime.now().strftime('%Y-%m-%d') + '.log'
    file_handler = RotatingFileHandler(
        os.path.join(LOG_DIR, log_filename),
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(detailed_formatter)

    # 2. Console handler - concise logs (INFO and above)
    # Ensure UTF-8 encoding on Windows to avoid garbled non-ASCII characters
    _ensure_utf8_stdout()
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(simple_formatter)

    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def get_logger(name: str = 'mirofish') -> logging.Logger:
    """
    Get a logger (creates one if it doesn't exist).

    Args:
        name: Logger name

    Returns:
        Logger instance
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        return setup_logger(name)
    return logger


# Create default logger
logger = setup_logger()


# Convenience methods
def debug(msg, *args, **kwargs):
    logger.debug(msg, *args, **kwargs)

def info(msg, *args, **kwargs):
    logger.info(msg, *args, **kwargs)

def warning(msg, *args, **kwargs):
    logger.warning(msg, *args, **kwargs)

def error(msg, *args, **kwargs):
    logger.error(msg, *args, **kwargs)

def critical(msg, *args, **kwargs):
    logger.critical(msg, *args, **kwargs)

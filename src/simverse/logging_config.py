"""Central logging configuration for Simverse."""

import logging
import os
from pathlib import Path
from typing import Optional

_DEFAULT_LEVEL = logging.INFO
_DEFAULT_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_LOG_FILE_ENV = "SIMVERSE_LOG_FILE"


def _ensure_log_dir(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)


def configure_logging(level: int = _DEFAULT_LEVEL, log_file: Optional[str] = None) -> None:
    """Configure root logger with console + optional file handlers."""
    logging.basicConfig(level=level, format=_DEFAULT_FORMAT)

    log_file = log_file or os.environ.get(_LOG_FILE_ENV)
    if not log_file:
        return

    log_path = Path(log_file).expanduser().resolve()
    _ensure_log_dir(log_path)

    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(level)
    file_handler.setFormatter(logging.Formatter(_DEFAULT_FORMAT))

    logging.getLogger().addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    """Convenience wrapper to fetch module loggers."""
    return logging.getLogger(name)

import logging
import os
import sys


def get_logger(name: str, drone_id: str | None = None) -> logging.Logger:
    """Structured stdout logger. Lines look like `[drone-3] message...` so
    `docker compose logs` is grep-friendly across all containers."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    level = os.environ.get("LOG_LEVEL", "INFO").upper()
    logger.setLevel(level)

    handler = logging.StreamHandler(sys.stdout)
    prefix = f"[{drone_id}] " if drone_id else ""
    handler.setFormatter(logging.Formatter(f"{prefix}%(message)s"))
    logger.addHandler(handler)
    logger.propagate = False
    return logger

import logging
import os
import sys
from typing import Any

__all__ = [
    "critical",
    "debug",
    "error",
    "info",
    "log_level",
    "logger",
    "logging",
    "set_log_level",
    "warning",
]

# Create a logger
logger = logging.getLogger("space_robotics_bench")

# Set the logger level
logger.setLevel(os.environ.get("SRB_LOG_LEVEL", "DEBUG").upper())

# Create a handler for logs below WARNING (DEBUG and INFO) to STDOUT
stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.setLevel(logging.DEBUG)  # Allow DEBUG and INFO
stdout_handler.addFilter(lambda record: record.levelno < logging.WARNING)

# Create a handler for logs WARNING and above (WARNING, ERROR, CRITICAL) to STDERR
stderr_handler = logging.StreamHandler(sys.stderr)
stderr_handler.setLevel(logging.WARNING)  # Allow WARNING and above

# Define a formatter and set it for both handlers
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
stdout_handler.setFormatter(formatter)
stderr_handler.setFormatter(formatter)

# Add handlers to the logger
logger.addHandler(stdout_handler)
logger.addHandler(stderr_handler)


def debug(msg: Any, *args, **kwargs):
    logger.debug(msg, *args, **kwargs)


def info(msg: Any, *args, **kwargs):
    logger.info(msg, *args, **kwargs)


def warning(msg: Any, *args, **kwargs):
    logger.warning(msg, *args, **kwargs)


def error(msg: Any, *args, **kwargs):
    logger.error(msg, *args, **kwargs)


def critical(msg: Any, *args, **kwargs):
    logger.critical(msg, *args, **kwargs)


def log_level() -> int:
    return logger.level


def set_log_level(level: str | int):
    def _log_level_from_str(level: str) -> int:
        match level.lower():
            case "debug":
                return logging.DEBUG
            case "info":
                return logging.INFO
            case "warning":
                return logging.WARNING
            case "error":
                return logging.ERROR
            case "critical":
                return logging.CRITICAL
            case _:
                return logging.NOTSET

    if isinstance(level, str):
        logger.setLevel(_log_level_from_str(level))
    else:
        logger.setLevel(level)

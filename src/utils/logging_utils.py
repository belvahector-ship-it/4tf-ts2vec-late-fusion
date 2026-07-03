"""
src/utils/logging_utils.py

Standardized logging setup (M0 — Project Bootstrap).

Purpose
-------
Every module and script in this repository gets its logger through
`get_logger` so that log format, level, and destination (console +
`logs/`) are consistent project-wide. Per DS-04, several validations
(e.g. V-DATA, V-LEAK) require an inspectable log trail of what a
pipeline stage did — this module is what produces that trail.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from src.utils.paths import LOGS_DIR

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_configured_loggers: set[str] = set()


def get_logger(
    name: str,
    log_file: Path | None = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """
    Return a configured logger that writes to console and (optionally) a file.

    Parameters
    ----------
    name : str
        Logger name, conventionally the calling module's `__name__`
        (e.g. "src.data.acquisition") so log lines are traceable to
        their source.
    log_file : Path, optional
        If given, a `FileHandler` is attached writing to this path
        (parent directories created automatically). If omitted, logs
        go to `logs/{name}.log` by default.
    level : int, optional
        Logging level, default `logging.INFO`.

    Returns
    -------
    logging.Logger
        A logger with both a console (stdout) and file handler
        attached. Calling this function multiple times with the same
        `name` returns the same logger without duplicating handlers.
    """
    logger = logging.getLogger(name)

    if name in _configured_loggers:
        return logger

    logger.setLevel(level)
    logger.propagate = False

    formatter = logging.Formatter(fmt=_LOG_FORMAT, datefmt=_DATE_FORMAT)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_file is None:
        safe_name = name.replace(".", "_")
        log_file = LOGS_DIR / f"{safe_name}.log"

    log_file = Path(log_file)
    log_file.parent.mkdir(parents=True, exist_ok=True)

    file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    _configured_loggers.add(name)
    return logger

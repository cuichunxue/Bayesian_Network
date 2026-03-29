"""Structured logging configuration for the bayesian_network package.

Call :func:`setup_logging` once at application startup to configure
console and optional file logging with consistent formatting.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional


_LOG_FORMAT = "%(asctime)s [%(levelname)-8s] %(name)s: %(message)s"
_LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_configured = False


def setup_logging(
    level: int | str = logging.INFO,
    log_file: Optional[str] = None,
) -> None:
    """Configure package-wide logging.

    Parameters
    ----------
    level : int or str
        Logging level (e.g. ``logging.DEBUG``, ``"INFO"``).
    log_file : str, optional
        If given, also write logs to this file.
    """
    global _configured
    if _configured:
        return

    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)

    root = logging.getLogger("bayesian_network")
    root.setLevel(level)

    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_LOG_DATE_FORMAT)

    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(level)
    console.setFormatter(formatter)
    root.addHandler(console)

    # Optional file handler
    if log_file:
        p = Path(log_file)
        p.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(str(p), encoding="utf-8")
        fh.setLevel(level)
        fh.setFormatter(formatter)
        root.addHandler(fh)

    # Prevent propagation to root logger (avoids duplicate lines)
    root.propagate = False
    _configured = True

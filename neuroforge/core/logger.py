"""
neuroforge/core/logger.py
==========================
Structured JSON-lines logger with console + rotating file output.
All subsystems import `get_logger(__name__)`.
"""
from __future__ import annotations
import json
import logging
import logging.handlers
import os
import time
from pathlib import Path


LOG_DIR = Path(__file__).parents[2] / "logs"
LOG_DIR.mkdir(exist_ok=True)


class _JsonFormatter(logging.Formatter):
    """Emit log records as single-line JSON for machine parsing."""

    def format(self, record: logging.LogRecord) -> str:  # type: ignore[override]
        payload = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)),
            "ms": int(record.msecs),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload)


def get_logger(name: str, level: int = logging.DEBUG) -> logging.Logger:
    """Return a module-level logger configured with JSON file + console handlers."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger   # already configured

    logger.setLevel(level)

    # ── Console (human-readable) ─────────────────────────────────────────────
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(
        logging.Formatter(
            "%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    logger.addHandler(ch)

    # ── Rotating JSON file ───────────────────────────────────────────────────
    fh = logging.handlers.RotatingFileHandler(
        LOG_DIR / "neuroforge.jsonl",
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=5,
        encoding="utf-8",
    )
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(_JsonFormatter())
    logger.addHandler(fh)

    return logger

"""Structured logging utilities.

Logs contain request IDs, endpoints, timings, dimensions, model and device —
never raw image bytes, base64 payloads, sensitive paths, or secrets.
"""

from __future__ import annotations

import json
import logging
import sys
from contextvars import ContextVar
from typing import Any

request_id_var: ContextVar[str] = ContextVar("request_id", default="-")


class JsonLineFormatter(logging.Formatter):
    """Minimal JSON-lines formatter with a fixed allow-list of fields."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": request_id_var.get(),
        }
        extra = getattr(record, "fields", None)
        if isinstance(extra, dict):
            payload.update(extra)
        if record.exc_info:
            # Only the exception class name is exposed in logs, not full traces
            # containing file paths.
            exc = record.exc_info[1]
            payload["error_class"] = type(exc).__name__
        return json.dumps(payload, default=str)


def setup_logging(level: str = "INFO") -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonLineFormatter())
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    # Quieten noisy third-party loggers.
    for name in ("uvicorn.access", "PIL"):
        logging.getLogger(name).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def log_fields(logger: logging.Logger, level: str, message: str, **fields: Any) -> None:
    """Emit a structured log line with an attached field dictionary."""
    logger.log(
        getattr(logging, level.upper(), logging.INFO),
        message,
        extra={"fields": fields},
    )

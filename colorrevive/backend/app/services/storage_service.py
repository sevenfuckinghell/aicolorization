"""Temporary-file and optional persistent storage management.

Privacy default: nothing is persisted. Uploads are streamed to a temp file
with a server-generated name (never the client's filename) and deleted in a
finally block. If ENABLE_PERSISTENT_STORAGE=true, results are written under
``storage_dir`` with a retention sweep on each request.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path

from ..config import Settings
from ..utils.logging import get_logger, log_fields

logger = get_logger("colorrevive.storage")


class TempWorkspace:
    """Context manager creating an isolated temp dir per request."""

    def __init__(self, settings: Settings):
        self.settings = settings
        base = Path(settings.temp_dir)
        base.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def session(self, request_id: str):
        directory = Path(
            tempfile.mkdtemp(prefix=f"cr-{request_id[:8]}-", dir=self.settings.temp_dir)
        )
        try:
            yield directory
        finally:
            # Always clean up temporary files, even on error paths.
            shutil.rmtree(directory, ignore_errors=True)


def maybe_persist_result(settings: Settings, directory: Path, filename: str) -> str | None:
    """Copy a result file to persistent storage when explicitly enabled."""
    if not settings.enable_persistent_storage:
        return None
    storage = Path(settings.storage_dir)
    storage.mkdir(parents=True, exist_ok=True)
    dest = storage / filename
    src = directory / filename
    if src.is_file():
        shutil.copy2(src, dest)
        log_fields(logger, "info", "result persisted", filename=filename)
        return str(dest)
    return None


def sweep_expired(settings: Settings) -> int:
    """Delete persisted results older than the configured retention window."""
    if not settings.enable_persistent_storage or settings.retention_seconds <= 0:
        return 0
    storage = Path(settings.storage_dir)
    removed = 0
    cutoff = time.time() - settings.retention_seconds
    if storage.is_dir():
        for entry in storage.iterdir():
            try:
                if entry.is_file() and os.path.getmtime(entry) < cutoff:
                    entry.unlink()
                    removed += 1
            except OSError:
                continue
    if removed:
        log_fields(logger, "info", "retention sweep removed expired files", count=removed)
    return removed

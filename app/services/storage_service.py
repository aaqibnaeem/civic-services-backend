"""File storage strategy.

Two genuinely different implementations sit behind one abstract interface, and the
one in use is chosen by configuration at import time. This is real polymorphism: the
caller (``ComplaintManager``) never branches on environment, and swapping in an S3
backend later means adding a subclass, not editing call sites.
"""

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from pathlib import Path

from app.core.config import Settings, settings
from app.core.logging_config import get_logger

log = get_logger(__name__)

_ALLOWED_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".heic", ".pdf"}


class StorageService(ABC):
    """Abstract file sink for citizen-uploaded evidence photos.

    Responsibility: turn raw bytes into a URL the API can hand back, and answer
    whether storage is available at all. Nothing above this class knows where the
    bytes physically land.
    """

    #: Human-readable name, surfaced by ``/health``.
    backend: str = "abstract"

    @abstractmethod
    async def save(self, data: bytes, *, filename: str) -> str:
        """Persist ``data`` and return the URL that serves it."""

    @abstractmethod
    def url_for(self, key: str) -> str:
        """Return the public URL for a previously stored key."""

    @property
    @abstractmethod
    def available(self) -> bool:
        """Whether writes are expected to succeed."""

    @staticmethod
    def _safe_key(filename: str) -> str:
        """Never trust a client-supplied filename; keep only a vetted extension."""
        suffix = Path(filename or "").suffix.lower()
        if suffix not in _ALLOWED_SUFFIXES:
            suffix = ".bin"
        return f"{uuid.uuid4().hex}{suffix}"


class LocalStorageService(StorageService):
    """Writes uploads to ``settings.UPLOAD_DIR`` and serves them from ``/uploads``.

    Right for a hackathon demo and for local development; the directory is created
    lazily so a fresh checkout needs no setup step.
    """

    backend = "local"

    def __init__(self, upload_dir: str | Path, *, url_prefix: str = "/uploads") -> None:
        self._root = Path(upload_dir)
        self._url_prefix = url_prefix.rstrip("/")

    @property
    def root(self) -> Path:
        return self._root

    @property
    def available(self) -> bool:
        try:
            self._ensure_root()
        except OSError:
            return False
        return True

    def _ensure_root(self) -> Path:
        self._root.mkdir(parents=True, exist_ok=True)
        return self._root

    async def save(self, data: bytes, *, filename: str) -> str:
        key = self._safe_key(filename)
        target = self._ensure_root() / key
        target.write_bytes(data)
        log.info("upload.stored", backend=self.backend, key=key, bytes=len(data))
        return self.url_for(key)

    def url_for(self, key: str) -> str:
        return f"{self._url_prefix}/{key}"


class NoopStorageService(StorageService):
    """Accepts nothing and stores nothing.

    Used on read-only/ephemeral hosts (Render's free tier has no persistent disk),
    where silently writing files that vanish on the next deploy would be worse than
    an explicit "unavailable". Uploads become a no-op instead of a 500.
    """

    backend = "noop"

    @property
    def available(self) -> bool:
        return False

    async def save(self, data: bytes, *, filename: str) -> str:
        log.warning("upload.discarded", backend=self.backend, filename=filename, bytes=len(data))
        return ""

    def url_for(self, key: str) -> str:
        return ""


def build_storage_service(config: Settings | None = None) -> StorageService:
    """Factory: pick the concrete backend from configuration."""
    config = config or settings
    if not config.UPLOAD_DIR or config.UPLOAD_DIR.lower() in {"none", "off", "disabled"}:
        return NoopStorageService()
    return LocalStorageService(config.UPLOAD_DIR)


#: Process-wide singleton, injected via ``app.core.deps.get_storage_service``.
storage_service: StorageService = build_storage_service()

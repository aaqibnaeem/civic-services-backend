"""Async engine / session factory and the Neon-safe URL normaliser.

The single most common deployment failure for this stack is handing an asyncpg
engine a Neon connection string verbatim. Neon issues URLs shaped like::

    postgresql://user:pass@host/db?sslmode=require&channel_binding=require

asyncpg does not accept ``sslmode``/``channel_binding``/``options`` as query
parameters (they are libpq concepts) and raises ``TypeError: connect() got an
unexpected keyword argument 'sslmode'`` at first connect. ``normalize_database_url``
rewrites the scheme and moves TLS into ``connect_args``. It is unit-tested.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any
from urllib.parse import parse_qsl, urlencode

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.core.config import settings
from app.core.logging_config import get_logger

log = get_logger(__name__)

# libpq-only parameters that asyncpg rejects if passed through the DSN.
_STRIPPED_QUERY_PARAMS = {"sslmode", "channel_binding", "options"}

# sslmode values that mean "do not use TLS".
_SSL_DISABLED_VALUES = {"disable", "allow", "prefer"}


def normalize_database_url(raw_url: str) -> tuple[str, dict[str, Any]]:
    """Return ``(sqlalchemy_url, connect_args)`` safe for the async driver.

    * ``postgres://`` / ``postgresql://`` / ``postgresql+psycopg2://`` are upgraded
      to ``postgresql+asyncpg://``.
    * ``sslmode``, ``channel_binding`` and ``options`` query params are removed, and
      TLS is expressed as ``connect_args={"ssl": True}`` instead (Neon requires TLS,
      so it stays on unless the URL explicitly disabled it).
    * ``sqlite`` is upgraded to ``sqlite+aiosqlite`` and gets no connect args
      (``check_same_thread`` is a pysqlite concept and is irrelevant to aiosqlite).
    """
    url = (raw_url or "").strip()
    if not url:
        raise ValueError("DATABASE_URL is empty")

    scheme, separator, remainder = url.partition("://")
    if not separator:
        raise ValueError(f"DATABASE_URL is not a valid connection URL: {url!r}")

    scheme = scheme.lower()
    body, _, query = remainder.partition("?")
    connect_args: dict[str, Any] = {}

    # ---- SQLite ---------------------------------------------------------------
    if scheme.split("+", 1)[0] == "sqlite":
        # aiosqlite gives every connection its own thread, so `check_same_thread`
        # (a pysqlite concern) is meaningless here — drop it if someone set it.
        kept = [
            (k, v)
            for k, v in parse_qsl(query, keep_blank_values=True)
            if k.lower() != "check_same_thread"
        ]
        suffix = f"?{urlencode(kept, doseq=True)}" if kept else ""
        return f"sqlite+aiosqlite://{body}{suffix}", {}

    # ---- Postgres -------------------------------------------------------------
    if scheme.split("+", 1)[0] not in {"postgres", "postgresql"}:
        # Unknown backend: leave it alone rather than guessing.
        return url, {}

    kept = []
    ssl_required = True
    for key, value in parse_qsl(query, keep_blank_values=True):
        lowered = key.lower()
        if lowered in _STRIPPED_QUERY_PARAMS:
            if lowered == "sslmode" and value.lower() in _SSL_DISABLED_VALUES:
                ssl_required = False
            continue
        kept.append((key, value))

    if ssl_required:
        # Neon (and every managed Postgres) requires TLS; asyncpg takes it as a
        # connect arg, never as a DSN query parameter.
        connect_args["ssl"] = True

    suffix = f"?{urlencode(kept, doseq=True)}" if kept else ""
    return f"postgresql+asyncpg://{body}{suffix}", connect_args


def _build_engine() -> AsyncEngine:
    url, connect_args = normalize_database_url(settings.DATABASE_URL)
    kwargs: dict[str, Any] = {
        "echo": False,
        "future": True,
        "pool_pre_ping": True,
        "connect_args": connect_args,
    }
    if url.startswith("postgresql"):
        # Neon's free tier idles connections aggressively; recycle before it does.
        kwargs["pool_size"] = 5
        kwargs["max_overflow"] = 5
        kwargs["pool_recycle"] = 280
    return create_async_engine(url, **kwargs)


engine: AsyncEngine = _build_engine()

SessionLocal: async_sessionmaker[AsyncSession] = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
)


async def get_session() -> AsyncIterator[AsyncSession]:
    """FastAPI dependency yielding a session that is always closed."""
    async with SessionLocal() as session:
        yield session


async def dispose_engine() -> None:
    """Close the pool on shutdown."""
    await engine.dispose()

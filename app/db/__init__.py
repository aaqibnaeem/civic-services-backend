"""Database package: engine, session factory, and schema bootstrap."""

from __future__ import annotations

from app.db.base import Base
from app.db.session import SessionLocal, engine, get_session, normalize_database_url


async def create_all() -> None:
    """Create every table declared on ``Base.metadata``.

    Alembic is deliberately out of scope for this hackathon build: the schema is
    created once at startup and the demo database is disposable, so a migration tool
    would be pure ceremony. Production-grade successors should add one.
    """
    import app.models  # noqa: F401  -- registers every mapper before create_all

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


__all__ = [
    "Base",
    "SessionLocal",
    "create_all",
    "engine",
    "get_session",
    "normalize_database_url",
]

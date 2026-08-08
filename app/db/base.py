"""Declarative base, shared column mixins, and the enum-column helper.

This module deliberately imports **no** model modules: models import from here, so
adding a reverse import would create a circular-import trap the moment a model is
imported before ``app.db.base``. Model registration happens in ``app.models``'s
package ``__init__`` — import that (``import app.models``) when you need
``Base.metadata`` to be complete, e.g. before ``create_all``.
"""

from __future__ import annotations

import enum
import uuid
from datetime import UTC, datetime

from sqlalchemy import DateTime, String, TypeDecorator, func
from sqlalchemy import Enum as SAEnum
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


def utcnow() -> datetime:
    """Timezone-aware UTC now. SQLite has no tz storage, so we normalise in Python."""
    return datetime.now(UTC)


class UTCDateTime(TypeDecorator):
    """A ``DateTime`` that is always timezone-aware UTC in Python.

    SQLite silently discards ``tzinfo`` on write and hands back naive datetimes on
    read, which would make the API emit ``2026-08-08T10:00:00`` — a string the
    browser parses as *local* time. The contract mandates UTC, so this decorator
    stamps UTC on the way in and on the way out, giving byte-identical behaviour on
    SQLite and Postgres.
    """

    impl = DateTime(timezone=True)
    cache_ok = True

    def process_bind_param(self, value: datetime | None, dialect) -> datetime | None:
        if value is None:
            return None
        return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)

    def process_result_value(self, value: datetime | None, dialect) -> datetime | None:
        if value is None:
            return None
        return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


def new_uuid() -> str:
    return str(uuid.uuid4())


class Base(DeclarativeBase):
    """Root of the ORM hierarchy for every model in the app."""


class UUIDPrimaryKeyMixin:
    """Gives a model a string UUID primary key.

    Stored as ``String(36)`` rather than a native ``UUID`` column so the identical
    models run unmodified on SQLite (local dev + tests) and Postgres (Neon).
    """

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)


class TimestampMixin:
    """Adds ORM-maintained ``created_at`` / ``updated_at`` columns."""

    created_at: Mapped[datetime] = mapped_column(
        UTCDateTime,
        default=utcnow,
        server_default=func.now(),
        nullable=False,
        index=True,
    )
    updated_at: Mapped[datetime] = mapped_column(
        UTCDateTime,
        default=utcnow,
        onupdate=utcnow,
        server_default=func.now(),
        nullable=False,
    )


def enum_column(enum_cls: type[enum.Enum], *, name: str) -> SAEnum:
    """Build a VARCHAR+CHECK enum column that stores the **wire value**.

    SQLAlchemy's default behaviour is to persist the *member name* (``ROAD``), not
    the value (``road``). Since the frozen contract fixes the lowercase wire
    strings, ``values_callable`` is mandatory here — this is the classic trap.
    ``native_enum=False`` keeps it a portable VARCHAR + CHECK constraint instead of
    a Postgres ENUM type (which would need migrations we deliberately do not have).
    """
    return SAEnum(
        enum_cls,
        name=name,
        native_enum=False,
        length=32,
        create_constraint=True,
        validate_strings=True,
        values_callable=lambda e: [member.value for member in e],
    )


__all__ = [
    "Base",
    "UTCDateTime",
    "TimestampMixin",
    "UUIDPrimaryKeyMixin",
    "enum_column",
    "new_uuid",
    "utcnow",
]

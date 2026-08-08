"""Shared wire types: enum re-exports, pagination envelope, error envelope.

Other agents should import the enums from here (or from ``app.models``) rather than
redeclaring them — the string values are frozen by the CONTRACT.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, PlainSerializer

from app.models.ai_analysis import AISource, Sentiment
from app.models.complaint import AIStatus, Category, Priority, Status
from app.models.user import Role

SortField = Literal["created_at", "priority", "status", "resolution_hours"]
SortOrder = Literal["asc", "desc"]


def _iso_utc(value: datetime) -> str:
    """Render a datetime as ``2026-08-08T10:00:00Z`` exactly as the CONTRACT shows."""
    aware = value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)
    return aware.isoformat().replace("+00:00", "Z")


# Every timestamp on the wire uses this alias so the frontend never has to guess a
# timezone (a naive ISO string is parsed as *local* time by the browser).
UTCDatetime = Annotated[
    datetime, PlainSerializer(_iso_utc, return_type=str, when_used="json")
]


class ORMModel(BaseModel):
    """Base for every response schema read out of a SQLAlchemy row."""

    model_config = ConfigDict(from_attributes=True)


class Page[T](BaseModel):
    """Paginated list envelope — CONTRACT §5.5 forbids unbounded arrays."""

    items: list[T]
    total: int
    page: int
    page_size: int
    pages: int

    @classmethod
    def build(cls, items: list[T], *, total: int, page: int, page_size: int) -> Page[T]:
        pages = (total + page_size - 1) // page_size if page_size else 0
        return cls(items=items, total=total, page=page, page_size=page_size, pages=pages)


class ErrorDetail(BaseModel):
    """One field-level problem inside an error envelope."""

    field: str | None = None
    issue: str


class ErrorBody(BaseModel):
    code: str
    message: str
    details: list[ErrorDetail] = Field(default_factory=list)
    request_id: str


class ErrorResponse(BaseModel):
    """The exact non-2xx body shape from CONTRACT §4."""

    error: ErrorBody


class HealthResponse(BaseModel):
    """Payload of ``GET /health``."""

    status: str
    database: str
    ai_provider: str
    version: str
    environment: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)


class MessageResponse(BaseModel):
    """Generic acknowledgement body."""

    message: str
    ok: bool = True


__all__ = [
    "AISource",
    "UTCDatetime",
    "AIStatus",
    "Category",
    "ErrorBody",
    "ErrorDetail",
    "ErrorResponse",
    "HealthResponse",
    "MessageResponse",
    "ORMModel",
    "Page",
    "Priority",
    "Role",
    "Sentiment",
    "SortField",
    "SortOrder",
    "Status",
]

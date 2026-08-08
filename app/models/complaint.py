"""Complaint aggregate root, its enums, and the ``resolution_hours`` computation.

The three citizen-facing enums (``Category``, ``Priority``, ``Status``) plus the AI
lifecycle flag (``AIStatus``) live here because the complaint is the aggregate that
owns them. Everything else in the codebase imports them from ``app.models``.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import TYPE_CHECKING

from sqlalchemy import Boolean, Float, ForeignKey, Index, String, Text, extract, func
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.ext.hybrid import hybrid_property
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql.functions import GenericFunction

from app.db.base import Base, TimestampMixin, UTCDateTime, UUIDPrimaryKeyMixin, enum_column

if TYPE_CHECKING:
    from app.models.ai_analysis import AIAnalysis
    from app.models.department import Department
    from app.models.status_event import StatusEvent
    from app.models.user import User


class Category(StrEnum):
    """Wire values: road | water | waste | electricity | drainage | safety | other."""

    ROAD = "road"
    WATER = "water"
    WASTE = "waste"
    ELECTRICITY = "electricity"
    DRAINAGE = "drainage"
    SAFETY = "safety"
    OTHER = "other"


class Priority(StrEnum):
    """Wire values: low | medium | high | critical."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class Status(StrEnum):
    """Wire values: open | assigned | in_progress | resolved | rejected."""

    OPEN = "open"
    ASSIGNED = "assigned"
    IN_PROGRESS = "in_progress"
    RESOLVED = "resolved"
    REJECTED = "rejected"


class AIStatus(StrEnum):
    """Lifecycle of the background enrichment: pending | complete | failed."""

    PENDING = "pending"
    COMPLETE = "complete"
    FAILED = "failed"


# Ordinal ranking used for `sort=priority` — enum members are not orderable by
# themselves, and the alphabetical order of the wire strings is meaningless.
PRIORITY_RANK: dict[Priority, int] = {
    Priority.LOW: 1,
    Priority.MEDIUM: 2,
    Priority.HIGH: 3,
    Priority.CRITICAL: 4,
}

STATUS_RANK: dict[Status, int] = {
    Status.OPEN: 1,
    Status.ASSIGNED: 2,
    Status.IN_PROGRESS: 3,
    Status.RESOLVED: 4,
    Status.REJECTED: 5,
}

#: The statuses that still cost a staff member time. Everything the assignment
#: rule calls "workload" is counted over exactly this set (CONTRACT §4b).
ACTIVE_STATUSES: tuple[Status, ...] = (Status.OPEN, Status.ASSIGNED, Status.IN_PROGRESS)


class hours_between(GenericFunction):  # noqa: N801 - SQL function names are lowercase
    """SQL function returning the number of hours between two timestamps.

    Needed because the contract allows ``sort=resolution_hours`` and analytics run
    percentile maths in SQL, but timestamp arithmetic is dialect specific. The
    ``@compiles`` hooks below emit the right expression per backend so the ORM layer
    stays portable between SQLite (dev/tests) and Postgres (Neon).
    """

    type = Float()
    name = "hours_between"
    inherit_cache = True


@compiles(hours_between)
def _compile_hours_between(element, compiler, **kw):  # pragma: no cover - via SQL
    start, end = list(element.clauses)
    return compiler.process(extract("epoch", end - start) / 3600.0, **kw)


@compiles(hours_between, "sqlite")
def _compile_hours_between_sqlite(element, compiler, **kw):  # pragma: no cover - via SQL
    start, end = list(element.clauses)
    return compiler.process((func.julianday(end) - func.julianday(start)) * 24.0, **kw)


class Complaint(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """A citizen-reported civic issue — the aggregate root of the whole system.

    Owns its AI analysis (1:1), its status timeline (1:N) and an optional pointer to
    the complaint it duplicates. Deletion is soft (``is_deleted``) so analytics keep
    a stable historical denominator and nothing referenced by a reference code ever
    disappears from an audit trail.
    """

    __tablename__ = "complaints"
    __table_args__ = (
        Index("ix_complaints_status_created", "status", "created_at"),
        Index("ix_complaints_category_created", "category", "created_at"),
        Index("ix_complaints_area_created", "area", "created_at"),
    )

    reference_code: Mapped[str] = mapped_column(
        String(16), unique=True, index=True, nullable=False
    )
    title: Mapped[str] = mapped_column(String(200), nullable=False, default="")
    description: Mapped[str] = mapped_column(Text, nullable=False)

    category: Mapped[Category] = mapped_column(
        enum_column(Category, name="category_enum"),
        default=Category.OTHER,
        nullable=False,
        index=True,
    )
    priority: Mapped[Priority] = mapped_column(
        enum_column(Priority, name="priority_enum"),
        default=Priority.MEDIUM,
        nullable=False,
        index=True,
    )
    status: Mapped[Status] = mapped_column(
        enum_column(Status, name="status_enum"), default=Status.OPEN, nullable=False, index=True
    )

    location_text: Mapped[str] = mapped_column(String(300), nullable=False)
    area: Mapped[str | None] = mapped_column(String(120), nullable=True, index=True)
    latitude: Mapped[float | None] = mapped_column(Float, nullable=True)
    longitude: Mapped[float | None] = mapped_column(Float, nullable=True)

    citizen_name: Mapped[str | None] = mapped_column(String(150), nullable=True)
    citizen_phone: Mapped[str | None] = mapped_column(String(40), nullable=True)
    citizen_email: Mapped[str | None] = mapped_column(String(255), nullable=True)
    image_url: Mapped[str | None] = mapped_column(String(500), nullable=True)

    department_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("departments.id", ondelete="SET NULL"), nullable=True, index=True
    )
    duplicate_of_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("complaints.id", ondelete="SET NULL"), nullable=True, index=True
    )

    #: The citizen account this complaint belongs to (CONTRACT §4b). Nullable
    #: because the ~800 seeded rows predate citizen accounts; every complaint filed
    #: through the API from now on has one.
    citizen_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True
    )
    #: The staff member carrying this complaint. Set by ``AssignmentService`` or by
    #: a human via ``PATCH /complaints/{id}``.
    assignee_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True
    )
    assigned_at: Mapped[datetime | None] = mapped_column(UTCDateTime, nullable=True)

    ai_status: Mapped[AIStatus] = mapped_column(
        enum_column(AIStatus, name="ai_status_enum"),
        default=AIStatus.PENDING,
        nullable=False,
        index=True,
    )
    #: True when a human chose the category — the citizen picked it on the submit
    #: form, or a staff member corrected it later. The AI pipeline stores its own
    #: opinion in ``ai_analysis`` either way, but it must not overwrite a human's
    #: choice on the complaint itself.
    category_locked: Mapped[bool] = mapped_column(
        Boolean, default=False, nullable=False, server_default="0"
    )

    resolved_at: Mapped[datetime | None] = mapped_column(
        UTCDateTime, nullable=True, index=True
    )
    is_deleted: Mapped[bool] = mapped_column(
        Boolean, default=False, nullable=False, index=True, server_default="0"
    )

    # --- relationships --------------------------------------------------------
    department: Mapped[Department | None] = relationship(
        "Department", back_populates="complaints", lazy="selectin"
    )
    ai: Mapped[AIAnalysis | None] = relationship(
        "AIAnalysis",
        back_populates="complaint",
        uselist=False,
        cascade="all, delete-orphan",
        lazy="selectin",
    )
    timeline: Mapped[list[StatusEvent]] = relationship(
        "StatusEvent",
        back_populates="complaint",
        cascade="all, delete-orphan",
        order_by="StatusEvent.created_at",
        lazy="selectin",
    )
    duplicate_of: Mapped[Complaint | None] = relationship(
        "Complaint", remote_side="Complaint.id", lazy="noload"
    )
    # Two foreign keys point at ``users``, so the join condition has to be spelled
    # out — SQLAlchemy cannot guess which one each relationship means.
    assignee: Mapped[User | None] = relationship(
        "User", foreign_keys="Complaint.assignee_id", lazy="selectin"
    )
    citizen: Mapped[User | None] = relationship(
        "User", foreign_keys="Complaint.citizen_id", lazy="noload"
    )

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<Complaint {self.reference_code} {self.category}/{self.status}>"

    # --- computed -------------------------------------------------------------
    @hybrid_property
    def resolution_hours(self) -> float | None:
        """Hours between creation and resolution; ``None`` while unresolved."""
        if self.resolved_at is None or self.created_at is None:
            return None
        start = self.created_at
        end = self.resolved_at
        # Rows loaded from SQLite come back naive; assume UTC so subtraction works.
        if start.tzinfo is None:
            start = start.replace(tzinfo=UTC)
        if end.tzinfo is None:
            end = end.replace(tzinfo=UTC)
        return round((end - start).total_seconds() / 3600.0, 2)

    @resolution_hours.inplace.expression
    @classmethod
    def _resolution_hours_expr(cls):
        """SQL form, so ``order_by(Complaint.resolution_hours)`` works on any backend."""
        return hours_between(cls.created_at, cls.resolved_at)

    @property
    def is_open(self) -> bool:
        return self.status in {Status.OPEN, Status.ASSIGNED, Status.IN_PROGRESS}

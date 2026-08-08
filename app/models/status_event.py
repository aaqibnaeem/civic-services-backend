"""Immutable audit entry for a complaint status change (CONTRACT §2 StatusEvent)."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import ForeignKey, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, UTCDateTime, UUIDPrimaryKeyMixin, enum_column, utcnow
from app.models.complaint import Status

if TYPE_CHECKING:
    from app.models.complaint import Complaint


class StatusEvent(UUIDPrimaryKeyMixin, Base):
    """One append-only row per state change, forming the complaint's timeline.

    Only ``ComplaintManager`` writes these — that is what guarantees the timeline is
    a faithful record rather than something a route handler can forget to append.
    """

    __tablename__ = "status_events"

    complaint_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("complaints.id", ondelete="CASCADE"), index=True, nullable=False
    )
    from_status: Mapped[Status | None] = mapped_column(
        enum_column(Status, name="event_from_status_enum"), nullable=True
    )
    to_status: Mapped[Status] = mapped_column(
        enum_column(Status, name="event_to_status_enum"), nullable=False
    )
    note: Mapped[str | None] = mapped_column(Text, nullable=True)
    actor: Mapped[str] = mapped_column(String(150), nullable=False, default="system")
    created_at: Mapped[datetime] = mapped_column(
        UTCDateTime,
        default=utcnow,
        server_default=func.now(),
        nullable=False,
        index=True,
    )

    complaint: Mapped[Complaint] = relationship(
        "Complaint", back_populates="timeline", lazy="noload"
    )

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<StatusEvent {self.from_status}->{self.to_status}>"

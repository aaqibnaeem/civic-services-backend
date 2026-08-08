"""User account model and the ``Role`` enum (CONTRACT §1)."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import TYPE_CHECKING

from sqlalchemy import Boolean, ForeignKey, String, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, UTCDateTime, UUIDPrimaryKeyMixin, enum_column, utcnow

if TYPE_CHECKING:
    from app.models.department import Department


class Role(StrEnum):
    """Wire values: citizen | staff | admin."""

    CITIZEN = "citizen"
    STAFF = "staff"
    ADMIN = "admin"


class User(UUIDPrimaryKeyMixin, Base):
    """A person who can sign in — a citizen, a staff member, or an administrator.

    Citizens never fill in a signup form: ``POST /complaints`` finds-or-creates a
    ``citizen`` row for the submitted email (CONTRACT §4b) so the person can log in
    later and see their own history. Staff rows additionally carry
    ``department_id`` + ``is_available``, which together are the input to the
    auto-assignment rule in :class:`~app.services.assignment_service.AssignmentService`.
    """

    __tablename__ = "users"

    email: Mapped[str] = mapped_column(String(255), unique=True, index=True, nullable=False)
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    full_name: Mapped[str] = mapped_column(String(150), nullable=False, default="")
    role: Mapped[Role] = mapped_column(
        enum_column(Role, name="role_enum"), default=Role.STAFF, nullable=False, index=True
    )
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)

    #: Which department a staff member belongs to. Nullable: citizens and admins
    #: have none, and a staff member can be created before being posted anywhere.
    department_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("departments.id", ondelete="SET NULL"), nullable=True, index=True
    )
    #: Cleared for new work? An unavailable staff member (leave, off-shift) keeps
    #: their existing complaints but is skipped by auto-assignment.
    is_available: Mapped[bool] = mapped_column(
        Boolean, default=True, nullable=False, server_default="1"
    )

    created_at: Mapped[datetime] = mapped_column(
        UTCDateTime, default=utcnow, server_default=func.now(), nullable=False
    )

    department: Mapped[Department | None] = relationship("Department", lazy="selectin")

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<User {self.email} role={self.role}>"

    @property
    def is_admin(self) -> bool:
        return self.role == Role.ADMIN

    @property
    def is_staff(self) -> bool:
        """True for anyone who may carry complaints (staff and admins)."""
        return self.role in {Role.STAFF, Role.ADMIN}

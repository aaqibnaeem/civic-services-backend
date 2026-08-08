"""User account model and the ``Role`` enum (CONTRACT §1)."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from sqlalchemy import Boolean, String, func
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base, UTCDateTime, UUIDPrimaryKeyMixin, enum_column, utcnow


class Role(StrEnum):
    """Wire values: citizen | staff | admin."""

    CITIZEN = "citizen"
    STAFF = "staff"
    ADMIN = "admin"


class User(UUIDPrimaryKeyMixin, Base):
    """A person who can sign in to the admin/staff console.

    Citizens do not need accounts — complaint submission is anonymous by design and
    tracked via ``reference_code`` — so this table only holds municipal staff.
    """

    __tablename__ = "users"

    email: Mapped[str] = mapped_column(String(255), unique=True, index=True, nullable=False)
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    full_name: Mapped[str] = mapped_column(String(150), nullable=False, default="")
    role: Mapped[Role] = mapped_column(
        enum_column(Role, name="role_enum"), default=Role.STAFF, nullable=False, index=True
    )
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        UTCDateTime, default=utcnow, server_default=func.now(), nullable=False
    )

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<User {self.email} role={self.role}>"

    @property
    def is_admin(self) -> bool:
        return self.role == Role.ADMIN

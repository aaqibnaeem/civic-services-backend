"""Municipal department model — the routing target for a complaint."""

from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy import JSON, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, TimestampMixin, UUIDPrimaryKeyMixin

if TYPE_CHECKING:
    from app.models.complaint import Complaint


class Department(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """A municipal unit that complaints get routed to.

    ``categories`` is the list of complaint categories this department handles; the
    routing rules in ``DepartmentService`` read it, which keeps assignment logic
    data-driven instead of hard-coded in a giant if/elif.
    """

    __tablename__ = "departments"

    name: Mapped[str] = mapped_column(String(150), nullable=False)
    slug: Mapped[str] = mapped_column(String(80), unique=True, index=True, nullable=False)
    categories: Mapped[list[str]] = mapped_column(JSON, default=list, nullable=False)
    contact_email: Mapped[str | None] = mapped_column(String(255), nullable=True)

    complaints: Mapped[list[Complaint]] = relationship(
        "Complaint", back_populates="department", lazy="noload"
    )

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<Department {self.slug}>"

    def handles(self, category: str) -> bool:
        """True when this department is responsible for ``category``."""
        return category in (self.categories or [])

"""Department wire schemas (CONTRACT §2 Department)."""

from __future__ import annotations

from pydantic import BaseModel, Field

from app.schemas.common import Category, ORMModel


class DepartmentRef(ORMModel):
    """Nested form embedded inside a Complaint payload."""

    id: str
    name: str
    slug: str


class DepartmentRead(ORMModel):
    """Full department row plus the live open-complaint count."""

    id: str
    name: str
    slug: str
    categories: list[str] = Field(default_factory=list)
    contact_email: str | None = None
    open_complaints: int = 0


class DepartmentCreate(BaseModel):
    """Used by the seeder; there is no public department-creation endpoint."""

    name: str = Field(min_length=2, max_length=150)
    slug: str = Field(min_length=2, max_length=80)
    categories: list[Category] = Field(default_factory=list)
    contact_email: str | None = None

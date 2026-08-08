"""Complaint wire schemas (CONTRACT §2 Complaint, §3 endpoints)."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field, field_validator

from app.schemas.ai import AIAnalysisRead
from app.schemas.common import AIStatus, Category, ORMModel, Priority, Status, UTCDatetime
from app.schemas.department import DepartmentRef


class ComplaintCreate(BaseModel):
    """Public submission body. ``consent`` must be true to store personal data."""

    description: str = Field(min_length=15, max_length=5000)
    location_text: str = Field(min_length=3, max_length=300)
    area: str | None = Field(default=None, max_length=120)
    latitude: float | None = Field(default=None, ge=-90, le=90)
    longitude: float | None = Field(default=None, ge=-180, le=180)
    citizen_name: str | None = Field(default=None, max_length=150)
    citizen_phone: str | None = Field(default=None, max_length=40)
    citizen_email: str | None = Field(default=None, max_length=255)
    image_url: str | None = Field(default=None, max_length=500)
    category: Category | None = None  # citizen hint; the AI may still override it
    consent: bool = True

    @field_validator("description", "location_text", mode="before")
    @classmethod
    def _strip(cls, value: object) -> object:
        return value.strip() if isinstance(value, str) else value

    @field_validator("area", "citizen_name", "citizen_phone", "citizen_email", "image_url")
    @classmethod
    def _blank_to_none(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = value.strip()
        return cleaned or None


class ComplaintUpdate(BaseModel):
    """`PATCH /complaints/{id}` body — every field optional."""

    status: Status | None = None
    priority: Priority | None = None
    category: Category | None = None
    department_id: str | None = None
    note: str | None = Field(default=None, max_length=1000)

    def has_changes(self) -> bool:
        return any(
            value is not None
            for value in (self.status, self.priority, self.category, self.department_id)
        )


class StatusEventRead(ORMModel):
    """One timeline entry."""

    id: str
    from_status: Status | None = None
    to_status: Status
    note: str | None = None
    actor: str
    created_at: UTCDatetime


class ComplaintRead(ORMModel):
    """The canonical Complaint payload from CONTRACT §2."""

    id: str
    reference_code: str
    title: str
    description: str
    category: Category
    priority: Priority
    status: Status
    location_text: str
    area: str | None = None
    latitude: float | None = None
    longitude: float | None = None
    citizen_name: str | None = None
    citizen_phone: str | None = None
    citizen_email: str | None = None
    image_url: str | None = None
    department: DepartmentRef | None = None
    duplicate_of_id: str | None = None
    ai_status: AIStatus
    ai: AIAnalysisRead | None = None
    created_at: UTCDatetime
    updated_at: UTCDatetime
    resolved_at: UTCDatetime | None = None
    resolution_hours: float | None = None


class ComplaintDetail(ComplaintRead):
    """`GET /complaints/{id}` — adds the status timeline."""

    timeline: list[StatusEventRead] = Field(default_factory=list)


class ComplaintFilters(BaseModel):
    """Normalised, validated view of the admin list query string.

    Built once in the router and handed to the repository, which turns it into SQL.
    Keeping it a real object (rather than 12 loose kwargs) is what lets the analytics
    module reuse the exact same filtering semantics.
    """

    q: str | None = None
    category: list[Category] = Field(default_factory=list)
    priority: list[Priority] = Field(default_factory=list)
    status: list[Status] = Field(default_factory=list)
    department_id: str | None = None
    area: str | None = None
    date_from: datetime | None = None
    date_to: datetime | None = None
    sort: str = "created_at"
    order: str = "desc"
    page: int = 1
    page_size: int = 20

"""Complaint wire schemas (CONTRACT §2 Complaint, §3 endpoints, §4b v2 additions)."""

from __future__ import annotations

import re
from datetime import datetime

from pydantic import BaseModel, Field, field_validator, model_validator

from app.schemas.ai import AIAnalysisRead
from app.schemas.common import AIStatus, Category, ORMModel, Priority, Status, UTCDatetime
from app.schemas.department import DepartmentRef
from app.schemas.user import AccountRead

#: Deliberately permissive: this gate exists to catch typos ("ali@gmail"), not to
#: adjudicate RFC 5322. Anything stricter rejects real addresses, and the only
#: authority on deliverability is a delivered email.
_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s.]+(\.[^@\s.]+)+$")

_EMAIL_REQUIRED = (
    "An email address is required — we use it to create your account so you can "
    "track this complaint after you submit it."
)
_EMAIL_INVALID = "Enter a valid email address, for example name@example.com."


class ComplaintCreate(BaseModel):
    """Public submission body. ``consent`` must be true to store personal data.

    ``citizen_email`` is **required** as of CONTRACT §4b: the API finds-or-creates a
    ``citizen`` account for it so the person can sign in and see their own history
    without ever filling in a signup form.
    """

    description: str = Field(min_length=15, max_length=5000)
    location_text: str = Field(min_length=3, max_length=300)
    citizen_email: str = Field(
        min_length=5,
        max_length=255,
        description="Required. A citizen account is created for this address on first use.",
    )
    area: str | None = Field(default=None, max_length=120)
    latitude: float | None = Field(default=None, ge=-90, le=90)
    longitude: float | None = Field(default=None, ge=-180, le=180)
    citizen_name: str | None = Field(default=None, max_length=150)
    citizen_phone: str | None = Field(default=None, max_length=40)
    image_url: str | None = Field(default=None, max_length=500)
    category: Category | None = None  # citizen hint; the AI may still override it
    consent: bool = True

    @model_validator(mode="before")
    @classmethod
    def _surface_missing_email(cls, data: object) -> object:
        """Route an omitted ``citizen_email`` through the field validator.

        Pydantic's stock message for an absent required field is "Field required",
        which tells a citizen nothing. Materialising the key as ``None`` means the
        validator below runs and the frontend gets an explanation it can render
        next to the input, still keyed to ``citizen_email`` in the error envelope.
        """
        if isinstance(data, dict) and "citizen_email" not in data:
            return {**data, "citizen_email": None}
        return data

    @field_validator("citizen_email", mode="before")
    @classmethod
    def _validate_email(cls, value: object) -> object:
        if value is None or isinstance(value, str):
            text = (value or "").strip()
            if not text:
                raise ValueError(_EMAIL_REQUIRED)
            if not _EMAIL_RE.match(text) or len(text) > 255:
                raise ValueError(_EMAIL_INVALID)
            return text.lower()
        raise ValueError(_EMAIL_INVALID)

    @field_validator("description", "location_text", mode="before")
    @classmethod
    def _strip(cls, value: object) -> object:
        return value.strip() if isinstance(value, str) else value

    @field_validator("area", "citizen_name", "citizen_phone", "image_url")
    @classmethod
    def _blank_to_none(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = value.strip()
        return cleaned or None


class ComplaintUpdate(BaseModel):
    """`PATCH /complaints/{id}` body — every field optional.

    ``assignee_id`` needs three states, not two: absent (leave the assignee alone),
    a uuid (assign), and an explicit ``null`` (unassign). A plain ``str | None =
    None`` collapses the first and third, so callers consult
    :attr:`assignee_provided`, which reads pydantic's ``model_fields_set``.
    """

    status: Status | None = None
    priority: Priority | None = None
    category: Category | None = None
    department_id: str | None = None
    assignee_id: str | None = None
    note: str | None = Field(default=None, max_length=1000)

    @property
    def assignee_provided(self) -> bool:
        """True when the caller sent ``assignee_id`` at all, ``null`` included."""
        return "assignee_id" in self.model_fields_set

    def has_changes(self) -> bool:
        return self.assignee_provided or any(
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


class AssigneeRef(ORMModel):
    """The staff member carrying a complaint, nested per CONTRACT §4b."""

    id: str
    full_name: str
    email: str
    department_id: str | None = None


class ComplaintRead(ORMModel):
    """The canonical Complaint payload from CONTRACT §2 + §4b."""

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
    citizen_id: str | None = None
    image_url: str | None = None
    department: DepartmentRef | None = None
    assignee: AssigneeRef | None = None
    assigned_at: UTCDatetime | None = None
    duplicate_of_id: str | None = None
    ai_status: AIStatus
    ai: AIAnalysisRead | None = None
    created_at: UTCDatetime
    updated_at: UTCDatetime
    resolved_at: UTCDatetime | None = None
    resolution_hours: float | None = None


class ComplaintCreated(ComplaintRead):
    """`POST /complaints` response — the complaint plus its citizen account block."""

    account: AccountRead


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
    assignee_id: str | None = None
    citizen_id: str | None = None
    area: str | None = None
    date_from: datetime | None = None
    date_to: datetime | None = None
    sort: str = "created_at"
    order: str = "desc"
    page: int = 1
    page_size: int = 20

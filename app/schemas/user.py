"""Auth and user wire schemas (CONTRACT §3 Auth, §4b staff/assignment)."""

from __future__ import annotations

from pydantic import BaseModel, Field

from app.schemas.common import ORMModel, Role
from app.schemas.department import DepartmentRef


class UserRead(ORMModel):
    """`{id, email, full_name, role}` plus the §4b staff fields.

    ``department`` is the nested object the contract asks for; ``department_id`` is
    kept alongside it so a client can PATCH the value straight back without digging
    into the nested shape. ``active_assignments`` is only populated on the staff
    directory endpoints — it costs an aggregate query, so ``/auth/me`` leaves it
    ``null`` rather than paying for it on every request.
    """

    id: str
    email: str
    full_name: str
    role: Role
    is_active: bool = True
    is_available: bool = True
    department_id: str | None = None
    department: DepartmentRef | None = None
    active_assignments: int | None = None


class StaffRead(UserRead):
    """A staff directory row — ``active_assignments`` is always a real number here."""

    active_assignments: int = 0

    @classmethod
    def from_user(cls, user: object, *, active_assignments: int) -> StaffRead:
        """Build a row from a ``User`` plus its separately-aggregated workload."""
        return cls(
            **UserRead.model_validate(user).model_dump(exclude={"active_assignments"}),
            active_assignments=active_assignments,
        )


class UserUpdate(BaseModel):
    """`PATCH /users/{id}` body (admin).

    ``department_id`` distinguishes "not provided" from "explicitly null" via
    :attr:`model_fields_set`, so an admin can clear a posting by sending
    ``{"department_id": null}`` without that being indistinguishable from omitting
    the field.
    """

    department_id: str | None = None
    is_available: bool | None = None
    full_name: str | None = Field(default=None, max_length=150)
    is_active: bool | None = None

    @property
    def department_provided(self) -> bool:
        return "department_id" in self.model_fields_set


class LoginRequest(BaseModel):
    email: str = Field(min_length=3, max_length=255)
    password: str = Field(min_length=1, max_length=200)


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserRead


class UserCreate(BaseModel):
    """Used by the seeder / bootstrap, not exposed as a public endpoint."""

    email: str = Field(min_length=3, max_length=255)
    password: str = Field(min_length=8, max_length=200)
    full_name: str = ""
    role: Role = Role.STAFF


class AccountRead(BaseModel):
    """The ``account`` block on a `POST /complaints` response (CONTRACT §4b).

    ``default_password`` is populated **only** when this request created the
    account. For an email that already had one it is ``null`` — returning it would
    hand the caller a working credential for somebody else's account.
    """

    email: str
    is_new: bool
    default_password: str | None = None

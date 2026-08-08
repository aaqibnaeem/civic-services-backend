"""Auth and user wire schemas (CONTRACT §3 Auth)."""

from __future__ import annotations

from pydantic import BaseModel, Field

from app.schemas.common import ORMModel, Role


class UserRead(ORMModel):
    """`{id, email, full_name, role}` — exactly what the contract returns."""

    id: str
    email: str
    full_name: str
    role: Role


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

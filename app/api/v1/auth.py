"""Auth endpoints (CONTRACT §3 Auth)."""

from __future__ import annotations

from fastapi import APIRouter

from app.core.deps import AuthServiceDep, CurrentUser
from app.schemas.user import LoginRequest, TokenResponse, UserRead

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/login", response_model=TokenResponse, summary="Sign in and receive a JWT")
async def login(payload: LoginRequest, auth: AuthServiceDep) -> TokenResponse:
    """Exchange email + password for a bearer token.

    Demo credentials seeded by ``scripts/seed.py``: ``admin@civic.gov.pk`` / ``Admin@123``.
    """
    return await auth.login(payload.email, payload.password)


@router.get("/me", response_model=UserRead, summary="The signed-in user")
async def me(user: CurrentUser) -> UserRead:
    return UserRead.model_validate(user)

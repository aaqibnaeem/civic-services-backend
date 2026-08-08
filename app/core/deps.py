"""FastAPI dependency providers.

This is the composition root: it wires a request-scoped ``AsyncSession`` into
repositories, repositories into services, and services into routers. Route handlers
depend on *services*, never on a session — that is what enforces the
router -> service -> repository -> DB layering.

Other agents: depend on ``CurrentUser`` / ``AdminUser`` for auth, and on
``ComplaintManagerDep`` if you need complaint reads. Do not open your own sessions
inside a request.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings, get_settings
from app.core.errors import UnauthorizedError
from app.db.session import get_session
from app.models.user import Role, User
from app.repositories.complaint_repo import ComplaintRepository
from app.repositories.department_repo import DepartmentRepository
from app.repositories.user_repo import UserRepository
from app.services.auth_service import AuthService
from app.services.complaint_service import ComplaintManager
from app.services.department_service import DepartmentService
from app.services.notification_service import NotificationDispatcher, notification_dispatcher
from app.services.storage_service import StorageService, storage_service

# auto_error=False so a missing header raises our envelope, not FastAPI's.
bearer_scheme = HTTPBearer(auto_error=False, scheme_name="BearerAuth")

SessionDep = Annotated[AsyncSession, Depends(get_session)]
SettingsDep = Annotated[Settings, Depends(get_settings)]


# --------------------------------------------------------------------- repositories
def get_complaint_repository(session: SessionDep) -> ComplaintRepository:
    return ComplaintRepository(session)


def get_department_repository(session: SessionDep) -> DepartmentRepository:
    return DepartmentRepository(session)


def get_user_repository(session: SessionDep) -> UserRepository:
    return UserRepository(session)


ComplaintRepoDep = Annotated[ComplaintRepository, Depends(get_complaint_repository)]
DepartmentRepoDep = Annotated[DepartmentRepository, Depends(get_department_repository)]
UserRepoDep = Annotated[UserRepository, Depends(get_user_repository)]


# -------------------------------------------------------------------- infrastructure
def get_storage_service() -> StorageService:
    """The configured storage backend (local disk or no-op)."""
    return storage_service


def get_notification_dispatcher() -> NotificationDispatcher:
    """The process-wide notification fan-out."""
    return notification_dispatcher


StorageDep = Annotated[StorageService, Depends(get_storage_service)]
NotifierDep = Annotated[NotificationDispatcher, Depends(get_notification_dispatcher)]


# ------------------------------------------------------------------------- services
def get_department_service(
    department_repo: DepartmentRepoDep, complaint_repo: ComplaintRepoDep
) -> DepartmentService:
    return DepartmentService(department_repo, complaint_repo)


DepartmentServiceDep = Annotated[DepartmentService, Depends(get_department_service)]


def get_complaint_manager(
    complaint_repo: ComplaintRepoDep,
    departments: DepartmentServiceDep,
    notifier: NotifierDep,
) -> ComplaintManager:
    return ComplaintManager(complaint_repo, departments=departments, notifier=notifier)


ComplaintManagerDep = Annotated[ComplaintManager, Depends(get_complaint_manager)]


def get_auth_service(user_repo: UserRepoDep) -> AuthService:
    return AuthService(user_repo)


AuthServiceDep = Annotated[AuthService, Depends(get_auth_service)]


# ----------------------------------------------------------------------------- auth
async def get_current_user(
    auth: AuthServiceDep,
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer_scheme)],
) -> User:
    """Resolve the caller from the ``Authorization: Bearer <jwt>`` header."""
    if credentials is None or not credentials.credentials:
        raise UnauthorizedError("Authentication is required.")
    return await auth.user_from_token(credentials.credentials)


CurrentUser = Annotated[User, Depends(get_current_user)]


async def require_admin(user: CurrentUser) -> User:
    """Admin-only guard (soft delete, destructive operations)."""
    return AuthService.require_role(user, Role.ADMIN)


AdminUser = Annotated[User, Depends(require_admin)]


async def require_staff(user: CurrentUser) -> User:
    """Staff-or-admin guard, used by the console endpoints."""
    return AuthService.require_role(user, Role.ADMIN, Role.STAFF)


StaffUser = Annotated[User, Depends(require_staff)]


def get_request_id(request: Request) -> str:
    """The current ``X-Request-ID`` (set by ``RequestContextMiddleware``)."""
    return getattr(request.state, "request_id", "-")


RequestIdDep = Annotated[str, Depends(get_request_id)]

__all__ = [
    "AdminUser",
    "AuthServiceDep",
    "ComplaintManagerDep",
    "ComplaintRepoDep",
    "CurrentUser",
    "DepartmentRepoDep",
    "DepartmentServiceDep",
    "NotifierDep",
    "RequestIdDep",
    "SessionDep",
    "SettingsDep",
    "StaffUser",
    "StorageDep",
    "UserRepoDep",
    "get_current_user",
    "require_admin",
    "require_staff",
]

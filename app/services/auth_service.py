"""Authentication use-cases: login, token issuance, current-user resolution."""

from __future__ import annotations

from app.core.errors import ForbiddenError, UnauthorizedError
from app.core.logging_config import get_logger
from app.core.security import (
    create_access_token,
    decode_access_token,
    hash_password,
    verify_password,
)
from app.models.user import Role, User
from app.repositories.user_repo import UserRepository
from app.schemas.user import TokenResponse, UserRead

log = get_logger(__name__)


class AuthService:
    """Owns every credential decision in the application.

    Why a class: authentication is a *policy* (which failures are indistinguishable,
    when an account counts as disabled, what goes in a token) and policy belongs in
    one auditable object rather than scattered across route handlers.
    """

    def __init__(self, user_repo: UserRepository) -> None:
        self._users = user_repo

    async def authenticate(self, email: str, password: str) -> User:
        """Verify credentials, or raise :class:`UnauthorizedError`.

        Wrong-email and wrong-password produce the identical message on purpose:
        distinguishing them leaks which accounts exist.
        """
        user = await self._users.get_by_email(email)
        if user is None or not verify_password(password, user.hashed_password):
            log.info("auth.login_failed", email=email)
            raise UnauthorizedError("Incorrect email or password.")
        if not user.is_active:
            raise ForbiddenError("This account has been deactivated.")
        log.info("auth.login_ok", user_id=user.id, role=str(user.role))
        return user

    async def login(self, email: str, password: str) -> TokenResponse:
        """Full login use-case: authenticate then mint a JWT."""
        user = await self.authenticate(email, password)
        token = create_access_token(user.id, role=str(user.role), email=user.email)
        return TokenResponse(
            access_token=token,
            token_type="bearer",
            user=UserRead.model_validate(user),
        )

    async def user_from_token(self, token: str) -> User:
        """Resolve a bearer token to a live, active user row."""
        payload = decode_access_token(token)
        user_id = payload.get("sub")
        if not user_id:
            raise UnauthorizedError("Invalid authentication token.")
        user = await self._users.get_by_id(str(user_id))
        if user is None:
            raise UnauthorizedError("The account for this token no longer exists.")
        if not user.is_active:
            raise ForbiddenError("This account has been deactivated.")
        return user

    async def ensure_user(
        self, *, email: str, password: str, full_name: str, role: Role = Role.ADMIN
    ) -> User:
        """Idempotently create an account — used by the seeder and startup bootstrap."""
        existing = await self._users.get_by_email(email)
        if existing is not None:
            return existing
        user = User(
            email=email.strip().lower(),
            hashed_password=hash_password(password),
            full_name=full_name,
            role=role,
            is_active=True,
        )
        return self._users.add(user)

    @staticmethod
    def require_role(user: User, *allowed: Role) -> User:
        """Guard helper shared by the ``require_admin`` dependency."""
        if user.role not in allowed:
            raise ForbiddenError("You do not have permission to perform this action.")
        return user

"""Authentication use-cases: login, token issuance, current-user resolution."""

from __future__ import annotations

from app.core.errors import ForbiddenError, NotFoundError, UnauthorizedError
from app.core.logging_config import get_logger
from app.core.security import (
    create_access_token,
    decode_access_token,
    hash_password,
    verify_password,
)
from app.models.user import Role, User
from app.repositories.user_repo import UserRepository
from app.schemas.user import TokenResponse, UserRead, UserUpdate

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
        self,
        *,
        email: str,
        password: str,
        full_name: str,
        role: Role = Role.ADMIN,
        department_id: str | None = None,
        is_available: bool = True,
    ) -> User:
        """Idempotently create an account — used by the seeder and startup bootstrap.

        An account that already exists is returned untouched: neither its password
        nor its posting is overwritten, so re-running the seeder cannot lock anyone
        out of an environment.
        """
        existing = await self._users.get_by_email(email)
        if existing is not None:
            return existing
        user = User(
            email=email.strip().lower(),
            hashed_password=hash_password(password),
            full_name=full_name,
            role=role,
            is_active=True,
            department_id=department_id,
            is_available=is_available,
        )
        return self._users.add(user)

    async def find_or_create_citizen(
        self, email: str, *, full_name: str | None = None, password: str | None = None
    ) -> tuple[User, bool]:
        """Get the citizen account behind a complaint submission (CONTRACT §4b).

        Returns ``(user, is_new)``. The caller needs that flag because the default
        password may only be echoed back when *this* request minted the account —
        returning it for a pre-existing one would hand the submitter working
        credentials to somebody else's history.

        An existing account is never modified. Not its password (that would let
        anyone reset any account by filing a complaint in its owner's name), and
        not its role (a staff member who files a complaint from their work address
        must not be silently demoted to ``citizen``).
        """
        from app.core.config import settings  # noqa: PLC0415 - avoids an import cycle

        normalised = email.strip().lower()
        existing = await self._users.get_by_email(normalised)
        if existing is not None:
            return existing, False

        user = User(
            email=normalised,
            hashed_password=hash_password(password or settings.CITIZEN_DEFAULT_PASSWORD),
            full_name=(full_name or "").strip() or normalised.split("@")[0],
            role=Role.CITIZEN,
            is_active=True,
        )
        self._users.add(user)
        log.info("auth.citizen_created", email=normalised)
        return user, True

    async def update_user(self, user_id: str, payload: UserUpdate) -> User:
        """Apply an admin's `PATCH /users/{id}` (CONTRACT §4b).

        Only the four fields an operator legitimately administers are writable —
        posting, availability, display name and the active flag. Email, role and
        password are deliberately not settable here: changing any of them is a
        privilege or identity change, not a rostering change.
        """
        user = await self._users.get_by_id(user_id)
        if user is None:
            raise NotFoundError(f"No user with id '{user_id}'.")

        if payload.department_provided:
            user.department_id = payload.department_id
        if payload.is_available is not None:
            user.is_available = payload.is_available
        if payload.full_name is not None:
            user.full_name = payload.full_name
        if payload.is_active is not None:
            user.is_active = payload.is_active

        await self._users.session.commit()
        await self._users.session.refresh(user)
        log.info("user.updated", user_id=user.id, role=str(user.role))
        return user

    @staticmethod
    def require_role(user: User, *allowed: Role) -> User:
        """Guard helper shared by the ``require_admin`` dependency."""
        if user.role not in allowed:
            raise ForbiddenError("You do not have permission to perform this action.")
        return user

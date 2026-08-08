"""All SQL for user accounts."""

from __future__ import annotations

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.user import Role, User


class UserRepository:
    """Data-access object for ``User``.

    Exists so ``AuthService`` can talk about "find the account for this email"
    without importing ``select()``; it also normalises email casing in exactly one
    place, which is where login bugs otherwise live.
    """

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def get_by_id(self, user_id: str) -> User | None:
        stmt = select(User).where(User.id == user_id)
        return (await self.session.execute(stmt)).scalar_one_or_none()

    async def get_by_email(self, email: str) -> User | None:
        stmt = select(User).where(func.lower(User.email) == email.strip().lower())
        return (await self.session.execute(stmt)).scalar_one_or_none()

    async def count(self) -> int:
        return int((await self.session.execute(select(func.count()).select_from(User))).scalar_one())

    def add(self, user: User) -> User:
        self.session.add(user)
        return user

    async def list_all(self, *, limit: int = 100) -> list[User]:
        stmt = select(User).order_by(User.created_at.desc()).limit(limit)
        return list((await self.session.execute(stmt)).scalars())

    async def list_staff(
        self,
        *,
        department_id: str | None = None,
        only_available: bool = False,
        include_admins: bool = False,
    ) -> list[User]:
        """Staff accounts, optionally narrowed to one department.

        Ordered by ``id`` because the assignment rule's final tie-break is on user
        id — sorting here means the caller's ``min()`` is deterministic without
        needing a secondary sort of its own.
        """
        roles = [Role.STAFF, Role.ADMIN] if include_admins else [Role.STAFF]
        stmt = select(User).where(User.role.in_(roles), User.is_active.is_(True))
        if department_id is not None:
            stmt = stmt.where(User.department_id == department_id)
        if only_available:
            stmt = stmt.where(User.is_available.is_(True))
        return list((await self.session.execute(stmt.order_by(User.id.asc()))).scalars())

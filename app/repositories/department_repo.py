"""All SQL for departments."""

from __future__ import annotations

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.department import Department


class DepartmentRepository:
    """Data-access object for ``Department``.

    Departments are small and read-mostly, so this repository is intentionally thin —
    but keeping it means the routing service never opens a ``select()`` of its own,
    which is the invariant that keeps the layering honest.
    """

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def list_all(self) -> list[Department]:
        stmt = select(Department).order_by(Department.name.asc())
        return list((await self.session.execute(stmt)).scalars())

    async def get_by_id(self, department_id: str) -> Department | None:
        stmt = select(Department).where(Department.id == department_id)
        return (await self.session.execute(stmt)).scalar_one_or_none()

    async def get_by_slug(self, slug: str) -> Department | None:
        stmt = select(Department).where(Department.slug == slug.strip().lower())
        return (await self.session.execute(stmt)).scalar_one_or_none()

    async def get_by_name(self, name: str) -> Department | None:
        stmt = select(Department).where(func.lower(Department.name) == name.strip().lower())
        return (await self.session.execute(stmt)).scalar_one_or_none()

    async def count(self) -> int:
        stmt = select(func.count()).select_from(Department)
        return int((await self.session.execute(stmt)).scalar_one())

    def add(self, department: Department) -> Department:
        self.session.add(department)
        return department

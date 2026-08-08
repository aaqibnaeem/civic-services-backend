"""Department reads and category-based routing."""

from __future__ import annotations

from app.core.errors import NotFoundError
from app.models.complaint import Category
from app.models.department import Department
from app.repositories.complaint_repo import ComplaintRepository
from app.repositories.department_repo import DepartmentRepository
from app.schemas.department import DepartmentRead

#: Fallback used when no department claims a category (e.g. `other`).
DEFAULT_DEPARTMENT_SLUG = "general"


class DepartmentService:
    """Answers "who owns this complaint?" and serves the department directory.

    The routing table is not hard-coded here — it is read from each department's
    ``categories`` list, so an operator can re-route a whole category by editing one
    row instead of shipping new code.
    """

    def __init__(
        self,
        department_repo: DepartmentRepository,
        complaint_repo: ComplaintRepository | None = None,
    ) -> None:
        self._departments = department_repo
        self._complaints = complaint_repo

    async def list_with_counts(self) -> list[DepartmentRead]:
        """Directory for ``GET /departments``, including live open-complaint counts."""
        departments = await self._departments.list_all()
        counts: dict[str, int] = {}
        if self._complaints is not None:
            counts = await self._complaints.count_open_by_department()
        return [
            DepartmentRead(
                id=d.id,
                name=d.name,
                slug=d.slug,
                categories=list(d.categories or []),
                contact_email=d.contact_email,
                open_complaints=counts.get(d.id, 0),
            )
            for d in departments
        ]

    async def get_or_404(self, department_id: str) -> Department:
        department = await self._departments.get_by_id(department_id)
        if department is None:
            raise NotFoundError(f"No department with id '{department_id}'.")
        return department

    async def route_for_category(self, category: Category | str) -> Department | None:
        """Pick the department that handles ``category``, falling back to general."""
        value = category.value if isinstance(category, Category) else str(category)
        for department in await self._departments.list_all():
            if department.handles(value):
                return department
        return await self._departments.get_by_slug(DEFAULT_DEPARTMENT_SLUG)

    async def resolve_by_name_or_slug(self, needle: str | None) -> Department | None:
        """Best-effort lookup used to honour the AI's ``department_suggestion``."""
        if not needle:
            return None
        return await self._departments.get_by_slug(needle) or await self._departments.get_by_name(
            needle
        )

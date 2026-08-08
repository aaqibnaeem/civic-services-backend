"""Department directory endpoints (CONTRACT §3 Public, §4b staff roster)."""

from __future__ import annotations

from fastapi import APIRouter

from app.core.deps import AssignmentServiceDep, DepartmentServiceDep, StaffUser
from app.schemas.department import DepartmentRead
from app.schemas.user import StaffRead

router = APIRouter(prefix="/departments", tags=["departments"])


@router.get("", response_model=list[DepartmentRead], summary="List departments")
async def list_departments(service: DepartmentServiceDep) -> list[DepartmentRead]:
    """Public directory including each department's live open-complaint count."""
    return await service.list_with_counts()


@router.get(
    "/{department_id}/staff",
    response_model=list[StaffRead],
    summary="Staff in one department, with workload (staff)",
)
async def department_staff(
    department_id: str,
    _user: StaffUser,
    departments: DepartmentServiceDep,
    assignments: AssignmentServiceDep,
) -> list[StaffRead]:
    """The roster the assignment rule draws from, with each member's active count.

    404s on an unknown department rather than returning an empty list, so a typo in
    the id is distinguishable from a department that genuinely has nobody posted.
    """
    await departments.get_or_404(department_id)
    rows = await assignments.staff_directory(department_id=department_id)
    return [StaffRead.from_user(r.user, active_assignments=r.count) for r in rows]

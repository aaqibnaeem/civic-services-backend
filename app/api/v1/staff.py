"""Staff directory and user administration (CONTRACT §4b).

``GET /staff`` and ``PATCH /users/{id}`` are the two halves of managing who can be
auto-assigned: the first shows the current workload distribution, the second is how
an admin changes a posting or marks someone unavailable.
"""

from __future__ import annotations

from fastapi import APIRouter

from app.core.deps import AdminUser, AssignmentServiceDep, AuthServiceDep
from app.core.logging_config import get_logger
from app.schemas.user import StaffRead, UserRead, UserUpdate

log = get_logger(__name__)

router = APIRouter(tags=["staff"])


@router.get("/staff", response_model=list[StaffRead], summary="Staff directory (admin)")
async def list_staff(_user: AdminUser, assignments: AssignmentServiceDep) -> list[StaffRead]:
    """Every staff account with its department and live active-complaint count.

    Unavailable staff are included and flagged via ``is_available`` — the directory
    exists precisely so an admin can see who is out, which is the opposite of what
    the assignment rule wants from the same data.
    """
    rows = await assignments.staff_directory()
    return [StaffRead.from_user(r.user, active_assignments=r.count) for r in rows]


@router.patch("/users/{user_id}", response_model=UserRead, summary="Update a user (admin)")
async def update_user(
    user_id: str, payload: UserUpdate, actor: AdminUser, auth: AuthServiceDep
) -> UserRead:
    """Set a user's department, availability, display name or active flag.

    ``department_id`` accepts an explicit ``null`` to clear a posting; omitting the
    field leaves it untouched (see :class:`~app.schemas.user.UserUpdate`).
    """
    user = await auth.update_user(user_id, payload)
    log.info("user.updated_by", user_id=user.id, actor=actor.email)
    return UserRead.model_validate(user)

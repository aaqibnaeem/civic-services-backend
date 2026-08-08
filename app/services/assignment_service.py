"""Who should carry this complaint? — the staff-assignment decision (CONTRACT §4b).

``AssignmentService`` has exactly one responsibility: **deciding** which staff
member a complaint belongs to, and refusing decisions that are not allowed. It
does not mutate the complaint. Applying a decision — writing ``assignee_id``,
moving ``open -> assigned``, appending the ``StatusEvent`` — stays inside
``ComplaintManager``, which is the only object permitted to change a complaint and
the only place the status state machine lives.

That split is deliberate. Load-balancing policy changes for business reasons
("weight critical cases double", "prefer the person who filed the original"); the
complaint state machine changes for correctness reasons. Keeping them in separate
objects means neither edit can quietly break the other, and the rule below is
testable without a complaint, a session commit, or an HTTP request in sight.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.core.errors import ConflictError, NotFoundError
from app.core.logging_config import get_logger
from app.models.complaint import PRIORITY_RANK, Complaint
from app.models.user import User
from app.repositories.complaint_repo import ComplaintRepository
from app.repositories.user_repo import UserRepository

log = get_logger(__name__)

#: Priority weights from CONTRACT §4b: critical 4, high 3, medium 2, low 1.
#: Identical to ``PRIORITY_RANK`` by construction — aliased rather than copied so
#: the two can never drift apart.
PRIORITY_WEIGHTS = PRIORITY_RANK


@dataclass(frozen=True, slots=True)
class Workload:
    """One staff member's current load, as the assignment rule measures it.

    ``count`` is how many active complaints they hold; ``weight`` is those
    complaints' summed priority weights. The ordering below *is* the tie-break
    chain from the contract, which is why it lives on the type rather than being
    re-spelled at each call site.
    """

    user: User
    count: int = 0
    weight: int = 0

    @property
    def sort_key(self) -> tuple[int, int, str]:
        """Fewest active complaints, then least urgent load, then lowest user id."""
        return (self.count, self.weight, self.user.id)


class AssignmentService:
    """Picks the staff member who should take a complaint, and validates manual picks.

    The rule (CONTRACT §4b), applied within the complaint's own department:

    1. only *available*, *active* staff in that department are candidates;
    2. among them, pick whoever holds the fewest **active** complaints
       (``open``, ``assigned`` or ``in_progress``);
    3. break ties towards the least urgent workload — the smallest sum of priority
       weights (critical 4, high 3, medium 2, low 1) across those active
       complaints, so a critical case does not land on someone already carrying two
       criticals while a colleague carries three low-priority ones;
    4. break any remaining tie on the lowest user id, so the outcome is
       deterministic and a test can assert on it.

    If the department has no eligible staff — or the complaint has no department
    yet — the answer is ``None`` and the complaint stays unassigned. It is never
    dropped, and never rerouted to some other department's queue.
    """

    def __init__(self, users: UserRepository, complaints: ComplaintRepository) -> None:
        self._users = users
        self._complaints = complaints

    # =================================================================== decision
    async def choose_assignee(self, complaint: Complaint) -> User | None:
        """The staff member who should take ``complaint``, or ``None`` if nobody can."""
        if complaint.department_id is None:
            log.info("assignment.skipped", complaint_id=complaint.id, reason="no_department")
            return None

        candidates = await self._users.list_staff(
            department_id=complaint.department_id, only_available=True
        )
        if not candidates:
            log.info(
                "assignment.no_candidates",
                complaint_id=complaint.id,
                department_id=complaint.department_id,
            )
            return None

        workloads = await self.workloads_for(candidates)
        winner = min(workloads, key=lambda w: w.sort_key)
        log.info(
            "assignment.chosen",
            complaint_id=complaint.id,
            assignee_id=winner.user.id,
            active_count=winner.count,
            active_weight=winner.weight,
            candidates=len(candidates),
        )
        return winner.user

    async def workloads_for(self, staff: list[User]) -> list[Workload]:
        """Attach each staff member's active count and priority weight, in one query."""
        if not staff:
            return []
        totals = await self._complaints.workload_by_assignee([u.id for u in staff])
        rows: list[Workload] = []
        for member in staff:
            count, weight = totals.get(member.id, (0, 0))
            rows.append(Workload(user=member, count=count, weight=weight))
        return rows

    async def staff_directory(self, *, department_id: str | None = None) -> list[Workload]:
        """Rows for ``GET /staff`` and ``GET /departments/{id}/staff``.

        Includes unavailable staff — the point of the directory is to *show* who is
        unavailable, unlike :meth:`choose_assignee`, which filters them out.
        """
        staff = await self._users.list_staff(department_id=department_id)
        workloads = await self.workloads_for(staff)
        workloads.sort(key=lambda w: (w.user.full_name.lower(), w.user.id))
        return workloads

    # ================================================================= validation
    async def resolve_assignee(self, complaint: Complaint, assignee_id: str) -> User:
        """Load the user a human asked for, or explain why they cannot have them.

        Every rejection is a ``409 conflict`` with a message that names the actual
        problem, because "assignment failed" sends a staff member hunting through
        four different screens to find out which rule they tripped.
        """
        user = await self._users.get_by_id(assignee_id)
        if user is None:
            raise NotFoundError(f"No user with id '{assignee_id}'.")

        if not user.is_staff:
            raise ConflictError(
                f"{user.email} is a {user.role} account and cannot be assigned complaints.",
                details=[{"field": "assignee_id", "issue": "not a staff or admin account"}],
            )
        if not user.is_active:
            raise ConflictError(
                f"{user.email} is deactivated and cannot be assigned complaints.",
                details=[{"field": "assignee_id", "issue": "account is deactivated"}],
            )
        if not user.is_available:
            raise ConflictError(
                f"{user.full_name or user.email} is marked unavailable and cannot take new work.",
                details=[{"field": "assignee_id", "issue": "staff member is unavailable"}],
            )
        if complaint.department_id is not None and user.department_id != complaint.department_id:
            raise ConflictError(
                f"{user.full_name or user.email} does not belong to this complaint's department.",
                details=[{"field": "assignee_id", "issue": "different department"}],
            )
        return user

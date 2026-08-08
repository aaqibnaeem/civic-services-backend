"""All SQL for complaints, status events and AI analyses lives here.

Nothing above this layer builds a ``select()``. That is the point: the services stay
readable domain logic, and when the analytics agent needs the same filter semantics
it reuses :meth:`ComplaintRepository.build_filter_clauses` instead of re-deriving
them (and drifting).
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import Select, and_, case, delete, func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql.elements import ColumnElement

from app.core.security import generate_reference_code
from app.models.ai_analysis import AIAnalysis
from app.models.complaint import (
    ACTIVE_STATUSES,
    PRIORITY_RANK,
    STATUS_RANK,
    AIStatus,
    Complaint,
    Status,
)
from app.models.status_event import StatusEvent

_MAX_REFERENCE_ATTEMPTS = 12

_SORTABLE = {"created_at", "priority", "status", "resolution_hours", "updated_at"}


class ComplaintRepository:
    """Data-access object for the complaint aggregate.

    Owns query construction, pagination, sorting and the reference-code collision
    loop. It never commits — transaction boundaries belong to the service layer so a
    complaint plus its first status event land atomically.
    """

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _base_select() -> Select[tuple[Complaint]]:
        """Every read excludes soft-deleted rows unless a caller opts out."""
        return select(Complaint).where(Complaint.is_deleted.is_(False))

    @staticmethod
    def build_filter_clauses(filters: Any) -> list[ColumnElement[bool]]:
        """Translate a ``ComplaintFilters`` object into SQL predicates.

        Public and static so the analytics module can apply identical filtering to
        its own aggregate queries.
        """
        clauses: list[ColumnElement[bool]] = []

        term = (getattr(filters, "q", None) or "").strip()
        if term:
            like = f"%{term.lower()}%"
            clauses.append(
                or_(
                    func.lower(Complaint.title).like(like),
                    func.lower(Complaint.description).like(like),
                    func.lower(Complaint.location_text).like(like),
                    func.lower(Complaint.reference_code).like(like),
                )
            )

        if categories := getattr(filters, "category", None):
            clauses.append(Complaint.category.in_(list(categories)))
        if priorities := getattr(filters, "priority", None):
            clauses.append(Complaint.priority.in_(list(priorities)))
        if statuses := getattr(filters, "status", None):
            clauses.append(Complaint.status.in_(list(statuses)))
        if department_id := getattr(filters, "department_id", None):
            clauses.append(Complaint.department_id == department_id)
        if assignee_id := getattr(filters, "assignee_id", None):
            clauses.append(Complaint.assignee_id == assignee_id)
        if citizen_id := getattr(filters, "citizen_id", None):
            # The whole isolation guarantee of GET /complaints/mine is this one
            # predicate: it runs in SQL, so there is no code path that loads a row
            # belonging to another citizen and then forgets to filter it out.
            clauses.append(Complaint.citizen_id == citizen_id)
        if area := getattr(filters, "area", None):
            clauses.append(func.lower(Complaint.area) == area.strip().lower())
        if date_from := getattr(filters, "date_from", None):
            clauses.append(Complaint.created_at >= _as_utc(date_from))
        if date_to := getattr(filters, "date_to", None):
            clauses.append(Complaint.created_at <= _as_utc(date_to))
        return clauses

    @staticmethod
    def _order_by(sort: str, order: str):
        """Map the contract's sort keys onto real, index-friendly SQL expressions."""
        descending = (order or "desc").lower() == "desc"
        field = sort if sort in _SORTABLE else "created_at"

        if field == "priority":
            # Rank ordinally: alphabetical order of the wire strings is meaningless.
            column = case(PRIORITY_RANK, value=Complaint.priority, else_=0)
        elif field == "status":
            column = case(STATUS_RANK, value=Complaint.status, else_=0)
        elif field == "resolution_hours":
            column = Complaint.resolution_hours
        else:
            column = getattr(Complaint, field)

        primary = column.desc() if descending else column.asc()
        # Stable tiebreaker so pagination never repeats or drops a row.
        return [primary, Complaint.id.asc()]

    # -------------------------------------------------------------------- reads
    async def get_by_id(self, complaint_id: str, *, include_deleted: bool = False) -> Complaint | None:
        stmt = select(Complaint).where(Complaint.id == complaint_id)
        if not include_deleted:
            stmt = stmt.where(Complaint.is_deleted.is_(False))
        return (await self.session.execute(stmt)).scalar_one_or_none()

    async def get_by_reference(self, reference_code: str) -> Complaint | None:
        stmt = self._base_select().where(
            func.upper(Complaint.reference_code) == reference_code.strip().upper()
        )
        return (await self.session.execute(stmt)).scalar_one_or_none()

    async def reference_exists(self, reference_code: str) -> bool:
        stmt = select(func.count()).select_from(Complaint).where(
            Complaint.reference_code == reference_code
        )
        return bool((await self.session.execute(stmt)).scalar_one())

    async def list_paginated(self, filters: Any) -> tuple[list[Complaint], int]:
        """Filter + sort + paginate entirely in SQL. Returns ``(rows, total)``."""
        clauses = self.build_filter_clauses(filters)

        count_stmt = select(func.count()).select_from(Complaint).where(
            Complaint.is_deleted.is_(False)
        )
        if clauses:
            count_stmt = count_stmt.where(and_(*clauses))
        total = int((await self.session.execute(count_stmt)).scalar_one())

        page = max(1, int(getattr(filters, "page", 1) or 1))
        page_size = min(100, max(1, int(getattr(filters, "page_size", 20) or 20)))

        stmt = self._base_select()
        if clauses:
            stmt = stmt.where(and_(*clauses))
        stmt = (
            stmt.order_by(*self._order_by(getattr(filters, "sort", "created_at"), getattr(filters, "order", "desc")))
            .offset((page - 1) * page_size)
            .limit(page_size)
        )
        rows = list((await self.session.execute(stmt)).scalars().unique())
        return rows, total

    async def list_recent(self, *, limit: int = 50) -> list[Complaint]:
        stmt = self._base_select().order_by(Complaint.created_at.desc()).limit(limit)
        return list((await self.session.execute(stmt)).scalars().unique())

    async def find_similar(
        self, *, description: str, exclude_id: str | None = None, limit: int = 5
    ) -> list[Complaint]:
        """Cheap SQL-side candidate generation for duplicate detection.

        Deliberately naive (token overlap on the longest words); the AI agent layers
        real similarity scoring on top of these candidates.
        """
        tokens = sorted(
            {w.lower() for w in description.split() if len(w) > 4},
            key=len,
            reverse=True,
        )[:5]
        if not tokens:
            return []
        stmt = self._base_select().where(
            or_(*[func.lower(Complaint.description).like(f"%{t}%") for t in tokens])
        )
        if exclude_id:
            stmt = stmt.where(Complaint.id != exclude_id)
        stmt = stmt.order_by(Complaint.created_at.desc()).limit(limit * 4)
        return list((await self.session.execute(stmt)).scalars().unique())[:limit]

    async def count_all(self, *, include_deleted: bool = False) -> int:
        stmt = select(func.count()).select_from(Complaint)
        if not include_deleted:
            stmt = stmt.where(Complaint.is_deleted.is_(False))
        return int((await self.session.execute(stmt)).scalar_one())

    async def workload_by_assignee(
        self, user_ids: list[str] | None = None
    ) -> dict[str, tuple[int, int]]:
        """``{user_id: (active_count, summed_priority_weight)}`` — the assignment rule's input.

        Both numbers are aggregated in SQL over the *active* statuses only, so a
        staff member who has closed fifty cases looks exactly as free as one who has
        closed none. Users with no active work simply do not appear in the mapping;
        callers treat a miss as ``(0, 0)``.
        """
        weight = case(PRIORITY_RANK, value=Complaint.priority, else_=0)
        stmt = (
            select(
                Complaint.assignee_id,
                func.count(),
                func.coalesce(func.sum(weight), 0),
            )
            .where(
                Complaint.is_deleted.is_(False),
                Complaint.assignee_id.is_not(None),
                Complaint.status.in_(ACTIVE_STATUSES),
            )
            .group_by(Complaint.assignee_id)
        )
        if user_ids is not None:
            if not user_ids:
                return {}
            stmt = stmt.where(Complaint.assignee_id.in_(user_ids))
        rows = (await self.session.execute(stmt)).all()
        return {row[0]: (int(row[1]), int(row[2])) for row in rows}

    async def count_open_by_department(self) -> dict[str, int]:
        """`{department_id: open_complaint_count}` for the departments endpoint."""
        stmt = (
            select(Complaint.department_id, func.count())
            .where(
                Complaint.is_deleted.is_(False),
                Complaint.department_id.is_not(None),
                Complaint.status.in_([Status.OPEN, Status.ASSIGNED, Status.IN_PROGRESS]),
            )
            .group_by(Complaint.department_id)
        )
        return {row[0]: int(row[1]) for row in (await self.session.execute(stmt)).all()}

    # ------------------------------------------------------------------- writes
    async def allocate_reference_code(self) -> str:
        """Generate a unique ``CIV-XXXXXX``, retrying on collision."""
        for _ in range(_MAX_REFERENCE_ATTEMPTS):
            candidate = generate_reference_code()
            if not await self.reference_exists(candidate):
                return candidate
        raise RuntimeError("Unable to allocate a unique reference code")

    def add(self, complaint: Complaint) -> Complaint:
        self.session.add(complaint)
        return complaint

    def add_event(self, event: StatusEvent) -> StatusEvent:
        self.session.add(event)
        return event

    async def get_analysis(self, complaint_id: str) -> AIAnalysis | None:
        stmt = select(AIAnalysis).where(AIAnalysis.complaint_id == complaint_id)
        return (await self.session.execute(stmt)).scalar_one_or_none()

    async def replace_analysis(self, complaint_id: str, analysis: AIAnalysis) -> AIAnalysis:
        """Upsert semantics for the 1:1 analysis row (``/reanalyze`` re-runs it)."""
        await self.session.execute(delete(AIAnalysis).where(AIAnalysis.complaint_id == complaint_id))
        analysis.complaint_id = complaint_id
        self.session.add(analysis)
        return analysis

    async def set_ai_status(self, complaint_id: str, ai_status: AIStatus) -> None:
        complaint = await self.get_by_id(complaint_id, include_deleted=True)
        if complaint is not None:
            complaint.ai_status = ai_status

    async def soft_delete(self, complaint: Complaint) -> None:
        complaint.is_deleted = True

    async def timeline_for(self, complaint_id: str) -> list[StatusEvent]:
        stmt = (
            select(StatusEvent)
            .where(StatusEvent.complaint_id == complaint_id)
            .order_by(StatusEvent.created_at.asc())
        )
        return list((await self.session.execute(stmt)).scalars())


def _as_utc(value: datetime) -> datetime:
    """Naive datetimes from query strings are interpreted as UTC."""
    return value if value.tzinfo else value.replace(tzinfo=UTC)

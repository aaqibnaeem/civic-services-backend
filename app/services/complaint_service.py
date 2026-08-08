"""Complaint orchestration — the domain core of the service.

``ComplaintManager`` is the only object allowed to mutate a complaint. It owns the
status state machine, reference-code allocation, department routing, timeline
emission and the AI hand-off. Routers call it; it calls repositories; nothing skips
a layer.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from app.core.errors import IllegalStatusTransition, NotFoundError, ValidationError
from app.core.logging_config import get_logger
from app.db.base import utcnow
from app.models.ai_analysis import AIAnalysis
from app.models.complaint import AIStatus, Category, Complaint, Priority, Status
from app.models.status_event import StatusEvent
from app.models.user import User
from app.repositories.complaint_repo import ComplaintRepository
from app.schemas.ai import AIAnalysisResult
from app.schemas.common import Page
from app.schemas.complaint import (
    ComplaintCreate,
    ComplaintDetail,
    ComplaintFilters,
    ComplaintRead,
    ComplaintUpdate,
)
from app.schemas.user import AccountRead
from app.services.department_service import DepartmentService
from app.services.notification_service import NotificationDispatcher, notification_dispatcher

if TYPE_CHECKING:
    from fastapi import BackgroundTasks

    from app.services.assignment_service import AssignmentService
    from app.services.auth_service import AuthService

log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class ComplaintSubmission:
    """What ``POST /complaints`` produced: the row, plus the citizen account behind it.

    The account block is returned separately rather than hung off the complaint
    because it is *not* part of the complaint — it describes something that
    happened to a user record, and its ``default_password`` must never be
    persisted, re-read, or served by any later GET.
    """

    complaint: Complaint
    account: AccountRead

#: The complaint status state machine. ``resolved`` is terminal on purpose — a
#: closed case must not silently reopen, because that would corrupt every
#: resolution-time statistic. Recurrences are filed as new complaints.
ALLOWED_TRANSITIONS: dict[Status, set[Status]] = {
    Status.OPEN: {Status.ASSIGNED, Status.IN_PROGRESS, Status.RESOLVED, Status.REJECTED},
    Status.ASSIGNED: {Status.IN_PROGRESS, Status.RESOLVED, Status.REJECTED, Status.OPEN},
    Status.IN_PROGRESS: {Status.RESOLVED, Status.REJECTED, Status.ASSIGNED},
    Status.RESOLVED: set(),
    Status.REJECTED: {Status.OPEN},
}

#: Coarse area buckets used by the analytics module. Matched against free-text
#: location when the citizen does not pick one explicitly.
KNOWN_AREAS: tuple[str, ...] = (
    "Gulshan-e-Iqbal",
    "North Nazimabad",
    "Nazimabad",
    "Saddar",
    "Clifton",
    "Korangi",
    "Malir",
    "Gulistan-e-Johar",
    "Lyari",
    "DHA",
    "Orangi Town",
    "Landhi",
)

_TITLE_MAX = 90
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


class ComplaintManager:
    """Use-case orchestrator for the complaint aggregate.

    Responsibilities:
      * create a complaint (reference code, title, area inference, routing);
      * validate and apply status transitions, emitting a ``StatusEvent`` for each;
      * hand analysis results from the AI pipeline onto the aggregate;
      * expose read models (single, paginated, timeline) to the routers.

    Why a class rather than loose functions: these operations share invariants
    (a status change *always* produces a timeline row and *always* notifies the
    citizen) and collaborators (repo, routing, notifications). Bundling them makes
    those invariants enforceable in one place instead of hoping every caller
    remembers.
    """

    def __init__(
        self,
        repo: ComplaintRepository,
        *,
        departments: DepartmentService | None = None,
        notifier: NotificationDispatcher | None = None,
        accounts: AuthService | None = None,
        assignments: AssignmentService | None = None,
    ) -> None:
        self._repo = repo
        self._departments = departments
        self._notifier = notifier or notification_dispatcher
        self._accounts = accounts
        self._assignments = assignments

    # ================================================================== creation
    async def create(
        self,
        payload: ComplaintCreate,
        *,
        background_tasks: BackgroundTasks | None = None,
    ) -> ComplaintSubmission:
        """Persist a complaint, link it to a citizen account, schedule AI enrichment.

        CONTRACT §5.1: this never blocks on the LLM. The row is committed with
        ``ai_status="pending"`` and the analysis happens after the response is sent.
        CONTRACT §4b: the submitted email is turned into a ``citizen`` account
        (created on first use, reused afterwards) and the complaint is linked to it,
        which is what makes ``GET /complaints/mine`` possible without a signup form.
        """
        if not payload.consent:
            raise ValidationError(
                "Consent is required to file a complaint.",
                details=[{"field": "consent", "issue": "must be true"}],
            )

        citizen, is_new_account = await self._resolve_citizen(payload)

        reference_code = await self._repo.allocate_reference_code()
        category = payload.category or Category.OTHER

        complaint = Complaint(
            reference_code=reference_code,
            title=self._derive_title(payload.description),
            description=payload.description,
            category=category,
            priority=Priority.MEDIUM,
            status=Status.OPEN,
            location_text=payload.location_text,
            area=payload.area or self._infer_area(payload.location_text),
            latitude=payload.latitude,
            longitude=payload.longitude,
            citizen_name=payload.citizen_name,
            citizen_phone=payload.citizen_phone,
            citizen_email=payload.citizen_email,
            citizen_id=citizen.id if citizen is not None else None,
            image_url=payload.image_url,
            ai_status=AIStatus.PENDING,
            category_locked=payload.category is not None,
        )

        if payload.category is not None and self._departments is not None:
            department = await self._departments.route_for_category(payload.category)
            if department is not None:
                complaint.department_id = department.id

        self._repo.add(complaint)
        self._repo.add_event(
            StatusEvent(
                complaint=complaint,
                from_status=None,
                to_status=Status.OPEN,
                note="Complaint submitted by citizen.",
                actor=payload.citizen_email or payload.citizen_name or "citizen",
            )
        )
        await self._repo.session.commit()
        await self._repo.session.refresh(complaint)

        log.info(
            "complaint.created",
            complaint_id=complaint.id,
            reference_code=complaint.reference_code,
            area=complaint.area,
            citizen_id=complaint.citizen_id,
            new_account=is_new_account,
        )

        if background_tasks is not None:
            self.schedule_enrichment(background_tasks, complaint.id)

        from app.core.config import settings  # noqa: PLC0415 - avoids an import cycle

        return ComplaintSubmission(
            complaint=complaint,
            account=AccountRead(
                email=payload.citizen_email,
                is_new=is_new_account,
                # Only ever populated for an account this request created.
                default_password=settings.CITIZEN_DEFAULT_PASSWORD if is_new_account else None,
            ),
        )

    async def _resolve_citizen(self, payload: ComplaintCreate) -> tuple[User | None, bool]:
        """Find-or-create the citizen account for the submitted email.

        Returns ``(user, is_new)``. When no ``AuthService`` was wired in — the
        ``ComplaintManager`` used by the AI pipeline has no need for one — the
        complaint is still stored, just without an account link. Intake must not
        fail because of an account-management concern.
        """
        if self._accounts is None:
            return None, False
        citizen, is_new = await self._accounts.find_or_create_citizen(
            payload.citizen_email, full_name=payload.citizen_name
        )
        if is_new:
            # The primary key is a Python-side column default, so it only exists
            # after a flush — and ``citizen.id`` is needed for the FK below. The
            # surrounding create() still commits everything in one transaction.
            await self._repo.session.flush()
        return citizen, is_new

    @staticmethod
    def schedule_enrichment(background_tasks: BackgroundTasks, complaint_id: str) -> None:
        """Queue AI enrichment to run after the HTTP response is flushed."""
        background_tasks.add_task(run_ai_enrichment, complaint_id)

    # ===================================================================== reads
    async def get_or_404(self, complaint_id: str) -> Complaint:
        complaint = await self._repo.get_by_id(complaint_id)
        if complaint is None:
            raise NotFoundError(f"No complaint with id '{complaint_id}'.")
        return complaint

    async def get_by_reference_or_404(self, reference_code: str) -> Complaint:
        complaint = await self._repo.get_by_reference(reference_code)
        if complaint is None:
            raise NotFoundError(f"No complaint found for reference '{reference_code}'.")
        return complaint

    async def get_detail(self, complaint_id: str) -> ComplaintDetail:
        """Full complaint plus its timeline (``GET /complaints/{id}``).

        ``Complaint.timeline`` is a ``selectin`` relationship, so the events arrive
        with the row — no N+1, no lazy-load-in-async explosion.
        """
        complaint = await self.get_or_404(complaint_id)
        return ComplaintDetail.model_validate(complaint)

    async def list(self, filters: ComplaintFilters) -> Page[ComplaintRead]:
        """Paginated admin list. All filtering, sorting and slicing happens in SQL."""
        rows, total = await self._repo.list_paginated(filters)
        return Page[ComplaintRead].build(
            [ComplaintRead.model_validate(row) for row in rows],
            total=total,
            page=filters.page,
            page_size=filters.page_size,
        )

    async def find_duplicates(
        self, complaint_id: str, *, limit: int = 5, threshold: float = 0.18
    ) -> list[tuple[Complaint, float, str]]:
        """Candidates for ``GET /complaints/{id}/duplicates``.

        Delegates to the TF-IDF engine in ``app.ai.duplicates``, which scores cosine
        similarity over word and character n-grams and gates candidates by category,
        geographic radius and a recency window. The Jaccard pass below is the fallback
        for when the AI package is unavailable: dependency-free, weaker, always present.
        """
        try:
            from app.ai.duplicates import find_duplicates as tfidf_duplicates
        except ImportError:
            log.warning("ai.duplicates unavailable, falling back to Jaccard overlap")
        else:
            await self.get_or_404(complaint_id)  # 404 before doing any work
            # Wider gating than the pipeline's automatic duplicate flagging uses. That
            # path links complaints without asking anyone, so it must be conservative;
            # this list is a suggestion a staff member reads and accepts or ignores, so
            # recall matters more than precision here.
            candidates = await tfidf_duplicates(
                complaint_id,
                session=self._repo.session,
                limit=limit,
                window_days=180,
                radius_m=5000.0,
                threshold=0.45,
            )
            return [(c.complaint, round(c.similarity, 3), c.reason) for c in candidates]

        complaint = await self.get_or_404(complaint_id)
        candidates = await self._repo.find_similar(
            description=complaint.description, exclude_id=complaint.id, limit=limit * 4
        )
        target_tokens = self._tokenize(complaint.description)

        scored: list[tuple[Complaint, float, str]] = []
        for candidate in candidates:
            similarity = self._jaccard(target_tokens, self._tokenize(candidate.description))
            if candidate.area and complaint.area and candidate.area == complaint.area:
                similarity = min(1.0, similarity + 0.1)  # same area is corroborating evidence
            if similarity < threshold:
                continue
            reasons = [f"{int(similarity * 100)}% wording overlap"]
            if candidate.category == complaint.category:
                reasons.append(f"same category ({complaint.category})")
            if candidate.area and candidate.area == complaint.area:
                reasons.append(f"same area ({complaint.area})")
            scored.append((candidate, round(similarity, 3), "; ".join(reasons)))

        scored.sort(key=lambda row: row[1], reverse=True)
        return scored[:limit]

    # =================================================================== updates
    async def apply_update(
        self, complaint_id: str, payload: ComplaintUpdate, *, actor: str
    ) -> Complaint:
        """Apply a `PATCH` — validating the transition and appending a timeline row."""
        complaint = await self.get_or_404(complaint_id)

        if payload.category is not None:
            complaint.category = payload.category
            complaint.category_locked = True  # a re-analysis must not undo this
        if payload.priority is not None:
            complaint.priority = payload.priority
        if payload.department_id is not None:
            await self._assign_department(complaint, payload.department_id)
        if payload.assignee_provided:
            # ``assignee_provided`` (not ``is not None``) is what makes an explicit
            # ``"assignee_id": null`` mean *unassign* rather than *leave alone*.
            await self._set_assignee(
                complaint, payload.assignee_id, actor=actor, note=payload.note
            )

        status_note = payload.note
        if (
            payload.status is Status.ASSIGNED
            and not payload.assignee_provided
            and complaint.assignee_id is None
            and self._assignments is not None
        ):
            # "Assigned to nobody" is not a real state — it is what a triager gets if
            # they set the status from a dropdown and the system asks no follow-up
            # question. Pick an owner with the same rule the AI pipeline uses, which
            # also performs the open -> assigned transition, so the status block below
            # then finds nothing left to change.
            chosen = await self._assignments.choose_assignee(complaint)
            if chosen is not None:
                self._apply_assignee(complaint, chosen, actor=actor, note=payload.note)
            else:
                # Let the status change through anyway — the complaint really is with a
                # department — but explain the empty assignee. The explanation rides on
                # the transition's own note rather than a second event, because two rows
                # for one PATCH makes the timeline read like it happened twice.
                log.info("assignment.none_available", complaint_id=complaint.id)
                status_note = " ".join(
                    filter(None, [payload.note, "No staff member was available to take it."])
                )

        previous_status: Status | None = None
        if payload.status is not None and payload.status != complaint.status:
            self._assert_transition_allowed(complaint.status, payload.status)
            previous_status = complaint.status
            self._apply_status(complaint, payload.status, note=status_note, actor=actor)
        elif payload.note and not payload.assignee_provided:
            # Note-only edit still deserves an audit row. Skipped when an assignment
            # already ran: that branch folded the note into its own event, and two
            # rows for one PATCH makes the timeline read like it happened twice.
            self._repo.add_event(
                StatusEvent(
                    complaint_id=complaint.id,
                    from_status=complaint.status,
                    to_status=complaint.status,
                    note=payload.note,
                    actor=actor,
                )
            )

        await self._repo.session.commit()
        await self._repo.session.refresh(complaint)

        if previous_status is not None:
            await self._notify_transition(
                complaint, payload.note, previous=previous_status, actor=actor
            )
        return complaint

    async def transition(
        self, complaint_id: str, to_status: Status, *, actor: str, note: str | None = None
    ) -> Complaint:
        """Explicit status change, used by staff tooling and tests."""
        return await self.apply_update(
            complaint_id, ComplaintUpdate(status=to_status, note=note), actor=actor
        )

    async def assign(self, complaint_id: str, department_id: str, *, actor: str) -> Complaint:
        """Route a complaint to a department and move it to ``assigned``."""
        complaint = await self.get_or_404(complaint_id)
        await self._assign_department(complaint, department_id)
        if complaint.status == Status.OPEN:
            self._apply_status(complaint, Status.ASSIGNED, note="Assigned to department.", actor=actor)
        await self._repo.session.commit()
        await self._repo.session.refresh(complaint)
        return complaint

    # ================================================================ assignment
    async def auto_assign(self, complaint_id: str, *, actor: str) -> Complaint:
        """Run the CONTRACT §4b workload rule and apply its verdict.

        The *decision* belongs to :class:`AssignmentService`; this method only
        applies it, so the state machine and the timeline stay owned by one object.
        A complaint whose department has no available staff is left exactly as it
        was — unassigned, still open, still in its own department's queue.
        """
        complaint = await self.get_or_404(complaint_id)
        if self._assignments is None:  # pragma: no cover - always wired in practice
            log.warning("assignment.service_missing", complaint_id=complaint_id)
            return complaint

        assignee = await self._assignments.choose_assignee(complaint)
        if assignee is None:
            return complaint

        self._apply_assignee(complaint, assignee, actor=actor)
        await self._repo.session.commit()
        await self._repo.session.refresh(complaint)
        return complaint

    async def _set_assignee(
        self, complaint: Complaint, assignee_id: str | None, *, actor: str, note: str | None = None
    ) -> None:
        """Validate and stage a manual (re)assignment. Does not commit."""
        if assignee_id is None:
            self._apply_assignee(complaint, None, actor=actor, note=note)
            return
        if self._assignments is None:  # pragma: no cover - always wired in practice
            raise ValidationError("Assignment is not available in this context.")
        assignee = await self._assignments.resolve_assignee(complaint, assignee_id)
        self._apply_assignee(complaint, assignee, actor=actor, note=note)

    def _apply_assignee(
        self, complaint: Complaint, assignee: User | None, *, actor: str, note: str | None = None
    ) -> None:
        """Write the assignee and emit the right timeline row.

        Setting an owner moves ``open -> assigned`` through the same
        ``_apply_status`` used by every other transition — the state machine is
        never bypassed. A complaint already past ``open`` keeps its status (a case
        being worked on does not regress just because it changed hands) and gets a
        note-only audit row instead. Clearing an owner likewise leaves the status
        alone: ``assigned`` also means "routed to a department", which is still true.
        """
        if assignee is None:
            complaint.assignee_id = None
            complaint.assigned_at = None
            self._repo.add_event(
                StatusEvent(
                    complaint_id=complaint.id,
                    from_status=complaint.status,
                    to_status=complaint.status,
                    note=note or "Assignee cleared.",
                    actor=actor,
                )
            )
            log.info("complaint.unassigned", complaint_id=complaint.id, actor=actor)
            return

        who = assignee.full_name or assignee.email
        complaint.assignee_id = assignee.id
        complaint.assigned_at = utcnow()

        if complaint.status == Status.OPEN:
            self._assert_transition_allowed(Status.OPEN, Status.ASSIGNED)
            self._apply_status(
                complaint, Status.ASSIGNED, note=note or f"Assigned to {who}.", actor=actor
            )
        else:
            self._repo.add_event(
                StatusEvent(
                    complaint_id=complaint.id,
                    from_status=complaint.status,
                    to_status=complaint.status,
                    note=note or f"Reassigned to {who}.",
                    actor=actor,
                )
            )
        log.info(
            "complaint.assigned",
            complaint_id=complaint.id,
            assignee_id=assignee.id,
            actor=actor,
        )

    async def soft_delete(self, complaint_id: str, *, actor: str) -> None:
        """Admin-only removal. The row stays for audit and analytics continuity."""
        complaint = await self.get_or_404(complaint_id)
        await self._repo.soft_delete(complaint)
        await self._repo.session.commit()
        log.info("complaint.soft_deleted", complaint_id=complaint_id, actor=actor)

    # ================================================================ ai hand-off
    async def attach_analysis(
        self, complaint_id: str, result: AIAnalysisResult, *, apply_to_complaint: bool = True
    ) -> Complaint:
        """Persist an analyzer result and (optionally) fold it into the complaint.

        This is the seam the AI agent writes through. Keeping it here means the
        pipeline never has to know about ``AIAnalysis`` columns, soft deletes, or how
        confident is confident enough to override a citizen's own category guess.
        """
        complaint = await self._repo.get_by_id(complaint_id, include_deleted=True)
        if complaint is None:
            raise NotFoundError(f"No complaint with id '{complaint_id}'.")

        analysis = AIAnalysis(
            complaint_id=complaint.id,
            category=result.category,
            priority=result.priority,
            summary=result.summary or "",
            department_suggestion=result.department_suggestion,
            confidence=float(result.confidence or 0.0),
            source=result.source,
            model_name=result.model_name or "",
            reasoning=result.reasoning,
            keywords=list(result.keywords or []),
            sentiment=result.sentiment,
            is_emergency=bool(result.is_emergency),
            latency_ms=int(result.latency_ms or 0),
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            cache_hit_tokens=result.cache_hit_tokens,
        )
        await self._repo.replace_analysis(complaint.id, analysis)

        if apply_to_complaint:
            await self._apply_analysis_to_complaint(complaint, result)

        complaint.ai_status = AIStatus.COMPLETE
        await self._repo.session.commit()
        await self._repo.session.refresh(complaint)
        log.info(
            "complaint.analysis_attached",
            complaint_id=complaint.id,
            source=str(result.source),
            confidence=result.confidence,
        )
        return complaint

    async def mark_ai_failed(self, complaint_id: str) -> None:
        """Flag enrichment as failed without touching anything else."""
        await self._repo.set_ai_status(complaint_id, AIStatus.FAILED)
        await self._repo.session.commit()

    async def reanalyze(self, complaint_id: str) -> Complaint:
        """`POST /complaints/{id}/reanalyze` — re-run the pipeline synchronously."""
        complaint = await self.get_or_404(complaint_id)
        try:
            from app.ai.pipeline import analyze_and_store  # noqa: PLC0415 - optional module
        except ImportError:
            log.warning("ai.pipeline_unavailable", complaint_id=complaint_id, phase="reanalyze")
            await self.mark_ai_failed(complaint_id)
            return await self.get_or_404(complaint_id)

        try:
            await analyze_and_store(complaint_id)
        except Exception as exc:  # noqa: BLE001 - never let AI break the API
            log.warning("ai.reanalyze_failed", complaint_id=complaint_id, error=str(exc))
            await self.mark_ai_failed(complaint_id)

        self._repo.session.expire(complaint)
        return await self.get_or_404(complaint_id)

    # ============================================================ private helpers
    @staticmethod
    def _assert_transition_allowed(current: Status, target: Status) -> None:
        if target not in ALLOWED_TRANSITIONS.get(current, set()):
            raise IllegalStatusTransition(str(current), str(target))

    def _apply_status(
        self, complaint: Complaint, target: Status, *, note: str | None, actor: str
    ) -> None:
        """Mutate status, stamp ``resolved_at``, and append the audit row."""
        previous = complaint.status
        complaint.status = target
        if target == Status.RESOLVED:
            complaint.resolved_at = utcnow()
        self._repo.add_event(
            StatusEvent(
                complaint_id=complaint.id,
                from_status=previous,
                to_status=target,
                note=note,
                actor=actor,
            )
        )

    async def _assign_department(self, complaint: Complaint, department_id: str) -> None:
        if self._departments is not None:
            department = await self._departments.get_or_404(department_id)
            complaint.department_id = department.id
        else:
            complaint.department_id = department_id

    async def _apply_analysis_to_complaint(
        self, complaint: Complaint, result: AIAnalysisResult
    ) -> None:
        """Fold a confident analysis into the complaint itself.

        The category is left alone when a human already chose it. The analyzer's
        own verdict is still recorded on the ``AIAnalysis`` row, so a disagreement
        stays visible to staff instead of being resolved silently in the AI's favour.
        """
        if not complaint.category_locked:
            complaint.category = result.category
        complaint.priority = result.priority
        if result.title:
            complaint.title = result.title[:_TITLE_MAX]
        elif result.summary and complaint.title == self._derive_title(complaint.description):
            complaint.title = self._derive_title(result.summary)

        if complaint.department_id is None and self._departments is not None:
            department = await self._departments.resolve_by_name_or_slug(
                result.department_suggestion
            ) or await self._departments.route_for_category(result.category)
            if department is not None:
                complaint.department_id = department.id

    async def _notify_transition(
        self, complaint: Complaint, note: str | None, *, previous: Status | None, actor: str
    ) -> None:
        await self._notifier.notify_status_change(
            reference_code=complaint.reference_code,
            recipient=complaint.citizen_email,
            from_status=str(previous) if previous else None,
            to_status=str(complaint.status),
            note=note,
        )
        log.info(
            "complaint.status_changed",
            complaint_id=complaint.id,
            to_status=str(complaint.status),
            actor=actor,
        )

    @staticmethod
    def _derive_title(description: str) -> str:
        """First sentence of the citizen's text, trimmed to a headline length."""
        text = " ".join((description or "").split())
        if not text:
            return "Untitled complaint"
        first = _SENTENCE_SPLIT.split(text)[0]
        if len(first) <= _TITLE_MAX:
            return first
        return first[: _TITLE_MAX - 1].rsplit(" ", 1)[0] + "…"

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        """Lowercase content words of length > 3, used for duplicate scoring."""
        return {w for w in re.findall(r"[a-z0-9]+", (text or "").lower()) if len(w) > 3}

    @staticmethod
    def _jaccard(left: set[str], right: set[str]) -> float:
        if not left or not right:
            return 0.0
        union = left | right
        return len(left & right) / len(union) if union else 0.0

    @staticmethod
    def _infer_area(location_text: str) -> str | None:
        """Map free-text location onto a known area bucket, else ``None``."""
        haystack = (location_text or "").lower().replace("_", " ")
        for area in KNOWN_AREAS:
            if area.lower() in haystack:
                return area
        return None


async def run_ai_enrichment(complaint_id: str) -> None:
    """Background entry point for post-submission AI analysis.

    Runs on its own session because the request-scoped one is already closed by the
    time FastAPI executes background tasks. Every failure mode — the AI module not
    existing yet, a provider outage, a bad response — ends with ``ai_status="failed"``
    and a log line. It can never roll back the complaint insert, which already
    committed before this function was scheduled (CONTRACT §5.1).
    """
    from app.db.session import SessionLocal  # noqa: PLC0415 - keeps import graph flat

    try:
        # Lazily imported: ``app.ai`` is owned by another agent and may not exist yet.
        from app.ai.pipeline import analyze_and_store  # noqa: PLC0415
    except ImportError as exc:
        log.warning("ai.pipeline_unavailable", complaint_id=complaint_id, error=str(exc))
        await _mark_failed(complaint_id)
        return

    try:
        await analyze_and_store(complaint_id)
    except Exception as exc:  # noqa: BLE001 - AI failure must never surface to the citizen
        log.warning("ai.enrichment_failed", complaint_id=complaint_id, error=str(exc))
        await _mark_failed(complaint_id)
        return

    # The pipeline owns its own transaction; verify it actually landed a result.
    async with SessionLocal() as session:
        repo = ComplaintRepository(session)
        complaint = await repo.get_by_id(complaint_id, include_deleted=True)
        if complaint is not None and complaint.ai_status == AIStatus.PENDING:
            complaint.ai_status = AIStatus.FAILED
            await session.commit()
            log.warning("ai.enrichment_left_pending", complaint_id=complaint_id)


async def _mark_failed(complaint_id: str) -> None:
    """Set ``ai_status='failed'`` on its own short-lived session."""
    from app.db.session import SessionLocal  # noqa: PLC0415

    try:
        async with SessionLocal() as session:
            repo = ComplaintRepository(session)
            await repo.set_ai_status(complaint_id, AIStatus.FAILED)
            await session.commit()
    except Exception as exc:  # noqa: BLE001 - best effort, never raise from a background task
        log.error("ai.mark_failed_error", complaint_id=complaint_id, error=str(exc))

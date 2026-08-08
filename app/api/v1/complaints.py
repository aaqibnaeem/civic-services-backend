"""Complaint endpoints (CONTRACT §3 Public + Admin).

Route handlers here are intentionally thin: parse the request, call a service,
serialise the result. No SQL, no session, no domain branching.
"""

from __future__ import annotations

import inspect
from datetime import datetime
from typing import Annotated, Any

from fastapi import APIRouter, BackgroundTasks, Query, Response
from fastapi import status as http_status

from app.core.deps import AdminUser, ComplaintManagerDep, StaffUser
from app.core.errors import AIUnavailableError
from app.core.logging_config import get_logger
from app.db.base import utcnow
from app.models.complaint import Category, Priority, Status
from app.schemas.ai import (
    AIAnalysisRead,
    AIAnalysisResult,
    AnalyzePreviewRequest,
    DuplicateCandidate,
    DuplicatesResponse,
)
from app.schemas.common import Page
from app.schemas.complaint import (
    ComplaintCreate,
    ComplaintDetail,
    ComplaintFilters,
    ComplaintRead,
    ComplaintUpdate,
)

log = get_logger(__name__)

router = APIRouter(prefix="/complaints", tags=["complaints"])


# =============================================================== public endpoints
@router.post(
    "",
    response_model=ComplaintRead,
    status_code=http_status.HTTP_201_CREATED,
    summary="File a complaint (public)",
)
async def create_complaint(
    payload: ComplaintCreate,
    background_tasks: BackgroundTasks,
    manager: ComplaintManagerDep,
) -> ComplaintRead:
    """Save the complaint and return immediately with ``ai_status="pending"``.

    CONTRACT §5.1 — the LLM is never on this request's critical path; enrichment runs
    in a background task once the response has been flushed.
    """
    complaint = await manager.create(payload, background_tasks=background_tasks)
    return ComplaintRead.model_validate(complaint)


# Declared before ``/{complaint_id}`` so the literal path always wins the match.
@router.post(
    "/analyze-preview",
    response_model=AIAnalysisRead,
    summary="Analyse text without saving (public)",
)
async def analyze_preview(payload: AnalyzePreviewRequest) -> AIAnalysisRead:
    """The one synchronous AI call in the system (CONTRACT §5.2).

    The analyzer itself lives in ``app.ai`` and is owned by another agent, so it is
    imported lazily: if that module does not exist, or the call fails, the endpoint
    answers ``503 ai_unavailable`` rather than breaking the submit screen.
    """
    try:
        from app.ai.pipeline import analyze_text  # noqa: PLC0415 - optional module
    except ImportError as exc:
        log.warning("ai.analyze_text_unavailable", error=str(exc))
        raise AIUnavailableError(
            "The AI analyzer is not available right now. You can still submit your complaint."
        ) from exc

    try:
        raw = analyze_text(payload.description)
        if inspect.isawaitable(raw):
            raw = await raw
    except Exception as exc:  # noqa: BLE001 - any analyzer failure is a 503, not a 500
        log.warning("ai.analyze_text_failed", error=str(exc))
        raise AIUnavailableError(
            "The AI analyzer could not process this text. You can still submit your complaint."
        ) from exc

    result = _coerce_analysis(raw)
    return AIAnalysisRead(**result.model_dump(exclude={"title"}), created_at=utcnow())


@router.get(
    "/track/{reference_code}",
    response_model=ComplaintRead,
    summary="Track a complaint by reference code (public)",
)
async def track_complaint(reference_code: str, manager: ComplaintManagerDep) -> ComplaintRead:
    """Public status lookup — the citizen's handle on their own submission."""
    complaint = await manager.get_by_reference_or_404(reference_code)
    return ComplaintRead.model_validate(complaint)


# ============================================================== admin / staff area
@router.get("", response_model=Page[ComplaintRead], summary="List complaints (staff)")
async def list_complaints(
    _user: StaffUser,
    manager: ComplaintManagerDep,
    q: Annotated[str | None, Query(description="Free-text search")] = None,
    category: Annotated[list[Category] | None, Query()] = None,
    priority: Annotated[list[Priority] | None, Query()] = None,
    status: Annotated[list[Status] | None, Query()] = None,
    department_id: Annotated[str | None, Query()] = None,
    area: Annotated[str | None, Query()] = None,
    date_from: Annotated[datetime | None, Query()] = None,
    date_to: Annotated[datetime | None, Query()] = None,
    sort: Annotated[str, Query(pattern="^(created_at|priority|status|resolution_hours)$")] = "created_at",
    order: Annotated[str, Query(pattern="^(asc|desc)$")] = "desc",
    page: Annotated[int, Query(ge=1)] = 1,
    page_size: Annotated[int, Query(ge=1, le=100)] = 20,
) -> Page[ComplaintRead]:
    """Every filter is applied in SQL — nothing is post-filtered in Python."""
    filters = ComplaintFilters(
        q=q,
        category=category or [],
        priority=priority or [],
        status=status or [],
        department_id=department_id,
        area=area,
        date_from=date_from,
        date_to=date_to,
        sort=sort,
        order=order,
        page=page,
        page_size=page_size,
    )
    return await manager.list(filters)


@router.get("/{complaint_id}", response_model=ComplaintDetail, summary="Complaint detail (staff)")
async def get_complaint(
    complaint_id: str, _user: StaffUser, manager: ComplaintManagerDep
) -> ComplaintDetail:
    """Full complaint including its ``timeline`` of status events."""
    return await manager.get_detail(complaint_id)


@router.patch("/{complaint_id}", response_model=ComplaintRead, summary="Update a complaint (staff)")
async def update_complaint(
    complaint_id: str,
    payload: ComplaintUpdate,
    user: StaffUser,
    manager: ComplaintManagerDep,
) -> ComplaintRead:
    """Applies the change, validates the status transition, appends a StatusEvent."""
    complaint = await manager.apply_update(complaint_id, payload, actor=user.email)
    return ComplaintRead.model_validate(complaint)


@router.post(
    "/{complaint_id}/reanalyze", response_model=ComplaintRead, summary="Re-run the AI pipeline"
)
async def reanalyze_complaint(
    complaint_id: str, _user: StaffUser, manager: ComplaintManagerDep
) -> ComplaintRead:
    """Re-runs analysis in-band and returns the refreshed complaint.

    If the AI module is missing the complaint comes back with ``ai_status="failed"``
    rather than an error — a staff action must not 500 because of a provider outage.
    """
    complaint = await manager.reanalyze(complaint_id)
    return ComplaintRead.model_validate(complaint)


@router.get(
    "/{complaint_id}/duplicates",
    response_model=DuplicatesResponse,
    summary="Likely duplicates of a complaint",
)
async def complaint_duplicates(
    complaint_id: str,
    _user: StaffUser,
    manager: ComplaintManagerDep,
    limit: Annotated[int, Query(ge=1, le=20)] = 5,
) -> DuplicatesResponse:
    """Scored duplicate candidates, most similar first."""
    matches = await manager.find_duplicates(complaint_id, limit=limit)
    return DuplicatesResponse(
        candidates=[
            DuplicateCandidate(
                complaint=ComplaintRead.model_validate(candidate).model_dump(mode="json"),
                similarity=similarity,
                reason=reason,
            )
            for candidate, similarity, reason in matches
        ]
    )


@router.delete(
    "/{complaint_id}",
    status_code=http_status.HTTP_204_NO_CONTENT,
    summary="Soft-delete a complaint (admin only)",
)
async def delete_complaint(
    complaint_id: str, user: AdminUser, manager: ComplaintManagerDep
) -> Response:
    """Soft delete: the row stays for audit, but drops out of every read."""
    await manager.soft_delete(complaint_id, actor=user.email)
    return Response(status_code=http_status.HTTP_204_NO_CONTENT)


def _coerce_analysis(raw: Any) -> AIAnalysisResult:
    """Accept whatever shape the analyzer returns and normalise it.

    The AI module is written by another agent against the same contract, but being
    liberal in what we accept here (pydantic model, dataclass, or plain dict) means a
    small signature mismatch degrades to a validation error instead of a 500.
    """
    if isinstance(raw, AIAnalysisResult):
        return raw
    if hasattr(raw, "model_dump"):
        return AIAnalysisResult.model_validate(raw.model_dump())
    if isinstance(raw, dict):
        return AIAnalysisResult.model_validate(raw)
    if hasattr(raw, "__dict__"):
        return AIAnalysisResult.model_validate(vars(raw))
    raise AIUnavailableError("The AI analyzer returned an unusable response.")

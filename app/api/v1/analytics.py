"""``/api/v1/analytics`` — the statistics dashboard API.

Every route here returns numbers **and** an explanation: each response carries an
``interpretation`` string, an ``insights`` array, or both. That is a hard rule for
this module — the project spec requires that we "explain what the statistics mean
rather than displaying numbers only", so a payload of bare numbers is treated as a
defect rather than a minimal implementation.

Auth: everything except ``/analytics/public-summary`` sits behind ``require_staff``
(staff or admin) — the dashboard is the service team's tool.
``/public-summary`` is deliberately anonymous and deliberately small — aggregate
counts and category shares only, never a reference code, an area breakdown fine
enough to identify a household, or anything about an individual complaint.

Dependency resolution is late-bound (see ``_resolve``): this module is written in
parallel with ``app.core.deps`` and ``app.db.session``, so it imports cleanly even
before those land and hardens automatically once they exist. If auth is genuinely
unavailable at import time the router logs a loud warning rather than silently
serving admin analytics to the world.
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Any

from fastapi import APIRouter, Depends, Query

from app.schemas.analytics import (
    AnalyticsFilters,
    AreasResponse,
    CategoriesResponse,
    DepartmentsResponse,
    InsightsResponse,
    OverviewResponse,
    PrioritiesResponse,
    PublicSummaryResponse,
    ResolutionTimesResponse,
    TrendsResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/analytics", tags=["analytics"])


# --------------------------------------------------------------------------- #
# late-bound dependencies
# --------------------------------------------------------------------------- #
def _resolve_session_dependency():
    try:
        from app.db.session import get_session

        return get_session
    except Exception:  # pragma: no cover - only during parallel bring-up
        logger.warning("app.db.session.get_session unavailable; analytics will serve empty frames.")

        async def _no_session():
            yield None

        return _no_session


def _resolve_console_dependency():
    """Staff-or-admin, not admin-only.

    The dashboard is the service team's tool — understanding which complaints are
    common, urgent or slow is the whole point of their job. Restricting it to admins
    locked out exactly the people it was built for. Admin-only is reserved for
    destructive operations.
    """
    try:
        from app.core.deps import require_staff

        return require_staff
    except Exception:  # pragma: no cover - only during parallel bring-up
        try:
            from app.core.deps import get_current_user

            logger.warning("require_staff unavailable; analytics falling back to get_current_user.")
            return get_current_user
        except Exception:
            logger.error(
                "SECURITY: app.core.deps is unavailable, so analytics endpoints are UNAUTHENTICATED. "
                "This must not reach production."
            )

            async def _open():
                return None

            return _open


_SESSION_DEP = _resolve_session_dependency()
_CONSOLE_DEP = _resolve_console_dependency()


def filter_params(
    date_from: date | None = Query(None, description="Include complaints created on or after this date."),
    date_to: date | None = Query(None, description="Include complaints created on or before this date."),
    category: str | None = Query(None, description="road|water|waste|electricity|drainage|safety|other"),
    area: str | None = Query(None, description="Exact area name to restrict to."),
) -> AnalyticsFilters:
    """Shared optional filter set — accepted by every analytics endpoint."""
    return AnalyticsFilters(date_from=date_from, date_to=date_to, category=category, area=area)


def _service(session: Any):
    from app.analytics.service import AnalyticsService

    return AnalyticsService(session)


# --------------------------------------------------------------------------- #
# routes
# --------------------------------------------------------------------------- #
@router.get(
    "/overview",
    response_model=OverviewResponse,
    summary="KPI cards plus the headline plain-English insights",
)
async def overview(
    filters: AnalyticsFilters = Depends(filter_params),
    session: Any = Depends(_SESSION_DEP),
    _user: Any = Depends(_CONSOLE_DEP),
) -> Any:
    """Dashboard header: totals, resolution rate, median resolution time, critical
    backlog, average AI confidence, this week's volume with its week-over-week
    delta — and the top-ranked insights explaining what those numbers mean."""
    return await _service(session).overview(filters)


@router.get(
    "/categories",
    response_model=CategoriesResponse,
    summary="Category frequency distribution, mode and per-category resolution speed",
)
async def categories(
    filters: AnalyticsFilters = Depends(filter_params),
    session: Any = Depends(_SESSION_DEP),
    _user: Any = Depends(_CONSOLE_DEP),
) -> Any:
    """Absolute, relative and cumulative frequency by category, with the modal
    category called out and a category × status contingency table."""
    return await _service(session).categories(filters)


@router.get(
    "/priorities",
    response_model=PrioritiesResponse,
    summary="Priority distribution, category × priority cross-tab and a chi-square test",
)
async def priorities(
    filters: AnalyticsFilters = Depends(filter_params),
    session: Any = Depends(_SESSION_DEP),
    _user: Any = Depends(_CONSOLE_DEP),
) -> Any:
    """Priority frequencies plus the full contingency table against category, tested
    for independence with chi-square (expected-frequency assumption reported), and a
    Spearman check on whether higher priority actually means faster resolution."""
    return await _service(session).priorities(filters)


@router.get(
    "/resolution-times",
    response_model=ResolutionTimesResponse,
    summary="Full descriptive statistics, quartiles, Tukey fences and outliers",
)
async def resolution_times(
    filters: AnalyticsFilters = Depends(filter_params),
    session: Any = Depends(_SESSION_DEP),
    _user: Any = Depends(_CONSOLE_DEP),
) -> Any:
    """n, mean, median, mode, min/max/range, sample variance and standard deviation
    (ddof=1), Q1/Q2/Q3, IQR, Tukey fences, skewness, kurtosis, coefficient of
    variation, a Freedman–Diaconis histogram, per-category quartiles, and the
    individual outlier complaints listed by reference code."""
    return await _service(session).resolution_times(filters)


@router.get(
    "/trends",
    response_model=TrendsResponse,
    summary="Daily volume, 7-day moving average, week-over-week and a naive forecast",
)
async def trends(
    days: int = Query(90, ge=7, le=365, description="Length of the analysis window in days."),
    filters: AnalyticsFilters = Depends(filter_params),
    session: Any = Depends(_SESSION_DEP),
    _user: Any = Depends(_CONSOLE_DEP),
) -> Any:
    """Gap-free daily counts (missing days filled with zero so the moving average
    covers real calendar weeks), a 7-day rolling mean, per-category series,
    week-over-week change and a 7-day naive forecast with its method and assumptions
    stated on the wire."""
    return await _service(session).trends(filters, days=days)


@router.get(
    "/departments",
    response_model=DepartmentsResponse,
    summary="Per-department volume, median resolution time and backlog",
)
async def departments(
    filters: AnalyticsFilters = Depends(filter_params),
    session: Any = Depends(_SESSION_DEP),
    _user: Any = Depends(_CONSOLE_DEP),
) -> Any:
    """Departments compared on medians rather than means, with departments holding
    fewer than five resolved complaints excluded from the fastest/slowest ranking
    instead of being presented as though their averages were reliable."""
    return await _service(session).departments(filters)


@router.get(
    "/areas",
    response_model=AreasResponse,
    summary="Per-area volume, dominant category and hotspot flags",
)
async def areas(
    filters: AnalyticsFilters = Depends(filter_params),
    session: Any = Depends(_SESSION_DEP),
    _user: Any = Depends(_CONSOLE_DEP),
) -> Any:
    """Complaint volume by area with the dominant category for each, and a hotspot
    flag whose rule (>= 5 complaints and >= 1.5x an even share) is returned alongside
    the data so it can be audited."""
    return await _service(session).areas(filters)


@router.get(
    "/insights",
    response_model=InsightsResponse,
    summary="The plain-English narrative layer",
)
async def insights(
    limit: int | None = Query(None, ge=1, le=50, description="Cap the number of insights returned."),
    filters: AnalyticsFilters = Depends(filter_params),
    session: Any = Depends(_SESSION_DEP),
    _user: Any = Depends(_CONSOLE_DEP),
) -> Any:
    """Deterministic, rules-based interpretations of every computed statistic, ranked
    critical → warn → info. No language model is involved, so every number in every
    sentence is the exact value computed upstream and cannot be hallucinated."""
    return await _service(session).insights(filters, limit=limit)


@router.get(
    "/public-summary",
    response_model=PublicSummaryResponse,
    summary="Small anonymous subset for the public landing page (no auth)",
)
async def public_summary(
    filters: AnalyticsFilters = Depends(filter_params),
    session: Any = Depends(_SESSION_DEP),
) -> Any:
    """Aggregate counts, resolution rate, median resolution time and the category
    mix. No authentication, and deliberately nothing that could identify an
    individual complaint or complainant."""
    return await _service(session).public_summary(filters)


__all__ = ["router"]

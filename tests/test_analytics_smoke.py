"""Smoke tests for the seams other agents build on.

These do not test the analytics or AI modules themselves (owned elsewhere) — they
assert that the contracts those modules depend on still hold: health, departments,
optional-router tolerance, and the AI hand-off behaviour.
"""

from __future__ import annotations

from app.api.v1.router import OPTIONAL_MODULES
from app.models import Category, Priority, Status
from app.schemas.ai import AIAnalysisResult
from tests.conftest import make_complaint


async def test_health_reports_database_and_ai_provider(client):
    response = await client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["database"] == "ok"
    assert set(body) >= {"status", "database", "ai_provider", "version"}


async def test_departments_endpoint_reports_open_counts(client, session, department):
    await make_complaint(session, reference_code="CIV-JJJ000", status=Status.OPEN)
    complaint = await make_complaint(session, reference_code="CIV-KKK111", status=Status.OPEN)
    complaint.department_id = department.id
    await session.commit()

    response = await client.get("/api/v1/departments")
    assert response.status_code == 200
    body = response.json()
    assert len(body) == 1
    assert body[0]["slug"] == "roads"
    assert body[0]["categories"] == ["road"]
    assert body[0]["open_complaints"] == 1


def test_optional_router_registration_never_breaks_the_app():
    """The AI/analytics modules are owned by other agents and may not exist yet."""
    assert OPTIONAL_MODULES == (
        "app.api.v1.analytics",
        "app.api.v1.ai",
        "app.api.v1.assistant",
    )
    from app.main import app  # importing must succeed regardless of what is missing

    paths = set(app.openapi()["paths"])
    # Every path this module owns must be served whether or not the optional
    # modules happen to be present in the working tree right now.
    assert {
        "/health",
        "/api/v1/auth/login",
        "/api/v1/auth/me",
        "/api/v1/complaints",
        "/api/v1/complaints/track/{reference_code}",
        "/api/v1/complaints/analyze-preview",
        "/api/v1/complaints/{complaint_id}",
        "/api/v1/departments",
    } <= paths


async def test_analyze_preview_never_500s(client):
    """CONTRACT §4/§5.2 — a missing or failing analyzer is ``ai_unavailable``.

    ``app.ai`` is owned by another agent, so this asserts the *seam*: either the
    pipeline answers with a well-formed AIAnalysis, or we degrade to a clean 503.
    Never a 500, and never a broken submit screen.
    """
    response = await client.post(
        "/api/v1/complaints/analyze-preview",
        json={"description": "Sewage water has flooded the whole street near the school."},
    )
    assert response.status_code in (200, 503)
    if response.status_code == 503:
        assert response.json()["error"]["code"] == "ai_unavailable"
    else:
        body = response.json()
        assert body["source"] in {"llm", "ml", "rules"}
        assert body["category"] in {c.value for c in Category}
        assert 0.0 <= body["confidence"] <= 1.0
        assert body["created_at"].endswith("Z")


async def test_attach_analysis_folds_the_result_into_the_complaint(session, manager, department):
    """The seam the AI agent writes through."""
    complaint = await make_complaint(
        session, reference_code="CIV-LLL222", category=Category.OTHER, priority=Priority.LOW
    )
    result = AIAnalysisResult(
        category=Category.ROAD,
        priority=Priority.HIGH,
        summary="Large pothole causing a traffic hazard.",
        department_suggestion="Roads & Infrastructure",
        confidence=0.91,
        source="ml",
        model_name="tfidf-linearsvc-v1",
        keywords=["pothole", "road"],
        sentiment="concerned",
        latency_ms=17,
    )

    updated = await manager.attach_analysis(complaint.id, result)

    assert updated.ai_status == "complete"
    assert updated.category == Category.ROAD
    assert updated.priority == Priority.HIGH
    assert updated.department_id == department.id  # routed from the suggestion
    assert updated.ai is not None
    assert updated.ai.source == "ml"
    assert updated.ai.keywords == ["pothole", "road"]


async def test_mark_ai_failed_never_touches_the_complaint_content(session, manager):
    complaint = await make_complaint(session, reference_code="CIV-MMM333")
    await manager.mark_ai_failed(complaint.id)
    await session.refresh(complaint)
    assert complaint.ai_status == "failed"
    assert complaint.status == Status.OPEN
    assert complaint.is_deleted is False


async def test_resolution_hours_sorting_works_in_sql(client, session, auth_headers):
    """``sort=resolution_hours`` must be a SQL order-by, not a Python sort."""
    from datetime import UTC, datetime, timedelta

    slow = await make_complaint(session, reference_code="CIV-NNN444", status=Status.RESOLVED)
    fast = await make_complaint(session, reference_code="CIV-OOO555", status=Status.RESOLVED)
    now = datetime.now(UTC)
    slow.created_at, slow.resolved_at = now - timedelta(days=10), now
    fast.created_at, fast.resolved_at = now - timedelta(hours=2), now
    await session.commit()

    response = await client.get(
        "/api/v1/complaints",
        params={"sort": "resolution_hours", "order": "desc"},
        headers=auth_headers,
    )
    codes = [item["reference_code"] for item in response.json()["items"]]
    assert codes == ["CIV-NNN444", "CIV-OOO555"]

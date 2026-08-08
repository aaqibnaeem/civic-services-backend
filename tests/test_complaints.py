"""Complaint lifecycle: create, track, list/filter, transition rules, soft delete."""

from __future__ import annotations

import pytest

from app.core.errors import IllegalStatusTransition
from app.db.session import normalize_database_url
from app.models import Category, Priority, Status
from app.schemas.complaint import ComplaintUpdate
from tests.conftest import make_complaint


# ============================================================ URL normalisation
@pytest.mark.parametrize(
    ("raw", "expected_url", "expected_args"),
    [
        (
            # The exact shape Neon hands out — the #1 deployment failure mode.
            "postgresql://user:pw@ep-cool-1.eu-central-1.aws.neon.tech/civic"
            "?sslmode=require&channel_binding=require",
            "postgresql+asyncpg://user:pw@ep-cool-1.eu-central-1.aws.neon.tech/civic",
            {"ssl": True},
        ),
        (
            "postgres://user:pw@host:5432/db",
            "postgresql+asyncpg://user:pw@host:5432/db",
            {"ssl": True},
        ),
        (
            "postgresql+psycopg2://user:pw@host/db?sslmode=disable",
            "postgresql+asyncpg://user:pw@host/db",
            {},
        ),
        ("sqlite:///./civic.db", "sqlite+aiosqlite:///./civic.db", {}),
        ("sqlite+aiosqlite:///./civic.db", "sqlite+aiosqlite:///./civic.db", {}),
        (
            "sqlite:///./civic.db?check_same_thread=false",
            "sqlite+aiosqlite:///./civic.db",
            {},
        ),
    ],
)
def test_normalize_database_url(raw, expected_url, expected_args):
    url, connect_args = normalize_database_url(raw)
    assert url == expected_url
    assert connect_args == expected_args


def test_normalize_database_url_keeps_unrelated_query_params():
    url, args = normalize_database_url(
        "postgresql://u:p@host/db?sslmode=require&application_name=civic"
    )
    assert url == "postgresql+asyncpg://u:p@host/db?application_name=civic"
    assert args == {"ssl": True}


def test_normalize_database_url_rejects_garbage():
    with pytest.raises(ValueError):
        normalize_database_url("not-a-url")


@pytest.mark.parametrize(
    "raw, expected",
    [
        # The exact value that took the first Render deploy down: a single bare
        # origin is not valid JSON, and pydantic-settings decodes complex fields
        # in the env source before any validator runs.
        ("https://civic-services-frontend.vercel.app", ["https://civic-services-frontend.vercel.app"]),
        ("https://a.vercel.app, https://b.com", ["https://a.vercel.app", "https://b.com"]),
        ("https://a.vercel.app,", ["https://a.vercel.app"]),
        ('["https://a.vercel.app","https://b.com"]', ["https://a.vercel.app", "https://b.com"]),
        ("", []),
    ],
)
def test_cors_origins_accepts_env_friendly_values(monkeypatch, raw, expected):
    from app.core.config import Settings

    monkeypatch.setenv("CORS_ORIGINS", raw)
    assert expected == Settings(_env_file=None).CORS_ORIGINS


def test_cors_origins_rejects_malformed_json(monkeypatch):
    from app.core.config import Settings

    monkeypatch.setenv("CORS_ORIGINS", "[not valid json")
    with pytest.raises(ValueError):
        Settings(_env_file=None)


# ================================================================ create + track
async def test_create_complaint_then_track_by_reference(client, complaint_payload):
    created = await client.post("/api/v1/complaints", json=complaint_payload)
    assert created.status_code == 201, created.text
    body = created.json()

    assert body["ai_status"] == "pending"  # CONTRACT §5.1 — never blocks on the LLM
    assert body["status"] == "open"
    assert body["reference_code"].startswith("CIV-")
    assert len(body["reference_code"]) == 10
    assert not set(body["reference_code"][4:]) & set("01OIL")  # unambiguous alphabet
    assert body["area"] == "Gulshan-e-Iqbal"  # inferred from the free-text location
    assert body["resolution_hours"] is None

    tracked = await client.get(f"/api/v1/complaints/track/{body['reference_code']}")
    assert tracked.status_code == 200
    assert tracked.json()["id"] == body["id"]


async def test_track_unknown_reference_returns_contract_error_envelope(client):
    response = await client.get("/api/v1/complaints/track/CIV-NOPE99")
    assert response.status_code == 404
    error = response.json()["error"]
    assert error["code"] == "not_found"
    assert error["request_id"]
    assert response.headers["X-Request-ID"] == error["request_id"]


async def test_short_description_uses_the_same_error_envelope(client, complaint_payload):
    response = await client.post(
        "/api/v1/complaints", json={**complaint_payload, "description": "too short"}
    )
    assert response.status_code == 422
    error = response.json()["error"]
    assert error["code"] == "validation_error"
    assert any(detail["field"] == "description" for detail in error["details"])


# ======================================================================= listing
async def test_list_requires_authentication(client):
    response = await client.get("/api/v1/complaints")
    assert response.status_code == 401
    assert response.json()["error"]["code"] == "unauthorized"


async def test_list_filters_and_paginates_in_sql(client, session, auth_headers):
    await make_complaint(session, category=Category.ROAD, area="Korangi", reference_code="CIV-AAA111")
    await make_complaint(
        session, category=Category.WATER, area="Lyari", reference_code="CIV-BBB222",
        priority=Priority.HIGH,
    )
    await make_complaint(
        session, category=Category.WATER, area="Korangi", reference_code="CIV-CCC333",
        status=Status.RESOLVED,
    )

    everything = await client.get("/api/v1/complaints", headers=auth_headers)
    assert everything.status_code == 200
    assert everything.json()["total"] == 3

    by_category = await client.get(
        "/api/v1/complaints", params={"category": "water"}, headers=auth_headers
    )
    assert by_category.json()["total"] == 2

    by_area_and_status = await client.get(
        "/api/v1/complaints",
        params={"area": "korangi", "status": "resolved"},
        headers=auth_headers,
    )
    body = by_area_and_status.json()
    assert body["total"] == 1
    assert body["items"][0]["reference_code"] == "CIV-CCC333"

    paged = (
        await client.get(
            "/api/v1/complaints", params={"page_size": 2, "page": 2}, headers=auth_headers
        )
    ).json()
    assert paged["pages"] == 2
    assert paged["page"] == 2
    assert len(paged["items"]) == 1


async def test_list_search_matches_description(client, session, auth_headers):
    await make_complaint(
        session, reference_code="CIV-DDD444", description="The transformer near the market is sparking badly."
    )
    await make_complaint(
        session, reference_code="CIV-EEE555", description="Garbage has not been collected for ten days."
    )
    response = await client.get(
        "/api/v1/complaints", params={"q": "transformer"}, headers=auth_headers
    )
    body = response.json()
    assert body["total"] == 1
    assert body["items"][0]["reference_code"] == "CIV-DDD444"


# ============================================================ status transitions
async def test_illegal_status_transition_raises_domain_error(session, manager):
    complaint = await make_complaint(session, status=Status.RESOLVED, reference_code="CIV-FFF666")
    with pytest.raises(IllegalStatusTransition):
        await manager.apply_update(
            complaint.id, ComplaintUpdate(status=Status.OPEN), actor="admin@civic.gov.pk"
        )


async def test_illegal_transition_over_http_is_a_409(client, session, auth_headers):
    complaint = await make_complaint(session, status=Status.RESOLVED, reference_code="CIV-GGG777")
    response = await client.patch(
        f"/api/v1/complaints/{complaint.id}", json={"status": "open"}, headers=auth_headers
    )
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "conflict"


async def test_legal_transition_records_timeline_and_resolution_hours(
    client, session, auth_headers
):
    complaint = await make_complaint(session, reference_code="CIV-HHH888")

    for target in ("assigned", "in_progress", "resolved"):
        response = await client.patch(
            f"/api/v1/complaints/{complaint.id}",
            json={"status": target, "note": f"moving to {target}"},
            headers=auth_headers,
        )
        assert response.status_code == 200, response.text

    detail = await client.get(f"/api/v1/complaints/{complaint.id}", headers=auth_headers)
    body = detail.json()
    assert body["status"] == "resolved"
    assert body["resolved_at"] is not None
    assert body["resolution_hours"] is not None and body["resolution_hours"] >= 0
    assert [event["to_status"] for event in body["timeline"]] == [
        "assigned",
        "in_progress",
        "resolved",
    ]


# ==================================================================== soft delete
async def test_soft_delete_hides_the_complaint_but_keeps_the_row(
    client, session, auth_headers, admin_user
):
    complaint = await make_complaint(session, reference_code="CIV-III999")

    deleted = await client.delete(f"/api/v1/complaints/{complaint.id}", headers=auth_headers)
    assert deleted.status_code == 204

    assert (await client.get("/api/v1/complaints/track/CIV-III999")).status_code == 404
    listed = await client.get("/api/v1/complaints", headers=auth_headers)
    assert listed.json()["total"] == 0

    await session.refresh(complaint)
    assert complaint.is_deleted is True


async def test_setting_status_assigned_picks_an_owner(client, session, auth_headers):
    """A triager choosing "assigned" from a dropdown must not create an ownerless case.

    Reported from the deployed admin console: the status flipped to `assigned` and the
    assignee column stayed empty, which is not a state the system should be able to
    reach — `assigned` means a named person is accountable for it.
    """
    from app.models.department import Department
    from app.models.user import Role
    from app.repositories.user_repo import UserRepository
    from app.services.auth_service import AuthService

    department = Department(name="Roads & Infrastructure", slug="roads", categories=["road"])
    session.add(department)
    await session.flush()

    await AuthService(UserRepository(session)).ensure_user(
        email="picker.staff@civic.gov.pk",
        password="Staff@123",
        full_name="Picker Staff",
        role=Role.STAFF,
        department_id=department.id,
    )
    complaint = await make_complaint(
        session, category=Category.ROAD, reference_code="CIV-ASN001", status=Status.OPEN
    )
    complaint.department_id = department.id
    await session.commit()

    response = await client.patch(
        f"/api/v1/complaints/{complaint.id}",
        json={"status": "assigned"},
        headers=auth_headers,
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["status"] == "assigned"
    assert body["assignee"] is not None, "status=assigned must come with an owner"
    assert body["assignee"]["full_name"] == "Picker Staff"
    assert body["assigned_at"] is not None


async def test_status_assigned_without_available_staff_still_succeeds(
    client, session, auth_headers
):
    """No available staff is a real situation, not an error — but it must be visible."""
    complaint = await make_complaint(
        session, category=Category.ROAD, reference_code="CIV-ASN002", status=Status.OPEN
    )
    await session.commit()

    response = await client.patch(
        f"/api/v1/complaints/{complaint.id}",
        json={"status": "assigned"},
        headers=auth_headers,
    )
    assert response.status_code == 200, response.text
    assert response.json()["assignee"] is None

    detail = await client.get(f"/api/v1/complaints/{complaint.id}", headers=auth_headers)
    timeline = detail.json()["timeline"]
    # One PATCH must produce exactly one timeline row, and it should say why nobody
    # is named on it rather than leaving that silently blank.
    assert [event["to_status"] for event in timeline] == ["assigned"]
    assert "No staff member was available" in (timeline[0]["note"] or "")

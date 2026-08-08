"""Citizen accounts (CONTRACT §4b): auto-signup on submit, /complaints/mine, role boundaries."""

from __future__ import annotations

import pytest

from app.core.config import settings
from app.models import Role, Status
from tests.conftest import login, make_complaint, make_user

CITIZEN_PASSWORD = settings.CITIZEN_DEFAULT_PASSWORD


# ==================================================== citizen_email is now required
async def test_missing_citizen_email_is_a_field_level_422(client, complaint_payload):
    """A citizen must be told *which* box to fill in, not just "field required"."""
    payload = {k: v for k, v in complaint_payload.items() if k != "citizen_email"}
    response = await client.post("/api/v1/complaints", json=payload)

    assert response.status_code == 422
    error = response.json()["error"]
    assert error["code"] == "validation_error"
    detail = next(d for d in error["details"] if d["field"] == "citizen_email")
    assert detail["issue"].startswith("An email address is required")
    # Pydantic's internal "Value error, " prefix must not reach the citizen.
    assert "Value error" not in detail["issue"]


@pytest.mark.parametrize("bad", ["", "   ", "not-an-email", "ali@gmail", "a@b", "@example.com"])
async def test_malformed_citizen_email_is_rejected(client, complaint_payload, bad):
    response = await client.post(
        "/api/v1/complaints", json={**complaint_payload, "citizen_email": bad}
    )
    assert response.status_code == 422, bad
    assert any(d["field"] == "citizen_email" for d in response.json()["error"]["details"])


async def test_citizen_email_is_normalised_to_lowercase(client, complaint_payload):
    response = await client.post(
        "/api/v1/complaints", json={**complaint_payload, "citizen_email": "  Ayesha@Example.COM "}
    )
    assert response.status_code == 201, response.text
    assert response.json()["citizen_email"] == "ayesha@example.com"


# ============================================================ account creation block
async def test_first_submission_creates_an_account_and_returns_the_password(
    client, complaint_payload
):
    response = await client.post("/api/v1/complaints", json=complaint_payload)
    assert response.status_code == 201, response.text
    body = response.json()

    assert body["account"] == {
        "email": complaint_payload["citizen_email"],
        "is_new": True,
        "default_password": CITIZEN_PASSWORD,
    }
    assert body["citizen_id"] is not None


async def test_second_submission_reuses_the_account_and_hides_the_password(
    client, complaint_payload
):
    """The contract is explicit: never leak a password for an account we did not create."""
    first = await client.post("/api/v1/complaints", json=complaint_payload)
    second = await client.post(
        "/api/v1/complaints",
        json={**complaint_payload, "description": "A completely different water leak on my street."},
    )
    assert second.status_code == 201, second.text

    account = second.json()["account"]
    assert account["is_new"] is False
    assert account["default_password"] is None
    # Same person, so both complaints hang off one citizen row.
    assert second.json()["citizen_id"] == first.json()["citizen_id"]


async def test_existing_account_password_is_never_overwritten(client, session, complaint_payload):
    """Filing a complaint in someone's name must not reset their credentials."""
    await make_user(
        session,
        email=complaint_payload["citizen_email"],
        password="TheirOwnPassword@9",
        role=Role.CITIZEN,
        full_name="Existing Citizen",
    )

    created = await client.post("/api/v1/complaints", json=complaint_payload)
    assert created.status_code == 201
    assert created.json()["account"]["is_new"] is False

    # The original password still works; the default one never did.
    await login(client, complaint_payload["citizen_email"], "TheirOwnPassword@9")
    rejected = await client.post(
        "/api/v1/auth/login",
        json={"email": complaint_payload["citizen_email"], "password": CITIZEN_PASSWORD},
    )
    assert rejected.status_code == 401


async def test_a_staff_account_filing_a_complaint_is_not_demoted(
    client, session, department, complaint_payload
):
    staff = await make_user(
        session, email="imran.baloch@civic.gov.pk", department_id=department.id
    )
    response = await client.post(
        "/api/v1/complaints",
        json={**complaint_payload, "citizen_email": "imran.baloch@civic.gov.pk"},
    )
    assert response.status_code == 201
    assert response.json()["account"]["is_new"] is False

    await session.refresh(staff)
    assert staff.role == Role.STAFF
    assert staff.department_id == department.id


# =================================================================== login as citizen
async def test_the_created_citizen_can_log_in_and_read_auth_me(client, complaint_payload):
    created = await client.post("/api/v1/complaints", json=complaint_payload)
    account = created.json()["account"]

    headers = await login(client, account["email"], account["default_password"])
    me = await client.get("/api/v1/auth/me", headers=headers)
    assert me.status_code == 200
    body = me.json()
    assert body["email"] == account["email"]
    assert body["role"] == "citizen"
    assert body["department"] is None


# ================================================================= /complaints/mine
async def test_mine_returns_only_the_callers_own_complaints(client, session):
    """The isolation guarantee: citizen A must never see citizen B's complaint."""
    alice = await make_user(
        session, email="alice@example.com", password=CITIZEN_PASSWORD, role=Role.CITIZEN
    )
    bob = await make_user(
        session, email="bob@example.com", password=CITIZEN_PASSWORD, role=Role.CITIZEN
    )
    await make_complaint(session, reference_code="CIV-ALICE1", citizen_id=alice.id)
    await make_complaint(session, reference_code="CIV-ALICE2", citizen_id=alice.id)
    await make_complaint(session, reference_code="CIV-BOBBY1", citizen_id=bob.id)
    await make_complaint(session, reference_code="CIV-NOBODY")  # pre-accounts row

    alice_headers = await login(client, "alice@example.com", CITIZEN_PASSWORD)
    mine = await client.get("/api/v1/complaints/mine", headers=alice_headers)
    assert mine.status_code == 200
    body = mine.json()
    assert body["total"] == 2
    assert {item["reference_code"] for item in body["items"]} == {"CIV-ALICE1", "CIV-ALICE2"}

    bob_headers = await login(client, "bob@example.com", CITIZEN_PASSWORD)
    bobs = await client.get("/api/v1/complaints/mine", headers=bob_headers)
    codes = {item["reference_code"] for item in bobs.json()["items"]}
    assert codes == {"CIV-BOBBY1"}
    assert "CIV-ALICE1" not in codes and "CIV-ALICE2" not in codes


async def test_mine_uses_the_same_page_envelope_as_the_admin_list(client, session):
    citizen = await make_user(
        session, email="page@example.com", password=CITIZEN_PASSWORD, role=Role.CITIZEN
    )
    for index in range(5):
        await make_complaint(
            session, reference_code=f"CIV-PAGE0{index}", citizen_id=citizen.id
        )

    headers = await login(client, "page@example.com", CITIZEN_PASSWORD)
    response = await client.get(
        "/api/v1/complaints/mine", params={"page_size": 2, "page": 2}, headers=headers
    )
    body = response.json()
    assert set(body) == {"items", "total", "page", "page_size", "pages"}
    assert (body["total"], body["page"], body["page_size"], body["pages"]) == (5, 2, 2, 3)
    assert len(body["items"]) == 2


async def test_mine_can_be_filtered_by_status(client, session):
    citizen = await make_user(
        session, email="filter@example.com", password=CITIZEN_PASSWORD, role=Role.CITIZEN
    )
    await make_complaint(
        session, reference_code="CIV-OPEN01", citizen_id=citizen.id, status=Status.OPEN
    )
    await make_complaint(
        session, reference_code="CIV-DONE01", citizen_id=citizen.id, status=Status.RESOLVED
    )

    headers = await login(client, "filter@example.com", CITIZEN_PASSWORD)
    response = await client.get(
        "/api/v1/complaints/mine", params={"status": "resolved"}, headers=headers
    )
    assert [i["reference_code"] for i in response.json()["items"]] == ["CIV-DONE01"]


async def test_mine_requires_authentication(client):
    response = await client.get("/api/v1/complaints/mine")
    assert response.status_code == 401
    assert response.json()["error"]["code"] == "unauthorized"


async def test_mine_is_reachable_end_to_end_from_a_fresh_submission(client, complaint_payload):
    created = await client.post("/api/v1/complaints", json=complaint_payload)
    account = created.json()["account"]
    headers = await login(client, account["email"], account["default_password"])

    mine = await client.get("/api/v1/complaints/mine", headers=headers)
    assert [i["id"] for i in mine.json()["items"]] == [created.json()["id"]]


# ==================================================== the citizen/staff role boundary
async def test_a_citizen_is_locked_out_of_every_staff_endpoint(client, session):
    citizen = await make_user(
        session, email="citizen@example.com", password=CITIZEN_PASSWORD, role=Role.CITIZEN
    )
    complaint = await make_complaint(
        session, reference_code="CIV-GUARD1", citizen_id=citizen.id
    )
    headers = await login(client, "citizen@example.com", CITIZEN_PASSWORD)

    forbidden_reads = (
        "/api/v1/complaints",
        "/api/v1/analytics/overview",
        "/api/v1/analytics/insights",
        "/api/v1/staff",
        f"/api/v1/complaints/{complaint.id}",
    )
    for path in forbidden_reads:
        response = await client.get(path, headers=headers)
        assert response.status_code == 403, path
        assert response.json()["error"]["code"] == "forbidden", path

    # ...including on their *own* complaint: reading it is fine via /mine, but the
    # staff console and every mutation stay closed.
    patched = await client.patch(
        f"/api/v1/complaints/{complaint.id}", json={"status": "resolved"}, headers=headers
    )
    assert patched.status_code == 403

    assigned = await client.post(
        f"/api/v1/complaints/{complaint.id}/auto-assign", headers=headers
    )
    assert assigned.status_code == 403

    deleted = await client.delete(f"/api/v1/complaints/{complaint.id}", headers=headers)
    assert deleted.status_code == 403


async def test_public_reference_tracking_stays_open_to_everyone(client, complaint_payload):
    """Regression guard: adding accounts must not put a login in front of tracking."""
    created = await client.post("/api/v1/complaints", json=complaint_payload)
    reference = created.json()["reference_code"]

    tracked = await client.get(f"/api/v1/complaints/track/{reference}")
    assert tracked.status_code == 200
    assert tracked.json()["reference_code"] == reference
    # And no password is smuggled out through the public read.
    assert "account" not in tracked.json()

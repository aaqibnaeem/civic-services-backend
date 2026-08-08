"""Staff departments and auto-assignment (CONTRACT §4b).

The rule under test: within the complaint's department, pick the *available* staff
member with the fewest active complaints; break ties on the lowest summed priority
weight; break what remains on the lowest user id.
"""

from __future__ import annotations

import pytest

from app.core.errors import ConflictError
from app.models import Priority, Role, Status
from app.schemas.complaint import ComplaintUpdate
from app.services.assignment_service import PRIORITY_WEIGHTS
from tests.conftest import ADMIN_EMAIL, ADMIN_PASSWORD, login, make_complaint, make_user


async def _staff(session, department, *, email, **kwargs):
    return await make_user(session, email=email, department_id=department.id, **kwargs)


# =============================================================== the rule itself
def test_priority_weights_match_the_contract():
    assert {str(k): v for k, v in PRIORITY_WEIGHTS.items()} == {
        "critical": 4,
        "high": 3,
        "medium": 2,
        "low": 1,
    }


async def test_picks_the_staff_member_with_the_fewest_active_complaints(
    session, department, assignments
):
    busy = await _staff(session, department, email="busy@civic.gov.pk")
    idle = await _staff(session, department, email="idle@civic.gov.pk")
    for index in range(3):
        await make_complaint(
            session,
            reference_code=f"CIV-BUSY0{index}",
            department_id=department.id,
            assignee_id=busy.id,
            status=Status.IN_PROGRESS,
        )

    target = await make_complaint(
        session, reference_code="CIV-TARGET", department_id=department.id
    )
    assert (await assignments.choose_assignee(target)).id == idle.id


async def test_resolved_complaints_do_not_count_as_workload(session, department, assignments):
    """A closed case costs nobody anything — otherwise the best worker gets punished."""
    veteran = await _staff(session, department, email="veteran@civic.gov.pk")
    rookie = await _staff(session, department, email="rookie@civic.gov.pk")
    for index in range(5):
        await make_complaint(
            session,
            reference_code=f"CIV-DONE0{index}",
            department_id=department.id,
            assignee_id=veteran.id,
            status=Status.RESOLVED,
        )
    await make_complaint(
        session,
        reference_code="CIV-LIVE01",
        department_id=department.id,
        assignee_id=rookie.id,
        status=Status.OPEN,
    )

    target = await make_complaint(
        session, reference_code="CIV-TARGET", department_id=department.id
    )
    assert (await assignments.choose_assignee(target)).id == veteran.id


async def test_equal_counts_break_towards_the_lighter_priority_load(
    session, department, assignments
):
    """Two complaints each — but one person is carrying two criticals."""
    heavy = await _staff(session, department, email="heavy@civic.gov.pk")
    light = await _staff(session, department, email="light@civic.gov.pk")
    for index, holder in ((0, heavy), (1, heavy)):
        await make_complaint(
            session,
            reference_code=f"CIV-CRIT0{index}",
            department_id=department.id,
            assignee_id=holder.id,
            status=Status.ASSIGNED,
            priority=Priority.CRITICAL,
        )
    for index in (0, 1):
        await make_complaint(
            session,
            reference_code=f"CIV-LOW0{index}",
            department_id=department.id,
            assignee_id=light.id,
            status=Status.ASSIGNED,
            priority=Priority.LOW,
        )

    target = await make_complaint(
        session, reference_code="CIV-TARGET", department_id=department.id, priority=Priority.CRITICAL
    )
    assert (await assignments.choose_assignee(target)).id == light.id


async def test_a_perfect_tie_breaks_deterministically_on_user_id(
    session, department, assignments
):
    first = await _staff(session, department, email="one@civic.gov.pk")
    second = await _staff(session, department, email="two@civic.gov.pk")
    expected = min(first.id, second.id)

    target = await make_complaint(
        session, reference_code="CIV-TARGET", department_id=department.id
    )
    for _ in range(3):  # same answer every time, not a coin flip
        assert (await assignments.choose_assignee(target)).id == expected


async def test_unavailable_staff_are_skipped(session, department, assignments):
    await _staff(session, department, email="onleave@civic.gov.pk", is_available=False)
    on_duty = await _staff(session, department, email="onduty@civic.gov.pk")
    # The available person is already busier, and still wins.
    await make_complaint(
        session,
        reference_code="CIV-LOAD01",
        department_id=department.id,
        assignee_id=on_duty.id,
        status=Status.ASSIGNED,
    )

    target = await make_complaint(
        session, reference_code="CIV-TARGET", department_id=department.id
    )
    assert (await assignments.choose_assignee(target)).id == on_duty.id


async def test_staff_from_another_department_are_never_borrowed(
    session, department, other_department, assignments
):
    await _staff(session, other_department, email="elsewhere@civic.gov.pk")
    target = await make_complaint(
        session, reference_code="CIV-TARGET", department_id=department.id
    )
    assert await assignments.choose_assignee(target) is None


async def test_inactive_and_citizen_accounts_are_not_candidates(
    session, department, assignments
):
    await _staff(session, department, email="disabled@civic.gov.pk", is_active=False)
    await make_user(
        session, email="resident@example.com", role=Role.CITIZEN, department_id=department.id
    )
    target = await make_complaint(
        session, reference_code="CIV-TARGET", department_id=department.id
    )
    assert await assignments.choose_assignee(target) is None


async def test_a_complaint_with_no_department_stays_unassigned(
    session, department, assignments
):
    await _staff(session, department, email="ready@civic.gov.pk")
    orphan = await make_complaint(session, reference_code="CIV-ORPHAN")
    assert await assignments.choose_assignee(orphan) is None


# ======================================================= applying the decision
async def test_assigning_moves_open_to_assigned_and_writes_a_timeline_row(
    session, department, manager
):
    member = await _staff(session, department, email="applier@civic.gov.pk", full_name="Sana Rizvi")
    complaint = await make_complaint(
        session, reference_code="CIV-APPLY1", department_id=department.id
    )

    updated = await manager.auto_assign(complaint.id, actor="admin@civic.gov.pk")
    assert updated.assignee_id == member.id
    assert updated.status == Status.ASSIGNED
    assert updated.assigned_at is not None

    detail = await manager.get_detail(complaint.id)
    last = detail.timeline[-1]
    assert (last.from_status, last.to_status) == (Status.OPEN, Status.ASSIGNED)
    assert "Sana Rizvi" in last.note


async def test_assigning_a_complaint_already_in_progress_keeps_its_status(
    session, department, manager
):
    """Handing work over must not regress a case that is already being worked on."""
    await _staff(session, department, email="handover@civic.gov.pk")
    complaint = await make_complaint(
        session,
        reference_code="CIV-INPROG",
        department_id=department.id,
        status=Status.IN_PROGRESS,
    )

    updated = await manager.auto_assign(complaint.id, actor="admin@civic.gov.pk")
    assert updated.status == Status.IN_PROGRESS
    assert updated.assignee_id is not None
    detail = await manager.get_detail(complaint.id)
    assert detail.timeline[-1].to_status == Status.IN_PROGRESS


async def test_auto_assign_with_no_available_staff_leaves_it_untouched(
    session, department, manager
):
    await _staff(session, department, email="away@civic.gov.pk", is_available=False)
    complaint = await make_complaint(
        session, reference_code="CIV-NOSTAF", department_id=department.id
    )

    updated = await manager.auto_assign(complaint.id, actor="admin@civic.gov.pk")
    assert updated.assignee_id is None
    assert updated.status == Status.OPEN  # never dropped, never rerouted


# ============================================================ PATCH assignee_id
async def test_patch_sets_and_then_clears_the_assignee(client, session, department, auth_headers):
    member = await _staff(session, department, email="patch@civic.gov.pk")
    complaint = await make_complaint(
        session, reference_code="CIV-PATCH1", department_id=department.id
    )

    assigned = await client.patch(
        f"/api/v1/complaints/{complaint.id}",
        json={"assignee_id": member.id, "note": "taking this one"},
        headers=auth_headers,
    )
    assert assigned.status_code == 200, assigned.text
    body = assigned.json()
    assert body["assignee"]["id"] == member.id
    # One PATCH must leave exactly one audit row behind, not two.
    timeline = (
        await client.get(f"/api/v1/complaints/{complaint.id}", headers=auth_headers)
    ).json()["timeline"]
    assert [e["to_status"] for e in timeline] == ["assigned"]
    assert body["assignee"]["email"] == "patch@civic.gov.pk"
    assert body["assignee"]["department_id"] == department.id
    assert body["status"] == "assigned"
    assert body["assigned_at"] is not None

    cleared = await client.patch(
        f"/api/v1/complaints/{complaint.id}", json={"assignee_id": None}, headers=auth_headers
    )
    assert cleared.status_code == 200, cleared.text
    assert cleared.json()["assignee"] is None
    assert cleared.json()["assigned_at"] is None


def test_omitting_assignee_id_is_distinguishable_from_sending_null():
    """The Pydantic contract that makes "unassign" expressible at all."""
    omitted = ComplaintUpdate(status=Status.RESOLVED)
    explicit = ComplaintUpdate.model_validate({"assignee_id": None})

    assert omitted.assignee_provided is False
    assert explicit.assignee_provided is True
    assert omitted.assignee_id is explicit.assignee_id is None


async def test_patch_leaves_the_assignee_alone_when_the_field_is_omitted(
    client, session, department, auth_headers
):
    member = await _staff(session, department, email="keep@civic.gov.pk")
    complaint = await make_complaint(
        session,
        reference_code="CIV-KEEP01",
        department_id=department.id,
        assignee_id=member.id,
        status=Status.ASSIGNED,
    )

    response = await client.patch(
        f"/api/v1/complaints/{complaint.id}",
        json={"priority": "critical"},
        headers=auth_headers,
    )
    assert response.status_code == 200, response.text
    assert response.json()["assignee"]["id"] == member.id


# =============================================== validation: every rejection is a 409
async def test_rejects_an_assignee_who_is_not_staff(client, session, department, auth_headers):
    citizen = await make_user(session, email="resident@example.com", role=Role.CITIZEN)
    complaint = await make_complaint(
        session, reference_code="CIV-REJ001", department_id=department.id
    )

    response = await client.patch(
        f"/api/v1/complaints/{complaint.id}",
        json={"assignee_id": citizen.id},
        headers=auth_headers,
    )
    assert response.status_code == 409
    error = response.json()["error"]
    assert error["code"] == "conflict"
    assert "citizen" in error["message"]


async def test_rejects_a_deactivated_assignee(client, session, department, auth_headers):
    disabled = await _staff(
        session, department, email="disabled@civic.gov.pk", is_active=False
    )
    complaint = await make_complaint(
        session, reference_code="CIV-REJ002", department_id=department.id
    )

    response = await client.patch(
        f"/api/v1/complaints/{complaint.id}",
        json={"assignee_id": disabled.id},
        headers=auth_headers,
    )
    assert response.status_code == 409
    assert "deactivated" in response.json()["error"]["message"]


async def test_rejects_an_unavailable_assignee(client, session, department, auth_headers):
    away = await _staff(session, department, email="away@civic.gov.pk", is_available=False)
    complaint = await make_complaint(
        session, reference_code="CIV-REJ003", department_id=department.id
    )

    response = await client.patch(
        f"/api/v1/complaints/{complaint.id}",
        json={"assignee_id": away.id},
        headers=auth_headers,
    )
    assert response.status_code == 409
    assert "unavailable" in response.json()["error"]["message"]


async def test_rejects_an_assignee_from_another_department(
    client, session, department, other_department, auth_headers
):
    outsider = await _staff(session, other_department, email="outsider@civic.gov.pk")
    complaint = await make_complaint(
        session, reference_code="CIV-REJ004", department_id=department.id
    )

    response = await client.patch(
        f"/api/v1/complaints/{complaint.id}",
        json={"assignee_id": outsider.id},
        headers=auth_headers,
    )
    assert response.status_code == 409
    assert "department" in response.json()["error"]["message"]


async def test_rejects_an_unknown_assignee_with_a_404(
    client, session, department, auth_headers
):
    complaint = await make_complaint(
        session, reference_code="CIV-REJ005", department_id=department.id
    )
    response = await client.patch(
        f"/api/v1/complaints/{complaint.id}",
        json={"assignee_id": "00000000-0000-0000-0000-000000000000"},
        headers=auth_headers,
    )
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "not_found"


async def test_resolve_assignee_raises_a_domain_error_not_an_http_one(
    session, department, other_department, assignments
):
    outsider = await _staff(session, other_department, email="outsider@civic.gov.pk")
    complaint = await make_complaint(
        session, reference_code="CIV-DOMAIN", department_id=department.id
    )
    with pytest.raises(ConflictError):
        await assignments.resolve_assignee(complaint, outsider.id)


# ================================================================== the endpoints
async def test_auto_assign_endpoint_requires_staff(client, session, department):
    citizen = await make_user(session, email="nosy@example.com", role=Role.CITIZEN)
    complaint = await make_complaint(
        session, reference_code="CIV-AUTH01", department_id=department.id
    )
    headers = await login(client, "nosy@example.com", "Staff@123")
    response = await client.post(
        f"/api/v1/complaints/{complaint.id}/auto-assign", headers=headers
    )
    assert response.status_code == 403
    assert citizen.role == Role.CITIZEN


async def test_auto_assign_endpoint_returns_the_assigned_complaint(
    client, session, department, auth_headers
):
    member = await _staff(session, department, email="endpoint@civic.gov.pk")
    complaint = await make_complaint(
        session, reference_code="CIV-AUTO01", department_id=department.id
    )

    response = await client.post(
        f"/api/v1/complaints/{complaint.id}/auto-assign", headers=auth_headers
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["assignee"]["id"] == member.id
    assert body["status"] == "assigned"


async def test_auto_assign_endpoint_is_a_200_with_a_null_assignee_when_nobody_is_free(
    client, session, department, auth_headers
):
    complaint = await make_complaint(
        session, reference_code="CIV-AUTO02", department_id=department.id
    )
    response = await client.post(
        f"/api/v1/complaints/{complaint.id}/auto-assign", headers=auth_headers
    )
    assert response.status_code == 200
    assert response.json()["assignee"] is None


async def test_staff_directory_reports_department_and_workload(
    client, session, department, auth_headers
):
    member = await _staff(
        session, department, email="listed@civic.gov.pk", full_name="Zainab Khan"
    )
    await _staff(session, department, email="resting@civic.gov.pk", is_available=False)
    await make_complaint(
        session,
        reference_code="CIV-WORK01",
        department_id=department.id,
        assignee_id=member.id,
        status=Status.ASSIGNED,
    )
    await make_complaint(
        session,
        reference_code="CIV-WORK02",
        department_id=department.id,
        assignee_id=member.id,
        status=Status.RESOLVED,  # closed: must not inflate the counter
    )

    response = await client.get("/api/v1/staff", headers=auth_headers)
    assert response.status_code == 200, response.text
    rows = {row["email"]: row for row in response.json()}
    assert rows["listed@civic.gov.pk"]["active_assignments"] == 1
    assert rows["listed@civic.gov.pk"]["department"]["slug"] == department.slug
    assert rows["resting@civic.gov.pk"]["is_available"] is False
    assert rows["resting@civic.gov.pk"]["active_assignments"] == 0


async def test_staff_directory_is_admin_only(client, session, department):
    await _staff(session, department, email="peon@civic.gov.pk")
    headers = await login(client, "peon@civic.gov.pk", "Staff@123")
    assert (await client.get("/api/v1/staff", headers=headers)).status_code == 403


async def test_department_staff_endpoint_lists_only_that_department(
    client, session, department, other_department, auth_headers
):
    await _staff(session, department, email="inside@civic.gov.pk")
    await _staff(session, other_department, email="outside@civic.gov.pk")

    response = await client.get(
        f"/api/v1/departments/{department.id}/staff", headers=auth_headers
    )
    assert response.status_code == 200, response.text
    assert [row["email"] for row in response.json()] == ["inside@civic.gov.pk"]


async def test_department_staff_404s_for_an_unknown_department(client, auth_headers):
    response = await client.get(
        "/api/v1/departments/00000000-0000-0000-0000-000000000000/staff",
        headers=auth_headers,
    )
    assert response.status_code == 404


async def test_admin_can_post_a_staff_member_and_mark_them_unavailable(
    client, session, department, auth_headers
):
    member = await make_user(session, email="unposted@civic.gov.pk")
    assert member.department_id is None

    response = await client.patch(
        f"/api/v1/users/{member.id}",
        json={"department_id": department.id, "is_available": False},
        headers=auth_headers,
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["department_id"] == department.id
    assert body["department"]["slug"] == department.slug
    assert body["is_available"] is False


async def test_patch_users_can_clear_a_department_with_an_explicit_null(
    client, session, department, auth_headers
):
    member = await _staff(session, department, email="posted@civic.gov.pk")
    response = await client.patch(
        f"/api/v1/users/{member.id}", json={"department_id": None}, headers=auth_headers
    )
    assert response.status_code == 200, response.text
    assert response.json()["department_id"] is None
    assert response.json()["department"] is None


async def test_patch_users_404s_for_an_unknown_user(client, auth_headers):
    response = await client.patch(
        "/api/v1/users/00000000-0000-0000-0000-000000000000",
        json={"is_available": False},
        headers=auth_headers,
    )
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "not_found"


async def test_patch_users_is_admin_only(client, session, department):
    member = await _staff(session, department, email="notadmin@civic.gov.pk")
    headers = await login(client, "notadmin@civic.gov.pk", "Staff@123")
    response = await client.patch(
        f"/api/v1/users/{member.id}", json={"is_available": False}, headers=headers
    )
    assert response.status_code == 403


# ================================================= list filters: assignee_id + mine
async def test_list_can_filter_by_assignee_and_by_mine(client, session, department):
    mover = await _staff(session, department, email="mover@civic.gov.pk")
    other = await _staff(session, department, email="other@civic.gov.pk")
    await make_complaint(
        session, reference_code="CIV-MINE01", department_id=department.id, assignee_id=mover.id
    )
    await make_complaint(
        session, reference_code="CIV-MINE02", department_id=department.id, assignee_id=other.id
    )
    await make_complaint(session, reference_code="CIV-MINE03", department_id=department.id)

    headers = await login(client, "mover@civic.gov.pk", "Staff@123")
    mine = await client.get("/api/v1/complaints", params={"mine": "true"}, headers=headers)
    assert [i["reference_code"] for i in mine.json()["items"]] == ["CIV-MINE01"]

    by_id = await client.get(
        "/api/v1/complaints", params={"assignee_id": other.id}, headers=headers
    )
    assert [i["reference_code"] for i in by_id.json()["items"]] == ["CIV-MINE02"]


async def test_mine_wins_over_a_conflicting_assignee_id(client, session, department):
    mover = await _staff(session, department, email="mover@civic.gov.pk")
    other = await _staff(session, department, email="other@civic.gov.pk")
    await make_complaint(
        session, reference_code="CIV-WINS01", department_id=department.id, assignee_id=mover.id
    )
    await make_complaint(
        session, reference_code="CIV-WINS02", department_id=department.id, assignee_id=other.id
    )

    headers = await login(client, "mover@civic.gov.pk", "Staff@123")
    response = await client.get(
        "/api/v1/complaints",
        params={"mine": "true", "assignee_id": other.id},
        headers=headers,
    )
    assert [i["reference_code"] for i in response.json()["items"]] == ["CIV-WINS01"]


# ======================================== the AI pipeline hands off to assignment
async def test_the_pipeline_auto_assigns_once_the_department_is_routed(
    session, department, manager
):
    """CONTRACT §4b — assignment runs at the tail of ``analyze_and_store``."""
    from app.ai.pipeline import analyze_and_store

    member = await _staff(session, department, email="pipeline@civic.gov.pk")
    complaint = await make_complaint(
        session,
        reference_code="CIV-PIPE01",
        description="There is a very large pothole on Main University Road near the school.",
    )
    assert complaint.department_id is None  # routing happens inside the pipeline
    # Captured before expiring: the pipeline commits on its own session, so these
    # ORM objects go stale and re-reading an attribute would trigger a lazy load.
    complaint_id, member_id, department_id = complaint.id, member.id, department.id

    await analyze_and_store(complaint_id)

    session.expire_all()
    refreshed = await manager.get_or_404(complaint_id)
    assert refreshed.department_id == department_id
    assert refreshed.assignee_id == member_id
    assert refreshed.status == Status.ASSIGNED
    assert str(refreshed.ai_status) == "complete"


async def test_a_broken_assignment_never_breaks_ai_enrichment(
    session, department, manager, monkeypatch
):
    """Enrichment succeeding must not depend on assignment succeeding."""
    from app.ai.pipeline import analyze_and_store
    from app.services.assignment_service import AssignmentService

    await _staff(session, department, email="pipeline@civic.gov.pk")

    async def _explode(self, complaint):
        raise RuntimeError("assignment backend is down")

    monkeypatch.setattr(AssignmentService, "choose_assignee", _explode)

    complaint = await make_complaint(
        session,
        reference_code="CIV-PIPE02",
        description="There is a very large pothole on Main University Road near the school.",
    )
    complaint_id, department_id = complaint.id, department.id

    result = await analyze_and_store(complaint_id)  # must not raise
    assert result is not None

    session.expire_all()
    refreshed = await manager.get_or_404(complaint_id)
    assert str(refreshed.ai_status) == "complete"  # the analysis still landed
    assert refreshed.department_id == department_id  # so did the routing
    assert refreshed.assignee_id is None  # only the assignment was lost
    assert refreshed.status == Status.OPEN


async def test_submitting_over_http_ends_up_assigned(
    client, session, department, complaint_payload, auth_headers
):
    """End-to-end: submit -> background enrichment -> routed -> auto-assigned."""
    member = await _staff(session, department, email="e2e@civic.gov.pk")

    created = await client.post("/api/v1/complaints", json=complaint_payload)
    assert created.status_code == 201
    assert created.json()["assignee"] is None  # §5.1: nothing blocks on the pipeline

    detail = await client.get(
        f"/api/v1/complaints/{created.json()['id']}", headers=auth_headers
    )
    body = detail.json()
    assert body["ai_status"] == "complete"
    assert body["assignee"]["id"] == member.id
    assert body["status"] == "assigned"


async def test_admin_login_still_works_and_reports_no_department(client, admin_user):
    headers = await login(client, ADMIN_EMAIL, ADMIN_PASSWORD)
    me = await client.get("/api/v1/auth/me", headers=headers)
    assert me.status_code == 200
    assert me.json()["role"] == "admin"
    assert me.json()["department"] is None
    assert me.json()["is_available"] is True

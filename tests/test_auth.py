"""Auth: login, token claims, protected-route rejection, role guards."""

from __future__ import annotations

from app.core.security import (
    create_access_token,
    decode_access_token,
    generate_reference_code,
    hash_password,
    verify_password,
)
from app.models import Role
from tests.conftest import ADMIN_EMAIL, ADMIN_PASSWORD


def test_password_hashing_roundtrip():
    hashed = hash_password("Admin@123")
    assert hashed != "Admin@123"
    assert verify_password("Admin@123", hashed) is True
    assert verify_password("wrong-password", hashed) is False
    assert verify_password("Admin@123", "not-a-bcrypt-hash") is False


def test_jwt_carries_subject_and_role():
    token = create_access_token("user-123", role=Role.ADMIN.value, email="a@b.pk")
    payload = decode_access_token(token)
    assert payload["sub"] == "user-123"
    assert payload["role"] == "admin"


def test_reference_codes_avoid_ambiguous_characters():
    codes = {generate_reference_code() for _ in range(300)}
    assert len(codes) > 290  # collisions must be vanishingly rare
    for code in codes:
        assert code.startswith("CIV-")
        assert len(code) == 10
        assert not set(code[4:]) & set("01OIL")


async def test_login_returns_token_and_user(client, admin_user):
    response = await client.post(
        "/api/v1/auth/login", json={"email": ADMIN_EMAIL, "password": ADMIN_PASSWORD}
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["token_type"] == "bearer"
    assert body["user"]["email"] == ADMIN_EMAIL
    assert body["user"]["role"] == "admin"
    assert "hashed_password" not in body["user"]


async def test_login_with_bad_password_is_401(client, admin_user):
    response = await client.post(
        "/api/v1/auth/login", json={"email": ADMIN_EMAIL, "password": "nope"}
    )
    assert response.status_code == 401
    assert response.json()["error"]["code"] == "unauthorized"


async def test_me_requires_a_token(client):
    assert (await client.get("/api/v1/auth/me")).status_code == 401


async def test_me_rejects_a_garbage_token(client):
    response = await client.get(
        "/api/v1/auth/me", headers={"Authorization": "Bearer not.a.jwt"}
    )
    assert response.status_code == 401
    assert response.json()["error"]["code"] == "unauthorized"


async def test_me_returns_the_signed_in_user(client, auth_headers):
    response = await client.get("/api/v1/auth/me", headers=auth_headers)
    assert response.status_code == 200
    assert response.json()["email"] == ADMIN_EMAIL


# ================================================ role boundaries on the console
async def test_staff_can_read_analytics_but_cannot_delete(client, session):
    """The dashboard is the service team's tool, so staff must reach it.

    Analytics was originally guarded by ``require_admin``, which locked out exactly
    the people it was built for — a staff login got 403 on every dashboard route.
    Admin-only is reserved for destructive operations.
    """
    from app.models.user import Role
    from app.repositories.user_repo import UserRepository
    from app.services.auth_service import AuthService

    await AuthService(UserRepository(session)).ensure_user(
        email="staff@example.gov.pk",
        password="Staff@123",
        full_name="Test Staff",
        role=Role.STAFF,
    )
    await session.commit()

    login = await client.post(
        "/api/v1/auth/login",
        json={"email": "staff@example.gov.pk", "password": "Staff@123"},
    )
    assert login.status_code == 200, login.text
    headers = {"Authorization": f"Bearer {login.json()['access_token']}"}

    for path in ("/api/v1/analytics/overview", "/api/v1/analytics/resolution-times",
                 "/api/v1/analytics/insights", "/api/v1/complaints"):
        assert (await client.get(path, headers=headers)).status_code == 200, path

    forbidden = await client.delete(
        "/api/v1/complaints/00000000-0000-0000-0000-000000000000", headers=headers
    )
    assert forbidden.status_code == 403
    assert forbidden.json()["error"]["code"] == "forbidden"

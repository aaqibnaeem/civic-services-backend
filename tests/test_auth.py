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

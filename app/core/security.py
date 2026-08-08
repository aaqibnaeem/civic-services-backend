"""Password hashing and JWT issuing/verification.

bcrypt is called directly (passlib is unmaintained and breaks on bcrypt >= 4.1).
Tokens are HS256 with ``sub`` = user id and a ``role`` claim.
"""

from __future__ import annotations

import secrets
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

import bcrypt
import jwt

from app.core.config import settings
from app.core.errors import UnauthorizedError

ALGORITHM = "HS256"

# Unambiguous alphabet for public reference codes: no 0/O/1/I/L.
REFERENCE_ALPHABET = "23456789ABCDEFGHJKMNPQRSTUVWXYZ"
REFERENCE_PREFIX = "CIV-"
REFERENCE_LENGTH = 6

# bcrypt refuses inputs longer than 72 bytes.
_BCRYPT_MAX_BYTES = 72


def hash_password(plain: str) -> str:
    """Hash a plaintext password with a per-password bcrypt salt."""
    payload = plain.encode("utf-8")[:_BCRYPT_MAX_BYTES]
    return bcrypt.hashpw(payload, bcrypt.gensalt()).decode("utf-8")


def verify_password(plain: str, hashed: str) -> bool:
    """Constant-time password check; never raises on malformed hashes."""
    if not plain or not hashed:
        return False
    try:
        return bcrypt.checkpw(plain.encode("utf-8")[:_BCRYPT_MAX_BYTES], hashed.encode("utf-8"))
    except (ValueError, TypeError):
        return False


def generate_reference_code() -> str:
    """``CIV-`` + 6 chars from an unambiguous alphabet (collision-checked by the repo)."""
    body = "".join(secrets.choice(REFERENCE_ALPHABET) for _ in range(REFERENCE_LENGTH))
    return f"{REFERENCE_PREFIX}{body}"


def create_access_token(
    subject: str | uuid.UUID,
    *,
    role: str,
    email: str | None = None,
    expires_minutes: int | None = None,
) -> str:
    """Issue a signed JWT for ``subject`` (the user id)."""
    expire_delta = timedelta(minutes=expires_minutes or settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    now = datetime.now(UTC)
    payload: dict[str, Any] = {
        "sub": str(subject),
        "role": role,
        "iat": int(now.timestamp()),
        "exp": int((now + expire_delta).timestamp()),
        "jti": uuid.uuid4().hex,
    }
    if email:
        payload["email"] = email
    return jwt.encode(payload, settings.SECRET_KEY, algorithm=ALGORITHM)


def decode_access_token(token: str) -> dict[str, Any]:
    """Verify and decode a JWT, raising :class:`UnauthorizedError` on any problem."""
    try:
        return jwt.decode(token, settings.SECRET_KEY, algorithms=[ALGORITHM])
    except jwt.ExpiredSignatureError as exc:
        raise UnauthorizedError("Your session has expired. Please sign in again.") from exc
    except jwt.PyJWTError as exc:
        raise UnauthorizedError("Invalid authentication token.") from exc

"""Domain exception hierarchy and the wire error envelope.

Every failure the API can produce is expressed as a subclass of :class:`CivicError`,
which carries both the machine ``code`` from the frozen CONTRACT and the HTTP status
to answer with. Services raise these; the router layer never builds HTTPExceptions.
"""

from __future__ import annotations

from typing import Any


class CivicError(Exception):
    """Base class for every expected (non-bug) failure in the application.

    Why a class hierarchy: the API contract fixes a small closed set of error codes.
    Modelling them as exception types lets deep service code fail loudly without
    knowing anything about HTTP, while a single handler in ``main.py`` renders the
    exact envelope the frontend expects.
    """

    code: str = "internal_error"
    status_code: int = 500
    message: str = "Something went wrong."

    def __init__(
        self,
        message: str | None = None,
        *,
        details: list[dict[str, Any]] | None = None,
        code: str | None = None,
        status_code: int | None = None,
    ) -> None:
        self.message = message or self.message
        self.details = details or []
        if code is not None:
            self.code = code
        if status_code is not None:
            self.status_code = status_code
        super().__init__(self.message)

    def to_envelope(self, request_id: str) -> dict[str, Any]:
        """Render the CONTRACT §4 error envelope."""
        return {
            "error": {
                "code": self.code,
                "message": self.message,
                "details": self.details,
                "request_id": request_id,
            }
        }


class ValidationError(CivicError):
    """Input failed domain validation (as opposed to schema validation)."""

    code = "validation_error"
    status_code = 422
    message = "The submitted data is not valid."


class NotFoundError(CivicError):
    """A referenced resource does not exist (or is soft-deleted)."""

    code = "not_found"
    status_code = 404
    message = "The requested resource was not found."


class UnauthorizedError(CivicError):
    """Missing or invalid credentials."""

    code = "unauthorized"
    status_code = 401
    message = "Authentication is required."


class ForbiddenError(CivicError):
    """Authenticated, but not allowed to perform this action."""

    code = "forbidden"
    status_code = 403
    message = "You do not have permission to perform this action."


class ConflictError(CivicError):
    """The request conflicts with current state (duplicate key, illegal transition)."""

    code = "conflict"
    status_code = 409
    message = "The request conflicts with the current state of the resource."


class AIUnavailableError(CivicError):
    """No analyzer tier could satisfy the request."""

    code = "ai_unavailable"
    status_code = 503
    message = "The AI service is temporarily unavailable."


class RateLimitedError(CivicError):
    """Too many requests from this caller."""

    code = "rate_limited"
    status_code = 429
    message = "Too many requests. Please slow down."


class IllegalStatusTransition(ConflictError):
    """Raised by ``ComplaintManager`` when a status change violates the state machine."""

    def __init__(self, from_status: str, to_status: str) -> None:
        super().__init__(
            f"Cannot move a complaint from '{from_status}' to '{to_status}'.",
            details=[{"field": "status", "issue": f"illegal transition {from_status} -> {to_status}"}],
        )


__all__ = [
    "AIUnavailableError",
    "CivicError",
    "ConflictError",
    "ForbiddenError",
    "IllegalStatusTransition",
    "NotFoundError",
    "RateLimitedError",
    "UnauthorizedError",
    "ValidationError",
]

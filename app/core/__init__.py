"""Cross-cutting concerns: configuration, security, errors, logging, DI wiring."""

from app.core.config import Settings, get_settings, settings
from app.core.errors import (
    AIUnavailableError,
    CivicError,
    ConflictError,
    ForbiddenError,
    IllegalStatusTransition,
    NotFoundError,
    RateLimitedError,
    UnauthorizedError,
    ValidationError,
)
from app.core.logging_config import configure_logging, get_logger

__all__ = [
    "AIUnavailableError",
    "CivicError",
    "ConflictError",
    "ForbiddenError",
    "IllegalStatusTransition",
    "NotFoundError",
    "RateLimitedError",
    "Settings",
    "UnauthorizedError",
    "ValidationError",
    "configure_logging",
    "get_logger",
    "get_settings",
    "settings",
]

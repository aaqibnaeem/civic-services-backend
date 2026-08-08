"""Health probe. Mounted at the ROOT path (``/health``) per CONTRACT §3."""

from __future__ import annotations

from fastapi import APIRouter
from sqlalchemy import text

from app.core.config import settings
from app.core.deps import SessionDep, StorageDep
from app.core.logging_config import get_logger
from app.schemas.common import HealthResponse

log = get_logger(__name__)

router = APIRouter(tags=["health"])

APP_VERSION = "1.0.0"


def _ai_provider() -> str:
    """Report which analyzer tier the deployment is configured for.

    Read purely from configuration so the probe never makes a network call —
    a health check that can time out is worse than no health check.
    """
    if not settings.AI_ENABLED:
        return "disabled"
    if settings.DEEPSEEK_API_KEY:
        return f"deepseek:{settings.DEEPSEEK_MODEL}"
    if settings.GEMINI_API_KEY:
        return "gemini"
    return "rules"


@router.get("/health", response_model=HealthResponse, summary="Liveness + dependency probe")
async def health(session: SessionDep, storage: StorageDep) -> HealthResponse:
    """Returns ``200`` even when a dependency is degraded — the body carries detail.

    Render uses this as the deploy health check; failing it on a transient database
    blip would roll back an otherwise good deploy.
    """
    database = "ok"
    try:
        await session.execute(text("SELECT 1"))
    except Exception as exc:  # noqa: BLE001 - degraded, not fatal
        database = "error"
        log.error("health.database_unreachable", error=str(exc))

    return HealthResponse(
        status="ok" if database == "ok" else "degraded",
        database=database,
        ai_provider=_ai_provider(),
        version=APP_VERSION,
        environment=settings.APP_ENV,
        details={"storage": storage.backend, "ai_enabled": settings.AI_ENABLED},
    )

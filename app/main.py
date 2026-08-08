"""FastAPI application factory, middleware, error handlers and lifespan.

Boot order: configure logging -> create tables -> bootstrap admin -> optional seed.
Shutdown disposes the connection pool.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.staticfiles import StaticFiles

from app.api.v1 import health as health_module
from app.api.v1.router import LOADED_OPTIONAL_ROUTERS, api_router
from app.core.config import CORS_ORIGIN_REGEX, settings
from app.core.errors import CivicError
from app.core.logging_config import configure_logging, get_logger
from app.db import create_all
from app.services.storage_service import LocalStorageService, storage_service

configure_logging(debug=settings.DEBUG, level="DEBUG" if settings.DEBUG else "INFO")
log = get_logger("app")

APP_VERSION = health_module.APP_VERSION
REQUEST_ID_HEADER = "X-Request-ID"

# Maps FastAPI/Starlette HTTP status codes onto the CONTRACT §4 error codes.
STATUS_TO_CODE: dict[int, str] = {
    400: "validation_error",
    401: "unauthorized",
    403: "forbidden",
    404: "not_found",
    405: "not_found",
    409: "conflict",
    422: "validation_error",
    429: "rate_limited",
    503: "ai_unavailable",
}


def _request_id(request: Request) -> str:
    return getattr(request.state, "request_id", "-")


def _envelope(
    *, code: str, message: str, request_id: str, details: list[dict] | None = None
) -> dict:
    """The single place the error body shape is constructed."""
    return {
        "error": {
            "code": code,
            "message": message,
            "details": details or [],
            "request_id": request_id,
        }
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Create the schema, bootstrap the admin account, optionally seed demo data."""
    log.info(
        "app.starting",
        env=settings.APP_ENV,
        version=APP_VERSION,
        optional_routers=LOADED_OPTIONAL_ROUTERS,
    )

    # Alembic is deliberately out of scope for this hackathon: the schema is created
    # from the models at startup and the demo database is disposable.
    await create_all()

    await _bootstrap_admin()
    if settings.SEED_ON_STARTUP:
        await _seed_if_empty()

    yield

    from app.db.session import dispose_engine

    await dispose_engine()
    log.info("app.stopped")


async def _bootstrap_admin() -> None:
    """Guarantee the demo admin exists so ``/auth/login`` works on a fresh database."""
    from app.db.session import SessionLocal
    from app.models.user import Role
    from app.repositories.user_repo import UserRepository
    from app.services.auth_service import AuthService

    try:
        async with SessionLocal() as session:
            auth = AuthService(UserRepository(session))
            await auth.ensure_user(
                email=settings.ADMIN_EMAIL,
                password=settings.ADMIN_PASSWORD,
                full_name="Civic Administrator",
                role=Role.ADMIN,
            )
            await session.commit()
        log.info("bootstrap.admin_ready", email=settings.ADMIN_EMAIL)
    except Exception as exc:  # noqa: BLE001 - never block boot on bootstrap
        log.error("bootstrap.admin_failed", error=str(exc))


async def _seed_if_empty() -> None:
    """Populate demo data when ``SEED_ON_STARTUP=true`` and there are no complaints."""
    try:
        from app.db.session import SessionLocal
        from app.repositories.complaint_repo import ComplaintRepository

        async with SessionLocal() as session:
            existing = await ComplaintRepository(session).count_all(include_deleted=True)
        if existing:
            log.info("seed.skipped", reason="complaints table is not empty", count=existing)
            return

        from scripts.seed import seed_database

        summary = await seed_database(reset=False)
        log.info("seed.completed", **summary)
    except Exception as exc:  # noqa: BLE001 - seeding must never prevent startup
        log.error("seed.failed", error=str(exc))


def create_app() -> FastAPI:
    """Build and wire the FastAPI application."""
    app = FastAPI(
        title="AI Smart Civic Services API",
        description=(
            "Backend for the AI Smart Civic Services platform: complaint intake, "
            "AI triage, department routing and civic analytics."
        ),
        version=APP_VERSION,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    # --- CORS -----------------------------------------------------------------
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins,
        allow_origin_regex=CORS_ORIGIN_REGEX,  # any https://*.vercel.app preview
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=[REQUEST_ID_HEADER],
    )

    # --- request context ------------------------------------------------------
    @app.middleware("http")
    async def request_context(
        request: Request, call_next: Callable[[Request], Awaitable]
    ):
        """Generate/propagate ``X-Request-ID`` and bind it into every log line.

        The same id is echoed in the response header and inside every error envelope,
        so a screenshot of a failed request is enough to find its logs.
        """
        request_id = request.headers.get(REQUEST_ID_HEADER) or uuid.uuid4().hex
        request.state.request_id = request_id

        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(
            request_id=request_id, method=request.method, path=request.url.path
        )
        started = time.perf_counter()
        try:
            response = await call_next(request)
        finally:
            structlog.contextvars.unbind_contextvars("method", "path")
        response.headers[REQUEST_ID_HEADER] = request_id
        duration_ms = round((time.perf_counter() - started) * 1000, 2)
        if request.url.path not in {"/health", "/openapi.json"}:
            log.info(
                "http.request",
                status_code=response.status_code,
                duration_ms=duration_ms,
                request_id=request_id,
            )
        return response

    # --- error handlers -------------------------------------------------------
    @app.exception_handler(CivicError)
    async def handle_civic_error(request: Request, exc: CivicError) -> JSONResponse:
        request_id = _request_id(request)
        log.warning("error.domain", code=exc.code, status=exc.status_code, message=exc.message)
        return JSONResponse(
            status_code=exc.status_code,
            content=exc.to_envelope(request_id),
            headers={REQUEST_ID_HEADER: request_id},
        )

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        """Override FastAPI's default 422 body so schema errors use the same envelope."""
        request_id = _request_id(request)
        details = [
            {
                "field": ".".join(str(part) for part in error.get("loc", ())[1:]) or None,
                "issue": error.get("msg", "invalid value"),
            }
            for error in exc.errors()
        ]
        return JSONResponse(
            status_code=422,
            content=_envelope(
                code="validation_error",
                message="The submitted data is not valid.",
                request_id=request_id,
                details=details,
            ),
            headers={REQUEST_ID_HEADER: request_id},
        )

    @app.exception_handler(StarletteHTTPException)
    async def handle_http_exception(
        request: Request, exc: StarletteHTTPException
    ) -> JSONResponse:
        request_id = _request_id(request)
        code = STATUS_TO_CODE.get(exc.status_code, "internal_error")
        message = exc.detail if isinstance(exc.detail, str) else "Request failed."
        return JSONResponse(
            status_code=exc.status_code,
            content=_envelope(code=code, message=message, request_id=request_id),
            headers={REQUEST_ID_HEADER: request_id, **(exc.headers or {})},
        )

    @app.exception_handler(Exception)
    async def handle_unexpected(request: Request, exc: Exception) -> JSONResponse:
        """Last resort: log the traceback, return a safe message."""
        request_id = _request_id(request)
        log.exception("error.unhandled", error=str(exc), exc_info=exc)
        return JSONResponse(
            status_code=500,
            content=_envelope(
                code="internal_error",
                message="Something went wrong on our side. Please try again.",
                request_id=request_id,
            ),
            headers={REQUEST_ID_HEADER: request_id},
        )

    # --- routes ---------------------------------------------------------------
    app.include_router(health_module.router)  # /health lives at the root per CONTRACT §3
    app.include_router(api_router, prefix="/api/v1")

    if isinstance(storage_service, LocalStorageService):
        storage_service.root.mkdir(parents=True, exist_ok=True)
        app.mount(
            "/uploads", StaticFiles(directory=str(storage_service.root)), name="uploads"
        )

    @app.get("/", tags=["health"], summary="Service banner")
    async def root() -> dict:
        return {
            "service": "AI Smart Civic Services API",
            "version": APP_VERSION,
            "docs": "/docs",
            "health": "/health",
            "api": "/api/v1",
        }

    return app


app = create_app()

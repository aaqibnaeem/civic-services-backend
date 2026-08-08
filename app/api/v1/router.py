"""Aggregates every ``/api/v1`` router.

Core routers (auth, complaints, departments) are registered directly. The AI,
assistant and analytics modules are built in parallel by other agents, so they are
registered through a fault-tolerant loop below.
"""

from __future__ import annotations

from importlib import import_module

from fastapi import APIRouter

from app.api.v1 import auth, complaints, departments, staff
from app.core.logging_config import get_logger

log = get_logger(__name__)

api_router = APIRouter()

# --- routers this module owns -------------------------------------------------
api_router.include_router(auth.router)
api_router.include_router(complaints.router)
api_router.include_router(departments.router)
api_router.include_router(staff.router)


# --- deliberate optional-module registration ----------------------------------
# These three modules are owned by other agents working in the same repository at
# the same time. Importing them eagerly would mean a teammate's half-finished file
# takes the whole API down — including the endpoints that *are* ready. So each is
# imported defensively: a missing module logs a warning and the app still boots with
# everything else served. Once the module lands, it is picked up on the next start
# with no change here.
OPTIONAL_MODULES: tuple[str, ...] = (
    "app.api.v1.analytics",
    "app.api.v1.ai",
    "app.api.v1.assistant",
)


def _register_optional_routers(target: APIRouter) -> list[str]:
    """Include every optional router that currently exists. Returns the names loaded."""
    loaded: list[str] = []
    for module_path in OPTIONAL_MODULES:
        try:
            module = import_module(module_path)
        except ImportError as exc:
            log.warning("router.optional_missing", module=module_path, error=str(exc))
            continue
        except Exception as exc:  # noqa: BLE001 - a broken teammate module must not block boot
            log.error("router.optional_broken", module=module_path, error=str(exc))
            continue

        router = getattr(module, "router", None)
        if router is None:
            log.warning("router.optional_no_router_attr", module=module_path)
            continue

        target.include_router(router)
        loaded.append(module_path)
        log.info("router.optional_loaded", module=module_path)
    return loaded


LOADED_OPTIONAL_ROUTERS = _register_optional_routers(api_router)

__all__ = ["LOADED_OPTIONAL_ROUTERS", "OPTIONAL_MODULES", "api_router"]

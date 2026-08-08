"""Application configuration.

A single ``Settings`` object (pydantic-settings) is the only place that reads the
environment. Every other module imports the ``settings`` singleton or calls
``get_settings()``. The field names here are part of the internal contract other
agents (AI, analytics, ML) build against — do not rename them.
"""

from __future__ import annotations

import json
from functools import lru_cache
from typing import Annotated

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

# Origins the SPA runs on during development. Always allowed, on top of CORS_ORIGINS.
DEFAULT_CORS_ORIGINS: tuple[str, ...] = (
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:4173",
    "http://127.0.0.1:4173",
    "http://localhost:3000",
)

# Preview/production frontends live on Vercel; allow any subdomain via regex.
CORS_ORIGIN_REGEX = r"https://.*\.vercel\.app"


class Settings(BaseSettings):
    """Typed, validated view of the process environment.

    Why a class: it gives every consumer one immutable, documented surface and lets
    pydantic coerce/validate at boot instead of scattering ``os.getenv`` string
    parsing across the codebase.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- app -----------------------------------------------------------------
    APP_ENV: str = "development"
    DEBUG: bool = True
    SECRET_KEY: str = "dev-secret-change-me-in-production"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24

    # --- database ------------------------------------------------------------
    # Local default is SQLite; production injects a Neon Postgres URL.
    DATABASE_URL: str = "sqlite+aiosqlite:///./civic.db"

    # --- cors ----------------------------------------------------------------
    # `NoDecode` is load-bearing. For any complex field (list/dict/set),
    # pydantic-settings JSON-decodes the raw env var inside the settings *source*,
    # which runs before field validators — so `CORS_ORIGINS=https://foo.vercel.app`
    # raised a JSONDecodeError and killed the app at import time, before
    # `_split_origins` below ever saw it. NoDecode hands the raw string to the
    # validator instead.
    CORS_ORIGINS: Annotated[list[str], NoDecode] = Field(default_factory=list)

    # --- ai providers --------------------------------------------------------
    DEEPSEEK_API_KEY: str = ""
    DEEPSEEK_BASE_URL: str = "https://api.deepseek.com"
    DEEPSEEK_MODEL: str = "deepseek-v4-flash"
    AI_ENABLED: bool = True
    AI_TIMEOUT_SECONDS: int = 25
    AI_MAX_RETRIES: int = 3
    GEMINI_API_KEY: str = ""

    # --- ml ------------------------------------------------------------------
    ML_MODEL_PATH: str = "ml/artifacts/model.joblib"

    # --- seeding / bootstrap admin -------------------------------------------
    SEED_ON_STARTUP: bool = False
    ADMIN_EMAIL: str = "admin@civic.gov.pk"
    ADMIN_PASSWORD: str = "Admin@123"

    # --- storage -------------------------------------------------------------
    UPLOAD_DIR: str = "uploads"

    @field_validator("CORS_ORIGINS", mode="before")
    @classmethod
    def _split_origins(cls, value: object) -> object:
        """Accept ``a,b,c`` (env-friendly) as well as a real JSON list.

        With ``NoDecode`` on the field, nothing else will parse JSON for us, so
        this validator has to handle both shapes itself.
        """
        if value is None or value == "":
            return []
        if isinstance(value, str):
            raw = value.strip()
            if raw.startswith("["):
                try:
                    return json.loads(raw)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"CORS_ORIGINS looks like JSON but does not parse: {exc}"
                    ) from exc
            return [item.strip() for item in raw.split(",") if item.strip()]
        return value

    @property
    def allowed_origins(self) -> list[str]:
        """Dev origins plus anything configured via ``CORS_ORIGINS``, de-duplicated."""
        merged = list(DEFAULT_CORS_ORIGINS) + list(self.CORS_ORIGINS)
        seen: dict[str, None] = {}
        for origin in merged:
            seen.setdefault(origin.rstrip("/"), None)
        return list(seen)

    @property
    def is_production(self) -> bool:
        return self.APP_ENV.lower() in {"production", "prod"}


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached accessor — safe to use as a FastAPI dependency."""
    return Settings()


settings: Settings = get_settings()

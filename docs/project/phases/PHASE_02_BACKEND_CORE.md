# Phase 02 — Backend Core

**Status:** `complete`
**Owner:** Backend Core agent
**Scope:** `civic-backend/` — app factory, config, DB layer, models, schemas,
repositories, services, auth, complaints & departments API, seed data, tests, deploy
config.

---

## 1. What was built

### Stack (as locked)
Python 3.12 (managed by `uv`, pinned in `uv.lock`), FastAPI + Uvicorn, Pydantic v2 +
pydantic-settings, SQLAlchemy 2.0 **async** + `asyncpg` (Neon) / `aiosqlite` (local),
PyJWT + bcrypt (called directly — no passlib), structlog, pytest + pytest-asyncio +
httpx + ruff. **No Alembic** — `Base.metadata.create_all` runs in the lifespan
startup, with a comment saying migrations are deliberately out of scope.

### Layout

```
civic-backend/
  app/main.py             app factory, CORS, request-id middleware, error handlers, lifespan
  app/core/               config.py  security.py  errors.py  logging_config.py  deps.py
  app/db/                 base.py (Base, mixins, UTCDateTime, enum_column)  session.py  __init__.py
  app/models/             user  department  complaint  ai_analysis  status_event  (+ all enums)
  app/schemas/            common  user  department  complaint  ai
  app/repositories/       complaint_repo  user_repo  department_repo
  app/services/           complaint_service  department_service  auth_service
                          notification_service  storage_service
  app/api/v1/             router  auth  complaints  departments  health
  scripts/seed.py         ~800-complaint reproducible generator
  tests/                  conftest  test_complaints  test_auth  test_analytics_smoke
  pyproject.toml  uv.lock  .env.example  .gitignore  README.md  render.yaml  Makefile
```

### Endpoints implemented (all match the frozen CONTRACT)

| Method | Path | Auth |
|---|---|---|
| `GET` | `/health` | public (root path, not under `/api/v1`) |
| `POST` | `/api/v1/complaints` | public — returns `201` + `ai_status:"pending"` |
| `GET` | `/api/v1/complaints/track/{reference_code}` | public |
| `POST` | `/api/v1/complaints/analyze-preview` | public — **delegated** to `app.ai` |
| `GET` | `/api/v1/departments` | public |
| `POST` | `/api/v1/auth/login` | public |
| `GET` | `/api/v1/auth/me` | bearer |
| `GET` | `/api/v1/complaints` | staff — all contract filters/sort/pagination |
| `GET` | `/api/v1/complaints/{id}` | staff — includes `timeline` |
| `PATCH` | `/api/v1/complaints/{id}` | staff — appends a `StatusEvent` |
| `POST` | `/api/v1/complaints/{id}/reanalyze` | staff |
| `GET` | `/api/v1/complaints/{id}/duplicates` | staff |
| `DELETE` | `/api/v1/complaints/{id}` | **admin only**, soft delete |

Verified live: 24 OpenAPI paths served with the analytics/ai/assistant modules loaded.

---

## 2. Key decisions

### Database URL normalisation (the #1 deployment failure mode)
`normalize_database_url()` in `app/db/session.py` returns `(url, connect_args)`:
upgrades `postgres://` / `postgresql://` / `postgresql+psycopg2://` to
`postgresql+asyncpg://`, strips `sslmode` / `channel_binding` / `options` (asyncpg
rejects these libpq params and raises `TypeError: connect() got an unexpected keyword
argument 'sslmode'`), and moves TLS into `connect_args={"ssl": True}`. SQLite is
upgraded to `sqlite+aiosqlite` and gets no connect args. **Paste the Neon URL
verbatim.** 8 unit tests cover it.

### Enums store the wire value, not the member name
All enum columns use `enum_column()` → `Enum(..., native_enum=False,
create_constraint=True, values_callable=lambda e: [m.value for m in e])`. The DB
stores `road`, not `ROAD`. Native Postgres ENUM types were avoided on purpose — they
would need migrations we do not have.

### All timestamps are timezone-aware UTC
A custom `UTCDateTime` `TypeDecorator` stamps UTC on write and read. SQLite otherwise
discards `tzinfo` and returns naive datetimes, which serialise as
`2026-08-08T10:00:00` — a string browsers parse as **local** time. Responses use a
`UTCDatetime` Pydantic alias that renders `...Z` exactly as the contract shows.

### `resolution_hours`
A `@hybrid_property` on `Complaint`: Python getter for serialisation, plus a SQL
expression via a custom `hours_between` `GenericFunction` with `@compiles` hooks for
the default dialect (`EXTRACT(epoch FROM ...)/3600`) and SQLite
(`(julianday(b)-julianday(a))*24`). That makes `sort=resolution_hours` a real
`ORDER BY` on both backends — **analytics can use `Complaint.resolution_hours`
directly in SQL.**

### Status state machine
`ALLOWED_TRANSITIONS` in `app/services/complaint_service.py`. **`resolved` is
terminal** — `resolved -> open` raises `IllegalStatusTransition` (→ HTTP 409
`conflict`). Reopening a closed case would corrupt every resolution-time statistic;
recurrences are filed as new complaints. `rejected -> open` is allowed (appeal).

### Soft delete
`DELETE` sets `is_deleted`. Every repository read filters it out. Analytics keep a
stable historical denominator and reference codes never dangle.

### OOP service layer (graded rubric item)
| Class | Why it is a class |
|---|---|
| `ComplaintManager` | Only object that mutates a complaint. Bundles operations sharing invariants (a status change *always* emits a `StatusEvent` and *always* notifies) and collaborators (repo, routing, notifier). Private helpers are `_`-prefixed. |
| `StorageService` (ABC) | Real polymorphism: `LocalStorageService` writes to `UPLOAD_DIR`; `NoopStorageService` refuses honestly on disk-less hosts. Selected by `build_storage_service(config)`. |
| `NotificationService` (ABC) | `ConsoleNotificationChannel` + `EmailNotificationChannel`, fanned out by `NotificationDispatcher` on every transition. Each send is individually guarded so a channel failure cannot abort a committed transition. |
| `AuthService` | Auth is *policy* (indistinguishable failure messages, disabled accounts, token claims) and belongs in one auditable object. |
| `DepartmentService` | Routing is data-driven off `Department.categories`, not a hard-coded if/elif. |
| `*Repository` | All query construction. Layering is `router -> service -> repository -> DB`; no router touches a session. |

### Errors
`CivicError` hierarchy in `app/core/errors.py` (`NotFoundError`, `ValidationError`,
`UnauthorizedError`, `ForbiddenError`, `ConflictError`, `AIUnavailableError`,
`RateLimitedError`, plus `IllegalStatusTransition`), each carrying `code` +
`status_code`. `main.py` registers handlers for `CivicError`,
`RequestValidationError` (so Pydantic 422s use the **same** envelope),
`StarletteHTTPException`, and bare `Exception`. Middleware generates/propagates
`X-Request-ID`, binds it into `structlog.contextvars`, echoes it in the response
header, and embeds it in every error envelope.

### Optional-router registration
`app/api/v1/router.py` registers `app.api.v1.analytics`, `app.api.v1.ai` and
`app.api.v1.assistant` inside a loop that catches `ImportError` (and any other
exception) and logs a warning. A half-finished teammate module can never take down
the endpoints that *are* ready. All three are currently loading successfully.

---

## 3. What other agents must know

### AI agent
* **The enrichment seam is `app.ai.pipeline.analyze_and_store(complaint_id)`** —
  imported lazily inside `run_ai_enrichment` in `app/services/complaint_service.py`.
  It must open its **own** session (the request session is closed by then) and commit
  itself. Any exception, or the module not existing, ends as `ai_status="failed"` and
  a log line — never a rolled-back complaint (CONTRACT §5.1).
* **`analyze-preview` calls `app.ai.pipeline.analyze_text(description)`.** Sync or
  async both work (`inspect.isawaitable` is handled). Return an `AIAnalysisResult`, a
  pydantic model, or a plain dict — `_coerce_analysis` normalises all three. Any
  failure becomes `503 ai_unavailable`, never a 500.
* **Prefer writing through `ComplaintManager.attach_analysis(complaint_id, result)`**
  rather than touching `AIAnalysis` directly. It upserts the 1:1 row (so `/reanalyze`
  is safe), folds category/priority/title into the complaint, routes the department
  from `department_suggestion`, and sets `ai_status="complete"` in one transaction.
  `mark_ai_failed(complaint_id)` is the failure counterpart.
* `AIAnalysisResult` (`app/schemas/ai.py`) carries an optional `title` plus
  `prompt_tokens` / `completion_tokens` / `cache_hit_tokens`. Token telemetry is
  LLM-only; seeded rows leave it `None` on purpose.
* **Never label a non-LLM result as `llm`** (CONTRACT §5.3). Seed rows are `ml` /
  `rules` only.

### Analytics agent
* **Reuse `ComplaintRepository.build_filter_clauses(filters)`** (public + static) so
  `date_from` / `date_to` / `category` / `area` semantics match the list endpoint
  exactly instead of drifting.
* `Complaint.resolution_hours` works as a **SQL expression** on both SQLite and
  Postgres — use it in aggregates and `ORDER BY` directly.
* Always exclude soft-deleted rows: `Complaint.is_deleted.is_(False)`.
* Timestamps come back timezone-aware UTC; no manual tz handling needed.
* `PRIORITY_RANK` and `STATUS_RANK` (`app/models/complaint.py`) give ordinal values
  for ranking — the wire strings are not alphabetically ordered.
* Depend on `CurrentUser` / `AdminUser` from `app/core/deps.py` for auth, and on
  `SessionDep` or a repository — **do not open your own session inside a request**.

### ML agent
* `scripts/seed.py` is your training corpus: 800 rows, 7 categories, 12 English +
  4 Roman-Urdu templates each (~112 templates) with street/landmark/duration/area slot
  variation. `description -> category` is always internally consistent, so labels are
  clean.
* ~15–18 % of rows use Roman-Urdu phrasing ("nali ka pani", "bijli nahi hai",
  "kachra") — the model must handle it.
* Fixed seed `20260808`; complaint UUIDs are drawn from the seeded RNG, so ids are
  stable across runs too.
* `ML_MODEL_PATH` defaults to `ml/artifacts/classifier.joblib`.

### Frontend agent
* Base URL `/api/v1`; `/health` is at the **root**.
* Error envelope on every non-2xx, including Pydantic 422s. `X-Request-ID` is on
  every response and inside the envelope.
* CORS allows `localhost:5173`, `localhost:4173`, `localhost:3000`, anything in
  `CORS_ORIGINS`, plus regex `https://.*\.vercel\.app`.
* `POST /complaints` returns `201` with `ai_status:"pending"` — poll
  `GET /complaints/track/{ref}` until `ai_status != "pending"` (measured ~1 s with the
  ML tier).
* Timestamps are `...Z`. Pagination envelope is `{items, total, page, page_size, pages}`.

---

## 4. Seed data

`uv run python -m scripts.seed [--reset] [--count N] [--seed N]`

800 complaints over 180 days, fixed seed, idempotent (skips a non-empty complaints
table unless `--reset`). Also runs at startup when `SEED_ON_STARTUP=true` **and** the
complaints table is empty. The admin user is bootstrapped on *every* startup so login
works on a fresh unseeded database.

Verified shape of the generated dataset:

| Property | Measured |
|---|---|
| Status mix | resolved 67 %, open 13 %, in_progress 9 %, assigned 8 %, rejected 2 % |
| Monsoon spike | July 251 complaints vs ~110/month baseline; drainage 67 in July vs 8–23 |
| Slow department | Sewerage & Drainage median 106 h vs 47–83 h for the others |
| Hotspots | Orangi Town 134, Lyari 112, Korangi 96 vs DHA/Clifton ~35 |
| Right skew | median ≈ 78 h, mean ≈ 128 h, Tukey upper fence ≈ 320 h, ~42 outliers, max ≈ 1665 h |
| Roman-Urdu | ~18 % of rows |
| Duplicates | 21 rows linked via `duplicate_of_id` |
| AI rows | 800 (`ml` ≈ 70 %, `rules` ≈ 30 %), zero network calls |

Seeded accounts: `admin@civic.gov.pk` / `Admin@123` (admin),
`staff@civic.gov.pk` / `Staff@123` (staff).

---

## 5. Verification performed

* `uv run ruff check` — clean across every file this phase owns.
* `uv run pytest -q` — **148 passed**, including the whole suite with the AI and
  analytics modules present.
* Booted `uv run uvicorn app.main:app` against SQLite with the seeded database and
  exercised live:
  * `GET /health` → `200 {"status":"ok","database":"ok",...}`
  * `POST /api/v1/complaints` → `201`, `ai_status:"pending"`, `CIV-N7NWXV`
  * background enrichment completed ~1 s later → `ai_status:"complete"`, category
    corrected `other -> road`, priority `medium -> critical`, department auto-routed
    to Roads & Infrastructure, `source:"ml"`
  * `GET /complaints/track/{ref}` → `200`
  * `POST /auth/login` + `GET /auth/me` → `200`
  * `GET /complaints?category=drainage&status=resolved&sort=resolution_hours&order=desc`
    → correct SQL ordering (1520 h, 897 h)
  * `PATCH` `open -> assigned` → `200` + timeline row
  * `PATCH` `resolved -> open` → `409 conflict` with the contract envelope
  * `GET /complaints/{id}/duplicates` → scored candidates with reasons
  * `DELETE /complaints/{id}` (admin) → `204`, row hidden from track and list
  * unauthenticated list → `401 unauthorized`; short description → `422
    validation_error` with per-field details
  * Server killed afterwards; zero errors/tracebacks in the uvicorn log.

---

## 6. Contract deviations

**None.** Every endpoint path, field name, enum wire value, error code and response
shape matches `docs/CONTRACT.md` exactly.

Two additions that are supersets, not deviations:
* `/health` also returns `environment` and `details{storage, ai_enabled}` alongside
  the four required fields.
* `AIAnalysis` responses include the three nullable token-telemetry fields
  (`prompt_tokens`, `completion_tokens`, `cache_hit_tokens`) in addition to the
  contract fields.

---

## 7. Running it

```bash
cd civic-backend
uv sync
uv run python -m scripts.seed --reset
uv run uvicorn app.main:app --reload --port 8000
```

Docs at <http://127.0.0.1:8000/docs>. Deployment blueprint in `render.yaml`
(free plan, `uv sync --frozen --no-dev` build, uvicorn start, `/health` check,
`UPLOAD_DIR=none` because the free plan has no persistent disk).

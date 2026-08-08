# AI Smart Civic Services — Backend

FastAPI + SQLAlchemy 2.0 (async) backend for the AI Smart Civic Services platform:
citizen complaint intake, AI triage, department routing, and civic analytics.

The HTTP surface is defined by [`../docs/CONTRACT.md`](../docs/CONTRACT.md), which is
frozen. Field names, enum wire values, endpoint paths and the error envelope all come
from there.

---

## Quick start

```bash
cd civic-backend
cp .env.example .env          # defaults work as-is for local SQLite
uv sync                       # creates .venv from uv.lock (Python 3.12)
uv run python -m scripts.seed # ~800 demo complaints + departments + admin
uv run uvicorn app.main:app --reload --port 8000
```

Then open <http://127.0.0.1:8000/docs>.

`uv` is the only prerequisite — it installs the pinned Python 3.12 itself. Do not use
the system `python3` (it is 3.9 on macOS and this project requires 3.12+).

### Prove it works

```bash
curl -s localhost:8000/health

REF=$(curl -s -X POST localhost:8000/api/v1/complaints \
  -H 'Content-Type: application/json' \
  -d '{"description":"Large pothole on Main University Road near the school gate.",
       "location_text":"Block 5, Gulshan-e-Iqbal, Karachi","consent":true}' \
  | python3 -c 'import sys,json;print(json.load(sys.stdin)["reference_code"])')

curl -s "localhost:8000/api/v1/complaints/track/$REF"

TOKEN=$(curl -s -X POST localhost:8000/api/v1/auth/login \
  -H 'Content-Type: application/json' \
  -d '{"email":"admin@civic.gov.pk","password":"Admin@123"}' \
  | python3 -c 'import sys,json;print(json.load(sys.stdin)["access_token"])')

curl -s -H "Authorization: Bearer $TOKEN" \
  "localhost:8000/api/v1/complaints?status=open&sort=priority&order=desc&page_size=5"
```

### Demo credentials

| Email | Password | Role |
|---|---|---|
| `admin@civic.gov.pk` | `Admin@123` | admin |
| `staff@civic.gov.pk` | `Staff@123` | staff (created by the seeder) |

The admin account is created on every startup, so login works even on a fresh,
unseeded database.

---

## Common commands

```bash
make dev       # uvicorn with autoreload
make seed      # seed if empty
make reseed    # wipe and regenerate the demo dataset
make test      # pytest
make lint      # ruff check --fix
```

Or without make:

```bash
uv run uvicorn app.main:app --reload --port 8000
uv run python -m scripts.seed --reset --count 800
uv run pytest -q
uv run ruff check --fix .
```

---

## Environment variables

Every variable maps to a field on `Settings` in `app/core/config.py`; nothing else in
the codebase reads the environment.

| Variable | Default | Notes |
|---|---|---|
| `APP_ENV` | `development` | `production` switches logs to JSON |
| `DEBUG` | `true` | console log rendering + verbose level |
| `SECRET_KEY` | dev placeholder | **must** be set in production; HS256 signing key |
| `ACCESS_TOKEN_EXPIRE_MINUTES` | `1440` | JWT lifetime |
| `DATABASE_URL` | `sqlite+aiosqlite:///./civic.db` | paste the Neon URL verbatim; see below |
| `CORS_ORIGINS` | empty | comma-separated extra origins |
| `DEEPSEEK_API_KEY` | empty | LLM tier; absent ⇒ pipeline falls back to ML/rules |
| `DEEPSEEK_BASE_URL` | `https://api.deepseek.com` | |
| `DEEPSEEK_MODEL` | `deepseek-v4-flash` | |
| `AI_ENABLED` | `true` | master switch for the analyzer |
| `AI_TIMEOUT_SECONDS` | `25` | hard cap for `analyze-preview` (CONTRACT §5.2) |
| `AI_MAX_RETRIES` | `3` | |
| `GEMINI_API_KEY` | empty | optional secondary provider |
| `ML_MODEL_PATH` | `ml/artifacts/classifier.joblib` | scikit-learn artifact |
| `SEED_ON_STARTUP` | `false` | seeds only when the complaints table is empty |
| `ADMIN_EMAIL` | `admin@civic.gov.pk` | bootstrap admin |
| `ADMIN_PASSWORD` | `Admin@123` | change in production |
| `UPLOAD_DIR` | `uploads` | set to `none` to disable uploads entirely |

### The Neon / asyncpg gotcha

Neon hands out URLs like:

```
postgresql://user:pass@ep-xxx.aws.neon.tech/civic?sslmode=require&channel_binding=require
```

`asyncpg` **rejects** `sslmode` and `channel_binding` — they are libpq concepts, not
asyncpg keyword arguments, and passing them through produces
`TypeError: connect() got an unexpected keyword argument 'sslmode'` at the first
query. `normalize_database_url()` in `app/db/session.py` handles this for you:

* upgrades the scheme to `postgresql+asyncpg://`
* strips `sslmode`, `channel_binding` and `options`
* returns `connect_args={"ssl": True}` so TLS stays on

So paste the URL exactly as Neon gives it. This is unit-tested in
`tests/test_complaints.py::test_normalize_database_url`.

---

## Project documentation

The full write-up for this hackathon submission lives in [`docs/`](docs/):

| Document | What it covers |
|---|---|
| [Project overview](docs/project/OVERVIEW.md) | The civic problem, features, and how the two repos fit together |
| [Architecture](docs/project/ARCHITECTURE.md) | System, sequence and class diagrams — exactly where the AI sits |
| [AI testing evidence](docs/AI_TESTING_EVIDENCE.md) | 40 hand-written complaints through all three tiers, with limitations |
| [Model evaluation](ml/artifacts/evaluation.md) | Held-out accuracy, per-class scores, confusion matrices, and why the CV score is worthless |
| [API contract](docs/project/CONTRACT.md) | The frozen contract both repos were built against |
| [Build phases](docs/project/PROGRESS.md) | Phase-by-phase plan, decisions log and what integration uncovered |

Frontend repository: <https://github.com/aaqibnaeem/civic-services-frontend>

---

## Architecture

```
app/
  main.py             app factory, middleware, error handlers, lifespan
  core/               config, security (JWT + bcrypt), errors, logging, DI wiring
  db/                 Base + mixins, async engine, URL normaliser, create_all
  models/             SQLAlchemy models + the CONTRACT enums
  schemas/            Pydantic wire types
  repositories/       ALL SQL lives here
  services/           domain orchestration (ComplaintManager, AuthService, …)
  api/v1/             thin route handlers
scripts/seed.py       reproducible demo + ML training data generator
```

The dependency direction is strict and one-way:

```
router  ->  service  ->  repository  ->  database
```

A route handler never touches a session, and a service never writes a `select()`.
`app/core/deps.py` is the composition root that wires the three layers together.

### Design decisions worth knowing

* **No Alembic.** Tables are created from the models in the lifespan startup
  (`Base.metadata.create_all`). Migrations are deliberately out of scope for a
  hackathon with a disposable database; a production successor should add them.
* **Soft delete.** `DELETE /complaints/{id}` sets `is_deleted`. Analytics keep a
  stable historical denominator and reference codes never dangle.
* **`resolved` is terminal.** The status state machine in
  `app/services/complaint_service.py` refuses `resolved -> open`; reopening would
  corrupt every resolution-time statistic. Recurrences are filed as new complaints.
* **Enums store wire values.** Columns use
  `Enum(..., native_enum=False, values_callable=...)`, so the database holds `road`,
  not `ROAD` — the classic SQLAlchemy trap.
* **All timestamps are UTC-aware.** A custom `UTCDateTime` column type normalises on
  read and write (SQLite otherwise returns naive datetimes, which browsers parse as
  local time), and responses serialise as `...Z`.
* **AI never blocks intake.** `POST /complaints` commits and returns `201` with
  `ai_status: "pending"`; enrichment runs in a FastAPI `BackgroundTask` on its own
  session. Any failure — including the AI module not existing — ends as
  `ai_status: "failed"`, never a rolled-back complaint.
* **Optional routers.** `app/api/v1/router.py` registers `analytics`, `ai` and
  `assistant` inside a try/except loop, so a teammate's unfinished module cannot stop
  the rest of the API from booting.

### Object-oriented service layer

| Class | Responsibility |
|---|---|
| `ComplaintManager` | The only object that mutates a complaint: creation, reference codes, status state machine, timeline emission, AI hand-off |
| `AuthService` | Every credential decision: authentication, token issuance, role guards |
| `DepartmentService` | Category → department routing, read models with live counts |
| `StorageService` (ABC) | `LocalStorageService` / `NoopStorageService`, chosen by config |
| `NotificationService` (ABC) | `ConsoleNotificationChannel` / `EmailNotificationChannel`, fanned out by `NotificationDispatcher` on status transitions |
| `*Repository` | All query construction |

---

## Seed data

`scripts/seed.py` generates ~800 Karachi complaints over the last 180 days from a
fixed random seed, so every run is identical. It is shaped to tell a story rather
than be uniform noise:

* **Monsoon spike** — volume roughly doubles in July–August and the drainage share
  triples.
* **A slow department** — Sewerage & Drainage has a median resolution time about 2×
  the fastest department.
* **Hotspots** — Orangi Town, Lyari and Korangi carry disproportionate volume.
* **Right-skewed resolution times** — log-normal, so median ≈ 78 h against a mean
  ≈ 128 h, and Tukey fences surface ~40 genuine outliers.
* **~15 % Roman-Urdu phrasing** ("nali ka pani", "bijli nahi hai", "kachra"), which is
  how a large share of Karachi residents actually write.
* Every complaint gets a plausible `AIAnalysis` row (`ml` or `rules` tier) so the
  dashboard is fully populated **with no network calls**.

```bash
uv run python -m scripts.seed --reset --count 800
```

---

## Deployment (Render)

`render.yaml` is a working blueprint: free plan, build with `uv sync --frozen`, start
with uvicorn, health check on `/health`.

1. Create a Neon Postgres database and copy its connection string.
2. In Render: **New + → Blueprint**, point at this repository.
3. Set the secret env vars: `DATABASE_URL`, `ADMIN_PASSWORD`, `DEEPSEEK_API_KEY`,
   `CORS_ORIGINS` (your Vercel URL).
4. Deploy. The first boot creates the schema and — because `SEED_ON_STARTUP=true` —
   seeds the demo dataset into the empty database.

`UPLOAD_DIR` is set to `none` in the blueprint because Render's free plan has no
persistent disk; `NoopStorageService` is selected automatically so uploads degrade
cleanly instead of writing files that vanish on the next deploy.

---

## Testing

```bash
uv run pytest -q
```

Tests run against a temp-file SQLite database (not `:memory:`, so background-task
code paths that open their own session still see the data). Coverage focuses on the
things that actually break: the Neon URL normaliser, create → track by reference,
SQL-side list filtering and pagination, the illegal status transition, auth login and
protected-route rejection, and the error envelope shape.

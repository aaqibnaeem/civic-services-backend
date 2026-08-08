"""Pytest fixtures: an isolated in-memory database and an httpx client.

Each test gets a fresh SQLite file in a temp dir (rather than ``:memory:``) so the
background-task code path, which opens its *own* session, still sees the same data.
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import AsyncIterator
from pathlib import Path

import pytest

# Configure the environment before anything imports `app.core.config`.
_TMP_DIR = Path(tempfile.mkdtemp(prefix="civic-tests-"))
os.environ.setdefault("APP_ENV", "test")
os.environ["DATABASE_URL"] = f"sqlite+aiosqlite:///{_TMP_DIR / 'test.db'}"
os.environ["SECRET_KEY"] = "test-secret-key-that-is-long-enough-for-hs256"
os.environ["SEED_ON_STARTUP"] = "false"
os.environ["AI_ENABLED"] = "false"
os.environ["UPLOAD_DIR"] = str(_TMP_DIR / "uploads")
os.environ["ADMIN_EMAIL"] = "admin@civic.gov.pk"
os.environ["ADMIN_PASSWORD"] = "Admin@123"

import httpx  # noqa: E402
from httpx import ASGITransport  # noqa: E402

from app.db import create_all  # noqa: E402
from app.db.session import SessionLocal, engine  # noqa: E402
from app.main import app  # noqa: E402
from app.models import Base, Category, Complaint, Department, Priority, Role, Status  # noqa: E402
from app.repositories.complaint_repo import ComplaintRepository  # noqa: E402
from app.repositories.department_repo import DepartmentRepository  # noqa: E402
from app.repositories.user_repo import UserRepository  # noqa: E402
from app.services.auth_service import AuthService  # noqa: E402
from app.services.complaint_service import ComplaintManager  # noqa: E402
from app.services.department_service import DepartmentService  # noqa: E402

ADMIN_EMAIL = "admin@civic.gov.pk"
ADMIN_PASSWORD = "Admin@123"


@pytest.fixture(autouse=True)
async def clean_database() -> AsyncIterator[None]:
    """Drop and recreate every table around each test for full isolation."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await create_all()
    yield


@pytest.fixture
async def session() -> AsyncIterator:
    async with SessionLocal() as db:
        yield db


@pytest.fixture
async def admin_user(session):
    auth = AuthService(UserRepository(session))
    user = await auth.ensure_user(
        email=ADMIN_EMAIL, password=ADMIN_PASSWORD, full_name="Test Admin", role=Role.ADMIN
    )
    await session.commit()
    return user


@pytest.fixture
async def department(session) -> Department:
    repo = DepartmentRepository(session)
    dept = Department(
        name="Roads & Infrastructure",
        slug="roads",
        categories=[Category.ROAD.value],
        contact_email="roads@civic.gov.pk",
    )
    repo.add(dept)
    await session.commit()
    await session.refresh(dept)
    return dept


@pytest.fixture
async def manager(session) -> ComplaintManager:
    """A ``ComplaintManager`` wired exactly as the API wires it."""
    complaint_repo = ComplaintRepository(session)
    departments = DepartmentService(DepartmentRepository(session), complaint_repo)
    return ComplaintManager(complaint_repo, departments=departments)


@pytest.fixture
async def client() -> AsyncIterator[httpx.AsyncClient]:
    """ASGI-transport client. Lifespan is not run — fixtures own the schema."""
    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest.fixture
async def auth_headers(client: httpx.AsyncClient, admin_user) -> dict[str, str]:
    response = await client.post(
        "/api/v1/auth/login", json={"email": ADMIN_EMAIL, "password": ADMIN_PASSWORD}
    )
    assert response.status_code == 200, response.text
    return {"Authorization": f"Bearer {response.json()['access_token']}"}


@pytest.fixture
def complaint_payload() -> dict:
    return {
        "description": "There is a very large pothole on Main University Road near the school gate.",
        "location_text": "Block 5, Gulshan-e-Iqbal, Karachi",
        "citizen_name": "Test Citizen",
        "citizen_email": "citizen@example.com",
        "consent": True,
    }


async def make_complaint(
    session,
    *,
    category: Category = Category.ROAD,
    priority: Priority = Priority.MEDIUM,
    status: Status = Status.OPEN,
    area: str = "Gulshan-e-Iqbal",
    reference_code: str = "CIV-TEST01",
    description: str = "A test complaint about a broken road surface in the neighbourhood.",
) -> Complaint:
    """Insert a complaint directly — used by the list-filtering tests."""
    complaint = Complaint(
        reference_code=reference_code,
        title=description[:60],
        description=description,
        category=category,
        priority=priority,
        status=status,
        location_text=f"Block 1, {area}, Karachi",
        area=area,
    )
    session.add(complaint)
    await session.commit()
    await session.refresh(complaint)
    return complaint

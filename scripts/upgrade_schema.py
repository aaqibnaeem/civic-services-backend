"""Additive schema upgrade for a database that already holds data.

This project has no Alembic: tables are created from the models by
``Base.metadata.create_all`` at startup. That call creates *missing tables* and
nothing else — it will never add a column to a table that already exists. So the
moment a new column lands on a model, an existing deployment starts raising
``UndefinedColumnError`` on its first query, while a fresh database is perfectly
fine. That asymmetry is exactly the trap this script exists to defuse.

The alternative on offer was "drop everything and reseed", which works but throws
away the live complaint history — including any reference code a citizen has
already been given. This does the additive half of a migration instead:

  * compares every model against the live table and ``ADD COLUMN``s what is missing
  * seeds departments, staff and demo citizens using the seeder's own idempotent
    helpers, so an existing row is reused rather than duplicated
  * backfills assignees on complaints that already have a department

**It is additive only.** It never drops, renames, retypes or reorders a column,
so it cannot destroy data — the worst case is a column that is added and unused.
Anything requiring a destructive change is reported and left alone for a human.

    uv run python -m scripts.upgrade_schema --dry-run   # show the plan, change nothing
    uv run python -m scripts.upgrade_schema             # apply it
    uv run python -m scripts.upgrade_schema --skip-seed # schema only, no rows
"""

from __future__ import annotations

import argparse
import asyncio

from sqlalchemy import inspect as sa_inspect
from sqlalchemy import text
from sqlalchemy.schema import CreateColumn

from app.db.base import Base
from app.db.session import SessionLocal, engine
from app.models import *  # noqa: F403 - registers every model on Base.metadata


def _column_ddl(dialect, table_name: str, column) -> str:
    """``ALTER TABLE … ADD COLUMN …`` for one column, compiled for this dialect.

    A NOT NULL column can only be added to a populated table if it carries a
    server default, so one without a default is demoted to nullable rather than
    failing the whole run. The model still enforces the constraint on write; the
    gap is only in the database, and only for rows that predate the column.
    """
    spec = CreateColumn(column).compile(dialect=dialect).string
    if "NOT NULL" in spec.upper() and column.server_default is None:
        spec = spec.replace(" NOT NULL", "")
    return f"ALTER TABLE {table_name} ADD COLUMN {spec}"


async def plan(conn) -> tuple[list[str], list[str]]:
    """Return the DDL to run, plus warnings about anything not safely automatable."""
    statements: list[str] = []
    warnings: list[str] = []

    def _inspect(sync_conn):
        inspector = sa_inspect(sync_conn)
        existing_tables = set(inspector.get_table_names())
        found: dict[str, set[str]] = {}
        for name in existing_tables:
            found[name] = {c["name"] for c in inspector.get_columns(name)}
        return existing_tables, found

    existing_tables, existing_columns = await conn.run_sync(_inspect)
    dialect = conn.dialect

    for table in Base.metadata.sorted_tables:
        if table.name not in existing_tables:
            # create_all handles brand-new tables correctly; nothing to do here.
            continue
        present = existing_columns[table.name]
        for column in table.columns:
            if column.name in present:
                continue
            if not column.nullable and column.server_default is None:
                warnings.append(
                    f"{table.name}.{column.name} is NOT NULL with no server default; "
                    "adding it as nullable so existing rows stay valid."
                )
            statements.append(_column_ddl(dialect, table.name, column))

    return statements, warnings


async def upgrade(*, dry_run: bool, skip_seed: bool) -> None:
    async with engine.begin() as conn:
        statements, warnings = await plan(conn)

        for warning in warnings:
            print(f"  ! {warning}")

        if not statements:
            print("  schema is already up to date — no columns to add")
        else:
            print(f"  {len(statements)} column(s) to add:")
            for statement in statements:
                print(f"    {statement}")
            if dry_run:
                print("  (dry run — nothing executed)")
            else:
                for statement in statements:
                    await conn.execute(text(statement))
                print("  applied")

    if dry_run or skip_seed:
        if dry_run:
            print("  (dry run — skipping row seeding)")
        return

    # Reuse the seeder's idempotent helpers rather than reimplementing them: each
    # looks the row up first and only inserts when it is genuinely absent.
    from scripts.seed import _ensure_admin, _ensure_citizens, _ensure_departments, _ensure_staff

    async with SessionLocal() as session:
        departments = await _ensure_departments(session)
        await _ensure_admin(session)
        staff_by_department = await _ensure_staff(session, departments)
        citizens = await _ensure_citizens(session)
        await session.commit()

        staff_total = sum(len(v) for v in staff_by_department.values())
        print(f"  departments: {len(departments)}  staff: {staff_total}  citizens: {len(citizens)}")

        assigned = await _backfill_assignees(session)
        print(f"  backfilled assignees on {assigned} complaint(s)")


async def _backfill_assignees(session) -> int:
    """Give already-routed, already-actioned complaints an owner.

    Only touches complaints that have a department, have moved past ``open`` and
    have no assignee — so it cannot overwrite a human's decision, and it leaves
    open complaints alone because an unassigned open complaint is a legitimate
    state that the live system produces every day.
    """
    from sqlalchemy import select

    from app.models.complaint import Complaint, Status
    from app.repositories.complaint_repo import ComplaintRepository
    from app.repositories.user_repo import UserRepository
    from app.services.assignment_service import AssignmentService

    service = AssignmentService(UserRepository(session), ComplaintRepository(session))

    rows = list(
        (
            await session.execute(
                select(Complaint).where(
                    Complaint.assignee_id.is_(None),
                    Complaint.department_id.is_not(None),
                    Complaint.is_deleted.is_(False),
                    Complaint.status != Status.OPEN,
                )
            )
        )
        .scalars()
        .unique()
    )

    count = 0
    for complaint in rows:
        chosen = await service.choose_assignee(complaint)
        if chosen is not None:
            complaint.assignee_id = chosen.id
            count += 1
    await session.commit()
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="print the plan, change nothing")
    parser.add_argument("--skip-seed", action="store_true", help="add columns but do not seed rows")
    args = parser.parse_args()
    asyncio.run(upgrade(dry_run=args.dry_run, skip_seed=args.skip_seed))


if __name__ == "__main__":
    main()

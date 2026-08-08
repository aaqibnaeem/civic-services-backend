"""Department directory endpoint (CONTRACT §3 Public)."""

from __future__ import annotations

from fastapi import APIRouter

from app.core.deps import DepartmentServiceDep
from app.schemas.department import DepartmentRead

router = APIRouter(prefix="/departments", tags=["departments"])


@router.get("", response_model=list[DepartmentRead], summary="List departments")
async def list_departments(service: DepartmentServiceDep) -> list[DepartmentRead]:
    """Public directory including each department's live open-complaint count."""
    return await service.list_with_counts()

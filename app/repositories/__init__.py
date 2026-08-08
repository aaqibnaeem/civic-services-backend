"""Repository layer — the only place in the app that builds SQL."""

from app.repositories.complaint_repo import ComplaintRepository
from app.repositories.department_repo import DepartmentRepository
from app.repositories.user_repo import UserRepository

__all__ = ["ComplaintRepository", "DepartmentRepository", "UserRepository"]

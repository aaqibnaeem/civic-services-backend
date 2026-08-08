"""Model registry.

Importing this package imports every mapped class, which is what makes
``Base.metadata`` complete. ``app.db.init_models()`` and the lifespan handler rely
on that, and so should any script that calls ``create_all``.
"""

from app.db.base import Base
from app.models.ai_analysis import AIAnalysis, AISource, Sentiment
from app.models.complaint import (
    ACTIVE_STATUSES,
    PRIORITY_RANK,
    STATUS_RANK,
    AIStatus,
    Category,
    Complaint,
    Priority,
    Status,
)
from app.models.department import Department
from app.models.status_event import StatusEvent
from app.models.user import Role, User

__all__ = [
    "ACTIVE_STATUSES",
    "PRIORITY_RANK",
    "STATUS_RANK",
    "AIAnalysis",
    "AISource",
    "AIStatus",
    "Base",
    "Category",
    "Complaint",
    "Department",
    "Priority",
    "Role",
    "Sentiment",
    "Status",
    "StatusEvent",
    "User",
]

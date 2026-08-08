"""Stored output of the analyzer pipeline for one complaint (CONTRACT §2 AIAnalysis).

Carries token/latency telemetry alongside the prediction so the AI agent can prove
cost and speed on the dashboard without a second table.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import TYPE_CHECKING

from sqlalchemy import JSON, Boolean, Float, ForeignKey, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, UTCDateTime, UUIDPrimaryKeyMixin, enum_column, utcnow
from app.models.complaint import Category, Priority

if TYPE_CHECKING:
    from app.models.complaint import Complaint


class AISource(StrEnum):
    """Which analyzer tier produced the result: llm | ml | rules."""

    LLM = "llm"
    ML = "ml"
    RULES = "rules"


class Sentiment(StrEnum):
    """Wire values: calm | concerned | angry."""

    CALM = "calm"
    CONCERNED = "concerned"
    ANGRY = "angry"


class AIAnalysis(UUIDPrimaryKeyMixin, Base):
    """One complaint's analysis result, whichever tier produced it.

    Separated from ``Complaint`` rather than inlined because it has a different
    lifecycle (written asynchronously, re-runnable via ``/reanalyze``) and because
    the tier that produced it (``source``) must be auditable — CONTRACT §5.3 forbids
    passing a rules result off as an LLM result.
    """

    __tablename__ = "ai_analyses"

    complaint_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("complaints.id", ondelete="CASCADE"),
        unique=True,
        index=True,
        nullable=False,
    )

    category: Mapped[Category] = mapped_column(
        enum_column(Category, name="ai_category_enum"), nullable=False
    )
    priority: Mapped[Priority] = mapped_column(
        enum_column(Priority, name="ai_priority_enum"), nullable=False
    )
    summary: Mapped[str] = mapped_column(Text, nullable=False, default="")
    department_suggestion: Mapped[str | None] = mapped_column(String(150), nullable=True)
    confidence: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    source: Mapped[AISource] = mapped_column(
        enum_column(AISource, name="ai_source_enum"), nullable=False, index=True
    )
    model_name: Mapped[str] = mapped_column(String(120), nullable=False, default="")
    reasoning: Mapped[str | None] = mapped_column(Text, nullable=True)
    keywords: Mapped[list[str]] = mapped_column(JSON, default=list, nullable=False)
    sentiment: Mapped[Sentiment | None] = mapped_column(
        enum_column(Sentiment, name="ai_sentiment_enum"), nullable=True
    )
    is_emergency: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    # --- telemetry ------------------------------------------------------------
    latency_ms: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    prompt_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    completion_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    cache_hit_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        UTCDateTime, default=utcnow, server_default=func.now(), nullable=False
    )

    complaint: Mapped[Complaint] = relationship("Complaint", back_populates="ai", lazy="noload")

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<AIAnalysis {self.source}:{self.category} conf={self.confidence}>"

    @property
    def total_tokens(self) -> int | None:
        """Prompt + completion tokens, or ``None`` for non-LLM tiers."""
        if self.prompt_tokens is None and self.completion_tokens is None:
            return None
        return (self.prompt_tokens or 0) + (self.completion_tokens or 0)

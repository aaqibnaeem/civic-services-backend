"""AI-facing wire schemas (CONTRACT §2 AIAnalysis, §3 AI).

``AIAnalysisResult`` is the plain data class the analyzer tiers return; the AI agent
should build one of these and hand it to ``ComplaintManager.attach_analysis`` rather
than touching the ORM. ``AIAnalysisRead`` is what goes over the wire.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from app.schemas.common import AISource, Category, ORMModel, Priority, Sentiment, UTCDatetime


class AIAnalysisRead(ORMModel):
    """Serialised ``AIAnalysis`` row, matching the contract field-for-field."""

    category: Category
    priority: Priority
    summary: str
    department_suggestion: str | None = None
    confidence: float
    source: AISource
    model_name: str
    reasoning: str | None = None
    keywords: list[str] = Field(default_factory=list)
    sentiment: Sentiment | None = None
    is_emergency: bool = False
    latency_ms: int = 0
    created_at: UTCDatetime
    # Telemetry — nullable because only the LLM tier reports token usage.
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    cache_hit_tokens: int | None = None


class AIAnalysisResult(BaseModel):
    """Transport object produced by an analyzer tier before it is persisted.

    Kept separate from the ORM model so the AI pipeline can be unit-tested and run
    with no database at all (``/complaints/analyze-preview`` never saves).
    """

    category: Category = Category.OTHER
    priority: Priority = Priority.MEDIUM
    summary: str = ""
    department_suggestion: str | None = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    source: AISource = AISource.RULES
    model_name: str = ""
    reasoning: str | None = None
    keywords: list[str] = Field(default_factory=list)
    sentiment: Sentiment | None = None
    is_emergency: bool = False
    latency_ms: int = 0
    title: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    cache_hit_tokens: int | None = None


class AnalyzePreviewRequest(BaseModel):
    """Body of ``POST /complaints/analyze-preview``."""

    description: str = Field(min_length=10, max_length=5000)
    location_text: str | None = Field(default=None, max_length=300)


class AIHealthResponse(BaseModel):
    """Body of ``GET /ai/health`` (implemented by the AI agent)."""

    llm_available: bool = False
    ml_model_loaded: bool = False
    rules_available: bool = True
    model_name: str = ""
    last_error: str | None = None


class DuplicateCandidate(BaseModel):
    """One entry of ``GET /complaints/{id}/duplicates``."""

    complaint: dict
    similarity: float
    reason: str


class DuplicatesResponse(BaseModel):
    candidates: list[DuplicateCandidate] = Field(default_factory=list)

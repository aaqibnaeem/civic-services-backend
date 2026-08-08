"""``POST /assistant/chat`` — the natural-language interface over complaint data.

Thin by design. All of the logic lives in ``app.ai.assistant``; this module only
validates the request and shapes the response to CONTRACT §3.

The endpoint is public because the assistant only ever reports aggregates and
reference codes, which are already public tracking handles — it exposes nothing a
citizen could not obtain from ``/analytics/public-summary`` and their own
complaint. If that ever changes, add ``StaffUser`` to the signature.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.core.logging_config import get_logger

router = APIRouter(tags=["ai"])
log = get_logger(__name__)

MAX_HISTORY_TURNS = 8


class ChatTurn(BaseModel):
    """One prior message. Only role and content are carried forward."""

    role: Literal["user", "assistant"] = "user"
    content: str = Field(default="", max_length=2000)


class ChatRequest(BaseModel):
    """Body of ``POST /assistant/chat``."""

    message: Annotated[str, Field(min_length=1, max_length=1000)]
    history: list[ChatTurn] = Field(default_factory=list)


class Citation(BaseModel):
    """A complaint the answer refers to. Verified against the executed query."""

    reference_code: str
    id: str


class ChatResponse(BaseModel):
    """Contract shape: ``{answer, citations, used_stats, source}``."""

    answer: str
    citations: list[Citation] = Field(default_factory=list)
    used_stats: dict[str, Any] = Field(default_factory=dict)
    #: ``llm`` (planner + writer), ``hybrid`` (one tier fell back), ``rules`` (no LLM).
    source: str = "rules"


@router.post("/assistant/chat", response_model=ChatResponse, summary="Ask about complaint data")
async def assistant_chat(payload: ChatRequest) -> ChatResponse:
    """Answer a question about the complaints, grounded in real aggregates.

    The LLM never counts. A planner call turns the question into a whitelisted
    query plan, this service executes real SQL, and a second call phrases the
    computed facts. With no API key both LLM steps degrade to a keyword planner and
    a template writer, so the endpoint always answers — ``source`` says which path
    ran.
    """
    try:
        from app.ai.assistant import answer_question
    except Exception as exc:  # noqa: BLE001 - never 500 the chat box
        log.error("assistant.import_failed", error=str(exc))
        return ChatResponse(
            answer="The assistant is not available right now. Please try again shortly.",
            used_stats={"error": "assistant_unavailable"},
            source="rules",
        )

    history = [
        {"role": turn.role, "content": turn.content}
        for turn in payload.history[-MAX_HISTORY_TURNS:]
        if turn.content.strip()
    ]

    try:
        result = await answer_question(payload.message, history=history)
    except Exception as exc:  # noqa: BLE001
        log.exception("assistant.failed", error=str(exc))
        return ChatResponse(
            answer=("I could not answer that just now. Try asking about complaint counts, "
                    "categories, priorities, areas or resolution times."),
            used_stats={"error": "assistant_error"},
            source="rules",
        )

    log.info(
        "assistant.answered",
        source=result.get("source"),
        citations=len(result.get("citations") or []),
        latency_ms=(result.get("used_stats") or {}).get("latency_ms"),
    )
    return ChatResponse(
        answer=result.get("answer", ""),
        citations=[Citation(**c) for c in (result.get("citations") or [])],
        used_stats=result.get("used_stats") or {},
        source=result.get("source", "rules"),
    )

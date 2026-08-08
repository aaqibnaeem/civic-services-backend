"""The analyzer orchestrator: fallback chain, cache, and persistence.

    analyze_text(text, context)
        DeepSeek ──fail──► MLAnalyzer ──fail──► RuleBasedAnalyzer ──► always returns

The chain is the whole reliability story. The rules tier has no dependencies and
cannot be unavailable, so ``analyze_text`` is total: it never raises, and there is
no input for which the system has no answer. Which tier actually won is recorded
in ``AnalysisResult.source`` and surfaced as a UI badge — CONTRACT §5.3 forbids
passing a rules result off as an LLM result, and this module is where that promise
is kept.

Two functions matter to the rest of the app:

``analyze_text``       pure analysis, no database. Powers
                       ``POST /complaints/analyze-preview`` (CONTRACT §5.2).
``analyze_and_store``  loads a complaint, analyses it, persists the result, routes
                       the department, runs duplicate detection and flips
                       ``ai_status``. Called from a BackgroundTask so complaint
                       submission never blocks on the LLM (CONTRACT §5.1).
"""

from __future__ import annotations

import logging
import threading
import time
from collections import OrderedDict
from typing import Any

from app.ai.base import (
    AIAnalyzer,
    AnalysisResult,
    normalise_text,
)
from app.ai.llm_analyzer import DeepSeekAnalyzer
from app.ai.ml_analyzer import MLAnalyzer
from app.ai.rule_analyzer import RuleBasedAnalyzer

logger = logging.getLogger("app.ai.pipeline")

#: Entries in the in-process result cache.
CACHE_MAX_ENTRIES = 256
#: Cache lifetime. Long enough for a demo loop, short enough that a re-analysis
#: after a prompt change is never served stale.
CACHE_TTL_SECONDS = 900.0


# --------------------------------------------------------------------------- #
# In-process LRU cache
# --------------------------------------------------------------------------- #

class _ResultCache:
    """Tiny TTL + LRU cache keyed on normalised complaint text.

    Demo submissions repeat constantly ("let me show you the AI again"), and a
    repeat is both a wasted API call and 2-6 s of dead air on stage. Deliberately
    in-process: a Redis dependency for a 256-entry cache would add a network hop
    to the component that exists to make things fast.
    """

    def __init__(self, max_entries: int = CACHE_MAX_ENTRIES, ttl: float = CACHE_TTL_SECONDS):
        self._data: OrderedDict[str, tuple[float, AnalysisResult]] = OrderedDict()
        self._lock = threading.Lock()
        self._max = max_entries
        self._ttl = ttl
        self.hits = 0
        self.misses = 0

    @staticmethod
    def key(text: str, context: dict[str, Any] | None) -> str:
        ctx = context or {}
        hint = ctx.get("category") or ""
        return f"{normalise_text(text)}|{hint}"

    def get(self, key: str) -> AnalysisResult | None:
        with self._lock:
            entry = self._data.get(key)
            if entry is None:
                self.misses += 1
                return None
            stored_at, result = entry
            if (time.monotonic() - stored_at) > self._ttl:
                del self._data[key]
                self.misses += 1
                return None
            self._data.move_to_end(key)
            self.hits += 1
            # Copy: callers mutate (latency, cached flag) and must not poison the
            # cached entry.
            return result.model_copy(deep=True)

    def put(self, key: str, result: AnalysisResult) -> None:
        with self._lock:
            self._data[key] = (time.monotonic(), result.model_copy(deep=True))
            self._data.move_to_end(key)
            while len(self._data) > self._max:
                self._data.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()
            self.hits = self.misses = 0

    def stats(self) -> dict[str, Any]:
        with self._lock:
            total = self.hits + self.misses
            return {
                "entries": len(self._data),
                "max_entries": self._max,
                "ttl_seconds": self._ttl,
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate": round(self.hits / total, 4) if total else 0.0,
            }


_cache = _ResultCache()


# --------------------------------------------------------------------------- #
# Analyzer registry
# --------------------------------------------------------------------------- #

_analyzers: list[AIAnalyzer] | None = None
_registry_lock = threading.Lock()


def get_analyzers(refresh: bool = False) -> list[AIAnalyzer]:
    """The fallback chain, highest quality first. Built once per process.

    All three are constructed even when unavailable, because availability is
    dynamic: a circuit breaker closes, an operator sets an API key and restarts,
    an artifact appears. ``is_available()`` is consulted per request, not here.
    """
    global _analyzers
    if _analyzers is not None and not refresh:
        return _analyzers
    with _registry_lock:
        if _analyzers is not None and not refresh:
            return _analyzers
        _analyzers = [DeepSeekAnalyzer(), MLAnalyzer(), RuleBasedAnalyzer()]
        logger.info("analyzer chain: %s", " -> ".join(a.name for a in _analyzers))
        return _analyzers


def get_analyzer(source: str) -> AIAnalyzer | None:
    """Fetch one tier by its ``source`` value. Used by the evidence harness."""
    for analyzer in get_analyzers():
        if analyzer.source == source:
            return analyzer
    return None


# --------------------------------------------------------------------------- #
# Core analysis
# --------------------------------------------------------------------------- #

def analyze_text_sync(text: str, context: dict[str, Any] | None = None,
                      *, use_cache: bool = True) -> AnalysisResult:
    """Blocking implementation of the fallback chain. Never raises.

    Prefer the async :func:`analyze_text` from request handlers — this one blocks
    on the network for up to ~45 s and would stall the event loop.
    """
    context = context or {}
    text = (text or "").strip()
    if not text:
        # Degenerate input still gets a legal, honest answer rather than an error.
        return AnalysisResult(
            category="other", priority="medium",
            summary="Empty complaint text; nothing to analyse.",
            department_suggestion="General Administration",
            confidence=0.0, source="rules", model_name="keyword-rules-v1",
            reasoning="No text was supplied, so no analysis was possible.",
        )

    cache_key = _ResultCache.key(text, context)
    if use_cache:
        cached = _cache.get(cache_key)
        if cached is not None:
            cached.cached = True
            logger.debug("cache hit (%s)", cached.source)
            return cached

    attempted: list[str] = []
    errors: list[str] = []

    for analyzer in get_analyzers():
        result, error = analyzer.safe_analyze(text, context)
        if result is not None:
            result.fallback_from = list(attempted)
            result.cached = False
            if attempted:
                logger.info(
                    "analysis served by %s after %s failed", analyzer.source, ", ".join(attempted)
                )
            if use_cache:
                _cache.put(cache_key, result)
            return result
        attempted.append(analyzer.source)
        errors.append(error or f"{analyzer.name}: unknown failure")
        logger.warning("tier %s unavailable: %s", analyzer.source, error)

    # Structurally unreachable: RuleBasedAnalyzer.is_available() is hard-coded True
    # and its analyze() has no failure path. Handled anyway so the function's
    # "never raises" promise holds even if someone edits the rules tier badly.
    logger.error("every analyzer tier failed: %s", errors)
    return AnalysisResult(
        category="other", priority="medium",
        summary=(text[:150] + "...") if len(text) > 150 else text,
        department_suggestion="General Administration",
        confidence=0.0, source="rules", model_name="unavailable",
        reasoning="All analyzer tiers failed: " + " | ".join(errors[:3]),
        fallback_from=attempted,
    )


async def analyze_text(text: str, context: dict[str, Any] | None = None,
                       *, use_cache: bool = True) -> AnalysisResult:
    """Async wrapper — runs the blocking chain on a worker thread.

    The DeepSeek SDK call is synchronous and can occupy up to ~45 s. Running it
    inline in an async handler would block the event loop and, on a 0.1-CPU Render
    instance, take the whole API down under a handful of concurrent submissions.
    """
    import anyio

    return await anyio.to_thread.run_sync(
        lambda: analyze_text_sync(text, context, use_cache=use_cache)
    )


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #

def _to_schema(result: AnalysisResult, *, title: str | None = None) -> Any:
    """Convert our internal result into backend-core's ``AIAnalysisResult``.

    Two deliberate mappings:
      * ``prompt_cache_hit_tokens`` -> ``cache_hit_tokens`` (the column's name).
      * ``title`` is derived from the summary so the complaint gets a readable
        headline instead of a truncated first line of the citizen's text.
    """
    from app.schemas.ai import AIAnalysisResult

    return AIAnalysisResult(
        category=result.category,
        priority=result.priority,
        summary=result.summary,
        department_suggestion=result.department_suggestion,
        confidence=result.confidence,
        source=result.source,
        model_name=result.model_name,
        reasoning=result.reasoning,
        keywords=result.keywords,
        sentiment=result.sentiment,
        is_emergency=result.is_emergency,
        latency_ms=result.latency_ms,
        title=title,
        prompt_tokens=result.prompt_tokens or None,
        completion_tokens=result.completion_tokens or None,
        cache_hit_tokens=result.prompt_cache_hit_tokens or None,
    )


def _derive_title(summary: str, fallback: str, limit: int = 120) -> str | None:
    """A short headline from the AI summary; None if it would not be an upgrade."""
    text = (summary or "").strip()
    if not text:
        return None
    # Drop our own "Roads & Potholes issue (high priority): " template prefix.
    for sep in (": ", " - "):
        if sep in text[:60]:
            head, _, tail = text.partition(sep)
            if len(head) < 55 and tail.strip():
                text = tail.strip()
                break
    text = text.rstrip(".")
    if len(text) > limit:
        text = text[: limit - 3].rstrip() + "..."
    return text or (fallback[:limit] if fallback else None)


#: An AI category only overrides a citizen's explicit choice above this confidence.
#: Below it the citizen is assumed to know their own street better than the model.
# (The analyzer no longer overrides a human category at any confidence.)


async def analyze_and_store(complaint_id: str) -> AnalysisResult | None:
    """Analyse a stored complaint and persist everything that follows from it.

    Opens its own session: by the time FastAPI runs a BackgroundTask the
    request-scoped session is closed. Steps:

      1. load the complaint (including soft-deleted, so a reanalyze still works);
      2. run the fallback chain off the event loop;
      3. write the ``AIAnalysis`` row via ``ComplaintManager.attach_analysis``
         (``apply_to_complaint=False`` — the stored analysis records what the AI
         actually said, unmodified);
      4. fold the result into the complaint under *our* rules, which respect an
         explicit citizen category;
      5. route the department from the suggestion;
      6. run duplicate detection and link if it is a strong match;
      7. commit, leaving ``ai_status`` complete (or failed).

    Never raises: it is called from a background task where an exception would be
    invisible, and from ``/reanalyze`` where it must not 500.
    """
    from app.db.session import SessionLocal
    from app.models.complaint import AIStatus, Category, Priority
    from app.repositories.complaint_repo import ComplaintRepository
    from app.repositories.department_repo import DepartmentRepository
    from app.services.complaint_service import ComplaintManager
    from app.services.department_service import DepartmentService

    result: AnalysisResult | None = None
    try:
        async with SessionLocal() as session:
            repo = ComplaintRepository(session)
            complaint = await repo.get_by_id(complaint_id, include_deleted=True)
            if complaint is None:
                logger.warning("analyze_and_store: no complaint %s", complaint_id)
                return None

            # ``category_locked`` is set whenever a human picked the category — the
            # citizen on the submit form, or a staff member correcting it later.
            citizen_category: str | None = (
                str(complaint.category) if complaint.category_locked else None
            )

            context = {
                "location_text": complaint.location_text,
                "area": complaint.area,
                "latitude": complaint.latitude,
                "longitude": complaint.longitude,
                "category": citizen_category,
            }
            result = await analyze_text(complaint.description, context)

            departments = DepartmentService(DepartmentRepository(session), repo)
            manager = ComplaintManager(repo, departments=departments)

            title = _derive_title(result.summary, complaint.description)
            # apply_to_complaint=False: the AIAnalysis row must record exactly what
            # the analyzer said. The complaint is updated separately, below, under
            # rules that can decline to override the citizen.
            await manager.attach_analysis(
                complaint_id, _to_schema(result, title=title), apply_to_complaint=False
            )

            complaint = await repo.get_by_id(complaint_id, include_deleted=True)
            if complaint is None:  # deleted mid-flight
                return result

            # --- category: an explicit human choice always wins ------------------
            # However confident the analyzer is, it does not get to overrule a person
            # who deliberately picked a category — the submit form offers that override
            # precisely because the AI is fallible, and silently reversing it would make
            # the control a lie. The analyzer's own verdict is still on the AIAnalysis
            # row above, so staff can see the disagreement and act on it.
            final_category = result.category
            if citizen_category and result.category != citizen_category:
                logger.info(
                    "keeping human category %s over analyzer's %s (confidence %.2f)",
                    citizen_category, result.category, result.confidence,
                )
                final_category = citizen_category
            complaint.category = Category(final_category)
            complaint.priority = Priority(result.priority)
            if title:
                complaint.title = title[:200]

            # --- department routing -------------------------------------------
            department = await departments.resolve_by_name_or_slug(
                result.department_suggestion
            ) or await departments.route_for_category(final_category)
            if department is not None:
                complaint.department_id = department.id

            # --- duplicate detection ------------------------------------------
            if complaint.duplicate_of_id is None:
                from app.ai.duplicates import best_duplicate_id

                duplicate_id = await best_duplicate_id(complaint_id, session=session)
                if duplicate_id and duplicate_id != complaint.id:
                    complaint.duplicate_of_id = duplicate_id
                    logger.info("complaint %s flagged as duplicate of %s",
                                complaint_id, duplicate_id)

            complaint.ai_status = AIStatus.COMPLETE
            await session.commit()
            logger.info(
                "analysed %s source=%s category=%s priority=%s confidence=%.2f latency=%dms",
                complaint.reference_code, result.source, complaint.category,
                complaint.priority, result.confidence, result.latency_ms,
            )

            # --- auto-assignment ----------------------------------------------
            # Runs last, after the department is settled and the analysis is
            # already committed, and cannot fail this function: AI enrichment
            # succeeding must never depend on there being staff to hand the work to.
            await _auto_assign_quietly(complaint_id, session, departments)
            return result
    except Exception:  # noqa: BLE001 - background task: log and mark failed, never raise
        logger.exception("analyze_and_store failed for %s", complaint_id)
        await _mark_failed(complaint_id)
        return result


async def _auto_assign_quietly(complaint_id: str, session: Any, departments: Any) -> None:
    """Hand the complaint to a staff member, swallowing every failure.

    Called at the tail of :func:`analyze_and_store`, once the department routing has
    been committed — the assignment rule reads ``complaint.department_id``, so it
    has to run after routing, not alongside it.

    Nothing in here may propagate. Auto-assignment is a convenience layered on top
    of triage: if the staff table is empty, the rule throws, or the rows are locked,
    the complaint must still come out of the pipeline correctly categorised, routed
    and marked ``ai_status="complete"`` — just unassigned, for a human to pick up.
    """
    try:
        from app.repositories.complaint_repo import ComplaintRepository
        from app.repositories.user_repo import UserRepository
        from app.services.assignment_service import AssignmentService
        from app.services.complaint_service import ComplaintManager

        repo = ComplaintRepository(session)
        assignments = AssignmentService(UserRepository(session), repo)
        manager = ComplaintManager(repo, departments=departments, assignments=assignments)
        await manager.auto_assign(complaint_id, actor="system:auto-assign")
    except Exception:  # noqa: BLE001 - assignment must never break AI enrichment
        logger.exception("auto-assignment failed for %s; leaving it unassigned", complaint_id)
        try:
            await session.rollback()  # leave the caller's session usable
        except Exception:  # noqa: BLE001
            logger.exception("could not roll back after a failed auto-assignment")


async def _mark_failed(complaint_id: str) -> None:
    """Best-effort ``ai_status='failed'`` on a fresh session."""
    try:
        from app.db.session import SessionLocal
        from app.models.complaint import AIStatus
        from app.repositories.complaint_repo import ComplaintRepository

        async with SessionLocal() as session:
            await ComplaintRepository(session).set_ai_status(complaint_id, AIStatus.FAILED)
            await session.commit()
    except Exception:  # noqa: BLE001
        logger.exception("could not mark %s as failed", complaint_id)


# --------------------------------------------------------------------------- #
# Health / diagnostics
# --------------------------------------------------------------------------- #

def health_snapshot() -> dict[str, Any]:
    """Everything ``GET /ai/health`` needs, with no network calls."""
    from app.ai.circuit_breaker import llm_breaker
    from app.ai.ml_analyzer import model_status

    analyzers = get_analyzers()
    llm = next((a for a in analyzers if a.source == "llm"), None)
    ml_status = model_status()
    # Probe the ML tier so a first request does not pay the load cost and so the
    # health endpoint reports the truth rather than "not loaded yet".
    ml_loaded = any(a.source == "ml" and a.is_available() for a in analyzers)
    if ml_loaded:
        ml_status = model_status()

    llm_configured = bool(llm and getattr(llm, "configured", lambda: False)())
    breaker = llm_breaker.snapshot()

    return {
        "llm_available": bool(llm_configured and breaker["state"] != "open"),
        "llm_configured": llm_configured,
        "ml_model_loaded": ml_loaded,
        "rules_available": True,
        "model_name": (llm.name if llm_configured and llm else
                       (ml_status.get("model_name") or "keyword-rules-v1")),
        "last_error": (getattr(llm, "last_error", None) if llm else None) or breaker["last_error"],
        "active_tier": ("llm" if llm_configured and breaker["state"] != "open"
                        else "ml" if ml_loaded else "rules"),
        "chain": [
            {"source": a.source, "name": a.name, "available": _safe_available(a)}
            for a in analyzers
        ],
        "circuit_breaker": breaker,
        "ml": ml_status,
        "cache": _cache.stats(),
    }


def _safe_available(analyzer: AIAnalyzer) -> bool:
    """``is_available`` must not break the health endpoint if a tier misbehaves."""
    try:
        if analyzer.source == "llm":
            # Do not consume the circuit breaker's half-open probe slot just to
            # render a health page.
            return bool(getattr(analyzer, "configured", lambda: False)())
        return analyzer.is_available()
    except Exception:  # noqa: BLE001
        return False


def clear_cache() -> None:
    """Drop the result cache. Used by tests and after a prompt change."""
    _cache.clear()


def cache_stats() -> dict[str, Any]:
    return _cache.stats()

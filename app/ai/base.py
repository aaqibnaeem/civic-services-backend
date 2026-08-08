"""Abstract base for every complaint analyzer, plus the shared result schema.

The design question this file answers
-------------------------------------
A civic triage system cannot depend on a single hosted LLM. DeepSeek's published
uptime is ~99.79% with multi-hour failure modes, and the API key may simply be
absent on a grader's machine. So the system needs *several* ways to analyse a
complaint that are interchangeable at the call site.

That is textbook polymorphism, and it is real here rather than decorative: three
genuinely different technologies (a hosted LLM, a locally-trained scikit-learn
model, and a deterministic keyword engine) implement one contract, and
``pipeline.py`` walks them in order without knowing which is which.

    AIAnalyzer (ABC)
    ├── DeepSeekAnalyzer   — hosted LLM, best quality, can be unavailable
    ├── MLAnalyzer         — local TF-IDF + LinearSVC, always available offline
    └── RuleBasedAnalyzer  — keyword/regex, cannot fail, lowest quality

What each subclass inherits rather than reimplements: the ``AnalysisResult``
schema, confidence clamping, keyword hygiene, department lookup, latency timing,
and the ``safe_analyze`` template method that turns any exception into a typed
failure. Subclasses supply only ``name``, ``source``, ``is_available()`` and
``analyze()`` — the parts that genuinely differ.
"""

from __future__ import annotations

import re
import time
from abc import ABC, abstractmethod
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, Field, field_validator

# ---------------------------------------------------------------------------
# Wire enums. These strings are frozen by docs/CONTRACT.md §1 and are what the
# frontend switches on. Never localise or re-case them.
# ---------------------------------------------------------------------------

Category = Literal["road", "water", "waste", "electricity", "drainage", "safety", "other"]
Priority = Literal["low", "medium", "high", "critical"]
AISource = Literal["llm", "ml", "rules"]
Sentiment = Literal["calm", "concerned", "angry"]

CATEGORIES: tuple[str, ...] = (
    "road", "water", "waste", "electricity", "drainage", "safety", "other",
)
PRIORITIES: tuple[str, ...] = ("low", "medium", "high", "critical")
SENTIMENTS: tuple[str, ...] = ("calm", "concerned", "angry")

#: Ordinal view of priority, used for escalation logic and for "was the model
#: off by one level or by three?" error analysis.
PRIORITY_RANK: dict[str, int] = {"low": 0, "medium": 1, "high": 2, "critical": 3}

#: Canonical department names. The DB is seeded by backend-core and may use
#: slightly different wording, so ``pipeline`` resolves these by fuzzy slug match
#: rather than by exact string equality.
DEPARTMENT_BY_CATEGORY: dict[str, str] = {
    "road": "Roads & Infrastructure",
    "water": "Water Supply",
    "waste": "Sanitation & Solid Waste",
    "electricity": "Electricity & Streetlights",
    "drainage": "Drainage & Sewerage",
    "safety": "Public Safety",
    "other": "General Administration",
}

#: Human labels for UI / prose. Mirrors CONTRACT.md §1.
CATEGORY_LABELS: dict[str, str] = {
    "road": "Roads & Potholes",
    "water": "Water Supply & Leakage",
    "waste": "Waste & Sanitation",
    "electricity": "Electricity & Streetlights",
    "drainage": "Drainage & Sewerage",
    "safety": "Public Safety",
    "other": "Other",
}


def normalise_category(value: Any) -> str:
    """Coerce anything an analyzer produced into a legal wire category.

    LLMs return ``"Road"``, ``"roads"``, ``"street_light"``, ``"garbage"``. The
    contract allows exactly seven lowercase strings, so everything is mapped here
    in one place instead of being re-guessed by each analyzer.
    """
    if not isinstance(value, str):
        return "other"
    v = value.strip().lower().replace("-", "_").replace(" ", "_")
    if v in CATEGORIES:
        return v
    aliases = {
        "roads": "road", "pothole": "road", "potholes": "road", "street": "road",
        "footpath": "road", "traffic": "road", "infrastructure": "road",
        "water_supply": "water", "watersupply": "water", "leakage": "water",
        "water_leakage": "water", "pipeline": "water", "tanker": "water",
        "garbage": "waste", "trash": "waste", "sanitation": "waste",
        "solid_waste": "waste", "rubbish": "waste", "cleanliness": "waste",
        "electric": "electricity", "electrical": "electricity", "power": "electricity",
        "streetlight": "electricity", "street_light": "electricity",
        "street_lights": "electricity", "streetlights": "electricity",
        "lighting": "electricity", "energy": "electricity",
        "sewerage": "drainage", "sewer": "drainage", "sewage": "drainage",
        "gutter": "drainage", "drain": "drainage", "drains": "drainage",
        "flooding": "drainage", "nala": "drainage", "nallah": "drainage",
        "security": "safety", "crime": "safety", "public_safety": "safety",
        "hazard": "safety", "encroachment": "other", "misc": "other",
        "general": "other", "others": "other", "unknown": "other", "none": "other",
    }
    if v in aliases:
        return aliases[v]
    # Last resort: substring probe, longest alias first so "street_light" beats
    # "street".
    for alias in sorted(aliases, key=len, reverse=True):
        if alias in v:
            return aliases[alias]
    for cat in CATEGORIES:
        if cat in v:
            return cat
    return "other"


def normalise_priority(value: Any) -> str:
    """Coerce anything an analyzer produced into a legal wire priority."""
    if isinstance(value, int | float) and not isinstance(value, bool):
        idx = max(0, min(3, int(value) - 1 if value >= 1 else 0))
        return PRIORITIES[idx]
    if not isinstance(value, str):
        return "medium"
    v = value.strip().lower().replace("-", "_").replace(" ", "_")
    if v in PRIORITIES:
        return v
    aliases = {
        "urgent": "critical", "emergency": "critical", "severe": "critical",
        "very_high": "critical", "p0": "critical", "p1": "critical",
        "important": "high", "major": "high", "p2": "high",
        "normal": "medium", "moderate": "medium", "average": "medium", "p3": "medium",
        "minor": "low", "trivial": "low", "routine": "low", "p4": "low",
    }
    if v in aliases:
        return aliases[v]
    for p in PRIORITIES:
        if p in v:
            return p
    return "medium"


def normalise_sentiment(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    v = value.strip().lower()
    if v in SENTIMENTS:
        return v
    aliases = {
        "neutral": "calm", "positive": "calm", "polite": "calm", "ok": "calm",
        "worried": "concerned", "anxious": "concerned", "negative": "concerned",
        "frustrated": "angry", "furious": "angry", "upset": "angry",
        "irritated": "angry", "aggressive": "angry",
    }
    return aliases.get(v)


_WORD_RE = re.compile(r"[^\w\s\-']", re.UNICODE)
_WS_RE = re.compile(r"\s+")


def normalise_text(text: str) -> str:
    """Canonical form used for cache keys and duplicate hashing.

    Lowercase, punctuation stripped, whitespace collapsed. Two demo submissions of
    the same complaint with different capitalisation must hit the same cache entry.
    """
    return _WS_RE.sub(" ", _WORD_RE.sub(" ", (text or "").lower())).strip()


class AnalysisResult(BaseModel):
    """The single output type every analyzer returns.

    Mirrors the ``AIAnalysis`` object in docs/CONTRACT.md §2 exactly, plus token
    telemetry fields that are internal-only (the API layer may drop them).
    """

    category: Category = "other"
    priority: Priority = "medium"
    summary: str = Field(default="", max_length=400)
    department_suggestion: str = Field(default="General Administration", max_length=120)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    source: AISource = "rules"
    model_name: str = "unknown"
    reasoning: str | None = Field(default=None, max_length=800)
    keywords: list[str] = Field(default_factory=list)
    sentiment: Sentiment | None = None
    is_emergency: bool = False
    latency_ms: int = 0

    # --- token telemetry (DeepSeek only; zero for local tiers) ---------------
    prompt_tokens: int = 0
    completion_tokens: int = 0
    prompt_cache_hit_tokens: int = 0
    prompt_cache_miss_tokens: int = 0

    # --- provenance ----------------------------------------------------------
    #: Tiers that were attempted and failed before this result was produced.
    #: Surfaced in `/ai/health` and the testing evidence, never hidden.
    fallback_from: list[str] = Field(default_factory=list)
    cached: bool = False

    model_config = {"extra": "ignore"}

    @field_validator("category", mode="before")
    @classmethod
    def _cat(cls, v: Any) -> str:
        return normalise_category(v)

    @field_validator("priority", mode="before")
    @classmethod
    def _pri(cls, v: Any) -> str:
        return normalise_priority(v)

    @field_validator("sentiment", mode="before")
    @classmethod
    def _sent(cls, v: Any) -> str | None:
        return normalise_sentiment(v)

    @field_validator("confidence", mode="before")
    @classmethod
    def _conf(cls, v: Any) -> float:
        try:
            f = float(v)
        except (TypeError, ValueError):
            return 0.0
        if f > 1.0:  # LLMs love returning 91 instead of 0.91
            f = f / 100.0 if f <= 100.0 else 1.0
        return max(0.0, min(1.0, f))

    @field_validator("keywords", mode="before")
    @classmethod
    def _kw(cls, v: Any) -> list[str]:
        if v is None:
            return []
        if isinstance(v, str):
            v = [p for p in re.split(r"[,;|]", v) if p.strip()]
        if not isinstance(v, list | tuple | set):
            return []
        out: list[str] = []
        seen: set[str] = set()
        for item in v:
            token = str(item).strip().lower()[:40]
            if token and token not in seen:
                seen.add(token)
                out.append(token)
            if len(out) >= 12:
                break
        return out

    @field_validator("summary", "department_suggestion", mode="before")
    @classmethod
    def _str(cls, v: Any) -> str:
        return "" if v is None else str(v).strip()

    def public_dict(self) -> dict[str, Any]:
        """The exact ``AIAnalysis`` shape from the contract, telemetry excluded."""
        return {
            "category": self.category,
            "priority": self.priority,
            "summary": self.summary,
            "department_suggestion": self.department_suggestion,
            "confidence": round(self.confidence, 4),
            "source": self.source,
            "model_name": self.model_name,
            "reasoning": self.reasoning,
            "keywords": self.keywords,
            "sentiment": self.sentiment,
            "is_emergency": self.is_emergency,
            "latency_ms": self.latency_ms,
        }


class AnalyzerUnavailable(RuntimeError):
    """Raised by an analyzer that cannot service the request *right now*.

    Distinct from a bug: this is the signal ``pipeline.py`` uses to drop to the
    next tier. A missing API key, an open circuit breaker or an absent model
    artifact all raise this.
    """


class AIAnalyzer(ABC):
    """Abstract base class for a complaint analyzer.

    INPUT
        ``text`` — the citizen's raw complaint description (15..5000 chars,
        English / Roman-Urdu / code-switched), plus an optional ``context`` dict
        carrying ``location_text``, ``area``, ``latitude``, ``longitude`` and any
        ``category`` the citizen hinted at.

    PROCESSING
        Defined entirely by the subclass. That is the point of the abstraction.

    OUTPUT
        An :class:`AnalysisResult`: category, priority, summary, department
        suggestion, confidence, keywords, sentiment, emergency flag and latency,
        stamped with the ``source`` tier that produced it.

    LIMITATIONS (shared by every subclass — see each subclass for its own)
        * Text only. No analyzer here reads the uploaded image: DeepSeek has no
          vision endpoint on the public API as of 2026-08-08, so an image is
          stored and displayed but never influences the classification.
        * ``priority`` is an opinion, not a fact. There is no ground truth for
          "how urgent is this"; two municipal officers routinely disagree.
        * No analyzer verifies that the complaint is *true*. A fabricated report
          is classified as confidently as a real one.
        * Confidence is a model-internal number. It is well-behaved for the ML
          tier (calibrated), self-reported and therefore softer for the LLM tier,
          and a fixed heuristic band for the rules tier.
    """

    #: Human/model identifier stored in ``AIAnalysis.model_name``.
    name: ClassVar[str] = "abstract"
    #: Wire tier recorded in ``AIAnalysis.source``. Drives the UI badge.
    source: ClassVar[AISource] = "rules"

    @abstractmethod
    def analyze(self, text: str, context: dict[str, Any] | None = None) -> AnalysisResult:
        """Analyse one complaint. Raise :class:`AnalyzerUnavailable` to fall through."""
        raise NotImplementedError

    def is_available(self) -> bool:
        """Can this tier service a request right now? Cheap, no network I/O."""
        return True

    # -- shared helpers (inherited, not reimplemented per subclass) -----------

    def _finalise(self, result: AnalysisResult, started: float) -> AnalysisResult:
        """Stamp tier identity and latency. Every subclass ends by calling this."""
        result.source = self.source
        if not result.model_name or result.model_name == "unknown":
            result.model_name = self.name
        if not result.department_suggestion:
            result.department_suggestion = DEPARTMENT_BY_CATEGORY.get(
                result.category, "General Administration"
            )
        if not result.summary:
            result.summary = "Complaint received; awaiting operator review."
        if result.latency_ms <= 0:
            result.latency_ms = max(0, int((time.perf_counter() - started) * 1000))
        return result

    def safe_analyze(self, text: str, context: dict[str, Any] | None = None
                     ) -> tuple[AnalysisResult | None, str | None]:
        """Template method: run ``analyze`` and never raise.

        Returns ``(result, None)`` on success or ``(None, error_message)`` on
        failure, so the orchestrator can walk the chain with plain control flow
        instead of nested try/except at every tier.
        """
        started = time.perf_counter()
        if not self.is_available():
            return None, f"{self.name}: unavailable"
        try:
            result = self.analyze(text, context or {})
        except AnalyzerUnavailable as exc:
            return None, f"{self.name}: {exc}"
        except Exception as exc:  # noqa: BLE001 - a broken tier must not break the chain
            return None, f"{self.name}: {type(exc).__name__}: {exc}"
        if result is None:
            return None, f"{self.name}: returned no result"
        return self._finalise(result, started), None

    def __repr__(self) -> str:  # pragma: no cover
        return f"<{type(self).__name__} name={self.name!r} source={self.source!r}>"

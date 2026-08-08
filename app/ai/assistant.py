"""Civic assistant — natural-language questions over the complaint database.

THE DESIGN PROBLEM
------------------
Ask an LLM "how many water complaints are open in Korangi?" with a pile of rows in
context and it will confidently answer "47". Sometimes it is right. It cannot
count, it is pattern-matching, and on a dashboard a confidently wrong number is
worse than no feature at all.

So **the LLM is never allowed to produce a number.**

    question
       │
       ▼
    (1) PLANNER LLM  ──► json query plan ──► Pydantic whitelist validation
       │                                     (unknown category? rejected.
       │                                      dangerous filter? rejected.)
       ▼
    (2) OUR CODE     ──► real SQLAlchemy aggregation against the real tables
       │                 counts, group-bys, medians, example rows
       ▼
    (3) WRITER LLM   ──► prose, given ONLY the computed facts, forbidden from
                         doing arithmetic, required to cite reference_codes

Step 2 is ordinary SQL written by hand. Every number in the answer comes from it.
The LLM's job is reduced to two things it is actually good at: parsing intent, and
writing a sentence.

If DeepSeek is unavailable at step 1, a keyword planner takes over. If it is
unavailable at step 3, a template writes the prose. **The assistant therefore works
end-to-end with no API key at all** — it just sounds more robotic. The `source`
field in the response says which path was taken, always.

LIMITATIONS
    * The planner understands the question types in its whitelist and nothing else.
      Anything outside them returns ``intent="unsupported"`` rather than guessing.
    * It answers only from the complaints table. It has no civic bylaws, no SOPs
      and no external knowledge, and is instructed to say so rather than invent.
    * Area matching is a case-insensitive substring match on a free-text field, so
      "Gulshan" matches "Gulshan-e-Iqbal" and also anything else containing it.
    * Multi-turn history is passed to the planner but the executed query is
      stateless — follow-ups like "and in Korangi?" work only when the planner
      restates the full filter set.
    * The writer can still phrase a correct number misleadingly. It cannot invent
      one.
"""

from __future__ import annotations

import json
import logging
import re
import statistics
import time
from datetime import UTC, datetime, timedelta
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from app.ai.base import CATEGORIES, CATEGORY_LABELS, PRIORITIES, normalise_category

logger = logging.getLogger("app.ai.assistant")

STATUSES = ("open", "assigned", "in_progress", "resolved", "rejected")
OPEN_STATUSES = ("open", "assigned", "in_progress")

Intent = Literal["count", "breakdown", "list", "trend", "resolution_time", "compare", "unsupported"]
GroupBy = Literal["category", "priority", "status", "area", "department", "day", "none"]

MAX_ROWS = 20
MAX_SCAN = 2000


# --------------------------------------------------------------------------- #
# 1. The query plan — a whitelist, not a query language
# --------------------------------------------------------------------------- #

class QueryFilters(BaseModel):
    """Validated filter set. Anything not on the whitelist is silently dropped.

    This is the security boundary. The LLM emits *strings*; nothing it writes ever
    reaches SQL as SQL. Values are coerced into known enum members or discarded,
    and free text is only ever used as a bound parameter in a LIKE.
    """

    category: list[str] = Field(default_factory=list)
    priority: list[str] = Field(default_factory=list)
    status: list[str] = Field(default_factory=list)
    area: list[str] = Field(default_factory=list)
    days: int = 30
    search: str | None = None

    @field_validator("category", mode="before")
    @classmethod
    def _cats(cls, v: Any) -> list[str]:
        if not v:
            return []
        if isinstance(v, str):
            v = [v]
        out = []
        for item in v:
            c = normalise_category(item)
            # Do not let a nonsense value collapse to "other" and silently change
            # the question's meaning.
            if c in CATEGORIES and str(item).strip().lower().startswith(c[:3]):
                out.append(c)
            elif str(item).strip().lower() in CATEGORIES:
                out.append(str(item).strip().lower())
        return list(dict.fromkeys(out))

    @field_validator("priority", mode="before")
    @classmethod
    def _pris(cls, v: Any) -> list[str]:
        if not v:
            return []
        if isinstance(v, str):
            v = [v]
        return list(dict.fromkeys(
            [str(i).strip().lower() for i in v if str(i).strip().lower() in PRIORITIES]
        ))

    @field_validator("status", mode="before")
    @classmethod
    def _stats(cls, v: Any) -> list[str]:
        if not v:
            return []
        if isinstance(v, str):
            v = [v]
        return list(dict.fromkeys(
            [str(i).strip().lower() for i in v if str(i).strip().lower() in STATUSES]
        ))

    @field_validator("area", mode="before")
    @classmethod
    def _areas(cls, v: Any) -> list[str]:
        if not v:
            return []
        if isinstance(v, str):
            v = [v]
        return [re.sub(r"[^\w\s\-]", "", str(i)).strip()[:80] for i in v if str(i).strip()][:4]

    @field_validator("days", mode="before")
    @classmethod
    def _days(cls, v: Any) -> int:
        try:
            n = int(v)
        except (TypeError, ValueError):
            return 30
        return max(1, min(n, 3650))

    @field_validator("search", mode="before")
    @classmethod
    def _search(cls, v: Any) -> str | None:
        if not v or not str(v).strip():
            return None
        return re.sub(r"[%_\\]", " ", str(v)).strip()[:80]


class QueryPlan(BaseModel):
    """What the planner is allowed to ask for. Nothing else is executable."""

    intent: Intent = "count"
    filters: QueryFilters = Field(default_factory=QueryFilters)
    group_by: GroupBy = "none"
    limit: int = 5
    needs_examples: bool = True
    clarification: str | None = None

    @field_validator("intent", mode="before")
    @classmethod
    def _intent(cls, v: Any) -> str:
        s = str(v or "").strip().lower()
        return s if s in {
            "count", "breakdown", "list", "trend", "resolution_time", "compare", "unsupported"
        } else "count"

    @field_validator("group_by", mode="before")
    @classmethod
    def _group(cls, v: Any) -> str:
        s = str(v or "none").strip().lower()
        return s if s in {
            "category", "priority", "status", "area", "department", "day", "none"
        } else "none"

    @field_validator("limit", mode="before")
    @classmethod
    def _limit(cls, v: Any) -> int:
        try:
            return max(1, min(int(v), MAX_ROWS))
        except (TypeError, ValueError):
            return 5


# --------------------------------------------------------------------------- #
# 1b. Keyword planner — the no-API-key path
# --------------------------------------------------------------------------- #

_CATEGORY_HINTS = {
    "road": ("road", "pothole", "street", "sarak", "footpath"),
    "water": ("water", "pani", "tanker", "supply", "leak"),
    "waste": ("waste", "garbage", "trash", "kachra", "rubbish", "sanitation", "sweeper"),
    "electricity": ("electric", "bijli", "light", "power", "transformer", "load shedding"),
    "drainage": ("drain", "sewer", "gutter", "nali", "sewage", "nallah"),
    "safety": ("safety", "crime", "snatch", "dog", "unsafe", "theft"),
    "other": ("encroach", "park", "qabza", "office"),
}


#: Questions this database cannot answer. Without this check the keyword planner
#: happily answered "what is the weather tomorrow?" with a complaint count, which
#: is worse than refusing — it looks like the system misunderstood reality rather
#: than the question.
_OUT_OF_SCOPE = re.compile(
    r"\b(weather|forecast|temperature|humidity|rain\s+tomorrow|"
    r"news|headline|joke|recipe|poem|song|lyrics|movie|"
    r"stock|share\s+price|bitcoin|exchange\s+rate|"
    r"cricket|football|match\s+score|"
    r"who\s+is|who\s+was|capital\s+of|translate|define|"
    r"prime\s+minister|president|election|politic\w*|"
    r"medical\s+advice|legal\s+advice|diagnos\w+|prescri\w+|symptom\w*|"
    r"write\s+(?:me\s+)?(?:a|an|some)|tell\s+me\s+a\s+story)\b",
    re.IGNORECASE,
)

#: Words that mean the question really is about this database.
_IN_SCOPE = re.compile(
    r"\b(complaint\w*|report\w*|shikayat|resolved|resolution|pending|open|backlog|"
    r"priority|priorities|critical|urgent|category|categories|department\w*|area\w*|"
    r"status|ticket\w*|citizen\w*|filed|logged|"
    r"road|water|waste|garbage|kachra|electricity|bijli|drainage|sewer|gutter|nali|"
    r"safety|pothole|street\s*light|sanitation)\b",
    re.IGNORECASE,
)

#: Requests to change data. The assistant is strictly read-only.
_MUTATION = re.compile(
    r"\b(delete|remove|drop|update|change|set|assign|close|resolve|reject|create|add)\b"
    r"[^?]{0,40}\b(complaint\w*|status|priority|database|table|record\w*)\b",
    re.IGNORECASE,
)


def is_out_of_scope(message: str) -> bool:
    """True when the question cannot be answered from the complaints table."""
    text = message or ""
    if _MUTATION.search(text):
        return True
    return bool(_OUT_OF_SCOPE.search(text)) and not _IN_SCOPE.search(text)


def plan_with_keywords(message: str) -> QueryPlan:
    """Deterministic planner. Not clever, but it always answers and never lies."""
    text = (message or "").lower()

    if is_out_of_scope(message):
        return QueryPlan(intent="unsupported", filters=QueryFilters(),
                         group_by="none", limit=5, needs_examples=False)

    categories = [c for c, hints in _CATEGORY_HINTS.items() if any(h in text for h in hints)]
    priorities = [p for p in PRIORITIES if p in text]
    if "urgent" in text or "emergency" in text:
        priorities.append("critical")

    statuses: list[str] = []
    if re.search(r"\b(open|pending|unresolved|outstanding|not resolved|backlog)\b", text):
        statuses = list(OPEN_STATUSES)
    elif re.search(r"\b(resolved|closed|fixed|completed)\b", text):
        statuses = ["resolved"]

    days = 30
    if re.search(r"\b(today|24 hours)\b", text):
        days = 1
    elif re.search(r"\b(this week|last week|7 days)\b", text):
        days = 7
    elif re.search(r"\b(quarter|90 days|3 months)\b", text):
        days = 90
    elif re.search(r"\b(year|12 months|365)\b", text):
        days = 365
    elif re.search(r"\b(all time|ever|overall|total)\b", text):
        days = 3650
    m = re.search(r"\blast\s+(\d{1,4})\s+days?\b", text)
    if m:
        days = int(m.group(1))

    areas: list[str] = []
    m = re.search(r"\b(?:in|at|from|for)\s+([A-Z][\w\-]*(?:[\s-][A-Z][\w\-]*)*)", message or "")
    if m:
        candidate = m.group(1).strip()
        if candidate.lower() not in {"the", "this", "last", "open", "all"}:
            areas = [candidate[:80]]

    if re.search(r"\b(how long|resolution time|take to|average time|median time)\b", text):
        intent, group_by = "resolution_time", ("category" if len(categories) > 1 else "none")
        statuses = statuses or ["resolved"]
    elif re.search(r"\b(trend|over time|per day|daily|rising|increasing)\b", text):
        intent, group_by = "trend", "day"
    elif re.search(r"\b(which|most common|breakdown|distribution|by category|by area|top)\b", text):
        intent = "breakdown"
        group_by = ("area" if "area" in text or "neighbourhood" in text else
                    "priority" if "priorit" in text else
                    "status" if "status" in text else
                    "department" if "department" in text else "category")
    elif re.search(r"\b(show|list|examples?|which ones|give me)\b", text):
        intent, group_by = "list", "none"
    else:
        intent, group_by = "count", "none"

    return QueryPlan(
        intent=intent,
        filters=QueryFilters(
            category=categories, priority=priorities, status=statuses,
            area=areas, days=days, search=None,
        ),
        group_by=group_by,
        limit=10 if intent == "list" else 7,
        needs_examples=intent in {"count", "list"},
        clarification=None,
    )


def plan_with_llm(message: str, history: list[dict[str, str]] | None = None) -> QueryPlan | None:
    """Ask DeepSeek for a plan. Returns None if the LLM tier is unavailable."""
    from app.ai.llm_analyzer import DeepSeekAnalyzer
    from app.ai.prompts import PLANNER_SYSTEM_PROMPT

    analyzer = DeepSeekAnalyzer()
    if not analyzer.is_available():
        return None

    convo = ""
    for turn in (history or [])[-4:]:
        role = str(turn.get("role", "user"))[:20]
        content = str(turn.get("content", ""))[:500]
        if content:
            convo += f"{role}: {content}\n"
    user = (f"PREVIOUS TURNS:\n{convo}\n" if convo else "") + \
           f"QUESTION: {message.strip()[:800]}\n\nReturn the json query plan now."

    try:
        raw, _usage = analyzer._call_with_retry(
            [
                {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
                {"role": "user", "content": user},
            ],
            max_tokens=400,
        )
        from app.ai.llm_analyzer import _strip_fences

        plan = QueryPlan.model_validate(json.loads(_strip_fences(raw)))
        from app.ai.circuit_breaker import llm_breaker

        llm_breaker.record_success()
        return plan
    except Exception as exc:  # noqa: BLE001 - fall back to the keyword planner
        from app.ai.circuit_breaker import llm_breaker

        llm_breaker.record_failure(f"planner: {exc}")
        logger.warning("LLM planner failed, using keyword planner: %s", exc)
        return None


# --------------------------------------------------------------------------- #
# 2. Execution — real SQL. Every number in the final answer originates here.
# --------------------------------------------------------------------------- #

async def execute_plan(plan: QueryPlan, session: Any = None) -> dict[str, Any]:
    """Run the plan against the database and return a facts dict.

    This function contains all of the arithmetic in the assistant. The LLM sees its
    output and may only quote it.
    """
    from sqlalchemy import func, or_, select

    from app.db.session import SessionLocal
    from app.models.complaint import Complaint
    from app.models.department import Department

    owns = session is None
    cm = SessionLocal() if owns else None
    if owns:
        session = await cm.__aenter__()

    try:
        f = plan.filters
        cutoff = datetime.now(UTC) - timedelta(days=f.days)

        conditions = [Complaint.is_deleted.is_(False), Complaint.created_at >= cutoff]
        if f.category:
            conditions.append(Complaint.category.in_(f.category))
        if f.priority:
            conditions.append(Complaint.priority.in_(f.priority))
        if f.status:
            conditions.append(Complaint.status.in_(f.status))
        if f.area:
            area_clauses = []
            for a in f.area:
                pattern = f"%{a.lower()}%"
                area_clauses.append(func.lower(Complaint.area).like(pattern))
                area_clauses.append(func.lower(Complaint.location_text).like(pattern))
            conditions.append(or_(*area_clauses))
        if f.search:
            pattern = f"%{f.search.lower()}%"
            conditions.append(or_(
                func.lower(Complaint.description).like(pattern),
                func.lower(Complaint.title).like(pattern),
            ))

        facts: dict[str, Any] = {
            "intent": plan.intent,
            "window_days": f.days,
            "filters_applied": {
                "category": f.category, "priority": f.priority, "status": f.status,
                "area": f.area, "search": f.search,
            },
        }

        total = (await session.execute(
            select(func.count()).select_from(Complaint).where(*conditions)
        )).scalar_one()
        facts["total_matching"] = int(total)

        grand_total = (await session.execute(
            select(func.count()).select_from(Complaint).where(Complaint.is_deleted.is_(False))
        )).scalar_one()
        facts["total_complaints_in_database"] = int(grand_total)

        # --- group-bys -------------------------------------------------------
        if plan.group_by in {"category", "priority", "status", "area"} or plan.intent == "breakdown":
            column = {
                "category": Complaint.category, "priority": Complaint.priority,
                "status": Complaint.status, "area": Complaint.area,
            }.get(plan.group_by, Complaint.category)
            rows = (await session.execute(
                select(column, func.count().label("n"))
                .where(*conditions).group_by(column).order_by(func.count().desc())
                .limit(plan.limit)
            )).all()
            facts["breakdown"] = {
                "group_by": plan.group_by if plan.group_by != "none" else "category",
                "groups": [
                    {"key": str(k) if k is not None else "unknown",
                     "label": CATEGORY_LABELS.get(str(k), str(k) if k else "unknown"),
                     "count": int(n),
                     "percent": round(100 * int(n) / total, 1) if total else 0.0}
                    for k, n in rows
                ],
            }
            if facts["breakdown"]["groups"]:
                top = facts["breakdown"]["groups"][0]
                facts["mode"] = {"key": top["key"], "label": top["label"], "count": top["count"]}

        if plan.group_by == "department":
            rows = (await session.execute(
                select(Department.name, func.count(Complaint.id))
                .select_from(Complaint).join(Department, Complaint.department_id == Department.id)
                .where(*conditions).group_by(Department.name)
                .order_by(func.count(Complaint.id).desc()).limit(plan.limit)
            )).all()
            facts["breakdown"] = {
                "group_by": "department",
                "groups": [{"key": str(name), "label": str(name), "count": int(n),
                            "percent": round(100 * int(n) / total, 1) if total else 0.0}
                           for name, n in rows],
            }

        # --- trend -----------------------------------------------------------
        if plan.intent == "trend" or plan.group_by == "day":
            rows = (await session.execute(
                select(Complaint.created_at)
                .where(*conditions).order_by(Complaint.created_at).limit(MAX_SCAN)
            )).scalars().all()
            buckets: dict[str, int] = {}
            for created in rows:
                dt = created.replace(tzinfo=UTC) if created.tzinfo is None else created
                buckets[dt.date().isoformat()] = buckets.get(dt.date().isoformat(), 0) + 1
            series = [{"date": d, "count": c} for d, c in sorted(buckets.items())]
            facts["trend"] = {
                "points": series[-30:],
                "days_with_data": len(series),
                "busiest_day": max(series, key=lambda p: p["count"]) if series else None,
                "mean_per_active_day": (
                    round(sum(p["count"] for p in series) / len(series), 2) if series else 0.0
                ),
            }

        # --- resolution time -------------------------------------------------
        if plan.intent in {"resolution_time", "compare"}:
            rows = (await session.execute(
                select(Complaint.category, Complaint.created_at, Complaint.resolved_at)
                .where(*conditions).where(Complaint.resolved_at.is_not(None)).limit(MAX_SCAN)
            )).all()
            per_category: dict[str, list[float]] = {}
            everything: list[float] = []
            for category, created, resolved in rows:
                c = created.replace(tzinfo=UTC) if created.tzinfo is None else created
                r = resolved.replace(tzinfo=UTC) if resolved.tzinfo is None else resolved
                hours = (r - c).total_seconds() / 3600.0
                if hours < 0:
                    continue
                everything.append(hours)
                per_category.setdefault(str(category), []).append(hours)

            def _describe(values: list[float]) -> dict[str, Any]:
                if not values:
                    return {"n": 0}
                ordered = sorted(values)
                return {
                    "n": len(ordered),
                    "median_hours": round(statistics.median(ordered), 1),
                    "mean_hours": round(statistics.fmean(ordered), 1),
                    "min_hours": round(ordered[0], 1),
                    "max_hours": round(ordered[-1], 1),
                    "median_days": round(statistics.median(ordered) / 24, 2),
                }

            facts["resolution"] = {
                "overall": _describe(everything),
                "by_category": {c: _describe(v) for c, v in sorted(per_category.items())},
                "note": ("median is the honest headline because a few very slow cases "
                         "drag the mean up"),
            }

        # --- example rows ----------------------------------------------------
        if plan.needs_examples or plan.intent == "list":
            rows = (await session.execute(
                select(Complaint).where(*conditions)
                .order_by(Complaint.created_at.desc())
                .limit(min(plan.limit, MAX_ROWS))
            )).scalars().unique().all()
            facts["examples"] = [
                {
                    "reference_code": c.reference_code,
                    "id": c.id,
                    "title": (c.title or c.description or "")[:120],
                    "category": str(c.category),
                    "priority": str(c.priority),
                    "status": str(c.status),
                    "area": c.area,
                    "location_text": c.location_text,
                    "created_at": (c.created_at.isoformat() if c.created_at else None),
                }
                for c in rows
            ]

        if facts["total_matching"] < 10:
            facts["sample_warning"] = (
                f"Only {facts['total_matching']} complaints match these filters, "
                "which is too few to draw a reliable conclusion from."
            )
        return facts
    finally:
        if owns and cm is not None:
            await cm.__aexit__(None, None, None)


# --------------------------------------------------------------------------- #
# 3. Prose — from facts only
# --------------------------------------------------------------------------- #

def _fmt_filters(f: QueryFilters) -> str:
    bits = []
    if f.category:
        bits.append(" or ".join(CATEGORY_LABELS.get(c, c) for c in f.category))
    if f.priority:
        bits.append(f"{'/'.join(f.priority)} priority")
    if f.status:
        bits.append("status " + "/".join(s.replace("_", " ") for s in f.status))
    if f.area:
        bits.append("in " + " or ".join(f.area))
    if f.search:
        bits.append(f"mentioning '{f.search}'")
    scope = ", ".join(bits) if bits else "all complaints"
    return f"{scope} in the last {f.days} days"


def write_answer_template(message: str, plan: QueryPlan, facts: dict[str, Any]) -> str:
    """Deterministic prose. This is what runs when there is no API key."""
    if plan.intent == "unsupported":
        return ("I can only answer questions about the complaints in this system — "
                "counts, categories, priorities, areas, statuses and resolution times. "
                "That question is outside the data I hold.")
    if plan.clarification:
        return plan.clarification

    scope = _fmt_filters(plan.filters)
    total = facts.get("total_matching", 0)
    parts: list[str] = []

    if total == 0:
        return (f"No complaints match {scope}. The database holds "
                f"{facts.get('total_complaints_in_database', 0)} complaints in total, so "
                "either the filters are too narrow or nothing has been reported yet.")

    parts.append(f"There are {total} complaints matching {scope}.")

    breakdown = facts.get("breakdown")
    if breakdown and breakdown.get("groups"):
        groups = breakdown["groups"]
        top = groups[0]
        parts.append(
            f"The largest group is {top['label']} with {top['count']} "
            f"({top['percent']}% of the matches)."
        )
        if len(groups) > 1:
            rest = ", ".join(f"{g['label']} {g['count']}" for g in groups[1:4])
            parts.append(f"Then {rest}.")

    resolution = facts.get("resolution", {}).get("overall")
    if resolution and resolution.get("n"):
        parts.append(
            f"Across {resolution['n']} resolved complaints the median resolution time is "
            f"{resolution['median_hours']} hours ({resolution['median_days']} days); "
            f"the mean is {resolution['mean_hours']} hours, and the median is the more "
            "honest headline because a few very slow cases pull the mean up."
        )

    trend = facts.get("trend")
    if trend and trend.get("busiest_day"):
        parts.append(
            f"Activity spans {trend['days_with_data']} days with a peak of "
            f"{trend['busiest_day']['count']} on {trend['busiest_day']['date']}, "
            f"averaging {trend['mean_per_active_day']} per active day."
        )

    examples = facts.get("examples") or []
    if examples:
        codes = ", ".join(e["reference_code"] for e in examples[:3])
        parts.append(f"Recent examples: {codes}.")

    if facts.get("sample_warning"):
        parts.append(facts["sample_warning"])

    return " ".join(parts)


def write_answer_llm(message: str, plan: QueryPlan, facts: dict[str, Any]) -> str | None:
    """Ask DeepSeek to phrase the computed facts. Returns None if unavailable."""
    from app.ai.llm_analyzer import DeepSeekAnalyzer
    from app.ai.prompts import WRITER_SYSTEM_PROMPT

    analyzer = DeepSeekAnalyzer()
    if not analyzer.is_available():
        return None

    # The writer sees the facts and nothing else — no raw rows, no free text beyond
    # the question itself.
    payload = json.dumps(facts, indent=2, default=str)[:6000]
    user = (
        f"QUESTION: {message.strip()[:600]}\n\n"
        f"FACTS (computed by SQL — every number in your answer must come from here):\n"
        f"{payload}\n\nWrite the answer now."
    )
    try:
        from openai import OpenAI  # noqa: F401  (import guard only)

        client = analyzer._get_client()
        response = client.chat.completions.create(
            model=analyzer.model,
            messages=[
                {"role": "system", "content": WRITER_SYSTEM_PROMPT},
                {"role": "user", "content": user},
            ],
            max_tokens=400,
            temperature=0.2,  # a little warmth in prose; the numbers are fixed anyway
            extra_body={"thinking": {"type": "disabled"}},
        )
        content = response.choices[0].message.content
        if not content or not content.strip():
            raise ValueError("empty content")
        usage = response.usage
        logger.info(
            "assistant writer cache_hit=%s miss=%s out=%s",
            getattr(usage, "prompt_cache_hit_tokens", 0),
            getattr(usage, "prompt_cache_miss_tokens", 0),
            getattr(usage, "completion_tokens", 0),
        )
        from app.ai.circuit_breaker import llm_breaker

        llm_breaker.record_success()
        return content.strip()
    except Exception as exc:  # noqa: BLE001 - template takes over
        from app.ai.circuit_breaker import llm_breaker

        llm_breaker.record_failure(f"writer: {exc}")
        logger.warning("LLM writer failed, using template: %s", exc)
        return None


_CODE_RE = re.compile(r"\bCIV-[A-Z0-9]{4,10}\b")


def _verify_citations(answer: str, facts: dict[str, Any]) -> tuple[str, list[dict[str, str]]]:
    """Strip any reference code the LLM invented, and return the surviving ones.

    Belt and braces: the writer prompt forbids inventing codes, but a hallucinated
    tracking code is exactly the kind of error that destroys trust in a demo, so it
    is verified against the executed query rather than trusted.
    """
    known = {e["reference_code"]: e for e in (facts.get("examples") or [])}
    cited = _CODE_RE.findall(answer or "")
    invented = [c for c in cited if c not in known]
    if invented:
        logger.warning("assistant invented reference codes, removing: %s", invented)
        for code in invented:
            answer = answer.replace(code, "a complaint in this set")
    citations = [
        {"reference_code": code, "id": known[code]["id"]}
        for code in dict.fromkeys(cited) if code in known
    ]
    if not citations:
        citations = [
            {"reference_code": e["reference_code"], "id": e["id"]}
            for e in (facts.get("examples") or [])[:3]
        ]
    return answer, citations


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #

async def answer_question(
    message: str,
    history: list[dict[str, str]] | None = None,
    session: Any = None,
) -> dict[str, Any]:
    """Full RAG-style pipeline. Returns the contract's ``/assistant/chat`` body.

    ``source`` is one of:
        ``llm``       both planning and prose came from DeepSeek
        ``hybrid``    one of the two steps fell back
        ``rules``     no LLM involved; keyword planner + template writer
    """
    started = time.perf_counter()
    message = (message or "").strip()
    if not message:
        return {
            "answer": "Ask me something about the complaints — for example, "
                      "'how many drainage complaints are open in Korangi?'",
            "citations": [], "used_stats": {}, "source": "rules",
        }

    import anyio

    plan = await anyio.to_thread.run_sync(lambda: plan_with_llm(message, history))
    planner_source = "llm" if plan is not None else "rules"
    if plan is None:
        plan = plan_with_keywords(message)

    # Belt and braces: the planner prompt asks for intent="unsupported" on
    # out-of-scope questions, but the scope boundary is a safety property and must
    # not depend on the model complying.
    if plan.intent != "unsupported" and is_out_of_scope(message):
        logger.info("overriding plan to unsupported for out-of-scope question")
        plan = QueryPlan(intent="unsupported")

    if plan.intent == "unsupported":
        return {
            "answer": ("I can only answer questions about the complaints stored in this "
                       "system — how many there are, their categories, priorities, areas, "
                       "statuses and how long they take to resolve. I do not have data "
                       "outside that."),
            "citations": [],
            "used_stats": {"intent": "unsupported", "plan": plan.model_dump()},
            "source": planner_source,
        }

    try:
        facts = await execute_plan(plan, session=session)
    except Exception:  # noqa: BLE001 - a query bug must not 500 the chat endpoint
        logger.exception("assistant query execution failed")
        return {
            "answer": "I could not query the complaint database just now. Please try again.",
            "citations": [], "used_stats": {"error": "query_failed"}, "source": "rules",
        }

    answer = await anyio.to_thread.run_sync(lambda: write_answer_llm(message, plan, facts))
    writer_source = "llm" if answer else "rules"
    if not answer:
        answer = write_answer_template(message, plan, facts)

    answer, citations = _verify_citations(answer, facts)

    if planner_source == "llm" and writer_source == "llm":
        source = "llm"
    elif planner_source == "llm" or writer_source == "llm":
        source = "hybrid"
    else:
        source = "rules"

    used_stats = {k: v for k, v in facts.items() if k != "examples"}
    used_stats["plan"] = plan.model_dump()
    used_stats["planner_source"] = planner_source
    used_stats["writer_source"] = writer_source
    used_stats["latency_ms"] = int((time.perf_counter() - started) * 1000)

    return {
        "answer": answer,
        "citations": citations,
        "used_stats": used_stats,
        "source": source,
    }

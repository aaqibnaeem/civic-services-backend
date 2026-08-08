"""``AnalyticsService`` — one query, one DataFrame, every metric derived from it.

Architecture decision
---------------------
Every analytics endpoint pulls **one** filtered result set out of Postgres/SQLite and
then computes every statistic from that single pandas DataFrame in memory. The
alternative — a ``COUNT``/``AVG``/``percentile_cont`` round trip per metric — would
mean 15+ queries to render one dashboard, would make the numbers mutually
inconsistent (each query sees a slightly different snapshot), and would push
statistics we need to control (Tukey fences, bias-corrected skewness, chi-square
expected counts) into dialect-specific SQL that behaves differently on SQLite and
Postgres. Pulling once and aggregating in pandas gives us identical maths everywhere
and one consistent snapshot per response.

Memory discipline, because the Render free tier has 512 MB:

* The ``SELECT`` names **only the columns the statistics need** — never ``SELECT *``,
  and in particular never the ``description`` text blob, which is by far the largest
  column and is used by no metric here.
* The result set is hard-capped at ``MAX_ROWS`` rows (newest first). At the cap the
  response says so rather than silently describing a subset.
* The TTL cache holds at most ``CACHE_MAXSIZE`` frames.

Caching: a 60-second TTL cache keyed on the filter set. A dashboard fires eight
analytics requests on load and users re-open it constantly; without the cache that is
eight identical scans per page view. 60 seconds is short enough that the numbers stay
live for a demo and long enough to collapse a page load into one query.

Column resolution is defensive on purpose: this service is built in parallel with the
model layer, so it introspects ``Complaint.__table__`` and uses whichever of the
expected columns actually exist, degrading a metric to ``None`` rather than raising if
one is missing.
"""

from __future__ import annotations

import threading
import time
from datetime import UTC, datetime, timedelta
from typing import Any

from app.analytics.descriptive import DescriptiveStats, _round
from app.analytics.distributions import (
    OPEN_STATUSES,
    ContingencyTable,
    FrequencyDistribution,
    label_for,
)
from app.analytics.inference import (
    category_priority_chi_square,
    priority_vs_resolution_spearman,
)
from app.analytics.narratives import InsightContext, InsightEngine, summarise
from app.analytics.outliers import OutlierDetector
from app.analytics.timeseries import TimeSeriesAnalyzer

# Columns the statistics engine needs. Anything not listed here is never fetched.
WANTED_COLUMNS: tuple[str, ...] = (
    "id",
    "reference_code",
    "title",
    "category",
    "priority",
    "status",
    "area",
    "location_text",
    "department_id",
    "created_at",
    "resolved_at",
    "resolution_hours",
    "ai_confidence",
)
MAX_ROWS = 50_000
CACHE_TTL_SECONDS = 60.0
CACHE_MAXSIZE = 16
MAX_OUTLIERS_ON_WIRE = 25


# --------------------------------------------------------------------------- #
# tiny TTL cache
# --------------------------------------------------------------------------- #
class TTLCache:
    """Minimal time-to-live cache with a size cap and oldest-first eviction.

    Deliberately not ``functools.lru_cache``: we need *time*-based expiry (numbers
    must go stale), and we need to be able to clear it from a test.
    """

    def __init__(self, ttl: float = CACHE_TTL_SECONDS, maxsize: int = CACHE_MAXSIZE) -> None:
        self.ttl = ttl
        self.maxsize = maxsize
        self._store: dict[Any, tuple[float, Any]] = {}
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def get(self, key: Any) -> Any | None:
        now = time.monotonic()
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                self.misses += 1
                return None
            stored_at, value = entry
            if now - stored_at > self.ttl:
                self._store.pop(key, None)
                self.misses += 1
                return None
            self.hits += 1
            return value

    def set(self, key: Any, value: Any) -> None:
        now = time.monotonic()
        with self._lock:
            if len(self._store) >= self.maxsize:
                oldest = min(self._store.items(), key=lambda kv: kv[1][0])[0]
                self._store.pop(oldest, None)
            self._store[key] = (now, value)

    def clear(self) -> None:
        with self._lock:
            self._store.clear()

    def stats(self) -> dict[str, Any]:
        return {"entries": len(self._store), "hits": self.hits, "misses": self.misses, "ttl": self.ttl}


#: Process-wide frame cache. Shared across requests on purpose — that is the point.
FRAME_CACHE = TTLCache()


def _utcnow() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


# --------------------------------------------------------------------------- #
# service
# --------------------------------------------------------------------------- #
class AnalyticsService:
    """Loads complaints once per request and derives every analytics payload.

    ``session`` is an ``AsyncSession``. It may be ``None`` in tests, in which case a
    pre-built ``frame`` can be injected — that is how the statistics are tested
    without a database.
    """

    def __init__(self, session: Any = None, *, frame: Any = None, use_cache: bool = True) -> None:
        self.session = session
        self._frame = frame
        self.use_cache = use_cache
        self.truncated = False
        self.load_notes: list[str] = []

    # ------------------------------------------------------------------ load
    async def load_frame(self, filters) -> Any:
        """Return the (cached) complaint DataFrame for this filter set."""
        import pandas as pd

        if self._frame is not None:
            return self._frame

        key = filters.cache_key() if filters is not None else ()
        if self.use_cache:
            cached = FRAME_CACHE.get(key)
            if cached is not None:
                self._frame = cached
                return cached

        rows, columns = await self._query(filters)
        frame = pd.DataFrame(rows, columns=columns) if rows else pd.DataFrame(columns=list(columns))
        frame = self._prepare(frame)

        if self.use_cache:
            FRAME_CACHE.set(key, frame)
        self._frame = frame
        return frame

    async def _query(self, filters) -> tuple[list[dict[str, Any]], list[str]]:
        """One SELECT, only the needed columns, filters pushed down to SQL."""
        if self.session is None:
            return [], list(WANTED_COLUMNS)

        from sqlalchemy import select

        from app.models.complaint import Complaint

        table = Complaint.__table__
        selected = [table.c[name].label(name) for name in WANTED_COLUMNS if name in table.c]
        columns = [name for name in WANTED_COLUMNS if name in table.c]
        source: Any = table

        # --- department name via LEFT JOIN (never a second round trip) --------
        try:
            from app.models.department import Department

            dept = Department.__table__
            if "department_id" in table.c and "name" in dept.c and "id" in dept.c:
                source = table.outerjoin(dept, table.c["department_id"] == dept.c["id"])
                selected.append(dept.c["name"].label("department"))
                columns.append("department")
        except Exception:  # pragma: no cover - department model optional
            self.load_notes.append("Department names unavailable; grouped by id instead.")

        # --- AI confidence, wherever it lives ---------------------------------
        ai_table = None
        if "ai_confidence" not in table.c:
            ai_table = self._find_ai_table()
            if ai_table is not None:
                try:
                    source = source.outerjoin(
                        ai_table, ai_table.c["complaint_id"] == table.c["id"]
                    )
                    selected.append(ai_table.c["confidence"].label("ai_confidence"))
                    columns.append("ai_confidence")
                except Exception:  # pragma: no cover
                    ai_table = None

        stmt = select(*selected).select_from(source)
        stmt = self._apply_filters(stmt, table, filters)
        if "created_at" in table.c:
            stmt = stmt.order_by(table.c["created_at"].desc())
        stmt = stmt.limit(MAX_ROWS + 1)

        try:
            result = await self.session.execute(stmt)
            records = [dict(row) for row in result.mappings().all()]
        except Exception:
            # A join into a model that is still being built must not 500 the
            # dashboard: retry with complaint columns only.
            if ai_table is None and "department" not in columns:
                raise
            self.load_notes.append(
                "Joined tables were unavailable; analytics fell back to complaint columns only."
            )
            selected = [table.c[name].label(name) for name in WANTED_COLUMNS if name in table.c]
            columns = [name for name in WANTED_COLUMNS if name in table.c]
            stmt = select(*selected).select_from(table)
            stmt = self._apply_filters(stmt, table, filters)
            if "created_at" in table.c:
                stmt = stmt.order_by(table.c["created_at"].desc())
            stmt = stmt.limit(MAX_ROWS + 1)
            result = await self.session.execute(stmt)
            records = [dict(row) for row in result.mappings().all()]

        if len(records) > MAX_ROWS:
            self.truncated = True
            records = records[:MAX_ROWS]
            self.load_notes.append(
                f"Result capped at the {MAX_ROWS:,} most recent complaints to protect memory; "
                "narrow the date range for a complete picture."
            )
        return records, columns

    @staticmethod
    def _find_ai_table():
        """Locate a mapped AI-analysis table exposing complaint_id + confidence."""
        import importlib

        for module_name in (
            "app.models.ai_analysis",
            "app.models.ai",
            "app.models.analysis",
            "app.models.complaint",
        ):
            try:
                module = importlib.import_module(module_name)
            except Exception:
                continue
            for obj in vars(module).values():
                table = getattr(obj, "__table__", None)
                if table is None:
                    continue
                if "complaint_id" in table.c and "confidence" in table.c:
                    return table
        return None

    @staticmethod
    def _apply_filters(stmt, table, filters):
        if filters is None:
            return AnalyticsService._apply_soft_delete(stmt, table)
        if getattr(filters, "date_from", None) and "created_at" in table.c:
            start = datetime.combine(filters.date_from, datetime.min.time())
            stmt = stmt.where(table.c["created_at"] >= start)
        if getattr(filters, "date_to", None) and "created_at" in table.c:
            end = datetime.combine(filters.date_to, datetime.min.time()) + timedelta(days=1)
            stmt = stmt.where(table.c["created_at"] < end)
        if getattr(filters, "category", None) and "category" in table.c:
            stmt = stmt.where(table.c["category"] == filters.category)
        if getattr(filters, "area", None) and "area" in table.c:
            stmt = stmt.where(table.c["area"] == filters.area)
        return AnalyticsService._apply_soft_delete(stmt, table)

    @staticmethod
    def _apply_soft_delete(stmt, table):
        if "is_deleted" in table.c:
            stmt = stmt.where(table.c["is_deleted"].is_(False))
        elif "deleted_at" in table.c:
            stmt = stmt.where(table.c["deleted_at"].is_(None))
        return stmt

    # --------------------------------------------------------------- prepare
    def _prepare(self, frame):
        """Normalise dtypes and derive ``resolution_hours`` / ``age_days`` once."""
        import numpy as np
        import pandas as pd

        for column in ("created_at", "resolved_at"):
            if column in frame.columns:
                series = pd.to_datetime(frame[column], utc=True, errors="coerce")
                try:
                    frame[column] = series.dt.tz_localize(None)
                except (TypeError, AttributeError):
                    frame[column] = series
            else:
                frame[column] = pd.NaT

        for column in ("category", "priority", "status", "area", "department", "reference_code"):
            if column not in frame.columns:
                frame[column] = None
        frame["category"] = frame["category"].astype("object").fillna("other")
        frame["priority"] = frame["priority"].astype("object").fillna("medium")
        frame["status"] = frame["status"].astype("object").fillna("open")
        frame["area"] = frame["area"].astype("object").fillna("Unspecified").replace("", "Unspecified")
        frame["department"] = (
            frame["department"].astype("object").fillna("Unassigned").replace("", "Unassigned")
        )

        # Resolution time: prefer the stored column, otherwise derive it. Either way
        # negative durations are data errors, not fast fixes, so they become NaN.
        derived = (frame["resolved_at"] - frame["created_at"]).dt.total_seconds() / 3600.0
        if "resolution_hours" in frame.columns:
            stored = pd.to_numeric(frame["resolution_hours"], errors="coerce")
            frame["resolution_hours"] = stored.fillna(derived)
        else:
            frame["resolution_hours"] = derived
        frame.loc[frame["resolution_hours"] < 0, "resolution_hours"] = np.nan
        # Only genuinely resolved complaints contribute a resolution time.
        frame.loc[frame["resolved_at"].isna(), "resolution_hours"] = np.nan

        now = _utcnow()
        frame["age_days"] = (now - frame["created_at"]).dt.total_seconds() / 86400.0
        frame["is_open"] = frame["status"].isin(OPEN_STATUSES)

        if "ai_confidence" in frame.columns:
            frame["ai_confidence"] = pd.to_numeric(frame["ai_confidence"], errors="coerce")
            # A LEFT JOIN onto a one-to-many AI table can duplicate complaints; keep
            # the highest-confidence row per complaint so counts stay correct.
            if "id" in frame.columns and frame["id"].duplicated().any():
                frame = (
                    frame.sort_values("ai_confidence", ascending=False)
                    .drop_duplicates(subset="id", keep="first")
                    .sort_values("created_at", ascending=False)
                )
        else:
            frame["ai_confidence"] = np.nan

        return frame.reset_index(drop=True)

    # ---------------------------------------------------------------- pieces
    @staticmethod
    def resolved_hours(frame):
        return frame.loc[frame["resolution_hours"].notna(), "resolution_hours"]

    def descriptive(self, frame) -> DescriptiveStats:
        return DescriptiveStats(self.resolved_hours(frame), unit="hours")

    def kpis(self, frame, trends: dict[str, Any] | None = None) -> dict[str, Any]:
        total = int(len(frame))
        counts = frame["status"].value_counts().to_dict() if total else {}
        resolved = int(counts.get("resolved", 0))
        stats = self.descriptive(frame)
        critical_open = (
            int(((frame["priority"] == "critical") & frame["is_open"]).sum()) if total else 0
        )
        confidence = frame["ai_confidence"].dropna() if total else None
        avg_conf = float(confidence.mean()) if confidence is not None and len(confidence) else None
        backlog = int(frame["is_open"].sum()) if total else 0
        open_ages = frame.loc[frame["is_open"], "age_days"] if total else None
        oldest = (
            float(open_ages.max()) if open_ages is not None and len(open_ages.dropna()) else None
        )

        wow = (trends or {}).get("week_over_week") or {}
        return {
            "total": total,
            "open": int(counts.get("open", 0)),
            "assigned": int(counts.get("assigned", 0)),
            "in_progress": int(counts.get("in_progress", 0)),
            "resolved": resolved,
            "rejected": int(counts.get("rejected", 0)),
            "resolution_rate": round(resolved / total * 100, 1) if total else 0.0,
            "median_resolution_hours": _round(stats.median, 1),
            "median_resolution_days": _round((stats.median or 0) / 24, 2) if stats.median else None,
            "mean_resolution_hours": _round(stats.mean, 1),
            "critical_open": critical_open,
            "avg_ai_confidence": round(avg_conf, 3) if avg_conf is not None else None,
            "complaints_this_week": int(wow.get("current_week") or 0),
            "complaints_last_week": int(wow.get("previous_week") or 0),
            "wow_change_pct": wow.get("change_pct"),
            "wow_direction": wow.get("direction", "flat"),
            "backlog": backlog,
            "oldest_open_days": round(oldest, 1) if oldest is not None else None,
        }

    def department_stats(self, frame) -> list[dict[str, Any]]:
        total = int(len(frame))
        if total == 0:
            return []
        out: list[dict[str, Any]] = []
        for name, chunk in frame.groupby("department", dropna=False):
            resolved_hours = chunk.loc[chunk["resolution_hours"].notna(), "resolution_hours"]
            stats = DescriptiveStats(resolved_hours, unit="hours")
            status_counts = chunk["status"].value_counts().to_dict()
            resolved = int(status_counts.get("resolved", 0))
            n = int(len(chunk))
            out.append(
                {
                    "department": str(name),
                    "n": n,
                    "open": int(status_counts.get("open", 0)),
                    "in_progress": int(status_counts.get("in_progress", 0)),
                    "resolved": resolved,
                    "backlog": int(chunk["is_open"].sum()),
                    "resolution_rate": round(resolved / n * 100, 1) if n else None,
                    "median_resolution_hours": _round(stats.median, 1),
                    "median_resolution_days": _round((stats.median or 0) / 24, 2)
                    if stats.median
                    else None,
                    "mean_resolution_hours": _round(stats.mean, 1),
                    "p90_resolution_hours": _round(stats.p90, 1),
                    "resolved_sample": stats.n,
                    "share_pct": round(n / total * 100, 1),
                    "sample_warning": stats.sample_warning if stats.n else None,
                }
            )
        out.sort(key=lambda d: -d["n"])
        return out

    def area_stats(self, frame, *, limit: int = 25) -> list[dict[str, Any]]:
        total = int(len(frame))
        if total == 0:
            return []
        groups = list(frame.groupby("area", dropna=False))
        even_share = total / max(1, len(groups))
        out: list[dict[str, Any]] = []
        for name, chunk in groups:
            n = int(len(chunk))
            cats = chunk["category"].value_counts()
            top_cat = str(cats.index[0]) if len(cats) else None
            top_count = int(cats.iloc[0]) if len(cats) else 0
            resolved_hours = chunk.loc[chunk["resolution_hours"].notna(), "resolution_hours"]
            median = float(resolved_hours.median()) if len(resolved_hours) else None
            status_counts = chunk["status"].value_counts().to_dict()
            critical = int((chunk["priority"] == "critical").sum())
            # Hotspot rule, stated on the wire: at least 5 complaints AND at least
            # 1.5x the volume an even split across areas would give.
            hotspot = n >= 5 and n >= even_share * 1.5
            reason = None
            if hotspot:
                reason = (
                    f"{n} complaints is {n / even_share:.1f}× the {even_share:.1f} an even split "
                    f"across {len(groups)} areas would produce."
                )
            out.append(
                {
                    "area": str(name),
                    "n": n,
                    "share_pct": round(n / total * 100, 1),
                    "open": int(chunk["is_open"].sum()),
                    "resolved": int(status_counts.get("resolved", 0)),
                    "critical_count": critical,
                    "top_category": top_cat,
                    "top_category_label": label_for("category", top_cat) if top_cat else None,
                    "top_category_count": top_count,
                    "top_category_share": round(top_count / n, 3) if n else 0.0,
                    "median_resolution_hours": _round(median, 1),
                    "hotspot": hotspot,
                    "hotspot_reason": reason,
                }
            )
        out.sort(key=lambda a: -a["n"])
        return out[:limit]

    def timeseries(self, frame, *, days: int = 90, end=None) -> TimeSeriesAnalyzer:
        return TimeSeriesAnalyzer(frame, days=days, end=end)

    # ------------------------------------------------------------- narrative
    def build_context(
        self,
        frame,
        *,
        filters=None,
        trends: dict[str, Any] | None = None,
        analyzer: TimeSeriesAnalyzer | None = None,
        outliers: dict[str, Any] | None = None,
        include_inference: bool = True,
    ) -> InsightContext:
        stats = self.descriptive(frame)
        return InsightContext(
            n_total=int(len(frame)),
            frame=frame,
            resolution=stats.to_dict(),
            outliers=outliers if outliers is not None else self.outlier_report(frame),
            category_dist=FrequencyDistribution(
                frame["category"].tolist(), variable="category"
            ).to_dict(),
            priority_dist=FrequencyDistribution(
                frame["priority"].tolist(), variable="priority"
            ).to_dict(),
            status_dist=FrequencyDistribution(frame["status"].tolist(), variable="status").to_dict(),
            area_dist=FrequencyDistribution(frame["area"].tolist(), variable="area").to_dict(),
            trends=trends,
            weekend_gap=analyzer.weekend_gap() if analyzer else None,
            chi_square=category_priority_chi_square(frame) if include_inference else None,
            spearman=priority_vs_resolution_spearman(frame) if include_inference else None,
            departments=self.department_stats(frame),
            areas=self.area_stats(frame),
            kpis=self.kpis(frame, trends),
            filters_description=filters.describe() if filters is not None else "all complaints",
        )

    def outlier_report(self, frame) -> dict[str, Any]:
        return OutlierDetector(frame).report()

    # ------------------------------------------------------------- endpoints
    async def overview(self, filters) -> dict[str, Any]:
        frame = await self.load_frame(filters)
        analyzer = self.timeseries(frame, days=90)
        trends = analyzer.to_dict(include_forecast=True, category_limit=7)
        context = self.build_context(frame, filters=filters, trends=trends, analyzer=analyzer)
        insights = InsightEngine(context).generate(limit=8)
        kpis = context.kpis or {}
        return {
            "generated_at": _utcnow(),
            "kpis": kpis,
            "cards": self._cards(kpis),
            "insights": insights,
            "interpretation": self._overview_interpretation(kpis, insights),
            "filters": filters,
        }

    def _overview_interpretation(self, kpis: dict[str, Any], insights: list[dict[str, Any]]) -> str:
        total = kpis.get("total", 0)
        if not total:
            return "No complaints match the current filters, so there are no KPIs to report."
        head = (
            f"{total} complaints are in view: {kpis.get('resolved', 0)} resolved "
            f"({kpis.get('resolution_rate', 0):.0f}%), {kpis.get('backlog', 0)} still open, and "
            f"{kpis.get('critical_open', 0)} of those are critical. "
        )
        median_days = kpis.get("median_resolution_days")
        if median_days:
            head += f"Half of resolved complaints closed within {median_days:.1f} days. "
        note = " ".join(item["title"] for item in insights[:2])
        return (head + note).strip()

    @staticmethod
    def _cards(kpis: dict[str, Any]) -> list[dict[str, Any]]:
        """KPI cards, each carrying its own one-line meaning."""
        rate = kpis.get("resolution_rate") or 0.0
        critical = kpis.get("critical_open") or 0
        wow = kpis.get("wow_change_pct")
        median_days = kpis.get("median_resolution_days")
        conf = kpis.get("avg_ai_confidence")
        cards = [
            {
                "key": "total",
                "label": "Total complaints",
                "value": float(kpis.get("total") or 0),
                "display": f"{kpis.get('total') or 0:,}",
                "unit": None,
                "hint": "Every complaint matching the current filters.",
                "severity": "info",
            },
            {
                "key": "open",
                "label": "Still open",
                "value": float(kpis.get("backlog") or 0),
                "display": f"{kpis.get('backlog') or 0:,}",
                "unit": None,
                "hint": "Open, assigned or in progress — the live workload.",
                "severity": "warn" if (kpis.get("backlog") or 0) > (kpis.get("resolved") or 0) else "info",
            },
            {
                "key": "resolved",
                "label": "Resolved",
                "value": float(kpis.get("resolved") or 0),
                "display": f"{kpis.get('resolved') or 0:,}",
                "unit": None,
                "hint": "Complaints that reached resolved status.",
                "severity": "info",
            },
            {
                "key": "resolution_rate",
                "label": "Resolution rate",
                "value": rate,
                "display": f"{rate:.0f}%",
                "unit": "%",
                "hint": "Share of complaints closed. A snapshot ratio — recent cases have not had time to close.",
                "severity": "info" if rate >= 50 else "warn",
            },
            {
                "key": "median_resolution",
                "label": "Median resolution time",
                "value": median_days,
                "display": f"{median_days:.1f} days" if median_days else "—",
                "unit": "days",
                "hint": "The median, not the mean — resolution times are right-skewed so the mean overstates the typical wait.",
                "severity": "info",
            },
            {
                "key": "critical_open",
                "label": "Critical & unresolved",
                "value": float(critical),
                "display": f"{critical:,}",
                "unit": None,
                "hint": "Highest-priority complaints still awaiting resolution.",
                "severity": "critical" if critical >= 5 else ("warn" if critical else "info"),
            },
            {
                "key": "ai_confidence",
                "label": "Avg AI confidence",
                "value": round(conf * 100, 1) if conf is not None else None,
                "display": f"{conf * 100:.0f}%" if conf is not None else "—",
                "unit": "%",
                "hint": "Mean classifier confidence behind the category and priority labels.",
                "severity": "warn" if (conf is not None and conf < 0.6) else "info",
            },
            {
                "key": "this_week",
                "label": "Filed this week",
                "value": float(kpis.get("complaints_this_week") or 0),
                "display": f"{kpis.get('complaints_this_week') or 0:,}"
                + (f" ({wow:+.0f}%)" if wow is not None else ""),
                "unit": None,
                "hint": (
                    f"Versus {kpis.get('complaints_last_week') or 0} the week before."
                    if wow is not None
                    else "No comparable week available."
                ),
                "severity": "warn" if (wow is not None and wow >= 25) else "info",
            },
        ]
        return cards

    async def categories(self, filters) -> dict[str, Any]:
        frame = await self.load_frame(filters)
        dist = FrequencyDistribution(frame["category"].tolist(), variable="category")
        by_status = ContingencyTable(
            frame["category"].tolist(),
            frame["status"].tolist(),
            row_variable="category",
            col_variable="status",
        )
        rows: list[dict[str, Any]] = []
        if len(frame):
            for name, chunk in frame.groupby("category", dropna=False):
                resolved_hours = chunk.loc[chunk["resolution_hours"].notna(), "resolution_hours"]
                median = float(resolved_hours.median()) if len(resolved_hours) else None
                rows.append(
                    {
                        "category": str(name),
                        "label": label_for("category", str(name)),
                        "n": int(len(chunk)),
                        "median_resolution_hours": _round(median, 1),
                        "median_resolution_days": _round((median or 0) / 24, 2) if median else None,
                        "open": int(chunk["is_open"].sum()),
                        "resolved": int((chunk["status"] == "resolved").sum()),
                    }
                )
            rows.sort(key=lambda r: -r["n"])

        context = self.build_context(frame, filters=filters, include_inference=False)
        insights = [
            item
            for item in InsightEngine(context).generate()
            if item["id"].startswith(("modal_category", "category_", "no_data", "low_sample"))
        ][:5]
        slowest = max(
            (r for r in rows if r["median_resolution_hours"] is not None),
            key=lambda r: r["median_resolution_hours"],
            default=None,
        )
        interpretation = dist.interpretation()
        if slowest:
            interpretation += (
                f" {slowest['label']} is the slowest category to resolve, at a median "
                f"{slowest['median_resolution_days']} days across {slowest['n']} complaints."
            )
        return {
            "distribution": dist.to_dict(),
            "resolution_by_category": rows,
            "by_status": by_status.to_dict(),
            "interpretation": interpretation,
            "insights": insights,
            "filters": filters,
        }

    async def priorities(self, filters) -> dict[str, Any]:
        frame = await self.load_frame(filters)
        dist = FrequencyDistribution(frame["priority"].tolist(), variable="priority")
        crosstab = ContingencyTable(
            frame["category"].tolist(),
            frame["priority"].tolist(),
            row_variable="category",
            col_variable="priority",
        )
        chi = category_priority_chi_square(frame)
        spearman = priority_vs_resolution_spearman(frame)

        rows = {r["value"]: r for r in dist.rows()}
        n = dist.n
        escalated = sum(rows.get(k, {}).get("count", 0) for k in ("high", "critical"))
        escalation_share = round(escalated / n * 100, 1) if n else 0.0

        context = self.build_context(frame, filters=filters)
        insights = [
            item
            for item in InsightEngine(context).generate()
            if item["id"].startswith(("priority_", "chi_square", "critical_", "no_data", "low_sample"))
        ][:5]
        return {
            "distribution": dist.to_dict(),
            "crosstab": crosstab.to_dict(),
            "chi_square": chi,
            "spearman_priority_vs_speed": spearman,
            "escalation_share_pct": escalation_share,
            "interpretation": " ".join(
                part
                for part in (dist.interpretation(), crosstab.interpretation(), chi.get("interpretation"))
                if part
            ),
            "insights": insights,
            "filters": filters,
        }

    async def resolution_times(self, filters) -> dict[str, Any]:
        """The statistics benchmark endpoint — full CONTRACT shape."""
        frame = await self.load_frame(filters)
        stats = self.descriptive(frame)
        base = stats.to_dict()
        outliers = self.outlier_report(frame)
        histogram = stats.histogram()

        by_category: list[dict[str, Any]] = []
        if len(frame):
            for name, chunk in frame.groupby("category", dropna=False):
                sub = DescriptiveStats(
                    chunk.loc[chunk["resolution_hours"].notna(), "resolution_hours"], unit="hours"
                )
                if sub.n == 0:
                    continue
                upper = sub.upper_fence
                out_count = (
                    int((chunk["resolution_hours"] > upper).sum()) if upper is not None else 0
                )
                by_category.append(
                    {
                        "category": str(name),
                        "n": sub.n,
                        "median": _round(sub.median, 1),
                        "q1": _round(sub.q1, 1),
                        "q3": _round(sub.q3, 1),
                        "iqr": _round(sub.iqr, 1),
                        "mean": _round(sub.mean, 1),
                        "upper_fence": _round(upper, 1),
                        "outlier_count": out_count,
                        "sample_warning": sub.sample_warning,
                    }
                )
            by_category.sort(key=lambda c: -(c["median"] or 0))

        resolved_count = stats.n
        total = int(len(frame))
        unresolved = total - resolved_count

        context = self.build_context(frame, filters=filters, outliers=outliers)
        insights = [
            item
            for item in InsightEngine(context).generate()
            if item["id"].startswith(
                ("resolution_", "category_outliers", "department_laggard", "no_data", "low_sample")
            )
        ][:6]

        payload = dict(base)
        payload.update(
            {
                "outliers": (outliers.get("outliers") or [])[:MAX_OUTLIERS_ON_WIRE],
                "outlier_report": outliers,
                "histogram": histogram,
                "histogram_method": "Freedman–Diaconis bin width",
                "by_category": by_category,
                "resolved_count": resolved_count,
                "unresolved_count": unresolved,
                "censoring_note": self._censoring_note(resolved_count, unresolved),
                "interpretation": self._resolution_interpretation(
                    stats, outliers, by_category, unresolved
                ),
                "insights": insights,
                "filters": filters,
            }
        )
        payload.pop("notes", None)
        payload.pop("kurtosis_type", None)
        return payload

    @staticmethod
    def _censoring_note(resolved: int, unresolved: int) -> str | None:
        if unresolved <= 0:
            return None
        total = resolved + unresolved
        return (
            f"These statistics describe only the {resolved} complaints that have actually been "
            f"resolved. The {unresolved} still open ({unresolved / total * 100:.0f}% of {total}) are "
            "right-censored: their final resolution time is unknown but is at least their current "
            "age. Because slow cases are the ones most likely to still be open, every figure here "
            "is an optimistic estimate of true resolution time — a survivorship bias we are naming "
            "rather than hiding."
        )

    @staticmethod
    def _resolution_interpretation(
        stats: DescriptiveStats,
        outliers: dict[str, Any],
        by_category: list[dict[str, Any]],
        unresolved: int,
    ) -> str:
        if stats.n == 0:
            return (
                "No complaint in this filter set has been resolved yet, so there is no resolution "
                "time to describe."
            )
        median_days = (stats.median or 0) / 24
        mean_days = (stats.mean or 0) / 24
        parts = [
            f"Across {stats.n} resolved complaints the median resolution time is "
            f"{median_days:.1f} days and the mean is {mean_days:.1f} days.",
        ]
        if stats.median and stats.mean and stats.mean > stats.median * 1.2:
            parts.append(
                f"Because the mean sits {((stats.mean / stats.median) - 1) * 100:.0f}% above the "
                "median, the distribution is right-skewed and the median is the honest headline "
                "figure — the mean is being pulled up by a slow minority."
            )
        if stats.q1 is not None and stats.q3 is not None:
            parts.append(
                f"The middle 50% of complaints close between {stats.q1 / 24:.1f} and "
                f"{stats.q3 / 24:.1f} days (IQR {(stats.iqr or 0) / 24:.1f} days)."
            )
        parts.append(outliers.get("interpretation", ""))
        if len(by_category) >= 2:
            slowest, fastest = by_category[0], by_category[-1]
            if slowest["median"] and fastest["median"]:
                parts.append(
                    f"{label_for('category', slowest['category'])} is the slowest category at a "
                    f"median {slowest['median'] / 24:.1f} days, against "
                    f"{label_for('category', fastest['category'])} at "
                    f"{fastest['median'] / 24:.1f} days."
                )
        if stats.sample_warning:
            parts.append(stats.sample_warning)
        if unresolved:
            parts.append(
                f"{unresolved} complaints are still open and contribute nothing to these figures."
            )
        return " ".join(p for p in parts if p)

    async def trends(self, filters, *, days: int = 90) -> dict[str, Any]:
        frame = await self.load_frame(filters)
        analyzer = self.timeseries(frame, days=days)
        payload = analyzer.to_dict(include_forecast=True, category_limit=7)
        context = self.build_context(
            frame, filters=filters, trends=payload, analyzer=analyzer, include_inference=False
        )
        payload["insights"] = [
            item
            for item in InsightEngine(context).generate()
            if item["id"].startswith(
                ("wow_", "category_wow", "forecast_", "weekend_", "volume_spike", "no_data")
            )
        ][:6]
        payload["filters"] = filters
        return payload

    async def departments(self, filters) -> dict[str, Any]:
        frame = await self.load_frame(filters)
        rows = self.department_stats(frame)
        stats = self.descriptive(frame)
        comparable = [r for r in rows if r["median_resolution_hours"] is not None and r["resolved_sample"] >= 5]
        slowest = max(comparable, key=lambda r: r["median_resolution_hours"], default=None)
        fastest = min(comparable, key=lambda r: r["median_resolution_hours"], default=None)
        largest_backlog = max(rows, key=lambda r: r["backlog"], default=None)

        context = self.build_context(frame, filters=filters, include_inference=False)
        insights = [
            item
            for item in InsightEngine(context).generate()
            if item["id"].startswith(("department_", "no_data", "low_sample", "backlog_"))
        ][:5]
        return {
            "n": len(rows),
            "total_complaints": int(len(frame)),
            "departments": rows,
            "overall_median_hours": _round(stats.median, 1),
            "slowest": slowest,
            "fastest": fastest,
            "largest_backlog": largest_backlog,
            "interpretation": self._department_interpretation(rows, slowest, fastest, stats),
            "insights": insights,
            "filters": filters,
        }

    @staticmethod
    def _department_interpretation(rows, slowest, fastest, stats) -> str:
        if not rows:
            return "No complaints match these filters, so there is nothing to compare by department."
        parts = [
            f"{len(rows)} departments handled the complaints in view; "
            f"{rows[0]['department']} carries the largest share at {rows[0]['share_pct']}% "
            f"({rows[0]['n']} complaints)."
        ]
        if slowest and fastest and slowest["department"] != fastest["department"]:
            parts.append(
                f"{slowest['department']} resolves in a median "
                f"{slowest['median_resolution_days']} days versus "
                f"{fastest['median_resolution_days']} days for {fastest['department']} — a "
                f"{(slowest['median_resolution_hours'] / max(fastest['median_resolution_hours'], 0.01)):.1f}× "
                "gap in the typical job, not in the worst one, because these are medians."
            )
        elif stats.median:
            parts.append(
                f"The citywide median resolution time is {stats.median / 24:.1f} days, but too few "
                "departments have enough resolved cases for a fair comparison."
            )
        thin = [r["department"] for r in rows if r["resolved_sample"] < 5]
        if thin:
            parts.append(
                f"{len(thin)} department(s) have fewer than 5 resolved complaints, so their medians "
                "are excluded from the fastest/slowest comparison rather than presented as if they "
                "were reliable."
            )
        return " ".join(parts)

    async def areas(self, filters) -> dict[str, Any]:
        frame = await self.load_frame(filters)
        rows = self.area_stats(frame)
        hotspots = [r for r in rows if r["hotspot"]]
        total = int(len(frame))
        top3 = sum(r["n"] for r in rows[:3])
        context = self.build_context(frame, filters=filters, include_inference=False)
        insights = [
            item
            for item in InsightEngine(context).generate()
            if item["id"].startswith(("hotspot_", "no_data", "low_sample"))
        ][:5]
        return {
            "n": len(rows),
            "total_complaints": total,
            "areas": rows,
            "hotspots": hotspots,
            "hotspot_rule": (
                "An area is flagged as a hotspot when it has at least 5 complaints AND at least "
                "1.5× the volume an even split across all areas would give it. The threshold is "
                "relative to the number of active areas, so it does not drift as coverage grows."
            ),
            "concentration_top3_pct": round(top3 / total * 100, 1) if total else 0.0,
            "interpretation": self._area_interpretation(rows, hotspots, total),
            "insights": insights,
            "filters": filters,
        }

    @staticmethod
    def _area_interpretation(rows, hotspots, total) -> str:
        if not rows:
            return "No complaints match these filters, so there are no areas to compare."
        top = rows[0]
        parts = [
            f"Complaints came from {len(rows)} areas. {top['area']} leads with {top['n']} "
            f"({top['share_pct']}% of {total}), most often about "
            f"{top['top_category_label'] or 'mixed issues'}."
        ]
        if hotspots:
            names = ", ".join(h["area"] for h in hotspots[:3])
            parts.append(
                f"{len(hotspots)} area(s) qualify as hotspots — {names} — meaning they report at "
                "least 1.5× their fair share of the city's complaints."
            )
        else:
            parts.append(
                "No area exceeds 1.5× its fair share, so complaint volume is spread fairly evenly "
                "across the city rather than concentrated."
            )
        slow = [r for r in rows if r["median_resolution_hours"] is not None]
        if len(slow) >= 2:
            slowest = max(slow, key=lambda r: r["median_resolution_hours"])
            parts.append(
                f"{slowest['area']} is the slowest to be served, at a median "
                f"{slowest['median_resolution_hours'] / 24:.1f} days."
            )
        return " ".join(parts)

    async def insights(self, filters, *, limit: int | None = None) -> dict[str, Any]:
        frame = await self.load_frame(filters)
        analyzer = self.timeseries(frame, days=90)
        trends = analyzer.to_dict(include_forecast=True, category_limit=7)
        context = self.build_context(frame, filters=filters, trends=trends, analyzer=analyzer)
        items = InsightEngine(context).generate(limit=limit)
        return {
            "generated_at": _utcnow(),
            "n": len(items),
            "insights": items,
            "interpretation": summarise(
                items,
                fallback="There is not enough data in this filter set to draw any conclusions.",
            ),
        }

    async def public_summary(self, filters) -> dict[str, Any]:
        """Small, non-identifying subset for the public landing page."""
        frame = await self.load_frame(filters)
        analyzer = self.timeseries(frame, days=30)
        trends = analyzer.to_dict(include_forecast=False, category_limit=None)
        kpis = self.kpis(frame, trends)
        dist = FrequencyDistribution(frame["category"].tolist(), variable="category")
        rows = dist.rows()

        context = self.build_context(
            frame, filters=filters, trends=trends, analyzer=analyzer, include_inference=False
        )
        safe_ids = {
            "modal_category",
            "resolution_skew",
            "resolution_symmetric",
            "resolution_rate",
            "wow_trend",
            "category_concentration",
        }
        highlights = [i for i in InsightEngine(context).generate() if i["id"] in safe_ids][:3]

        total = kpis.get("total", 0)
        top = rows[0] if rows else None
        interpretation = (
            f"{total} complaints have been reported. {kpis.get('resolved', 0)} are resolved "
            f"({kpis.get('resolution_rate', 0):.0f}%)"
            + (
                f", with half closed within {kpis['median_resolution_days']:.1f} days."
                if kpis.get("median_resolution_days")
                else "."
            )
            + (f" The most reported issue is {top['label']} ({top['percent']}%)." if top else "")
        ) if total else "No complaints have been reported yet."

        return {
            "generated_at": _utcnow(),
            "total_complaints": total,
            "resolved": kpis.get("resolved", 0),
            "resolution_rate": kpis.get("resolution_rate", 0.0),
            "median_resolution_days": kpis.get("median_resolution_days"),
            "complaints_this_week": kpis.get("complaints_this_week", 0),
            "active_areas": int(frame["area"].nunique()) if len(frame) else 0,
            "top_category": top["value"] if top else None,
            "top_category_label": top["label"] if top else None,
            "top_category_share_pct": top["percent"] if top else 0.0,
            "categories": rows[:7],
            "highlights": highlights,
            "interpretation": interpretation,
        }


__all__ = [
    "CACHE_TTL_SECONDS",
    "FRAME_CACHE",
    "MAX_ROWS",
    "WANTED_COLUMNS",
    "AnalyticsService",
    "TTLCache",
]

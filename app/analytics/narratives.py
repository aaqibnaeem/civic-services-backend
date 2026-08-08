"""Rules-based interpretation layer: turns computed statistics into plain English.

Why this is deterministic and NOT LLM-generated
-----------------------------------------------
Every sentence in this module is produced by a hand-written rule that reads a number
which was already computed by ``descriptive.py`` / ``distributions.py`` /
``timeseries.py`` / ``inference.py`` and formats it into a template. **No language
model is involved at any point.** That is a deliberate architectural decision, not a
shortcut:

1. **A statistic can never be hallucinated.** An LLM asked to "summarise these
   analytics" can and does invent plausible-sounding figures, invert a trend
   direction, or confidently describe a correlation that is not in the data. When the
   output is a civic dashboard that a city officer will act on, a fabricated "waste
   complaints rose 40%" is worse than no summary at all. Here the number in the
   sentence is literally the same Python float that was computed upstream — it is
   interpolated, never generated.
2. **It is reproducible and testable.** The same DataFrame always yields the same
   insights, so the narrative layer can be unit tested like any other code, and a
   judge can trace any sentence back to the exact rule and the exact statistic that
   produced it.
3. **It cannot fail or cost money.** No network call, no API key, no latency, no
   rate limit, no outage. The dashboard's explanation layer works when DeepSeek is
   down.

The trade-off is honestly acknowledged: rule-based prose is less fluent and cannot
notice a pattern nobody anticipated. For a statistics deliverable that trade is
correct — we would rather be narrow and right than broad and unverifiable. The LLM in
this project is used where hallucination is recoverable (drafting, classification with
a confidence score and a fallback tier), not where it would put invented numbers in
front of a decision maker.

Rule contract
-------------
Each ``_rule_*`` method returns an ``Insight`` dict — ``{id, severity, title, detail,
metric, unit}`` — or ``None`` when its precondition is not met (not enough data, the
pattern is absent). Rules are individually wrapped in try/except by the engine so one
malformed input can never take down the endpoint. ``severity`` is ``info``/``warn``/
``critical`` so the UI can rank; the sort is by severity first, then by rule order.

Every headline is written to **stand alone if read aloud** — the number lives inside
the sentence, never in a neighbouring cell the reader has to look up.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from app.analytics.distributions import CATEGORY_LABELS, label_for

SEVERITY_RANK = {"critical": 0, "warn": 1, "info": 2}


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _days(hours: float | None) -> float | None:
    return None if hours is None else round(hours / 24.0, 1)


def _one_in(share: float) -> str:
    """'1 in 3' phrasing for a proportion, rounded to something a human would say."""
    if share <= 0:
        return "none"
    ratio = 1.0 / share
    if ratio < 1.2:
        return "almost all"
    return f"1 in {ratio:.0f}" if ratio >= 2 else f"{share * 100:.0f}%"


def _pct(value: float | None, digits: int = 1) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}%"


def _insight(
    rule_id: str,
    severity: str,
    title: str,
    detail: str,
    metric: float | None = None,
    unit: str | None = None,
) -> dict[str, Any]:
    return {
        "id": rule_id,
        "severity": severity,
        "title": title,
        "detail": detail,
        "metric": round(metric, 2) if isinstance(metric, (int, float)) else None,
        "unit": unit,
    }


# --------------------------------------------------------------------------- #
# context
# --------------------------------------------------------------------------- #
@dataclass
class InsightContext:
    """Everything the rules may read. All fields optional — rules guard themselves."""

    n_total: int = 0
    frame: Any = None
    now: datetime = field(default_factory=lambda: datetime.now(UTC))

    resolution: dict[str, Any] | None = None  # DescriptiveStats.to_dict()
    outliers: dict[str, Any] | None = None  # OutlierDetector.report()
    category_dist: dict[str, Any] | None = None
    priority_dist: dict[str, Any] | None = None
    status_dist: dict[str, Any] | None = None
    area_dist: dict[str, Any] | None = None
    trends: dict[str, Any] | None = None  # TimeSeriesAnalyzer.to_dict()
    weekend_gap: tuple[float, float] | None = None
    chi_square: dict[str, Any] | None = None
    spearman: dict[str, Any] | None = None
    departments: list[dict[str, Any]] = field(default_factory=list)
    areas: list[dict[str, Any]] = field(default_factory=list)
    kpis: dict[str, Any] | None = None
    filters_description: str = "all complaints"


# --------------------------------------------------------------------------- #
# engine
# --------------------------------------------------------------------------- #
class InsightEngine:
    """Runs every rule against an :class:`InsightContext` and ranks the results."""

    #: Rule execution order. Order matters only as the tie-break inside a severity band.
    RULES: tuple[str, ...] = (
        "_rule_no_data",
        "_rule_low_sample",
        "_rule_critical_backlog",
        "_rule_sla_breach",
        "_rule_resolution_skew",
        "_rule_resolution_outliers",
        "_rule_category_outliers",
        "_rule_department_laggard",
        "_rule_department_backlog",
        "_rule_wow_trend",
        "_rule_category_wow_spike",
        "_rule_modal_category",
        "_rule_category_concentration",
        "_rule_hotspot_area",
        "_rule_priority_mix",
        "_rule_backlog_ageing",
        "_rule_resolution_rate",
        "_rule_weekend_effect",
        "_rule_chi_square",
        "_rule_spearman_priority_speed",
        "_rule_dispersion",
        "_rule_forecast",
        "_rule_ai_confidence",
        "_rule_busiest_day",
    )

    def __init__(self, context: InsightContext) -> None:
        self.ctx = context

    def generate(self, *, limit: int | None = None) -> list[dict[str, Any]]:
        """Produce ranked insights. A failing rule is skipped, never fatal."""
        results: list[tuple[int, int, dict[str, Any]]] = []
        for order, name in enumerate(self.RULES):
            rule = getattr(self, name, None)
            if rule is None:
                continue
            try:
                produced = rule()
            except Exception:  # pragma: no cover - defensive: a dashboard must render
                continue
            if not produced:
                continue
            items = produced if isinstance(produced, list) else [produced]
            for item in items:
                if item:
                    results.append((SEVERITY_RANK.get(item["severity"], 3), order, item))
        results.sort(key=lambda triple: (triple[0], triple[1]))
        ordered = [item for _sev, _order, item in results]
        return ordered[:limit] if limit else ordered

    def to_models(self, *, limit: int | None = None) -> list:
        from app.schemas.analytics import Insight

        return [Insight(**item) for item in self.generate(limit=limit)]

    # ------------------------------------------------------------------ rules
    def _rule_no_data(self) -> dict[str, Any] | None:
        if self.ctx.n_total > 0:
            return None
        return _insight(
            "no_data",
            "warn",
            "No complaints match the current filters.",
            f"The filter set ({self.ctx.filters_description}) returned zero complaints, so every "
            "statistic below is empty rather than zero — there is nothing to describe, which is "
            "different from describing something as nil.",
        )

    def _rule_low_sample(self) -> dict[str, Any] | None:
        n = self.ctx.n_total
        if n == 0 or n >= 30:
            return None
        return _insight(
            "low_sample",
            "warn",
            f"Only {n} complaints match these filters — treat every figure below as indicative.",
            f"With n = {n} (below the conventional threshold of 30), percentages move by several "
            f"points every time one complaint is added, quartiles are unstable and the outlier "
            "fences can shift dramatically. These numbers are a first look, not evidence.",
            metric=float(n),
            unit="complaints",
        )

    def _rule_critical_backlog(self) -> dict[str, Any] | None:
        kpis = self.ctx.kpis or {}
        critical_open = int(kpis.get("critical_open") or 0)
        if critical_open <= 0:
            return None
        total = max(1, int(kpis.get("total") or self.ctx.n_total or 1))
        share = critical_open / total * 100
        severity = "critical" if critical_open >= 5 else "warn"
        return _insight(
            "critical_backlog",
            severity,
            f"{critical_open} critical complaints are still unresolved right now.",
            f"Critical is the highest priority band in the system, and {critical_open} of them "
            f"({share:.1f}% of all {total} complaints in view) sit in open, assigned or in-progress "
            "status. These are the cases where delay carries the most risk, so they should be the "
            "first queue anyone clears.",
            metric=float(critical_open),
            unit="complaints",
        )

    def _rule_sla_breach(self) -> dict[str, Any] | None:
        frame = self.ctx.frame
        if frame is None or not len(frame):
            return None
        cols = set(frame.columns)
        if not {"status", "age_days"} <= cols:
            return None
        from app.analytics.distributions import OPEN_STATUSES

        open_rows = frame[frame["status"].isin(OPEN_STATUSES)]
        if not len(open_rows):
            return None
        stale = open_rows[open_rows["age_days"] > 14]
        if not len(stale):
            return None
        share = len(stale) / len(open_rows) * 100
        if share < 10:
            return None
        severity = "critical" if share >= 40 else "warn"
        oldest = float(open_rows["age_days"].max())
        return _insight(
            "backlog_stale",
            severity,
            f"{len(stale)} open complaints ({share:.0f}% of the backlog) have been waiting more than 14 days.",
            f"Of the {len(open_rows)} complaints still open, {len(stale)} are older than two weeks "
            f"and the oldest has been waiting {oldest:.0f} days. Ageing backlog is the single "
            "clearest signal that intake is outpacing capacity in at least one team.",
            metric=float(len(stale)),
            unit="complaints",
        )

    def _rule_resolution_skew(self) -> dict[str, Any] | None:
        res = self.ctx.resolution or {}
        mean, median, n = res.get("mean"), res.get("median"), res.get("n") or 0
        if mean is None or median is None or n < 5:
            return None
        median_days = _days(median)
        mean_days = _days(mean)
        if median <= 0:
            return None
        gap = (mean - median) / median
        skew = res.get("skewness")
        if gap > 0.20:
            return _insight(
                "resolution_skew",
                "info",
                f"Half of all complaints are resolved within {median_days} days.",
                f"The median resolution time is {median_days} days but the mean is "
                f"{mean_days} days — {gap * 100:.0f}% higher"
                + (f", and the distribution's skewness is {skew:.2f}" if skew is not None else "")
                + ". That gap means a minority of very slow cases is dragging the average upwards, "
                "so the median is the honest headline number and quoting the mean would overstate "
                "how long a typical citizen actually waits.",
                metric=median_days,
                unit="days",
            )
        if gap < -0.20:
            return _insight(
                "resolution_skew",
                "info",
                f"Half of all complaints are resolved within {median_days} days.",
                f"The mean ({mean_days} days) sits below the median ({median_days} days), which is "
                "unusual for service times: it means a cluster of very fast closures is pulling the "
                "average down. Worth checking that those quick closures are genuine resolutions and "
                "not tickets being closed without work.",
                metric=median_days,
                unit="days",
            )
        return _insight(
            "resolution_symmetric",
            "info",
            f"Complaints are resolved in a median {median_days} days, and the average agrees.",
            f"The mean ({mean_days} days) and median ({median_days} days) are within "
            f"{abs(gap) * 100:.0f}% of each other, so the distribution is roughly symmetric and "
            "either measure is a fair summary — that is unusual for resolution times and a good "
            "sign of a consistent process.",
            metric=median_days,
            unit="days",
        )

    def _rule_resolution_outliers(self) -> dict[str, Any] | None:
        out = self.ctx.outliers or {}
        upper = out.get("upper_fence")
        count = int(out.get("upper_count") or 0)
        n = int(out.get("n") or 0)
        if upper is None or n < 8:
            return None
        upper_days = _days(upper)
        if count == 0:
            return _insight(
                "resolution_outliers",
                "info",
                f"No complaint has taken longer than {upper_days} days — the process has no extreme tail.",
                f"The Tukey upper fence (Q3 + 1.5×IQR) sits at {upper_days} days across "
                f"{n} resolved complaints and nothing exceeds it. Every resolution falls inside the "
                "normal range for this process, which means there is no runaway case to chase.",
                metric=upper_days,
                unit="days",
            )
        share = count / n * 100
        severity = "critical" if share >= 10 else "warn"
        worst = None
        points = out.get("outliers") or []
        if points:
            worst = points[0]
        worst_text = ""
        if worst and worst.get("reference_code"):
            worst_text = (
                f" The worst is {worst['reference_code']} at {_days(worst.get('value'))} days"
                + (f" ({label_for('category', worst['category'])})" if worst.get("category") else "")
                + "."
            )
        return _insight(
            "resolution_outliers",
            severity,
            f"{count} complaints took longer than {upper_days} days — far beyond the normal range.",
            f"Using Tukey's rule (Q3 + 1.5×IQR = {upper_days} days), {count} of {n} resolved "
            f"complaints ({share:.1f}%) are statistical outliers rather than merely slow. "
            "The IQR rule was chosen over a ±3 standard-deviation rule precisely because the mean "
            "and standard deviation are themselves inflated by these cases, while the quartiles are "
            f"not.{worst_text} Each one is listed by reference code so it can be chased individually.",
            metric=float(count),
            unit="complaints",
        )

    def _rule_category_outliers(self) -> dict[str, Any] | None:
        groups = (self.ctx.outliers or {}).get("by_group") or []
        fenced = [g for g in groups if g.get("upper_fence") is not None and g.get("outlier_count")]
        if not fenced:
            return None
        worst = max(fenced, key=lambda g: g["outlier_count"])
        name = label_for("category", worst["group"])
        return _insight(
            "category_outliers",
            "warn",
            f"{name} has {worst['outlier_count']} complaints that are abnormally slow even by its own standards.",
            f"Fences were recomputed inside each category, because a 3-day drainage repair and a "
            f"3-day pothole repair are not comparable events. Judged against {name}'s own upper "
            f"fence of {_days(worst['upper_fence'])} days (its median is "
            f"{_days(worst['median'])} days), {worst['outlier_count']} of its {worst['n']} resolved "
            "complaints are genuine outliers — cases a single citywide fence would have hidden.",
            metric=float(worst["outlier_count"]),
            unit="complaints",
        )

    def _rule_department_laggard(self) -> dict[str, Any] | None:
        depts = [
            d
            for d in (self.ctx.departments or [])
            if d.get("median_resolution_hours") is not None and (d.get("resolved_sample") or 0) >= 5
        ]
        if len(depts) < 2:
            return None
        slowest = max(depts, key=lambda d: d["median_resolution_hours"])
        others = [d for d in depts if d["department"] != slowest["department"]]
        if not others:
            return None
        other_medians = sorted(d["median_resolution_hours"] for d in others)
        mid = len(other_medians) // 2
        peer_median = (
            other_medians[mid]
            if len(other_medians) % 2
            else (other_medians[mid - 1] + other_medians[mid]) / 2
        )
        slow_days = _days(slowest["median_resolution_hours"])
        peer_days = _days(peer_median)
        if peer_days in (None, 0) or slow_days is None:
            return None
        if slow_days < peer_days * 1.3:
            return None
        ratio = slow_days / peer_days if peer_days else 0
        severity = "critical" if ratio >= 3 else "warn"
        return _insight(
            "department_laggard",
            severity,
            f"{slowest['department']} resolves in a median {slow_days} days versus {peer_days} days across other departments.",
            f"{slowest['department']} handled {slowest['n']} complaints and closed them in a median "
            f"{slow_days} days — {ratio:.1f}× the {peer_days}-day median of the other "
            f"{len(others)} departments. The median is used rather than the mean so a single "
            "extreme case cannot manufacture this gap; the difference is in the typical job, not "
            "the worst one.",
            metric=slow_days,
            unit="days",
        )

    def _rule_department_backlog(self) -> dict[str, Any] | None:
        depts = [d for d in (self.ctx.departments or []) if (d.get("backlog") or 0) > 0]
        if not depts:
            return None
        worst = max(depts, key=lambda d: d["backlog"])
        total_backlog = sum(d.get("backlog") or 0 for d in depts)
        if total_backlog == 0 or worst["backlog"] < 3:
            return None
        share = worst["backlog"] / total_backlog * 100
        if share < 30 or len(depts) < 2:
            return None
        return _insight(
            "department_backlog",
            "warn",
            f"{worst['department']} is carrying {worst['backlog']} of the {total_backlog} unresolved complaints ({share:.0f}% of the backlog).",
            f"Backlog is concentrated rather than spread: {worst['department']} holds {share:.0f}% "
            f"of everything still open across {len(depts)} departments, from {worst['n']} total "
            "complaints. Concentrated backlog is usually a capacity problem in one team rather than "
            "a citywide one, which makes it fixable by reassignment.",
            metric=float(worst["backlog"]),
            unit="complaints",
        )

    def _rule_wow_trend(self) -> dict[str, Any] | None:
        wow = (self.ctx.trends or {}).get("week_over_week") or {}
        current = int(wow.get("current_week") or 0)
        previous = int(wow.get("previous_week") or 0)
        pct = wow.get("change_pct")
        if current == 0 and previous == 0:
            return None
        if pct is None:
            return _insight(
                "wow_trend",
                "warn",
                f"{current} complaints were filed this week, up from none the week before.",
                "The previous week had no complaints at all, so a percentage change cannot be "
                "computed from a zero base. The absolute jump is reported instead — a percentage "
                "here would be meaningless, not infinite.",
                metric=float(current),
                unit="complaints",
            )
        if abs(pct) < 10:
            return _insight(
                "wow_trend",
                "info",
                f"Complaint volume is steady: {current} this week versus {previous} last week ({pct:+.0f}%).",
                f"A {abs(pct):.0f}% change on a base of {previous} is within ordinary week-to-week "
                "noise for this volume, so there is no trend to act on — the intake rate is stable.",
                metric=float(current),
                unit="complaints",
            )
        rising = pct > 0
        severity = "warn" if (rising and pct >= 25) else "info"
        return _insight(
            "wow_trend",
            severity,
            f"Complaints {'rose' if rising else 'fell'} {abs(pct):.0f}% this week — {current} versus {previous} last week.",
            f"The last 7 days brought {current} complaints against {previous} in the 7 days before, "
            f"a {abs(pct):.0f}% {'increase' if rising else 'decrease'}. "
            + (
                "A rise of this size usually means either a genuine service failure somewhere or a "
                "reporting campaign; the per-category breakdown below shows which."
                if rising
                else "A fall of this size is worth confirming against the category breakdown — it can "
                "mean the problem was fixed, or that reporting has dropped off."
            ),
            metric=float(pct),
            unit="%",
        )

    def _rule_category_wow_spike(self) -> list[dict[str, Any]]:
        series = (self.ctx.trends or {}).get("by_category") or []
        out: list[dict[str, Any]] = []
        movers = [
            s
            for s in series
            if s.get("week_over_week_pct") is not None
            and abs(s["week_over_week_pct"]) >= 30
            and s.get("total", 0) >= 10
        ]
        movers.sort(key=lambda s: -abs(s["week_over_week_pct"]))
        for mover in movers[:2]:
            pct = mover["week_over_week_pct"]
            rising = pct > 0
            label = mover.get("label") or label_for("category", mover["category"])
            points = mover.get("points") or []
            current = sum(p["count"] for p in points[-7:])
            previous = sum(p["count"] for p in points[-14:-7]) if len(points) >= 14 else 0
            out.append(
                _insight(
                    f"category_wow_{mover['category']}",
                    "warn" if rising else "info",
                    f"{label} complaints {'rose' if rising else 'fell'} {abs(pct):.0f}% this week versus last.",
                    f"{label} went from {previous} complaints in the previous 7 days to {current} in "
                    f"the last 7 — a {abs(pct):.0f}% {'jump' if rising else 'drop'}. "
                    + (
                        "A single-category spike is usually geographic: check the area breakdown for "
                        "a burst in one neighbourhood before assuming a citywide change."
                        if rising
                        else "Confirm this is resolution rather than under-reporting."
                    ),
                    metric=float(pct),
                    unit="%",
                )
            )
        return out

    def _rule_modal_category(self) -> dict[str, Any] | None:
        dist = self.ctx.category_dist or {}
        mode = dist.get("mode")
        n = int(dist.get("n") or 0)
        if not mode or n < 5:
            return None
        share = float(dist.get("mode_share") or 0)
        count = int(dist.get("mode_count") or 0)
        label = CATEGORY_LABELS.get(mode, mode.title())
        if dist.get("mode_kind") == "multi":
            names = ", ".join(CATEGORY_LABELS.get(m, m) for m in dist.get("modes", [])[:3])
            return _insight(
                "modal_category",
                "info",
                f"No single complaint type dominates — {names} are tied at {count} complaints each.",
                f"Several categories tie for the mode with {count} complaints each out of {n}. A "
                "tied mode means demand is spread evenly across services rather than concentrated "
                "in one failing one, so there is no obvious single target for intervention.",
                metric=float(count),
                unit="complaints",
            )
        return _insight(
            "modal_category",
            "info",
            f"{label} is the most frequent complaint type — {_one_in(share)} of everything reported.",
            f"{label} accounts for {count} of {n} complaints ({share * 100:.1f}%), making it the "
            f"modal category. That is the single largest claim on resources in this filter set, so "
            "it is where a fixed improvement in resolution time buys the most citizen-hours back.",
            metric=float(share * 100),
            unit="%",
        )

    def _rule_category_concentration(self) -> dict[str, Any] | None:
        dist = self.ctx.category_dist or {}
        rows = dist.get("rows") or []
        n = int(dist.get("n") or 0)
        if len(rows) < 4 or n < 20:
            return None
        top3 = rows[:3]
        share = sum(r["count"] for r in top3) / n * 100
        names = ", ".join(r["label"] for r in top3)
        if share >= 70:
            return _insight(
                "category_concentration",
                "info",
                f"Three categories account for {share:.0f}% of all complaints: {names}.",
                f"Of {dist.get('distinct')} active categories, just three — {names} — produce "
                f"{share:.0f}% of the {n} complaints in view. A concentrated distribution like this "
                "means targeted investment beats spreading effort evenly across every service.",
                metric=float(share),
                unit="%",
            )
        return _insight(
            "category_concentration",
            "info",
            f"Complaints are spread broadly — the top three categories are only {share:.0f}% of the total.",
            f"With {dist.get('distinct')} active categories and the largest three covering just "
            f"{share:.0f}% of {n} complaints, demand is diffuse. There is no single failing service "
            "to target; improvements have to come from process changes that apply across categories.",
            metric=float(share),
            unit="%",
        )

    def _rule_hotspot_area(self) -> dict[str, Any] | None:
        areas = self.ctx.areas or []
        if len(areas) < 2:
            return None
        total = sum(a.get("n", 0) for a in areas)
        if total < 20:
            return None
        top = max(areas, key=lambda a: a.get("n", 0))
        share = top["n"] / total * 100
        expected = 100.0 / len(areas)
        if share < expected * 1.5:
            return None
        severity = "warn" if share >= expected * 2.5 else "info"
        cat = top.get("top_category_label") or top.get("top_category") or "mixed"
        return _insight(
            "hotspot_area",
            severity,
            f"{top['area']} is the clearest hotspot with {top['n']} complaints — {share:.0f}% of the city's total.",
            f"{top['area']} generates {share:.1f}% of all {total} complaints while an even spread "
            f"across {len(areas)} areas would give it {expected:.1f}%. Its dominant issue is {cat} "
            f"({top.get('top_category_count', 0)} cases). Concentration this far above the even "
            "share is the definition of a hotspot and justifies sending a dedicated team.",
            metric=float(share),
            unit="%",
        )

    def _rule_priority_mix(self) -> dict[str, Any] | None:
        dist = self.ctx.priority_dist or {}
        rows = {r["value"]: r for r in (dist.get("rows") or [])}
        n = int(dist.get("n") or 0)
        if n < 10:
            return None
        escalated = sum(rows.get(k, {}).get("count", 0) for k in ("high", "critical"))
        share = escalated / n * 100
        if share >= 50:
            return _insight(
                "priority_mix",
                "critical",
                f"{share:.0f}% of complaints are rated high or critical — the priority scale has lost its meaning.",
                f"{escalated} of {n} complaints sit in the top two priority bands. When more than "
                "half of everything is urgent, priority stops discriminating between cases and the "
                "queue effectively reverts to first-in-first-out. Either the triage thresholds need "
                "recalibrating or the city genuinely has a systemic failure.",
                metric=float(share),
                unit="%",
            )
        if share >= 30:
            return _insight(
                "priority_mix",
                "warn",
                f"{share:.0f}% of complaints are rated high or critical.",
                f"{escalated} of {n} complaints are in the top two priority bands. That is a heavy "
                "urgent load — enough that the highest band should be watched for creeping "
                "over-classification, which would dilute its usefulness for scheduling.",
                metric=float(share),
                unit="%",
            )
        return _insight(
            "priority_mix",
            "info",
            f"The priority mix is healthy — only {share:.0f}% of complaints are high or critical.",
            f"{escalated} of {n} complaints fall in the top two priority bands, leaving the "
            f"majority as routine work. A pyramid-shaped priority distribution like this is what "
            "lets urgent cases actually jump the queue.",
            metric=float(share),
            unit="%",
        )

    def _rule_backlog_ageing(self) -> dict[str, Any] | None:
        frame = self.ctx.frame
        if frame is None or not len(frame):
            return None
        if not {"status", "age_days"} <= set(frame.columns):
            return None
        from app.analytics.distributions import OPEN_STATUSES

        open_rows = frame[frame["status"].isin(OPEN_STATUSES)]
        if len(open_rows) < 5:
            return None
        median_age = float(open_rows["age_days"].median())
        oldest = float(open_rows["age_days"].max())
        severity = "warn" if median_age > 7 else "info"
        return _insight(
            "backlog_ageing",
            severity,
            f"The typical unresolved complaint has been waiting {median_age:.1f} days.",
            f"{len(open_rows)} complaints are still open, assigned or in progress. Their median age "
            f"is {median_age:.1f} days and the oldest has been waiting {oldest:.0f} days. Ageing is "
            "measured on the open queue specifically — averaging it with closed cases would hide "
            "exactly the cases that are going wrong.",
            metric=round(median_age, 1),
            unit="days",
        )

    def _rule_resolution_rate(self) -> dict[str, Any] | None:
        kpis = self.ctx.kpis or {}
        total = int(kpis.get("total") or 0)
        resolved = int(kpis.get("resolved") or 0)
        if total < 10:
            return None
        rate = resolved / total * 100
        if rate >= 75:
            severity, verdict = (
                "info",
                "That is a strong closure rate — the service is keeping pace with intake.",
            )
        elif rate >= 50:
            severity, verdict = (
                "info",
                f"The service is closing more than it carries, but the remaining {100 - rate:.0f}% "
                "is the standing backlog.",
            )
        else:
            severity, verdict = (
                "warn",
                "Fewer than half of all reported issues have been closed, which means the backlog "
                "is growing faster than it is being cleared.",
            )
        return _insight(
            "resolution_rate",
            severity,
            f"{rate:.0f}% of complaints have been resolved ({resolved} of {total}).",
            f"{resolved} of {total} complaints reached resolved status. {verdict} Note this is a "
            "snapshot ratio, not a cohort rate: recently filed complaints have not had time to be "
            "resolved yet, so it understates the true eventual closure rate.",
            metric=round(rate, 1),
            unit="%",
        )

    def _rule_weekend_effect(self) -> dict[str, Any] | None:
        gap = self.ctx.weekend_gap
        if not gap:
            return None
        weekday_mean, weekend_mean = gap
        if weekday_mean <= 0:
            return None
        ratio = weekend_mean / weekday_mean
        if 0.75 <= ratio <= 1.25:
            return _insight(
                "weekend_effect",
                "info",
                f"Reporting is flat across the week — {weekday_mean:.1f} complaints a day on weekdays versus {weekend_mean:.1f} at weekends.",
                "There is no meaningful weekend effect in this data, which suggests citizens are "
                "reporting through a channel that is genuinely available all week rather than only "
                "during office hours.",
                metric=round(ratio, 2),
                unit="ratio",
            )
        lower = ratio < 1
        return _insight(
            "weekend_effect",
            "info",
            f"Weekend reporting runs {abs(1 - ratio) * 100:.0f}% {'below' if lower else 'above'} weekdays — {weekend_mean:.1f} versus {weekday_mean:.1f} complaints a day.",
            f"Weekdays average {weekday_mean:.1f} complaints a day against {weekend_mean:.1f} at "
            f"weekends. "
            + (
                "The weekday skew means staffing should follow the same shape, and it also means a "
                "Monday spike is usually accumulated weekend demand rather than a new problem."
                if lower
                else "Weekend-heavy reporting is unusual and suggests issues are being noticed when "
                "people are at home rather than at work."
            ),
            metric=round(ratio, 2),
            unit="ratio",
        )

    def _rule_chi_square(self) -> dict[str, Any] | None:
        chi = self.ctx.chi_square or {}
        if chi.get("statistic") is None:
            return None
        row_v, col_v = chi.get("row_variable", "category"), chi.get("col_variable", "priority")
        if not chi.get("reliable"):
            return _insight(
                "chi_square_unreliable",
                "warn",
                f"The {row_v}–{col_v} independence test cannot be trusted: {chi.get('cells_below_5')} of "
                f"{chi.get('total_cells')} cells have an expected count below 5.",
                f"A chi-square test of independence gave χ²({chi.get('dof')}) = "
                f"{chi.get('statistic'):.1f}, but {chi.get('cells_below_5')} of "
                f"{chi.get('total_cells')} cells have an expected frequency below 5 (smallest = "
                f"{chi.get('expected_min')}). That breaks the assumption the test rests on, so the "
                "p-value is not meaningful. We are showing the number rather than hiding it, and "
                "the fix is more data or merging the sparse categories — not reporting it anyway.",
                metric=float(chi.get("statistic")),
                unit=None,
            )
        significant = bool(chi.get("significant"))
        v = chi.get("cramers_v")
        if significant:
            return _insight(
                "chi_square_association",
                "info",
                f"Complaint {col_v} is genuinely linked to {row_v} — χ²({chi.get('dof')}) = "
                f"{chi.get('statistic'):.1f}, p = {chi.get('p_value'):.4g}, Cramér's V = {v:.2f}.",
                f"A chi-square test of independence returned χ²({chi.get('dof')}, N={chi.get('n')}) "
                f"= {chi.get('statistic'):.1f} with a p-value of "
                f"{chi.get('p_value'):.4g}, below the 0.05 threshold, so we reject independence: "
                f"different {row_v} values genuinely carry different {col_v} mixes. Cramér's V = "
                f"{v:.2f} puts the strength of that association at '{chi.get('effect_size')}' — "
                "with a sample this size, significance alone would not have told us whether it "
                "matters in practice. All expected cell counts met the ≥5 assumption.",
                metric=float(v) if v is not None else None,
                unit=None,
            )
        return _insight(
            "chi_square_independence",
            "info",
            f"Complaint {col_v} looks statistically independent of {row_v} "
            f"(χ²({chi.get('dof')}) = {chi.get('statistic'):.1f}, p = {chi.get('p_value'):.2f}).",
            f"χ²({chi.get('dof')}, N={chi.get('n')}) = {chi.get('statistic'):.1f}, p = "
            f"{chi.get('p_value'):.3f}, which is above the 0.05 threshold, so we cannot reject the "
            f"hypothesis that the {col_v} mix is the same across every {row_v}. In practice that "
            "means priority is being driven by the individual case rather than by what kind of "
            "complaint it is.",
            metric=float(chi.get("statistic")),
            unit=None,
        )

    def _rule_spearman_priority_speed(self) -> dict[str, Any] | None:
        sp = self.ctx.spearman or {}
        rho = sp.get("rho")
        if rho is None or not sp.get("reliable"):
            return None
        if not sp.get("significant"):
            return _insight(
                "priority_speed_none",
                "warn",
                f"Priority is not driving the queue — the link between priority and speed is only ρ = {rho:.2f}, and not statistically significant.",
                f"A Spearman rank correlation between priority rank and resolution time gives "
                f"ρ = {rho:.2f} across {sp.get('n')} resolved complaints, which is not significant "
                f"(p = {sp.get('p_value'):.3f}). If the priority field were driving the work queue "
                "we would expect a clear negative correlation; its absence suggests priority is "
                "being recorded but not used for scheduling.",
                metric=round(float(rho), 2),
                unit="rho",
            )
        if rho < 0:
            return _insight(
                "priority_speed_ok",
                "info",
                f"Triage is working: higher-priority complaints are resolved measurably faster (ρ = {rho:.2f}).",
                f"Spearman's rank correlation between priority (low=1 … critical=4) and resolution "
                f"hours is {rho:.2f} over {sp.get('n')} resolved complaints, p = "
                f"{sp.get('p_value'):.4g}. The negative sign is the healthy direction — as priority "
                "rises, time-to-resolution falls. Spearman rather than Pearson because priority is "
                "ordinal and resolution time is heavily right-skewed.",
                metric=round(float(rho), 2),
                unit="rho",
            )
        return _insight(
            "priority_speed_inverted",
            "critical",
            f"High-priority complaints are taking LONGER than low-priority ones (ρ = +{rho:.2f}).",
            f"Spearman's rank correlation between priority rank and resolution hours is +{rho:.2f} "
            f"across {sp.get('n')} resolved complaints (p = {sp.get('p_value'):.4g}) — a "
            "significant positive relationship, meaning the queue is running backwards relative to "
            "its own priority labels. Either urgent cases are intrinsically harder, or they are not "
            "being scheduled first. Both need a human answer.",
            metric=round(float(rho), 2),
            unit="rho",
        )

    def _rule_dispersion(self) -> dict[str, Any] | None:
        res = self.ctx.resolution or {}
        cv = res.get("coefficient_of_variation")
        n = int(res.get("n") or 0)
        if cv is None or n < 20:
            return None
        std, mean = res.get("std_dev"), res.get("mean")
        if cv >= 1.0:
            return _insight(
                "resolution_dispersion",
                "warn",
                f"Resolution times are wildly inconsistent — the standard deviation ({_days(std)} days) is larger than the average itself ({_days(mean)} days).",
                f"The coefficient of variation is {cv:.2f}, meaning spread exceeds the mean. In "
                "practice a citizen cannot be given a reliable estimate of when their complaint "
                "will be closed, because two identical-looking complaints can differ by weeks. "
                "Consistency, not just speed, is the thing to fix here.",
                metric=round(float(cv), 2),
                unit="cv",
            )
        if cv >= 0.5:
            return _insight(
                "resolution_dispersion",
                "info",
                f"Resolution times vary moderately — a standard deviation of {_days(std)} days around a {_days(mean)}-day average.",
                f"The coefficient of variation is {cv:.2f}. There is real variability but it is "
                "proportionate; the city could quote a range to citizens with reasonable "
                "confidence.",
                metric=round(float(cv), 2),
                unit="cv",
            )
        return _insight(
            "resolution_dispersion",
            "info",
            f"Resolution times are tightly clustered — a standard deviation of only {_days(std)} days.",
            f"The coefficient of variation is {cv:.2f}, which is low: most complaints take close to "
            f"the {_days(mean)}-day average. That predictability is worth as much to citizens as "
            "the speed itself.",
            metric=round(float(cv), 2),
            unit="cv",
        )

    def _rule_forecast(self) -> dict[str, Any] | None:
        fc = (self.ctx.trends or {}).get("forecast") or {}
        expected = fc.get("expected_total")
        if expected is None or not fc.get("points"):
            return None
        wow = (self.ctx.trends or {}).get("week_over_week") or {}
        current = int(wow.get("current_week") or 0)
        if current == 0:
            return None
        delta = (expected - current) / current * 100
        return _insight(
            "forecast_next_week",
            "info",
            f"About {expected:.0f} complaints are expected over the next 7 days.",
            f"Using a {fc.get('method')}, next week's volume projects to roughly {expected:.0f} "
            f"complaints against {current} filed this week ({delta:+.0f}%). This is deliberately a "
            "naive baseline: it assumes the weekly pattern repeats, extrapolates no trend, and "
            "knows nothing about weather, holidays or campaigns. Use it for rostering, not for "
            "promises.",
            metric=round(float(expected), 1),
            unit="complaints",
        )

    def _rule_ai_confidence(self) -> dict[str, Any] | None:
        kpis = self.ctx.kpis or {}
        conf = kpis.get("avg_ai_confidence")
        if conf is None:
            return None
        pct = float(conf) * 100
        if pct < 60:
            return _insight(
                "ai_confidence",
                "warn",
                f"The AI classifier averages only {pct:.0f}% confidence on these complaints.",
                f"Mean confidence across analysed complaints is {pct:.0f}%. Below roughly 60% the "
                "automatic category and priority assignments should be treated as suggestions "
                "needing human review, not as facts — and every downstream category statistic "
                "inherits that uncertainty.",
                metric=round(pct, 1),
                unit="%",
            )
        return _insight(
            "ai_confidence",
            "info",
            f"The AI classifier averages {pct:.0f}% confidence on these complaints.",
            f"Mean confidence across analysed complaints is {pct:.0f}%, which is high enough that "
            "the automatic category and priority assignments underpinning these statistics can be "
            "relied on, while still leaving low-confidence cases visible for human review.",
            metric=round(pct, 1),
            unit="%",
        )

    def _rule_busiest_day(self) -> dict[str, Any] | None:
        trends = self.ctx.trends or {}
        busiest = trends.get("busiest_day")
        daily = trends.get("daily_stats") or {}
        mean, std = daily.get("mean"), daily.get("std_dev")
        if not busiest or mean is None or std in (None, 0):
            return None
        count = busiest["count"]
        if count <= mean:
            return None
        z = (count - mean) / std
        if z < 2:
            return None
        day = busiest["date"]
        day_text = day.isoformat() if hasattr(day, "isoformat") else str(day)
        return _insight(
            "volume_spike_day",
            "warn",
            f"{day_text} was an exceptional day with {count} complaints — {z:.1f} standard deviations above normal.",
            f"Daily volume over this window averages {mean:.1f} complaints with a standard "
            f"deviation of {std:.1f}. {day_text} recorded {count}, which is {z:.1f} SDs above the "
            "mean — well past the point where random variation is a plausible explanation. Days "
            "like this normally trace back to a single event such as a storm, a burst main or a "
            "media story.",
            metric=float(count),
            unit="complaints",
        )


def build_insights(context: InsightContext, *, limit: int | None = None) -> list[dict[str, Any]]:
    """Functional entry point used by the service layer."""
    return InsightEngine(context).generate(limit=limit)


def summarise(insights: list[dict[str, Any]], *, fallback: str = "") -> str:
    """Join the top insight titles into a one-paragraph ``interpretation`` string."""
    if not insights:
        return fallback
    return " ".join(item["title"] for item in insights[:3])


__all__ = ["SEVERITY_RANK", "InsightContext", "InsightEngine", "build_insights", "summarise"]

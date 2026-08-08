"""Tukey-fence outlier detection on resolution time — overall and *within category*.

The statistically honest bit
---------------------------
Computing one set of fences over all complaints is the obvious thing to do and it is
wrong for this dataset. Resolution time is not drawn from a single population: a
blown streetlight and a collapsed sewer line have completely different natural
service-time distributions. Pooling them inflates the IQR, which pushes the upper
fence out, which hides genuinely slow electrical jobs behind the drainage tail — and
simultaneously flags ordinary drainage work as "extreme" when it is merely typical
for drainage.

So we compute fences **twice**:

* ``detect_overall()`` — one global set of fences. Useful as a headline, honest about
  what it is: a mixture distribution.
* ``detect_by_group()`` — Tukey fences recomputed **inside each category**, which is
  the comparison an operations manager actually needs ("is this drainage job slow
  *for drainage*?"). Groups with fewer than ``min_group_n`` resolved cases get a
  sample warning instead of fences, because an IQR from 4 points is noise.

We also chose Tukey fences over a z-score / ±3σ rule deliberately: the mean and the
standard deviation are themselves dragged by the outliers we are hunting, whereas the
quartiles are not. For right-skewed service-time data, the IQR rule is the robust
choice.

Output is an **actionable list**, not a count: every flagged complaint carries its
reference code, category, current status and how far past the fence it sits, so the
dashboard can link straight to the case and someone can go and unblock it.
"""

from __future__ import annotations

from typing import Any

from app.analytics.descriptive import DescriptiveStats, _round
from app.analytics.distributions import label_for

# Below this many resolved cases in a group, quartiles are too unstable to fence on.
MIN_GROUP_N = 8
# Hard cap on how many outliers we serialise per scope — dashboards need a worklist,
# not a data dump.
MAX_OUTLIERS_RETURNED = 25

METHOD_LABEL = "Tukey fences (Q1 - 1.5*IQR, Q3 + 1.5*IQR)"


def _hours_to_days(hours: float | None) -> float | None:
    if hours is None:
        return None
    return round(hours / 24.0, 2)


class OutlierDetector:
    """Flag abnormal resolution times using robust IQR fences.

    Parameters
    ----------
    frame:
        DataFrame with at least ``value_col``. Optional identity columns
        (``reference_code``, ``id``, ``category``, ``priority``, ``status``,
        ``area``, ``department``, ``created_at``) are surfaced when present so the
        result is actionable.
    value_col:
        Numeric column to fence on. Defaults to ``resolution_hours``.
    group_col:
        Column defining the per-group fences. Defaults to ``category``.
    """

    def __init__(
        self,
        frame,
        *,
        value_col: str = "resolution_hours",
        group_col: str = "category",
        unit: str = "hours",
        min_group_n: int = MIN_GROUP_N,
        max_returned: int = MAX_OUTLIERS_RETURNED,
    ) -> None:
        self.frame = frame
        self.value_col = value_col
        self.group_col = group_col
        self.unit = unit
        self.min_group_n = min_group_n
        self.max_returned = max_returned

    # ------------------------------------------------------------------ utils
    def _usable(self):
        """Rows with a finite value in ``value_col``."""
        import pandas as pd

        if self.frame is None or self.value_col not in getattr(self.frame, "columns", []):
            return pd.DataFrame()
        sub = self.frame[self.frame[self.value_col].notna()]
        return sub

    def _point(self, row, *, fence: float, side: str, group: str | None = None) -> dict[str, Any]:
        value = float(row[self.value_col])
        exceeds = value - fence if side == "upper" else fence - value
        verdict = (
            "abnormally slow — investigate"
            if side == "upper"
            else "abnormally fast — verify the resolution was genuine"
        )
        created = row.get("created_at")
        raw_id = row.get("id")
        return {
            "reference_code": str(row.get("reference_code") or raw_id or "unknown"),
            "value": _round(value, 2),
            "id": str(raw_id) if raw_id is not None else None,
            "category": row.get("category") or group,
            "priority": row.get("priority"),
            "status": row.get("status"),
            "area": row.get("area"),
            "department": row.get("department"),
            "value_days": _hours_to_days(value),
            "fence": _round(fence, 2),
            "exceeds_fence_by": _round(abs(exceeds), 2),
            "side": side,
            "verdict": verdict,
            "created_at": created if created is not None and str(created) != "NaT" else None,
        }

    # ---------------------------------------------------------------- overall
    def detect_overall(self) -> dict[str, Any]:
        """Global Tukey fences across every resolved complaint in the frame."""
        sub = self._usable()
        n = int(len(sub))
        if n == 0:
            return {
                "method": METHOD_LABEL,
                "scope": "overall",
                "n": 0,
                "outliers": [],
                "interpretation": (
                    "No complaint in this filter set has been resolved yet, so there is no "
                    "resolution time to test for outliers."
                ),
            }

        stats = DescriptiveStats(sub[self.value_col], unit=self.unit)
        lower, upper = stats.lower_fence, stats.upper_fence
        points: list[dict[str, Any]] = []
        upper_count = lower_count = 0

        if upper is not None:
            high = sub[sub[self.value_col] > upper]
            upper_count = int(len(high))
            for _idx, row in high.sort_values(self.value_col, ascending=False).head(
                self.max_returned
            ).iterrows():
                points.append(self._point(row, fence=upper, side="upper"))
        if lower is not None and lower > 0:
            # A negative lower fence can never be crossed by a duration, so we only
            # report low outliers when the fence is actually reachable.
            low = sub[sub[self.value_col] < lower]
            lower_count = int(len(low))
            for _idx, row in low.sort_values(self.value_col).head(5).iterrows():
                points.append(self._point(row, fence=lower, side="lower"))

        total_out = upper_count + lower_count
        rate = total_out / n if n else 0.0
        return {
            "method": METHOD_LABEL,
            "scope": "overall",
            "n": n,
            "lower_fence": _round(lower, 2),
            "upper_fence": _round(upper, 2),
            "outlier_count": total_out,
            "outlier_rate": round(rate, 4),
            "upper_count": upper_count,
            "lower_count": lower_count,
            "outliers": points,
            "interpretation": self._overall_interpretation(
                n=n, upper=upper, lower=lower, upper_count=upper_count, lower_count=lower_count
            ),
        }

    def _overall_interpretation(
        self,
        *,
        n: int,
        upper: float | None,
        lower: float | None,
        upper_count: int,
        lower_count: int,
    ) -> str:
        if upper is None:
            return f"Only {n} resolved complaints — not enough to compute stable outlier fences."
        upper_days = upper / 24.0
        head = (
            f"Across all {n} resolved complaints the Tukey upper fence sits at {upper:.0f} hours "
            f"({upper_days:.1f} days): anything slower than that is beyond the normal range of this "
            "process, not just 'a bit late'."
        )
        if upper_count == 0:
            body = " No complaint currently exceeds it, so resolution times have no extreme tail right now."
        else:
            pct = upper_count / n * 100
            body = (
                f" {upper_count} complaints ({pct:.1f}%) exceed it and are listed individually below "
                "so they can be chased by reference code."
            )
        tail = ""
        if lower is not None and lower < 0:
            tail = (
                f" The lower fence is negative ({lower:.0f} hours), which is expected for a duration: "
                "it simply means no complaint can be an outlier for being resolved too quickly under "
                "this rule."
            )
        elif lower_count:
            tail = (
                f" {lower_count} complaints closed faster than the lower fence of {lower:.0f} hours — "
                "worth spot-checking that those closures were genuine."
            )
        return head + body + tail

    # --------------------------------------------------------------- by group
    def detect_by_group(self) -> list[dict[str, Any]]:
        """Recompute fences *inside* each group — the comparison that actually holds."""
        sub = self._usable()
        if len(sub) == 0 or self.group_col not in sub.columns:
            return []

        reports: list[dict[str, Any]] = []
        for group, chunk in sub.groupby(self.group_col, dropna=True):
            group_name = str(group)
            stats = DescriptiveStats(chunk[self.value_col], unit=self.unit)
            n = stats.n
            report: dict[str, Any] = {
                "group": group_name,
                "n": n,
                "median": _round(stats.median, 2),
                "q1": _round(stats.q1, 2),
                "q3": _round(stats.q3, 2),
                "iqr": _round(stats.iqr, 2),
                "lower_fence": _round(stats.lower_fence, 2),
                "upper_fence": _round(stats.upper_fence, 2),
                "outlier_count": 0,
                "outlier_rate": 0.0,
                "outliers": [],
                "sample_warning": stats.sample_warning,
            }

            if n < self.min_group_n:
                report["lower_fence"] = None
                report["upper_fence"] = None
                report["interpretation"] = (
                    f"{label_for(self.group_col, group_name)} has only {n} resolved complaints — "
                    f"below the {self.min_group_n} needed for a stable IQR, so we do not publish "
                    "outlier fences for it rather than flagging cases on the basis of noise."
                )
                reports.append(report)
                continue

            upper = stats.upper_fence
            points: list[dict[str, Any]] = []
            if upper is not None:
                high = chunk[chunk[self.value_col] > upper]
                report["outlier_count"] = int(len(high))
                report["outlier_rate"] = round(len(high) / n, 4)
                for _idx, row in high.sort_values(self.value_col, ascending=False).head(
                    self.max_returned
                ).iterrows():
                    points.append(self._point(row, fence=upper, side="upper", group=group_name))
            report["outliers"] = points
            report["interpretation"] = self._group_interpretation(
                group=group_name,
                n=n,
                median=stats.median,
                upper=upper,
                count=report["outlier_count"],
            )
            reports.append(report)

        reports.sort(key=lambda r: (-r["outlier_count"], -r["n"]))
        return reports

    @staticmethod
    def _group_interpretation(
        *, group: str, n: int, median: float | None, upper: float | None, count: int
    ) -> str:
        name = label_for("category", group)
        if upper is None or median is None:
            return f"{name}: not enough resolved cases to fence."
        if count == 0:
            return (
                f"{name} resolves in a median {median / 24:.1f} days and nothing exceeds its own "
                f"{upper / 24:.1f}-day fence — this category has no abnormal cases."
            )
        return (
            f"{name} resolves in a median {median / 24:.1f} days; {count} of its {n} resolved "
            f"complaints took longer than {upper / 24:.1f} days, which is abnormal *for this "
            "category* even where it would look ordinary against the citywide average."
        )

    # ----------------------------------------------------------------- report
    def report(self) -> dict[str, Any]:
        """Combined overall + per-group report ready for the wire."""
        overall = self.detect_overall()
        groups = self.detect_by_group()
        overall["by_group"] = groups
        if groups:
            flagged = sum(g["outlier_count"] for g in groups)
            fenced = [g for g in groups if g.get("upper_fence") is not None]
            if fenced:
                overall["interpretation"] += (
                    f" Fences were also recomputed within each {self.group_col}: "
                    f"{flagged} complaints are outliers relative to their own {self.group_col}'s "
                    "normal range, which is the fairer comparison because a 3-day drainage repair "
                    "and a 3-day pothole repair are not the same event."
                )
        return overall

    def to_model(self):
        from app.schemas.analytics import OutlierReport

        return OutlierReport(**self.report())


__all__ = ["MAX_OUTLIERS_RETURNED", "MIN_GROUP_N", "METHOD_LABEL", "OutlierDetector"]

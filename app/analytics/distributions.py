"""Frequency distributions and contingency tables for the categorical variables.

Covers the categorical half of the descriptive-statistics requirement: for
``category``, ``priority``, ``status``, ``department`` and ``area`` we produce a
proper frequency distribution — absolute frequency, **relative frequency**,
cumulative frequency and cumulative percentage — plus the **mode**, and a
cross-tabulation of category × priority as a real contingency table with row totals,
column totals and a grand total.

Conventions
-----------
* Rows are ordered by descending count (so the mode is row 0) unless a canonical
  order is supplied — priority and status have a natural ordinal sequence
  (low→critical, open→rejected) and shuffling them by frequency would destroy
  information, so those keep their contract order.
* Relative frequency is a proportion in 0..1; ``percent`` is the same number ×100.
  Both are emitted because dashboards want the percentage and downstream maths
  wants the proportion.
* Missing/blank values are counted separately as ``missing`` rather than being
  silently bucketed into a category, which would inflate that category's share.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

# ------------------------------------------------------------------ wire enums
CATEGORY_ORDER: tuple[str, ...] = (
    "road",
    "water",
    "waste",
    "electricity",
    "drainage",
    "safety",
    "other",
)
PRIORITY_ORDER: tuple[str, ...] = ("low", "medium", "high", "critical")
STATUS_ORDER: tuple[str, ...] = ("open", "assigned", "in_progress", "resolved", "rejected")

CATEGORY_LABELS: dict[str, str] = {
    "road": "Roads & Potholes",
    "water": "Water Supply & Leakage",
    "waste": "Waste & Sanitation",
    "electricity": "Electricity & Streetlights",
    "drainage": "Drainage & Sewerage",
    "safety": "Public Safety",
    "other": "Other",
}
PRIORITY_LABELS: dict[str, str] = {
    "low": "Low",
    "medium": "Medium",
    "high": "High",
    "critical": "Critical",
}
STATUS_LABELS: dict[str, str] = {
    "open": "Open",
    "assigned": "Assigned",
    "in_progress": "In Progress",
    "resolved": "Resolved",
    "rejected": "Rejected",
}

# Priority as an ordinal rank, used for the Spearman correlation in inference.py.
PRIORITY_RANK: dict[str, int] = {"low": 1, "medium": 2, "high": 3, "critical": 4}

# Statuses that mean "still costing the city money".
OPEN_STATUSES: tuple[str, ...] = ("open", "assigned", "in_progress")

MISSING_LABEL = "unspecified"


def label_for(variable: str, value: str) -> str:
    """Human display label for a wire enum value."""
    if variable == "category":
        return CATEGORY_LABELS.get(value, value.replace("_", " ").title())
    if variable == "priority":
        return PRIORITY_LABELS.get(value, value.replace("_", " ").title())
    if variable == "status":
        return STATUS_LABELS.get(value, value.replace("_", " ").title())
    return value if value else MISSING_LABEL


def canonical_order(variable: str) -> tuple[str, ...] | None:
    """Return the contract order for ordinal variables, ``None`` for nominal ones."""
    return {
        "priority": PRIORITY_ORDER,
        "status": STATUS_ORDER,
        "category": None,  # nominal: order by frequency so the mode leads
    }.get(variable)


def _normalise(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text.lower() in {"", "none", "nan", "null"}:
        return ""
    return text


class FrequencyDistribution:
    """Frequency / relative-frequency / cumulative distribution of one categorical.

    A class rather than a function because the mode, the cumulative column and the
    interpretation sentence all need the same sorted counts; computing them together
    guarantees the narrative and the table can never disagree.
    """

    def __init__(
        self,
        values: Iterable[Any] | None,
        *,
        variable: str,
        order: Sequence[str] | None = None,
        top_n: int | None = None,
    ) -> None:
        self.variable = variable
        self.top_n = top_n
        self._explicit_order = tuple(order) if order is not None else canonical_order(variable)

        cleaned: list[str] = []
        missing = 0
        for raw in values or []:
            text = _normalise(raw)
            if not text:
                missing += 1
            else:
                cleaned.append(text)
        self._values = cleaned
        self.missing = missing

        counts: dict[str, int] = {}
        for item in cleaned:
            counts[item] = counts.get(item, 0) + 1
        self.counts = counts

    # ------------------------------------------------------------------ basics
    @property
    def n(self) -> int:
        """Number of non-missing observations (the denominator for shares)."""
        return len(self._values)

    @property
    def distinct(self) -> int:
        return len(self.counts)

    def ordered_items(self) -> list[tuple[str, int]]:
        """(value, count) pairs in display order."""
        if self._explicit_order:
            known = [(v, self.counts.get(v, 0)) for v in self._explicit_order]
            extra = sorted(
                ((v, c) for v, c in self.counts.items() if v not in self._explicit_order),
                key=lambda kv: (-kv[1], kv[0]),
            )
            items = known + extra
        else:
            items = sorted(self.counts.items(), key=lambda kv: (-kv[1], kv[0]))
        if self.top_n:
            items = items[: self.top_n]
        return items

    # -------------------------------------------------------------------- mode
    def modes(self) -> tuple[list[str], str, int]:
        """``(modes, kind, max_count)`` — ties reported rather than hidden."""
        if not self.counts:
            return [], "none", 0
        max_count = max(self.counts.values())
        if max_count == 0:
            return [], "none", 0
        winners = sorted(v for v, c in self.counts.items() if c == max_count)
        kind = "unique" if len(winners) == 1 else "multi"
        return winners, kind, max_count

    @property
    def mode(self) -> str | None:
        winners, kind, _ = self.modes()
        return winners[0] if winners and kind != "none" else None

    @property
    def mode_share(self) -> float:
        _winners, _kind, max_count = self.modes()
        return (max_count / self.n) if self.n else 0.0

    # ---------------------------------------------------------------- exports
    def rows(self) -> list[dict[str, Any]]:
        """Frequency table rows including cumulative frequency and percentage."""
        total = self.n
        cumulative = 0
        out: list[dict[str, Any]] = []
        for value, count in self.ordered_items():
            cumulative += count
            rel = (count / total) if total else 0.0
            out.append(
                {
                    "value": value,
                    "label": label_for(self.variable, value),
                    "count": int(count),
                    "relative_frequency": round(rel, 4),
                    "percent": round(rel * 100, 2),
                    "cumulative_count": int(cumulative),
                    "cumulative_percent": round((cumulative / total * 100) if total else 0.0, 2),
                }
            )
        return out

    def interpretation(self) -> str:
        """Plain-English reading of this distribution."""
        if self.n == 0:
            return f"No complaints matched these filters, so there is no {self.variable} distribution to report."
        winners, kind, max_count = self.modes()
        share = self.mode_share * 100
        if kind == "none":
            return (
                f"All {self.n} complaints have a distinct {self.variable} value, so there is no modal "
                f"{self.variable}."
            )
        if kind == "multi":
            names = " and ".join(label_for(self.variable, w) for w in winners[:3])
            return (
                f"{names} tie as the most common {self.variable}, with {max_count} complaints each "
                f"({share:.1f}% of {self.n}). A tied mode means no single {self.variable} dominates."
            )
        top_label = label_for(self.variable, winners[0])
        one_in = (1 / self.mode_share) if self.mode_share else 0
        return (
            f"{top_label} is the modal {self.variable}: {max_count} of {self.n} complaints "
            f"({share:.1f}%, roughly 1 in {one_in:.1f}). "
            f"The {self.distinct} distinct values are spread across the {self.variable} axis."
        )

    def to_dict(self) -> dict[str, Any]:
        winners, kind, max_count = self.modes()
        return {
            "variable": self.variable,
            "n": self.n,
            "distinct": self.distinct,
            "rows": self.rows(),
            "mode": winners[0] if winners else None,
            "mode_label": label_for(self.variable, winners[0]) if winners else None,
            "modes": winners[:5],
            "mode_kind": kind,
            "mode_count": int(max_count),
            "mode_share": round(self.mode_share, 4),
            "missing": self.missing,
            "interpretation": self.interpretation(),
        }

    def to_model(self):
        from app.schemas.analytics import FrequencyDistributionModel

        return FrequencyDistributionModel(**self.to_dict())


class ContingencyTable:
    """Two-way cross-tabulation with margins — the input to the chi-square test.

    Built from two parallel sequences rather than a DataFrame so it can be unit
    tested without pandas, and so the same object can be handed straight to
    ``inference.ChiSquareIndependenceTest``.
    """

    def __init__(
        self,
        row_values: Iterable[Any],
        col_values: Iterable[Any],
        *,
        row_variable: str = "category",
        col_variable: str = "priority",
        row_order: Sequence[str] | None = None,
        col_order: Sequence[str] | None = None,
    ) -> None:
        self.row_variable = row_variable
        self.col_variable = col_variable

        pairs: list[tuple[str, str]] = []
        for r, c in zip(row_values, col_values, strict=False):
            rr, cc = _normalise(r), _normalise(c)
            if rr and cc:
                pairs.append((rr, cc))
        self._pairs = pairs

        cells: dict[tuple[str, str], int] = {}
        for pair in pairs:
            cells[pair] = cells.get(pair, 0) + 1
        self.cells = cells

        observed_rows = {r for r, _ in pairs}
        observed_cols = {c for _, c in pairs}

        base_rows = row_order if row_order is not None else canonical_order(row_variable)
        base_cols = col_order if col_order is not None else canonical_order(col_variable)

        self.row_labels = self._build_axis(base_rows, observed_rows, row_variable)
        self.col_labels = self._build_axis(base_cols, observed_cols, col_variable)

    @staticmethod
    def _build_axis(
        base: Sequence[str] | None, observed: set[str], variable: str
    ) -> list[str]:
        if base:
            ordered = [v for v in base if v in observed]
            extra = sorted(observed - set(base))
            return ordered + extra
        if variable == "category":  # keep contract order for readability
            ordered = [v for v in CATEGORY_ORDER if v in observed]
            return ordered + sorted(observed - set(CATEGORY_ORDER))
        return sorted(observed)

    # ---------------------------------------------------------------- margins
    def count(self, row: str, col: str) -> int:
        return int(self.cells.get((row, col), 0))

    def matrix(self) -> list[list[int]]:
        """Observed counts as a dense 2-D list, rows × cols."""
        return [[self.count(r, c) for c in self.col_labels] for r in self.row_labels]

    def row_totals(self) -> dict[str, int]:
        return {r: sum(self.count(r, c) for c in self.col_labels) for r in self.row_labels}

    def col_totals(self) -> dict[str, int]:
        return {c: sum(self.count(r, c) for r in self.row_labels) for c in self.col_labels}

    @property
    def grand_total(self) -> int:
        return len(self._pairs)

    # ---------------------------------------------------------------- exports
    def to_dict(self) -> dict[str, Any]:
        row_totals = self.row_totals()
        rows = [
            {
                "label": label_for(self.row_variable, r),
                "value": r,
                "cells": {c: self.count(r, c) for c in self.col_labels},
                "total": row_totals[r],
            }
            for r in self.row_labels
        ]
        row_pct = []
        for r in self.row_labels:
            total = row_totals[r] or 1
            row_pct.append({c: round(self.count(r, c) / total * 100, 2) for c in self.col_labels})
        return {
            "row_variable": self.row_variable,
            "col_variable": self.col_variable,
            "col_labels": list(self.col_labels),
            "rows": rows,
            "col_totals": self.col_totals(),
            "grand_total": self.grand_total,
            "row_percent_cells": row_pct,
            "interpretation": self.interpretation(),
        }

    def to_model(self):
        from app.schemas.analytics import ContingencyTableModel

        return ContingencyTableModel(**self.to_dict())

    def interpretation(self) -> str:
        """Describe the strongest row×column concentration in words."""
        if self.grand_total == 0:
            return (
                f"No complaints have both a {self.row_variable} and a {self.col_variable}, "
                "so the cross-tabulation is empty."
            )
        row_totals = self.row_totals()
        best_cell: tuple[str, str, float, int] | None = None
        for r in self.row_labels:
            total = row_totals.get(r, 0)
            if total < 5:  # tiny rows produce meaningless percentages
                continue
            for c in self.col_labels:
                share = self.count(r, c) / total
                if best_cell is None or share > best_cell[2]:
                    best_cell = (r, c, share, self.count(r, c))
        base = (
            f"The table cross-tabulates {len(self.row_labels)} {self.row_variable} values against "
            f"{len(self.col_labels)} {self.col_variable} values over {self.grand_total} complaints."
        )
        if best_cell is None:
            return base + " Every row is too small for a stable percentage breakdown."
        r, c, share, count = best_cell
        return (
            f"{base} The strongest concentration is "
            f"{label_for(self.row_variable, r)}: {share * 100:.0f}% of its {row_totals[r]} complaints "
            f"are {label_for(self.col_variable, c)} ({count} cases)."
        )


def distribution_from_frame(frame, column: str, *, variable: str | None = None, top_n: int | None = None):
    """Convenience: build a :class:`FrequencyDistribution` from a DataFrame column."""
    variable = variable or column
    if frame is None or column not in getattr(frame, "columns", []):
        return FrequencyDistribution([], variable=variable, top_n=top_n)
    return FrequencyDistribution(frame[column].tolist(), variable=variable, top_n=top_n)


__all__ = [
    "CATEGORY_LABELS",
    "CATEGORY_ORDER",
    "MISSING_LABEL",
    "OPEN_STATUSES",
    "PRIORITY_LABELS",
    "PRIORITY_ORDER",
    "PRIORITY_RANK",
    "STATUS_LABELS",
    "STATUS_ORDER",
    "ContingencyTable",
    "FrequencyDistribution",
    "canonical_order",
    "distribution_from_frame",
    "label_for",
]

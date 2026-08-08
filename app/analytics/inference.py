"""Inferential statistics: chi-square test of independence and Spearman correlation.

Knowing when a test is invalid is the point
-------------------------------------------
A chi-square test of independence is only trustworthy when its **expected-frequency
assumption** holds: the usual rule is that every expected cell count is at least 5
(a common relaxation allows up to 20% of cells between 1 and 5, but *no* expected
count below 1). With seven categories and four priority levels there are 28 cells, so
a few hundred complaints can easily leave the rare combinations — critical safety
complaints, say — with an expected count of 2 or 3. The chi-square statistic is an
asymptotic approximation to a discrete distribution; when the cells are that thin the
approximation breaks down and the p-value is simply wrong, usually too small.

This module therefore always computes the expected table, counts how many cells fall
below 5, records the minimum expected count, and sets ``reliable=False`` with an
explicit ``caveat`` when the assumption fails. The statistic is still returned so
nothing is hidden, but the interpretation says out loud that it should not be acted
on, and suggests the correct remedies (collapse sparse categories, or use Fisher's
exact test / a Monte-Carlo permutation version).

Two further deliberate choices:

* **Yates' continuity correction** is applied only for 2×2 tables (that is the only
  case where it is defined and where scipy applies it by default) and the response
  reports ``correction_applied`` so nobody has to guess which statistic they got.
* **Effect size is reported alongside significance.** With a large n, a trivially
  small association becomes "statistically significant"; p-value alone would let us
  claim a discovery that means nothing operationally. Cramér's V is graded against
  Cohen's thresholds, which depend on the table's degrees of freedom.
"""

from __future__ import annotations

import math
from typing import Any

from app.analytics.distributions import PRIORITY_RANK, ContingencyTable, label_for

ALPHA = 0.05
MIN_EXPECTED = 5.0

# Cohen's Cramér's V thresholds, indexed by df* = min(rows-1, cols-1).
_EFFECT_THRESHOLDS: dict[int, tuple[float, float, float]] = {
    1: (0.10, 0.30, 0.50),
    2: (0.07, 0.21, 0.35),
    3: (0.06, 0.17, 0.29),
    4: (0.05, 0.15, 0.25),
}
_EFFECT_DEFAULT = (0.05, 0.13, 0.22)


def _effect_label(v: float | None, df_star: int) -> str | None:
    if v is None or not math.isfinite(v):
        return None
    small, medium, large = _EFFECT_THRESHOLDS.get(max(1, df_star), _EFFECT_DEFAULT)
    if v < small:
        return "negligible"
    if v < medium:
        return "small"
    if v < large:
        return "medium"
    return "large"


def _format_p(p: float | None) -> str:
    if p is None or not math.isfinite(p):
        return "n/a"
    if p < 0.001:
        return "p < 0.001"
    return f"p = {p:.3f}"


class ChiSquareIndependenceTest:
    """Pearson chi-square test of independence between two categorical variables.

    Accepts either a :class:`~app.analytics.distributions.ContingencyTable` or a raw
    matrix plus labels, so it is unit-testable against a hand-computed table with no
    database and no DataFrame involved.
    """

    def __init__(
        self,
        table: ContingencyTable | None = None,
        *,
        matrix: list[list[int]] | None = None,
        row_labels: list[str] | None = None,
        col_labels: list[str] | None = None,
        row_variable: str = "category",
        col_variable: str = "priority",
        alpha: float = ALPHA,
    ) -> None:
        if table is not None:
            self.matrix = table.matrix()
            self.row_labels = list(table.row_labels)
            self.col_labels = list(table.col_labels)
            self.row_variable = table.row_variable
            self.col_variable = table.col_variable
        else:
            self.matrix = [list(map(int, row)) for row in (matrix or [])]
            self.row_labels = list(row_labels or [f"r{i}" for i in range(len(self.matrix))])
            self.col_labels = list(
                col_labels or [f"c{j}" for j in range(len(self.matrix[0]) if self.matrix else 0)]
            )
            self.row_variable = row_variable
            self.col_variable = col_variable
        self.alpha = alpha

    # ------------------------------------------------------------------ setup
    def _clean_matrix(self) -> tuple[list[list[int]], list[str], list[str]]:
        """Drop all-zero rows/columns — a zero margin makes the test undefined."""
        rows = [
            (label, row)
            for label, row in zip(self.row_labels, self.matrix, strict=False)
            if sum(row) > 0
        ]
        if not rows:
            return [], [], []
        keep_cols = [
            j for j in range(len(rows[0][1])) if sum(row[j] for _label, row in rows) > 0
        ]
        cleaned = [[row[j] for j in keep_cols] for _label, row in rows]
        return cleaned, [label for label, _ in rows], [self.col_labels[j] for j in keep_cols]

    def _empty(self, reason: str) -> dict[str, Any]:
        return {
            "test": "chi-square test of independence",
            "row_variable": self.row_variable,
            "col_variable": self.col_variable,
            "n": 0,
            "statistic": None,
            "dof": None,
            "p_value": None,
            "alpha": self.alpha,
            "significant": None,
            "cramers_v": None,
            "effect_size": None,
            "expected_min": None,
            "cells_below_5": 0,
            "total_cells": 0,
            "pct_cells_below_5": 0.0,
            "assumption_met": False,
            "reliable": False,
            "correction_applied": False,
            "h0": self._h0(),
            "h1": self._h1(),
            "interpretation": reason,
            "caveat": reason,
        }

    def _h0(self) -> str:
        return (
            f"H0: {self.row_variable} and {self.col_variable} are independent — knowing a "
            f"complaint's {self.row_variable} tells you nothing about its {self.col_variable}."
        )

    def _h1(self) -> str:
        return (
            f"H1: {self.row_variable} and {self.col_variable} are associated — the "
            f"{self.col_variable} mix differs between {self.row_variable} values."
        )

    # -------------------------------------------------------------------- run
    def run(self) -> dict[str, Any]:
        matrix, row_labels, col_labels = self._clean_matrix()
        if len(matrix) < 2 or len(col_labels) < 2:
            return self._empty(
                "The cross-tabulation has fewer than two non-empty rows or columns, so there is "
                "nothing to test for independence."
            )

        try:
            import numpy as np
            from scipy import stats as sps
        except ImportError:  # pragma: no cover - scipy is a declared dependency
            return self._empty("scipy is unavailable, so the chi-square test could not be run.")

        observed = np.asarray(matrix, dtype="float64")
        n = int(observed.sum())
        if n == 0:
            return self._empty("The contingency table is empty.")

        rows, cols = observed.shape
        correction = rows == 2 and cols == 2  # Yates only defined for 2x2
        statistic, p_value, dof, expected = sps.chi2_contingency(
            observed, correction=correction
        )

        total_cells = int(expected.size)
        cells_below = int((expected < MIN_EXPECTED).sum())
        expected_min = float(expected.min())
        pct_below = cells_below / total_cells * 100 if total_cells else 0.0
        # Standard rule: all expected >= 5. Relaxed rule (Cochran): <=20% of cells
        # between 1 and 5 and none below 1. We report both verdicts.
        assumption_met = cells_below == 0
        relaxed_met = pct_below <= 20.0 and expected_min >= 1.0
        reliable = assumption_met or relaxed_met

        df_star = min(rows - 1, cols - 1)
        cramers_v = math.sqrt(statistic / (n * df_star)) if n and df_star else None
        effect = _effect_label(cramers_v, df_star)
        significant = bool(p_value < self.alpha)

        result = {
            "test": "chi-square test of independence",
            "row_variable": self.row_variable,
            "col_variable": self.col_variable,
            "n": n,
            "statistic": round(float(statistic), 4),
            "dof": int(dof),
            "p_value": float(p_value),
            "alpha": self.alpha,
            "significant": significant,
            "cramers_v": round(float(cramers_v), 4) if cramers_v is not None else None,
            "effect_size": effect,
            "expected_min": round(expected_min, 3),
            "cells_below_5": cells_below,
            "total_cells": total_cells,
            "pct_cells_below_5": round(pct_below, 1),
            "assumption_met": assumption_met,
            "reliable": reliable,
            "correction_applied": bool(correction),
            "h0": self._h0(),
            "h1": self._h1(),
        }
        result["interpretation"] = self._interpret(result, relaxed_met=relaxed_met)
        result["caveat"] = self._caveat(result, relaxed_met=relaxed_met)
        return result

    # --------------------------------------------------------------- narrate
    def _interpret(self, r: dict[str, Any], *, relaxed_met: bool) -> str:
        row_v, col_v = r["row_variable"], r["col_variable"]
        stat, dof, p = r["statistic"], r["dof"], r["p_value"]
        head = (
            f"Chi-square test of independence between {row_v} and {col_v}: "
            f"χ²({dof}, N={r['n']}) = {stat:.2f}, {_format_p(p)}."
        )

        if not r["reliable"]:
            return (
                f"{head} This result is NOT reliable: {r['cells_below_5']} of {r['total_cells']} "
                f"cells have an expected count below 5 (smallest expected = {r['expected_min']}), "
                "which violates the assumption behind the chi-square approximation. We are "
                "reporting the number for transparency but you should not conclude anything from "
                "it until the sparse categories are merged or more data is collected."
            )

        if r["significant"]:
            verdict = (
                f" We reject the null hypothesis of independence at α = {r['alpha']}: the "
                f"{col_v} mix genuinely differs between {row_v} values — it is not random "
                "variation."
            )
        else:
            verdict = (
                f" We fail to reject the null hypothesis at α = {r['alpha']}: the {col_v} "
                f"breakdown looks statistically indistinguishable across {row_v} values, so "
                f"{col_v} appears to be assigned independently of {row_v}."
            )

        effect = ""
        if r["cramers_v"] is not None:
            effect = (
                f" Cramér's V = {r['cramers_v']:.3f}, a {r['effect_size']} association — "
                "significance says the pattern is real, effect size says how much it matters, "
                "and with a large sample the two can easily disagree."
            )

        if r["assumption_met"]:
            assumption = (
                " All expected cell counts are at least 5, so the chi-square approximation is valid."
            )
        elif relaxed_met:
            assumption = (
                f" {r['cells_below_5']} of {r['total_cells']} cells ({r['pct_cells_below_5']}%) have "
                "an expected count below 5 but none below 1, which satisfies Cochran's relaxed rule, "
                "so the result is usable with mild caution."
            )
        else:
            assumption = ""
        correction = (
            " Yates' continuity correction was applied because the table is 2×2."
            if r["correction_applied"]
            else ""
        )
        return head + verdict + effect + assumption + correction

    @staticmethod
    def _caveat(r: dict[str, Any], *, relaxed_met: bool) -> str | None:
        if r["reliable"] and r["assumption_met"]:
            return None
        if not r["reliable"]:
            return (
                f"Expected-frequency assumption violated: {r['cells_below_5']}/{r['total_cells']} "
                f"cells below 5 (min {r['expected_min']}). Merge sparse categories or use Fisher's "
                "exact / a Monte-Carlo permutation test instead."
            )
        return (
            f"{r['cells_below_5']}/{r['total_cells']} expected cells are below 5 but none below 1 — "
            "acceptable under Cochran's relaxed rule, interpret with mild caution."
        )

    def to_model(self):
        from app.schemas.analytics import ChiSquareResultModel

        return ChiSquareResultModel(**self.run())


class SpearmanCorrelation:
    """Spearman rank correlation — the right tool when the relationship is monotonic.

    Used here for questions like "do higher-priority complaints actually get resolved
    faster?". Spearman rather than Pearson because resolution time is heavily
    right-skewed and priority is ordinal, not interval: ranks are meaningful, the
    numeric gap between 'high' and 'critical' is not, and a handful of month-long
    cases would dominate a Pearson coefficient.
    """

    def __init__(
        self,
        x,
        y,
        *,
        x_variable: str = "x",
        y_variable: str = "y",
        alpha: float = ALPHA,
        min_n: int = 10,
    ) -> None:
        self.x_raw = x
        self.y_raw = y
        self.x_variable = x_variable
        self.y_variable = y_variable
        self.alpha = alpha
        self.min_n = min_n

    @staticmethod
    def _strength(rho: float) -> str:
        a = abs(rho)
        if a < 0.1:
            return "negligible"
        if a < 0.3:
            return "weak"
        if a < 0.5:
            return "moderate"
        if a < 0.7:
            return "strong"
        return "very strong"

    def run(self) -> dict[str, Any]:
        base = {
            "test": "Spearman rank correlation",
            "x_variable": self.x_variable,
            "y_variable": self.y_variable,
            "n": 0,
            "rho": None,
            "p_value": None,
            "alpha": self.alpha,
            "significant": None,
            "strength": None,
            "direction": None,
            "reliable": False,
            "interpretation": "",
            "caveat": None,
        }
        try:
            import numpy as np
            from scipy import stats as sps
        except ImportError:  # pragma: no cover
            base["interpretation"] = "scipy is unavailable, so the correlation was not computed."
            base["caveat"] = base["interpretation"]
            return base

        x = np.asarray(list(self.x_raw), dtype="float64")
        y = np.asarray(list(self.y_raw), dtype="float64")
        if x.size != y.size:
            size = min(x.size, y.size)
            x, y = x[:size], y[:size]
        mask = np.isfinite(x) & np.isfinite(y)
        x, y = x[mask], y[mask]
        n = int(x.size)
        base["n"] = n

        if n < 3 or np.unique(x).size < 2 or np.unique(y).size < 2:
            base["interpretation"] = (
                f"Only {n} paired observations with usable variation — too few to estimate a "
                "correlation between "
                f"{self.x_variable} and {self.y_variable}."
            )
            base["caveat"] = "Insufficient data for a correlation."
            return base

        rho, p_value = sps.spearmanr(x, y)
        rho = float(rho)
        p_value = float(p_value)
        reliable = n >= self.min_n
        direction = "positive" if rho > 0 else ("negative" if rho < 0 else "none")
        strength = self._strength(rho)
        significant = bool(p_value < self.alpha)

        base.update(
            {
                "rho": round(rho, 4),
                "p_value": p_value,
                "significant": significant,
                "strength": strength,
                "direction": direction,
                "reliable": reliable,
            }
        )
        base["interpretation"] = self._interpret(
            rho=rho, p_value=p_value, n=n, strength=strength, direction=direction,
            significant=significant, reliable=reliable,
        )
        if not reliable:
            base["caveat"] = (
                f"Only {n} pairs (< {self.min_n}); the coefficient is unstable and could swing "
                "sharply with a few more observations."
            )
        return base

    def _interpret(
        self, *, rho: float, p_value: float, n: int, strength: str, direction: str,
        significant: bool, reliable: bool,
    ) -> str:
        head = (
            f"Spearman rank correlation between {self.x_variable} and {self.y_variable}: "
            f"ρ = {rho:.3f} over {n} pairs ({_format_p(p_value)}) — a {strength} {direction} "
            "monotonic relationship."
        )
        if not significant:
            return (
                f"{head} That is not statistically significant at α = {self.alpha}, so on this "
                "evidence the two move independently."
            )
        if not reliable:
            return f"{head} It is statistically significant, but the sample is small enough that it should be treated as a hint, not a finding."
        return (
            f"{head} It is statistically significant at α = {self.alpha}. Spearman was chosen over "
            "Pearson because the relationship only needs to be monotonic, not linear, and it is "
            "robust to the long right tail in resolution times."
        )

    def to_model(self):
        from app.schemas.analytics import SpearmanResultModel

        return SpearmanResultModel(**self.run())


# --------------------------------------------------------------------------- #
# Frame-level convenience wrappers used by the service layer
# --------------------------------------------------------------------------- #
def category_priority_chi_square(frame) -> dict[str, Any]:
    """Chi-square of category × priority straight from the complaint frame."""
    if frame is None or not {"category", "priority"} <= set(getattr(frame, "columns", [])):
        return ChiSquareIndependenceTest(matrix=[], row_labels=[], col_labels=[])._empty(
            "The complaint set does not carry both a category and a priority."
        )
    table = ContingencyTable(
        frame["category"].tolist(),
        frame["priority"].tolist(),
        row_variable="category",
        col_variable="priority",
    )
    return ChiSquareIndependenceTest(table).run()


def area_category_chi_square(frame, *, max_areas: int = 10) -> dict[str, Any]:
    """Chi-square of area × category, limited to the busiest areas.

    Restricting to the top areas is not cherry-picking: it is the standard remedy for
    the sparse-cell problem. Dozens of areas with two complaints each would guarantee
    expected counts below 1 and an invalid test. The trimming is reported in the
    interpretation.
    """
    cols = set(getattr(frame, "columns", []))
    if frame is None or not {"area", "category"} <= cols:
        return ChiSquareIndependenceTest(matrix=[], row_labels=[], col_labels=[])._empty(
            "The complaint set does not carry both an area and a category."
        )
    top = frame["area"].value_counts().head(max_areas).index.tolist()
    sub = frame[frame["area"].isin(top)]
    table = ContingencyTable(
        sub["area"].tolist(),
        sub["category"].tolist(),
        row_variable="area",
        col_variable="category",
    )
    result = ChiSquareIndependenceTest(table).run()
    if result.get("n"):
        result["interpretation"] += (
            f" The test was restricted to the {len(top)} busiest areas; including every area would "
            "leave most cells with an expected count below 1 and invalidate the test outright."
        )
    return result


def priority_vs_resolution_spearman(frame) -> dict[str, Any]:
    """Does higher priority actually mean faster resolution?

    Priority is mapped to an ordinal rank (low=1 … critical=4) and correlated against
    resolution hours. A **negative** ρ is the desired outcome: higher priority ⇒ fewer
    hours. A positive ρ would mean the triage is inverted, which is a finding worth
    shouting about.
    """
    cols = set(getattr(frame, "columns", []))
    if frame is None or not {"priority", "resolution_hours"} <= cols:
        return SpearmanCorrelation([], [], x_variable="priority rank", y_variable="resolution hours").run()
    sub = frame[frame["resolution_hours"].notna()]
    ranks = [PRIORITY_RANK.get(str(p), float("nan")) for p in sub["priority"].tolist()]
    result = SpearmanCorrelation(
        ranks,
        sub["resolution_hours"].tolist(),
        x_variable="priority rank (low=1 … critical=4)",
        y_variable="resolution time in hours",
    ).run()
    rho = result.get("rho")
    if rho is not None and result.get("significant"):
        if rho < 0:
            result["interpretation"] += (
                " The negative sign is the healthy direction: higher-priority complaints are "
                "genuinely being resolved faster, so triage is working."
            )
        else:
            result["interpretation"] += (
                " The positive sign is a red flag: higher-priority complaints are taking LONGER, "
                "which means the priority label is not driving the queue."
            )
    return result


def label_pair(row_variable: str, value: str) -> str:
    """Small helper re-exported for narrative code."""
    return label_for(row_variable, value)


__all__ = [
    "ALPHA",
    "MIN_EXPECTED",
    "ChiSquareIndependenceTest",
    "SpearmanCorrelation",
    "area_category_chi_square",
    "category_priority_chi_square",
    "priority_vs_resolution_spearman",
]

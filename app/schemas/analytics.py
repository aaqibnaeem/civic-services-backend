"""Wire schemas for the statistics & analytics engine.

These models are the *typed* contract for everything under ``/api/v1/analytics``.
They mirror ``docs/CONTRACT.md`` exactly; any field beyond the contract is additive
(the frontend can ignore it) and exists because the statistics benchmark asks us to
show our working — e.g. ``ddof``, ``quartile_method``, ``sample_warning``,
``assumption_met``.

Every response model carries either an ``interpretation`` string or an ``insights``
array. That is deliberate: the project spec says "explain what the statistics mean
rather than displaying numbers only", so a bare-number response is considered a bug.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Severity = Literal["info", "warn", "critical"]
ModeKind = Literal["unique", "multi", "none"]
Direction = Literal["up", "down", "flat"]


class _Base(BaseModel):
    model_config = ConfigDict(populate_by_name=True)


# --------------------------------------------------------------------------- #
# Insight — the "explain the statistics" deliverable (CONTRACT §3)
# --------------------------------------------------------------------------- #
class Insight(_Base):
    """One plain-English statement derived deterministically from computed numbers.

    Shape is frozen by the contract: ``{id, severity, title, detail, metric, unit}``.
    """

    id: str = Field(description="Stable rule id, e.g. 'resolution_skew'.")
    severity: Severity = "info"
    title: str = Field(description="Headline sentence — must stand alone if read aloud.")
    detail: str = Field(description="Two-to-three sentence explanation, numbers inline.")
    metric: float | None = None
    unit: str | None = None


class InsightsResponse(_Base):
    generated_at: datetime
    n: int
    insights: list[Insight]
    interpretation: str


# --------------------------------------------------------------------------- #
# Filters echoed back on every response
# --------------------------------------------------------------------------- #
class AnalyticsFilters(_Base):
    """Optional filter set accepted by every analytics endpoint."""

    date_from: date | None = None
    date_to: date | None = None
    category: str | None = None
    area: str | None = None

    def cache_key(self) -> tuple:
        return (
            self.date_from.isoformat() if self.date_from else None,
            self.date_to.isoformat() if self.date_to else None,
            self.category,
            self.area,
        )

    def describe(self) -> str:
        """Human phrase describing the active filters, for interpretation strings."""
        parts: list[str] = []
        if self.date_from:
            parts.append(f"from {self.date_from.isoformat()}")
        if self.date_to:
            parts.append(f"to {self.date_to.isoformat()}")
        if self.category:
            parts.append(f"category={self.category}")
        if self.area:
            parts.append(f"area={self.area}")
        return ", ".join(parts) if parts else "all complaints"


# --------------------------------------------------------------------------- #
# Descriptive statistics
# --------------------------------------------------------------------------- #
class HistogramBin(_Base):
    bin_start: float
    bin_end: float
    count: int
    relative_frequency: float = 0.0
    label: str = ""


class DescriptiveStatsModel(_Base):
    """Full descriptive summary of one numeric variable.

    ``ddof`` is reported explicitly: variance and standard deviation are **sample**
    statistics (Bessel-corrected, ddof=1), because the stored complaints are a sample
    of an ongoing process rather than a closed population.
    """

    n: int
    unit: str = "hours"

    # central tendency
    mean: float | None = None
    median: float | None = None
    mode: float | None = None
    modes: list[float] = Field(default_factory=list)
    mode_kind: ModeKind = "none"
    modal_bin: str | None = None

    # spread
    min: float | None = None
    max: float | None = None
    range: float | None = None
    variance: float | None = None
    std_dev: float | None = None
    ddof: int = 1
    standard_error: float | None = None
    mean_ci95_low: float | None = None
    mean_ci95_high: float | None = None
    ci_method: str | None = None

    # position
    q1: float | None = None
    q2: float | None = None
    q3: float | None = None
    iqr: float | None = None
    p90: float | None = None
    lower_fence: float | None = None
    upper_fence: float | None = None
    quartile_method: str = "linear (numpy/R type-7)"

    # shape
    skewness: float | None = None
    kurtosis: float | None = None
    kurtosis_type: str = "excess (Fisher), bias-corrected G2"
    coefficient_of_variation: float | None = None

    sample_warning: str | None = None
    notes: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Outliers
# --------------------------------------------------------------------------- #
class OutlierPoint(_Base):
    """One flagged complaint. Actionable: it names the case, not just a count."""

    reference_code: str
    value: float
    id: str | None = None
    category: str | None = None
    priority: str | None = None
    status: str | None = None
    area: str | None = None
    department: str | None = None
    value_days: float | None = None
    fence: float | None = None
    exceeds_fence_by: float | None = None
    side: Literal["upper", "lower"] = "upper"
    verdict: str = "abnormally slow — investigate"
    created_at: datetime | None = None


class GroupOutlierReport(_Base):
    group: str
    n: int
    median: float | None = None
    q1: float | None = None
    q3: float | None = None
    iqr: float | None = None
    lower_fence: float | None = None
    upper_fence: float | None = None
    outlier_count: int = 0
    outlier_rate: float = 0.0
    outliers: list[OutlierPoint] = Field(default_factory=list)
    sample_warning: str | None = None
    interpretation: str = ""


class OutlierReport(_Base):
    method: str = "Tukey fences (Q1 - 1.5*IQR, Q3 + 1.5*IQR)"
    scope: str = "overall"
    n: int = 0
    lower_fence: float | None = None
    upper_fence: float | None = None
    outlier_count: int = 0
    outlier_rate: float = 0.0
    upper_count: int = 0
    lower_count: int = 0
    outliers: list[OutlierPoint] = Field(default_factory=list)
    by_group: list[GroupOutlierReport] = Field(default_factory=list)
    interpretation: str = ""


# --------------------------------------------------------------------------- #
# /analytics/resolution-times  (statistics benchmark lives here)
# --------------------------------------------------------------------------- #
class CategoryQuartiles(_Base):
    category: str
    n: int
    median: float | None = None
    q1: float | None = None
    q3: float | None = None
    iqr: float | None = None
    mean: float | None = None
    upper_fence: float | None = None
    outlier_count: int = 0
    sample_warning: str | None = None


class ResolutionTimesResponse(_Base):
    """Exact CONTRACT shape, plus additive rigour fields."""

    n: int
    unit: str = "hours"

    mean: float | None = None
    median: float | None = None
    mode: float | None = None
    modes: list[float] = Field(default_factory=list)
    mode_kind: ModeKind = "none"
    modal_bin: str | None = None

    min: float | None = None
    max: float | None = None
    range: float | None = None

    variance: float | None = None
    std_dev: float | None = None
    ddof: int = 1
    standard_error: float | None = None
    mean_ci95_low: float | None = None
    mean_ci95_high: float | None = None

    q1: float | None = None
    q2: float | None = None
    q3: float | None = None
    iqr: float | None = None
    p90: float | None = None
    lower_fence: float | None = None
    upper_fence: float | None = None
    quartile_method: str = "linear (numpy/R type-7)"

    skewness: float | None = None
    kurtosis: float | None = None
    coefficient_of_variation: float | None = None

    outliers: list[OutlierPoint] = Field(default_factory=list)
    outlier_report: OutlierReport | None = None
    histogram: list[HistogramBin] = Field(default_factory=list)
    histogram_method: str = "Freedman–Diaconis bin width"
    by_category: list[CategoryQuartiles] = Field(default_factory=list)

    resolved_count: int = 0
    unresolved_count: int = 0
    censoring_note: str | None = None

    interpretation: str = ""
    insights: list[Insight] = Field(default_factory=list)
    sample_warning: str | None = None
    filters: AnalyticsFilters | None = None


# --------------------------------------------------------------------------- #
# Frequency distributions
# --------------------------------------------------------------------------- #
class FrequencyRow(_Base):
    value: str
    label: str
    count: int
    relative_frequency: float
    percent: float
    cumulative_count: int
    cumulative_percent: float


class FrequencyDistributionModel(_Base):
    variable: str
    n: int
    distinct: int
    rows: list[FrequencyRow] = Field(default_factory=list)
    mode: str | None = None
    mode_label: str | None = None
    modes: list[str] = Field(default_factory=list)
    mode_kind: ModeKind = "none"
    mode_count: int = 0
    mode_share: float = 0.0
    missing: int = 0
    interpretation: str = ""


class ContingencyRow(_Base):
    label: str
    value: str
    cells: dict[str, int]
    total: int


class ContingencyTableModel(_Base):
    """A proper contingency table: cells, row totals, column totals, grand total."""

    row_variable: str
    col_variable: str
    col_labels: list[str] = Field(default_factory=list)
    rows: list[ContingencyRow] = Field(default_factory=list)
    col_totals: dict[str, int] = Field(default_factory=dict)
    grand_total: int = 0
    row_percentages: list[ContingencyRow] | None = None
    row_percent_cells: list[dict[str, float]] = Field(default_factory=list)
    interpretation: str = ""


# --------------------------------------------------------------------------- #
# Inference
# --------------------------------------------------------------------------- #
class ChiSquareResultModel(_Base):
    """Chi-square test of independence, with its assumption reported, not hidden."""

    test: str = "chi-square test of independence"
    row_variable: str
    col_variable: str
    n: int
    statistic: float | None = None
    dof: int | None = None
    p_value: float | None = None
    alpha: float = 0.05
    significant: bool | None = None
    cramers_v: float | None = None
    effect_size: str | None = None
    expected_min: float | None = None
    cells_below_5: int = 0
    total_cells: int = 0
    pct_cells_below_5: float = 0.0
    assumption_met: bool = False
    reliable: bool = False
    correction_applied: bool = False
    h0: str = ""
    h1: str = ""
    interpretation: str = ""
    caveat: str | None = None


class SpearmanResultModel(_Base):
    test: str = "Spearman rank correlation"
    x_variable: str
    y_variable: str
    n: int
    rho: float | None = None
    p_value: float | None = None
    alpha: float = 0.05
    significant: bool | None = None
    strength: str | None = None
    direction: str | None = None
    reliable: bool = False
    interpretation: str = ""
    caveat: str | None = None


# --------------------------------------------------------------------------- #
# Time series
# --------------------------------------------------------------------------- #
class TrendPoint(_Base):
    date: date
    count: int
    rolling_mean_7: float | None = None


class CategorySeries(_Base):
    category: str
    label: str
    total: int
    points: list[TrendPoint] = Field(default_factory=list)
    week_over_week_pct: float | None = None


class WeekOverWeek(_Base):
    current_week: int = 0
    previous_week: int = 0
    change: int = 0
    change_pct: float | None = None
    direction: Direction = "flat"
    window_days: int = 7
    interpretation: str = ""


class ForecastPoint(_Base):
    date: date
    forecast: float
    low: float | None = None
    high: float | None = None


class ForecastModel(_Base):
    method: str
    horizon_days: int
    assumptions: list[str] = Field(default_factory=list)
    points: list[ForecastPoint] = Field(default_factory=list)
    expected_total: float | None = None
    interpretation: str = ""


class TrendsResponse(_Base):
    days: int
    date_from: date | None = None
    date_to: date | None = None
    total: int = 0
    series: list[TrendPoint] = Field(default_factory=list)
    rolling_window: int = 7
    by_category: list[CategorySeries] = Field(default_factory=list)
    week_over_week: WeekOverWeek = Field(default_factory=WeekOverWeek)
    busiest_day: TrendPoint | None = None
    quietest_day: TrendPoint | None = None
    daily_stats: DescriptiveStatsModel | None = None
    weekday_effect: dict[str, float] = Field(default_factory=dict)
    forecast: ForecastModel | None = None
    gaps_filled: int = 0
    interpretation: str = ""
    insights: list[Insight] = Field(default_factory=list)
    filters: AnalyticsFilters | None = None


# --------------------------------------------------------------------------- #
# Departments / areas
# --------------------------------------------------------------------------- #
class DepartmentStat(_Base):
    department: str
    n: int
    open: int = 0
    in_progress: int = 0
    resolved: int = 0
    backlog: int = 0
    resolution_rate: float | None = None
    median_resolution_hours: float | None = None
    median_resolution_days: float | None = None
    mean_resolution_hours: float | None = None
    p90_resolution_hours: float | None = None
    resolved_sample: int = 0
    share_pct: float = 0.0
    sample_warning: str | None = None


class DepartmentsResponse(_Base):
    n: int
    total_complaints: int
    departments: list[DepartmentStat] = Field(default_factory=list)
    overall_median_hours: float | None = None
    slowest: DepartmentStat | None = None
    fastest: DepartmentStat | None = None
    largest_backlog: DepartmentStat | None = None
    interpretation: str = ""
    insights: list[Insight] = Field(default_factory=list)
    filters: AnalyticsFilters | None = None


class AreaStat(_Base):
    area: str
    n: int
    share_pct: float = 0.0
    open: int = 0
    resolved: int = 0
    critical_count: int = 0
    top_category: str | None = None
    top_category_label: str | None = None
    top_category_count: int = 0
    top_category_share: float = 0.0
    median_resolution_hours: float | None = None
    hotspot: bool = False
    hotspot_reason: str | None = None


class AreasResponse(_Base):
    n: int
    total_complaints: int
    areas: list[AreaStat] = Field(default_factory=list)
    hotspots: list[AreaStat] = Field(default_factory=list)
    hotspot_rule: str = ""
    concentration_top3_pct: float = 0.0
    interpretation: str = ""
    insights: list[Insight] = Field(default_factory=list)
    filters: AnalyticsFilters | None = None


# --------------------------------------------------------------------------- #
# Categories / priorities
# --------------------------------------------------------------------------- #
class CategoryResolutionRow(_Base):
    category: str
    label: str
    n: int
    median_resolution_hours: float | None = None
    median_resolution_days: float | None = None
    open: int = 0
    resolved: int = 0


class CategoriesResponse(_Base):
    distribution: FrequencyDistributionModel
    resolution_by_category: list[CategoryResolutionRow] = Field(default_factory=list)
    by_status: ContingencyTableModel | None = None
    interpretation: str = ""
    insights: list[Insight] = Field(default_factory=list)
    filters: AnalyticsFilters | None = None


class PrioritiesResponse(_Base):
    distribution: FrequencyDistributionModel
    crosstab: ContingencyTableModel
    chi_square: ChiSquareResultModel
    spearman_priority_vs_speed: SpearmanResultModel | None = None
    escalation_share_pct: float = 0.0
    interpretation: str = ""
    insights: list[Insight] = Field(default_factory=list)
    filters: AnalyticsFilters | None = None


# --------------------------------------------------------------------------- #
# Overview / public summary
# --------------------------------------------------------------------------- #
class OverviewKPIs(_Base):
    total: int = 0
    open: int = 0
    assigned: int = 0
    in_progress: int = 0
    resolved: int = 0
    rejected: int = 0
    resolution_rate: float = 0.0
    median_resolution_hours: float | None = None
    median_resolution_days: float | None = None
    mean_resolution_hours: float | None = None
    critical_open: int = 0
    avg_ai_confidence: float | None = None
    complaints_this_week: int = 0
    complaints_last_week: int = 0
    wow_change_pct: float | None = None
    wow_direction: Direction = "flat"
    backlog: int = 0
    oldest_open_days: float | None = None


class KPICard(_Base):
    key: str
    label: str
    value: float | None = None
    display: str
    unit: str | None = None
    hint: str = ""
    severity: Severity = "info"


class OverviewResponse(_Base):
    generated_at: datetime
    kpis: OverviewKPIs
    cards: list[KPICard] = Field(default_factory=list)
    insights: list[Insight] = Field(default_factory=list)
    interpretation: str = ""
    filters: AnalyticsFilters | None = None


class PublicSummaryResponse(_Base):
    """Deliberately small and non-identifying — served without authentication."""

    generated_at: datetime
    total_complaints: int = 0
    resolved: int = 0
    resolution_rate: float = 0.0
    median_resolution_days: float | None = None
    complaints_this_week: int = 0
    active_areas: int = 0
    top_category: str | None = None
    top_category_label: str | None = None
    top_category_share_pct: float = 0.0
    categories: list[FrequencyRow] = Field(default_factory=list)
    highlights: list[Insight] = Field(default_factory=list)
    interpretation: str = ""


__all__ = [
    "AnalyticsFilters",
    "AreaStat",
    "AreasResponse",
    "CategoriesResponse",
    "CategoryQuartiles",
    "CategoryResolutionRow",
    "CategorySeries",
    "ChiSquareResultModel",
    "ContingencyRow",
    "ContingencyTableModel",
    "DepartmentStat",
    "DepartmentsResponse",
    "DescriptiveStatsModel",
    "ForecastModel",
    "ForecastPoint",
    "FrequencyDistributionModel",
    "FrequencyRow",
    "GroupOutlierReport",
    "HistogramBin",
    "Insight",
    "InsightsResponse",
    "KPICard",
    "OutlierPoint",
    "OutlierReport",
    "OverviewKPIs",
    "OverviewResponse",
    "PrioritiesResponse",
    "PublicSummaryResponse",
    "ResolutionTimesResponse",
    "SpearmanResultModel",
    "TrendPoint",
    "TrendsResponse",
    "WeekOverWeek",
]

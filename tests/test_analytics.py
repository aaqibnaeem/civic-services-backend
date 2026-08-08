"""Tests for the statistics & analytics engine.

Philosophy: these are **pure-function tests against hand-computed fixtures**, not
snapshot tests and not database tests. Every expected value below was worked out by
hand (the derivation is in the comment above the assertion) so the suite actually
proves the maths is right rather than proving it has not changed. That also means the
statistics engine can be verified without any of the other agents' database fixtures.

The canonical fixture is the sample ``[2, 4, 4, 4, 5, 5, 7, 9]`` (n = 8):

    mean      = 40 / 8                       = 5.0
    median    = (4 + 5) / 2                  = 4.5
    mode      = 4        (occurs 3 times)
    range     = 9 - 2                        = 7
    Σ(x-x̄)²   = 9+1+1+1+0+0+4+16             = 32
    variance  = 32 / (8-1)                   = 4.571428…   (ddof = 1)
    std       = √(32/7)                      = 2.138089…
    Q1        pos (n-1)·0.25 = 1.75 → between x[1]=4 and x[2]=4 = 4.0
    Q2        pos (n-1)·0.50 = 3.50 → between x[3]=4 and x[4]=5 = 4.5
    Q3        pos (n-1)·0.75 = 5.25 → x[5]=5 + 0.25·(7-5)       = 5.5
    IQR       = 5.5 - 4.0                    = 1.5
    fences    = 4.0 - 1.5·1.5 = 1.75  and  5.5 + 1.5·1.5 = 7.75
    skewness  m2=4, m3=5.25 → g1=0.65625, G1 = g1·√(8·7)/6      = 0.818488…
    kurtosis  m4=44.5 → g2 = 44.5/16 - 3 = -0.21875,
              G2 = (7/30)·(9·(-0.21875) + 6)                    = 0.940625
    CV        = 2.138089… / 5                = 0.427618…
"""

from __future__ import annotations

from datetime import date, datetime, timedelta

import pytest

from app.analytics.descriptive import DescriptiveStats
from app.analytics.distributions import ContingencyTable, FrequencyDistribution
from app.analytics.inference import (
    ChiSquareIndependenceTest,
    SpearmanCorrelation,
    category_priority_chi_square,
    priority_vs_resolution_spearman,
)
from app.analytics.narratives import InsightContext, InsightEngine
from app.analytics.outliers import OutlierDetector
from app.analytics.service import AnalyticsService, TTLCache
from app.analytics.timeseries import TimeSeriesAnalyzer
from app.schemas.analytics import Insight, ResolutionTimesResponse

SAMPLE = [2, 4, 4, 4, 5, 5, 7, 9]


# =========================================================================== #
# descriptive.py — hand-computed fixture
# =========================================================================== #
class TestDescriptiveHandComputed:
    @pytest.fixture
    def stats(self) -> DescriptiveStats:
        return DescriptiveStats(SAMPLE, unit="hours")

    def test_n(self, stats):
        assert stats.n == 8

    def test_central_tendency(self, stats):
        assert stats.mean == pytest.approx(5.0)
        assert stats.median == pytest.approx(4.5)
        modes, kind, count = stats.modes()
        assert modes == [4.0]
        assert kind == "unique"
        assert count == 3

    def test_min_max_range(self, stats):
        assert stats.minimum == pytest.approx(2.0)
        assert stats.maximum == pytest.approx(9.0)
        assert stats.value_range == pytest.approx(7.0)

    def test_sample_variance_and_std_use_ddof_1(self, stats):
        # 32 / (8 - 1), NOT 32 / 8 (= 4.0, the population variance)
        assert stats.variance == pytest.approx(32 / 7)
        assert stats.variance == pytest.approx(4.571428571, abs=1e-6)
        assert stats.std_dev == pytest.approx(2.138089935, abs=1e-6)
        assert stats.ddof == 1
        assert stats.to_dict()["ddof"] == 1

    def test_population_variance_would_differ(self):
        """Guard against a silent switch to ddof=0."""
        population = DescriptiveStats(SAMPLE, ddof=0)
        assert population.variance == pytest.approx(4.0)
        assert DescriptiveStats(SAMPLE).variance != pytest.approx(4.0)

    def test_quartiles_and_iqr(self, stats):
        assert stats.q1 == pytest.approx(4.0)
        assert stats.q2 == pytest.approx(4.5)
        assert stats.q2 == pytest.approx(stats.median)
        assert stats.q3 == pytest.approx(5.5)
        assert stats.iqr == pytest.approx(1.5)

    def test_tukey_fences(self, stats):
        # Q1 - 1.5*IQR = 4.0 - 2.25 = 1.75 ; Q3 + 1.5*IQR = 5.5 + 2.25 = 7.75
        assert stats.lower_fence == pytest.approx(1.75)
        assert stats.upper_fence == pytest.approx(7.75)
        outliers = [x for x in SAMPLE if x > stats.upper_fence or x < stats.lower_fence]
        assert outliers == [9]

    def test_skewness_and_kurtosis(self, stats):
        assert stats.skewness == pytest.approx(0.818488, abs=1e-5)
        assert stats.kurtosis == pytest.approx(0.940625, abs=1e-5)

    def test_coefficient_of_variation(self, stats):
        assert stats.coefficient_of_variation == pytest.approx(2.138089935 / 5.0, abs=1e-6)

    def test_standard_error(self, stats):
        assert stats.standard_error == pytest.approx(2.138089935 / (8**0.5), abs=1e-6)

    def test_small_sample_warning_is_emitted(self, stats):
        assert stats.sample_warning is not None
        assert "8" in stats.sample_warning

    def test_no_warning_for_large_sample(self):
        assert DescriptiveStats(list(range(100))).sample_warning is None

    def test_to_model_is_typed_and_complete(self, stats):
        model = stats.to_model()
        assert model.n == 8
        assert model.ddof == 1
        assert model.mean == pytest.approx(5.0)
        assert model.q1 == pytest.approx(4.0)
        assert model.upper_fence == pytest.approx(7.75)
        assert "type-7" in model.quartile_method


class TestDescriptiveEdgeCases:
    def test_empty_series(self):
        stats = DescriptiveStats([])
        assert stats.n == 0
        for value in (stats.mean, stats.median, stats.variance, stats.std_dev,
                      stats.q1, stats.iqr, stats.lower_fence, stats.upper_fence,
                      stats.skewness, stats.kurtosis, stats.coefficient_of_variation):
            assert value is None
        assert stats.modes() == ([], "none", 0)
        assert stats.histogram() == []
        assert "No observations" in stats.sample_warning
        payload = stats.to_dict()
        assert payload["n"] == 0
        assert payload["mean"] is None

    def test_single_observation(self):
        """n=1: mean/median defined, every dispersion statistic undefined."""
        stats = DescriptiveStats([7.0])
        assert stats.n == 1
        assert stats.mean == pytest.approx(7.0)
        assert stats.median == pytest.approx(7.0)
        assert stats.minimum == stats.maximum == pytest.approx(7.0)
        assert stats.value_range == pytest.approx(0.0)
        # variance needs n > ddof; it must be None, never 0.0
        assert stats.variance is None
        assert stats.std_dev is None
        assert stats.skewness is None
        assert stats.kurtosis is None
        assert stats.coefficient_of_variation is None
        # quartiles of a single point all collapse onto that point
        assert stats.q1 == stats.q3 == pytest.approx(7.0)
        assert stats.iqr == pytest.approx(0.0)
        assert "1 observation" in stats.sample_warning

    def test_two_observations(self):
        """n=2: variance defined, shape statistics still undefined."""
        stats = DescriptiveStats([4.0, 8.0])
        assert stats.variance == pytest.approx(8.0)  # ((4-6)^2+(8-6)^2)/1 = 8
        assert stats.std_dev == pytest.approx(8**0.5)
        assert stats.skewness is None  # needs n >= 3
        assert stats.kurtosis is None  # needs n >= 4

    def test_three_observations_have_skew_but_no_kurtosis(self):
        stats = DescriptiveStats([1.0, 2.0, 6.0])
        assert stats.skewness is not None
        assert stats.kurtosis is None

    def test_no_mode_when_every_value_unique(self):
        modes, kind, count = DescriptiveStats([1, 2, 3, 4, 5]).modes()
        assert modes == []
        assert kind == "none"
        assert count == 1
        assert DescriptiveStats([1, 2, 3, 4, 5]).to_dict()["mode"] is None

    def test_multimodal_is_reported_as_multi(self):
        modes, kind, count = DescriptiveStats([1, 1, 2, 2, 3]).modes()
        assert modes == [1.0, 2.0]
        assert kind == "multi"
        assert count == 2

    def test_non_numeric_and_missing_values_are_dropped(self):
        stats = DescriptiveStats([1.0, None, float("nan"), "bad", 3.0, float("inf")])
        assert stats.n == 2
        assert stats.mean == pytest.approx(2.0)

    def test_all_identical_values(self):
        stats = DescriptiveStats([5.0] * 10)
        assert stats.variance == pytest.approx(0.0)
        assert stats.std_dev == pytest.approx(0.0)
        assert stats.iqr == pytest.approx(0.0)
        assert stats.coefficient_of_variation == pytest.approx(0.0)
        assert stats.skewness is None  # m2 == 0 -> undefined, not 0.0

    def test_cv_refused_for_non_positive_mean(self):
        assert DescriptiveStats([-5.0, 5.0, 0.0]).coefficient_of_variation is None

    def test_histogram_counts_sum_to_n(self):
        stats = DescriptiveStats([float(i) for i in range(200)])
        bins = stats.histogram()
        assert bins
        assert sum(b["count"] for b in bins) == 200
        assert all(b["bin_end"] >= b["bin_start"] for b in bins)
        assert stats.modal_bin(bins) is not None

    def test_json_safety_no_nan_leaks(self):
        payload = DescriptiveStats([]).to_dict()
        for value in payload.values():
            if isinstance(value, float):
                assert value == value  # not NaN


# =========================================================================== #
# distributions.py
# =========================================================================== #
class TestFrequencyDistribution:
    @pytest.fixture
    def dist(self) -> FrequencyDistribution:
        values = ["road"] * 5 + ["water"] * 3 + ["waste"] * 2
        return FrequencyDistribution(values, variable="category")

    def test_counts_and_relative_frequency(self, dist):
        rows = dist.rows()
        assert dist.n == 10
        assert dist.distinct == 3
        assert rows[0]["value"] == "road"
        assert rows[0]["count"] == 5
        assert rows[0]["relative_frequency"] == pytest.approx(0.5)
        assert rows[0]["percent"] == pytest.approx(50.0)
        assert rows[1]["count"] == 3
        assert rows[1]["relative_frequency"] == pytest.approx(0.3)

    def test_cumulative_frequency(self, dist):
        rows = dist.rows()
        assert [r["cumulative_count"] for r in rows] == [5, 8, 10]
        assert rows[-1]["cumulative_percent"] == pytest.approx(100.0)

    def test_mode(self, dist):
        assert dist.mode == "road"
        assert dist.mode_share == pytest.approx(0.5)
        assert dist.to_dict()["mode_kind"] == "unique"
        assert dist.to_dict()["mode_label"] == "Roads & Potholes"

    def test_interpretation_contains_the_number(self, dist):
        text = dist.interpretation()
        assert "Roads & Potholes" in text
        assert "5" in text and "10" in text

    def test_missing_values_counted_separately(self):
        dist = FrequencyDistribution(["road", None, "", "water"], variable="category")
        assert dist.n == 2
        assert dist.missing == 2

    def test_tied_mode_reported_as_multi(self):
        dist = FrequencyDistribution(["road", "road", "water", "water"], variable="category")
        modes, kind, count = dist.modes()
        assert modes == ["road", "water"]
        assert kind == "multi"
        assert count == 2
        assert "tie" in dist.interpretation()

    def test_empty_distribution(self):
        dist = FrequencyDistribution([], variable="category")
        assert dist.n == 0
        assert dist.rows() == []
        assert dist.mode is None
        assert "No complaints" in dist.interpretation()

    def test_ordinal_variables_keep_contract_order(self):
        """Priority is ordinal: sorting it by frequency would destroy information.

        Unobserved levels are kept as explicit zero rows so a bar chart shows the gap
        rather than silently omitting the level.
        """
        dist = FrequencyDistribution(
            ["critical", "low", "low", "medium"], variable="priority"
        )
        rows = dist.rows()
        assert [r["value"] for r in rows] == ["low", "medium", "high", "critical"]
        assert [r["count"] for r in rows] == [2, 1, 0, 1]
        assert rows[2]["percent"] == pytest.approx(0.0)
        assert rows[-1]["cumulative_count"] == 4

    def test_model_round_trip(self, dist):
        model = dist.to_model()
        assert model.n == 10
        assert model.mode == "road"
        assert len(model.rows) == 3


class TestContingencyTable:
    @pytest.fixture
    def table(self) -> ContingencyTable:
        #            low  high
        #   road      2     1     -> 3
        #   water     1     2     -> 3
        rows = ["road", "road", "road", "water", "water", "water"]
        cols = ["low", "low", "high", "low", "high", "high"]
        return ContingencyTable(rows, cols, row_variable="category", col_variable="priority")

    def test_cells_and_margins(self, table):
        assert table.count("road", "low") == 2
        assert table.count("road", "high") == 1
        assert table.count("water", "low") == 1
        assert table.count("water", "high") == 2
        assert table.row_totals() == {"road": 3, "water": 3}
        assert table.col_totals() == {"low": 3, "high": 3}
        assert table.grand_total == 6

    def test_matrix_shape(self, table):
        matrix = table.matrix()
        assert len(matrix) == 2
        assert all(len(row) == 2 for row in matrix)
        assert sum(sum(r) for r in matrix) == 6

    def test_column_order_is_ordinal_for_priority(self, table):
        assert table.col_labels == ["low", "high"]

    def test_dict_has_totals_and_interpretation(self, table):
        payload = table.to_dict()
        assert payload["grand_total"] == 6
        assert payload["col_totals"] == {"low": 3, "high": 3}
        assert payload["rows"][0]["total"] == 3
        assert payload["interpretation"]

    def test_empty_table(self):
        table = ContingencyTable([], [])
        assert table.grand_total == 0
        assert table.matrix() == []
        assert "empty" in table.interpretation()


# =========================================================================== #
# inference.py — known chi-square result
# =========================================================================== #
class TestChiSquare:
    """Hand-computed 2x3 table.

            B1  B2  B3   total
        A1  10  20  30     60
        A2  30  20  10     60
        tot 40  40  40    120

    Every expected cell = row_total * col_total / N = 60*40/120 = 20.
        chi2 = Σ (O-E)²/E = 5 + 0 + 5 + 5 + 0 + 5 = 20.0
        dof  = (2-1)(3-1) = 2
        p    = P(χ²₂ > 20) = e^(-10) = 4.53999e-05
        V    = √(20 / (120 * min(1,2))) = √(1/6) = 0.4082483
    A 2x3 table is used on purpose so Yates' continuity correction does not apply.
    """

    @pytest.fixture
    def result(self):
        return ChiSquareIndependenceTest(
            matrix=[[10, 20, 30], [30, 20, 10]],
            row_labels=["A1", "A2"],
            col_labels=["B1", "B2", "B3"],
        ).run()

    def test_statistic_and_dof(self, result):
        assert result["statistic"] == pytest.approx(20.0, abs=1e-9)
        assert result["dof"] == 2
        assert result["n"] == 120

    def test_p_value(self, result):
        import math

        assert result["p_value"] == pytest.approx(math.exp(-10.0), rel=1e-6)
        assert result["significant"] is True

    def test_cramers_v_and_effect_size(self, result):
        assert result["cramers_v"] == pytest.approx((1 / 6) ** 0.5, abs=1e-4)
        # Cohen's thresholds for df* = min(r-1, c-1) = 1 are .10/.30/.50, so
        # V = 0.408 is a *medium* association, not a large one. The thresholds must
        # scale with df* — grading every table against .10/.30/.50 would overstate
        # effect sizes on wider tables.
        assert result["effect_size"] == "medium"

    def test_expected_frequency_assumption_is_checked_and_met(self, result):
        assert result["expected_min"] == pytest.approx(20.0)
        assert result["cells_below_5"] == 0
        assert result["total_cells"] == 6
        assert result["assumption_met"] is True
        assert result["reliable"] is True
        assert result["caveat"] is None

    def test_no_yates_correction_on_2x3(self, result):
        assert result["correction_applied"] is False

    def test_interpretation_is_plain_english_with_numbers(self, result):
        text = result["interpretation"]
        assert "20.0" in text or "20.00" in text
        assert "reject the null hypothesis" in text
        assert "expected cell counts are at least 5" in text

    def test_hypotheses_are_stated(self, result):
        assert result["h0"].startswith("H0:")
        assert result["h1"].startswith("H1:")


class TestChiSquareAssumptions:
    def test_sparse_table_is_flagged_unreliable(self):
        """Expected counts far below 5 -> the result must be marked unreliable."""
        result = ChiSquareIndependenceTest(
            matrix=[[1, 2], [2, 1]], row_labels=["a", "b"], col_labels=["x", "y"]
        ).run()
        assert result["statistic"] is not None  # nothing is hidden
        assert result["expected_min"] < 5
        assert result["cells_below_5"] == 4
        assert result["assumption_met"] is False
        assert result["reliable"] is False
        assert result["caveat"] is not None
        assert "NOT reliable" in result["interpretation"]
        assert "Fisher" in result["caveat"]

    def test_yates_correction_applied_to_2x2(self):
        result = ChiSquareIndependenceTest(
            matrix=[[20, 30], [30, 20]], row_labels=["a", "b"], col_labels=["x", "y"]
        ).run()
        assert result["correction_applied"] is True
        # With Yates: Σ(|O-E|-0.5)²/E = 4 * (4.5²/25) = 3.24
        assert result["statistic"] == pytest.approx(3.24, abs=1e-6)
        assert result["dof"] == 1

    def test_degenerate_table_returns_empty_result(self):
        result = ChiSquareIndependenceTest(
            matrix=[[5, 5]], row_labels=["only"], col_labels=["x", "y"]
        ).run()
        assert result["statistic"] is None
        assert result["reliable"] is False
        assert "nothing to test" in result["interpretation"]

    def test_all_zero_rows_are_dropped(self):
        result = ChiSquareIndependenceTest(
            matrix=[[10, 20, 30], [30, 20, 10], [0, 0, 0]],
            row_labels=["A1", "A2", "Z"],
            col_labels=["B1", "B2", "B3"],
        ).run()
        assert result["statistic"] == pytest.approx(20.0, abs=1e-9)
        assert result["dof"] == 2


class TestSpearman:
    def test_perfect_monotonic_non_linear_relationship(self):
        result = SpearmanCorrelation(
            [1, 2, 3, 4, 5], [1, 4, 9, 16, 25], x_variable="x", y_variable="y", min_n=5
        ).run()
        assert result["rho"] == pytest.approx(1.0)
        assert result["direction"] == "positive"
        assert result["strength"] == "very strong"
        assert result["significant"] is True

    def test_perfect_negative_relationship(self):
        result = SpearmanCorrelation(
            list(range(10)), list(range(10, 0, -1)), min_n=5
        ).run()
        assert result["rho"] == pytest.approx(-1.0)
        assert result["direction"] == "negative"

    def test_too_few_pairs(self):
        result = SpearmanCorrelation([1, 2], [3, 4]).run()
        assert result["rho"] is None
        assert result["reliable"] is False
        assert "too few" in result["interpretation"].lower()

    def test_constant_input_is_rejected(self):
        result = SpearmanCorrelation([1, 1, 1, 1], [1, 2, 3, 4]).run()
        assert result["rho"] is None

    def test_small_sample_gets_a_caveat(self):
        result = SpearmanCorrelation([1, 2, 3, 4, 5], [2, 4, 6, 8, 10], min_n=10).run()
        assert result["reliable"] is False
        assert result["caveat"] is not None


# =========================================================================== #
# frames used by the timeseries / outlier / service tests
# =========================================================================== #
def make_frame(records: list[dict]):
    """Build a prepared complaint frame the way the service would."""
    import pandas as pd

    return AnalyticsService()._prepare(pd.DataFrame(records))


def synthetic_frame(n: int = 240, *, end: datetime | None = None):
    """Deterministic synthetic complaint set — no randomness, so assertions are stable."""
    end = end or datetime(2026, 8, 8, 12, 0, 0)
    categories = ["road", "water", "waste", "electricity", "drainage", "safety", "other"]
    priorities = ["low", "medium", "high", "critical"]
    areas = ["Gulshan-e-Iqbal", "Saddar", "Clifton", "North Nazimabad"]
    departments = ["Roads & Infrastructure", "Water Board", "Sanitation", "Sewerage"]
    records = []
    for i in range(n):
        created = end - timedelta(days=(i % 90), hours=(i * 7) % 24)
        category = categories[i % len(categories)]
        # Drainage is deliberately slow so per-category fences differ from global ones.
        base_hours = 240 if category == "drainage" else 36 + (i % 5) * 12
        resolved_at = None
        if i % 4 != 0:  # 75% resolved
            resolved_at = created + timedelta(hours=base_hours + (i % 11))
        if i == 3:  # one extreme outlier
            resolved_at = created + timedelta(hours=2000)
        records.append(
            {
                "id": f"id-{i:04d}",
                "reference_code": f"CIV-{i:05d}",
                "category": category,
                "priority": priorities[i % len(priorities)],
                "status": "resolved" if resolved_at is not None else "open",
                "area": areas[i % len(areas)],
                "department": departments[i % len(departments)],
                "created_at": created,
                "resolved_at": resolved_at,
                "ai_confidence": 0.7 + (i % 25) / 100.0,
            }
        )
    return make_frame(records)


# =========================================================================== #
# timeseries.py — the gap-filling requirement
# =========================================================================== #
class TestTimeSeriesGaps:
    @pytest.fixture
    def gappy(self):
        """2 complaints on Jan 1, 1 on Jan 2, nothing on Jan 3-4, 3 on Jan 5."""
        records = []
        plan = {date(2026, 1, 1): 2, date(2026, 1, 2): 1, date(2026, 1, 5): 3}
        i = 0
        for day, count in plan.items():
            for _ in range(count):
                records.append(
                    {
                        "id": f"id-{i}",
                        "reference_code": f"CIV-{i}",
                        "category": "road",
                        "priority": "low",
                        "status": "open",
                        "area": "Saddar",
                        "department": "Roads & Infrastructure",
                        "created_at": datetime.combine(day, datetime.min.time()) + timedelta(hours=9),
                        "resolved_at": None,
                        "ai_confidence": 0.8,
                    }
                )
                i += 1
        return make_frame(records)

    def test_missing_dates_are_filled_with_zero(self, gappy):
        analyzer = TimeSeriesAnalyzer(gappy, days=5, end=date(2026, 1, 5))
        counts = analyzer.daily_counts()
        assert len(counts) == 5  # a gap-free calendar index, not 3 observed days
        assert list(counts.values) == [2, 1, 0, 0, 3]
        assert analyzer.gaps_filled() == 2

    def test_moving_average_uses_calendar_days_not_rows(self, gappy):
        """The whole point of gap-filling.

        With a gap-free index the 3-day mean ending Jan 5 is (0 + 0 + 3)/3 = 1.0.
        A naive implementation that grouped only the observed dates would compute
        (2 + 1 + 3)/3 = 2.0 — double the truth. This asserts the honest number.
        """
        analyzer = TimeSeriesAnalyzer(gappy, days=5, end=date(2026, 1, 5), rolling_window=3)
        rolling = analyzer.rolling_mean()
        values = list(rolling.values)
        assert values[0] != values[0]  # NaN: window not yet full
        assert values[1] != values[1]  # NaN
        assert values[2] == pytest.approx(1.0)  # (2+1+0)/3
        assert values[3] == pytest.approx(1 / 3)  # (1+0+0)/3
        assert values[4] == pytest.approx(1.0)  # (0+0+3)/3 -- NOT 2.0

    def test_rolling_mean_is_null_until_the_window_is_full(self, gappy):
        analyzer = TimeSeriesAnalyzer(gappy, days=5, end=date(2026, 1, 5))
        points = analyzer.series_points()
        assert len(points) == 5
        # window is 7 but only 5 days exist -> no honest 7-day average is possible
        assert all(p["rolling_mean_7"] is None for p in points)

    def test_series_points_carry_dates_and_counts(self, gappy):
        points = TimeSeriesAnalyzer(gappy, days=5, end=date(2026, 1, 5)).series_points()
        assert points[0]["date"] == date(2026, 1, 1)
        assert points[0]["count"] == 2
        assert points[2]["count"] == 0

    def test_empty_frame_still_produces_a_gap_free_index(self):
        analyzer = TimeSeriesAnalyzer(make_frame([]), days=10, end=date(2026, 1, 10))
        counts = analyzer.daily_counts()
        assert len(counts) == 10
        assert int(counts.sum()) == 0
        assert "No complaints" in analyzer.interpretation()


class TestTimeSeriesMetrics:
    @pytest.fixture
    def analyzer(self):
        return TimeSeriesAnalyzer(synthetic_frame(), days=90, end=date(2026, 8, 8))

    def test_week_over_week(self, analyzer):
        wow = analyzer.week_over_week()
        assert wow["current_week"] >= 0
        assert wow["direction"] in {"up", "down", "flat"}
        assert wow["window_days"] == 7
        assert wow["interpretation"]

    def test_wow_from_zero_base_returns_none_not_infinity(self):
        records = [
            {
                "id": "a",
                "reference_code": "CIV-A",
                "category": "road",
                "priority": "low",
                "status": "open",
                "area": "Saddar",
                "department": "Roads & Infrastructure",
                "created_at": datetime(2026, 1, 20, 9),
                "resolved_at": None,
                "ai_confidence": 0.9,
            }
        ]
        analyzer = TimeSeriesAnalyzer(make_frame(records), days=14, end=date(2026, 1, 20))
        wow = analyzer.week_over_week()
        assert wow["previous_week"] == 0
        assert wow["change_pct"] is None
        assert "zero base" in wow["interpretation"]

    def test_forecast_declares_its_method_and_assumptions(self, analyzer):
        forecast = analyzer.forecast()
        assert len(forecast["points"]) == 7
        assert "naive" in forecast["method"].lower()
        assert len(forecast["assumptions"]) >= 3
        assert all(p["forecast"] >= 0 for p in forecast["points"])
        assert forecast["expected_total"] is not None
        assert "naive" in forecast["interpretation"].lower()

    def test_forecast_dates_follow_the_window(self, analyzer):
        first = analyzer.forecast()["points"][0]["date"]
        assert first == date(2026, 8, 9)

    def test_per_category_series_are_gap_free(self, analyzer):
        series = analyzer.by_category()
        assert series
        for entry in series:
            assert len(entry["points"]) == 90
            assert entry["total"] == sum(p["count"] for p in entry["points"])

    def test_weekday_effect_covers_seven_days(self, analyzer):
        effect = analyzer.weekday_effect()
        assert len(effect) == 7
        assert all(v >= 0 for v in effect.values())

    def test_to_dict_is_complete(self, analyzer):
        payload = analyzer.to_dict()
        for key in (
            "days", "date_from", "date_to", "total", "series", "by_category",
            "week_over_week", "busiest_day", "daily_stats", "forecast",
            "gaps_filled", "interpretation",
        ):
            assert key in payload
        assert payload["interpretation"]


# =========================================================================== #
# outliers.py
# =========================================================================== #
class TestOutliers:
    def test_overall_fences_flag_the_extreme_case(self):
        frame = synthetic_frame()
        report = OutlierDetector(frame).detect_overall()
        assert report["n"] > 0
        assert report["upper_fence"] is not None
        assert report["outlier_count"] >= 1
        codes = {o["reference_code"] for o in report["outliers"]}
        assert "CIV-00003" in codes  # the 2000-hour case planted in the fixture

    def test_outliers_are_actionable_not_just_a_count(self):
        report = OutlierDetector(synthetic_frame()).detect_overall()
        point = report["outliers"][0]
        for key in ("reference_code", "value", "value_days", "category", "status", "verdict", "fence"):
            assert key in point
        assert "investigate" in point["verdict"]

    def test_per_category_fences_differ_from_the_global_fence(self):
        """Drainage is slow by nature; its own fence must be higher than the global one."""
        frame = synthetic_frame()
        detector = OutlierDetector(frame)
        global_fence = detector.detect_overall()["upper_fence"]
        groups = {g["group"]: g for g in detector.detect_by_group()}
        assert "drainage" in groups
        assert groups["drainage"]["upper_fence"] is not None
        assert groups["drainage"]["median"] > global_fence / 4
        # each group carries its own quartiles
        for group in groups.values():
            if group["upper_fence"] is not None:
                assert group["q1"] is not None and group["q3"] is not None

    def test_small_groups_get_a_warning_instead_of_fences(self):
        records = [
            {
                "id": f"id-{i}",
                "reference_code": f"CIV-{i}",
                "category": "safety",
                "priority": "low",
                "status": "resolved",
                "area": "Saddar",
                "department": "Public Safety",
                "created_at": datetime(2026, 8, 1, 9),
                "resolved_at": datetime(2026, 8, 1, 9) + timedelta(hours=10 + i),
                "ai_confidence": 0.9,
            }
            for i in range(4)
        ]
        groups = OutlierDetector(make_frame(records)).detect_by_group()
        assert len(groups) == 1
        assert groups[0]["upper_fence"] is None
        assert groups[0]["outliers"] == []
        assert "below the" in groups[0]["interpretation"]

    def test_empty_frame(self):
        report = OutlierDetector(make_frame([])).detect_overall()
        assert report["n"] == 0
        assert report["outliers"] == []
        assert "no resolution time" in report["interpretation"].lower()

    def test_report_mentions_per_category_reasoning(self):
        report = OutlierDetector(synthetic_frame()).report()
        assert "by_group" in report
        assert "drainage" in report["interpretation"] or "category" in report["interpretation"]

    def test_model_round_trip(self):
        model = OutlierDetector(synthetic_frame()).to_model()
        assert model.n > 0
        assert model.method.startswith("Tukey")


# =========================================================================== #
# narratives.py
# =========================================================================== #
class TestNarratives:
    @pytest.fixture
    def insights(self):
        frame = synthetic_frame()
        service = AnalyticsService(frame=frame, use_cache=False)
        analyzer = TimeSeriesAnalyzer(frame, days=90, end=date(2026, 8, 8))
        trends = analyzer.to_dict()
        context = service.build_context(frame, trends=trends, analyzer=analyzer)
        return InsightEngine(context).generate()

    def test_produces_a_meaningful_number_of_insights(self, insights):
        assert len(insights) >= 8

    def test_every_insight_matches_the_contract_shape(self, insights):
        for item in insights:
            assert set(item) == {"id", "severity", "title", "detail", "metric", "unit"}
            assert item["severity"] in {"info", "warn", "critical"}
            assert item["title"] and item["detail"]
            model = Insight(**item)  # must validate against the wire schema
            assert model.id == item["id"]

    def test_titles_stand_alone_with_a_number_inside(self, insights):
        for item in insights:
            assert any(ch.isdigit() for ch in item["title"]), item["title"]
            assert item["title"].endswith((".", "!"))

    def test_insights_are_ranked_by_severity(self, insights):
        from app.analytics.narratives import SEVERITY_RANK

        ranks = [SEVERITY_RANK[i["severity"]] for i in insights]
        assert ranks == sorted(ranks)

    def test_expected_rules_fire_on_realistic_data(self, insights):
        ids = {i["id"] for i in insights}
        for expected in (
            "modal_category",
            "resolution_rate",
            "priority_mix",
            "resolution_outliers",
        ):
            assert expected in ids, f"rule {expected} did not fire; got {sorted(ids)}"

    def test_skew_rule_says_use_the_median(self):
        """A deliberately right-skewed sample must produce the mean-vs-median warning."""
        values = [24.0] * 40 + [1500.0] * 5
        context = InsightContext(
            n_total=45, resolution=DescriptiveStats(values, unit="hours").to_dict()
        )
        items = {i["id"]: i for i in InsightEngine(context).generate()}
        assert "resolution_skew" in items
        detail = items["resolution_skew"]["detail"]
        assert "median" in detail
        assert "right-skewed" in detail or "dragging the average" in detail

    def test_no_data_rule(self):
        items = InsightEngine(InsightContext(n_total=0)).generate()
        assert items[0]["id"] == "no_data"

    def test_low_sample_rule_fires_below_30(self):
        ids = {i["id"] for i in InsightEngine(InsightContext(n_total=12)).generate()}
        assert "low_sample" in ids

    def test_unreliable_chi_square_is_reported_as_such(self):
        chi = ChiSquareIndependenceTest(
            matrix=[[1, 2], [2, 1]], row_labels=["a", "b"], col_labels=["x", "y"]
        ).run()
        items = {i["id"] for i in InsightEngine(InsightContext(n_total=6, chi_square=chi)).generate()}
        assert "chi_square_unreliable" in items

    def test_a_broken_rule_cannot_break_the_engine(self):
        engine = InsightEngine(InsightContext(n_total=50, resolution={"mean": "not-a-number"}))
        assert isinstance(engine.generate(), list)  # no exception escapes

    def test_rule_count_meets_the_brief(self):
        assert len(InsightEngine.RULES) >= 12


# =========================================================================== #
# service.py
# =========================================================================== #
class TestTTLCache:
    def test_hit_and_miss(self):
        cache = TTLCache(ttl=60)
        assert cache.get("k") is None
        cache.set("k", 123)
        assert cache.get("k") == 123
        assert cache.hits == 1
        assert cache.misses == 1

    def test_expiry(self):
        cache = TTLCache(ttl=-1)  # already expired
        cache.set("k", 1)
        assert cache.get("k") is None

    def test_size_cap_evicts_oldest(self):
        cache = TTLCache(ttl=60, maxsize=2)
        cache.set("a", 1)
        cache.set("b", 2)
        cache.set("c", 3)
        assert len(cache._store) == 2

    def test_clear(self):
        cache = TTLCache()
        cache.set("a", 1)
        cache.clear()
        assert cache.get("a") is None


class TestPrepareFrame:
    def test_resolution_hours_derived_from_timestamps(self):
        frame = make_frame(
            [
                {
                    "id": "a",
                    "reference_code": "CIV-A",
                    "category": "road",
                    "priority": "low",
                    "status": "resolved",
                    "area": "Saddar",
                    "department": "Roads",
                    "created_at": datetime(2026, 8, 1, 0, 0),
                    "resolved_at": datetime(2026, 8, 3, 0, 0),
                    "ai_confidence": 0.9,
                }
            ]
        )
        assert frame.loc[0, "resolution_hours"] == pytest.approx(48.0)
        assert bool(frame.loc[0, "is_open"]) is False

    def test_unresolved_rows_have_no_resolution_time(self):
        frame = make_frame(
            [
                {
                    "id": "a",
                    "reference_code": "CIV-A",
                    "category": "road",
                    "priority": "low",
                    "status": "open",
                    "area": "Saddar",
                    "department": "Roads",
                    "created_at": datetime(2026, 8, 1),
                    "resolved_at": None,
                    "ai_confidence": 0.9,
                }
            ]
        )
        assert frame["resolution_hours"].isna().all()
        assert bool(frame.loc[0, "is_open"]) is True

    def test_negative_durations_become_nan(self):
        frame = make_frame(
            [
                {
                    "id": "a",
                    "reference_code": "CIV-A",
                    "category": "road",
                    "priority": "low",
                    "status": "resolved",
                    "area": "Saddar",
                    "department": "Roads",
                    "created_at": datetime(2026, 8, 5),
                    "resolved_at": datetime(2026, 8, 1),  # before creation: data error
                    "ai_confidence": 0.9,
                }
            ]
        )
        assert frame["resolution_hours"].isna().all()

    def test_missing_columns_are_defaulted(self):
        import pandas as pd

        frame = AnalyticsService()._prepare(pd.DataFrame([{"id": "x"}]))
        for column in ("category", "priority", "status", "area", "department",
                       "resolution_hours", "age_days", "is_open", "ai_confidence"):
            assert column in frame.columns


class TestServiceEndpoints:
    @pytest.fixture
    def service(self):
        return AnalyticsService(frame=synthetic_frame(), use_cache=False)

    @pytest.fixture
    def filters(self):
        from app.schemas.analytics import AnalyticsFilters

        return AnalyticsFilters()

    async def test_overview(self, service, filters):
        payload = await service.overview(filters)
        kpis = payload["kpis"]
        assert kpis["total"] == 240
        assert kpis["resolved"] > 0
        assert 0 <= kpis["resolution_rate"] <= 100
        assert kpis["median_resolution_hours"] is not None
        assert kpis["avg_ai_confidence"] is not None
        assert len(payload["cards"]) >= 8
        assert payload["insights"]
        assert payload["interpretation"]

    async def test_resolution_times_matches_the_contract(self, service, filters):
        payload = await service.resolution_times(filters)
        for key in (
            "n", "unit", "mean", "median", "mode", "min", "max", "range",
            "variance", "std_dev", "ddof", "q1", "q2", "q3", "iqr",
            "lower_fence", "upper_fence", "outliers", "histogram",
            "by_category", "interpretation", "sample_warning",
        ):
            assert key in payload, f"missing contract key: {key}"
        assert payload["ddof"] == 1
        assert payload["unit"] == "hours"
        assert payload["q2"] == pytest.approx(payload["median"])
        assert payload["range"] == pytest.approx(payload["max"] - payload["min"])
        assert payload["iqr"] == pytest.approx(payload["q3"] - payload["q1"], abs=0.02)
        assert payload["skewness"] is not None
        assert payload["kurtosis"] is not None
        assert payload["coefficient_of_variation"] is not None
        assert payload["by_category"]
        assert payload["censoring_note"]
        # validates against the frozen wire schema
        model = ResolutionTimesResponse(**payload)
        assert model.n == payload["n"]

    async def test_resolution_times_outliers_are_named(self, service, filters):
        payload = await service.resolution_times(filters)
        assert payload["outliers"]
        assert payload["outliers"][0]["reference_code"].startswith("CIV-")

    async def test_categories(self, service, filters):
        payload = await service.categories(filters)
        assert payload["distribution"]["n"] == 240
        assert payload["distribution"]["mode"]
        assert payload["resolution_by_category"]
        assert payload["by_status"]["grand_total"] == 240
        assert payload["interpretation"]

    async def test_priorities(self, service, filters):
        payload = await service.priorities(filters)
        assert payload["distribution"]["n"] == 240
        assert payload["crosstab"]["grand_total"] == 240
        assert payload["chi_square"]["statistic"] is not None
        assert "assumption_met" in payload["chi_square"]
        assert payload["spearman_priority_vs_speed"] is not None
        assert payload["interpretation"]

    async def test_trends(self, service, filters):
        payload = await service.trends(filters, days=90)
        assert len(payload["series"]) == 90
        assert payload["forecast"]["points"]
        assert payload["week_over_week"]["interpretation"]
        assert payload["interpretation"]

    async def test_departments(self, service, filters):
        payload = await service.departments(filters)
        assert payload["n"] == 4
        assert sum(d["n"] for d in payload["departments"]) == 240
        assert payload["interpretation"]

    async def test_areas(self, service, filters):
        payload = await service.areas(filters)
        assert payload["n"] == 4
        assert sum(a["n"] for a in payload["areas"]) == 240
        assert "hotspot_rule" in payload
        assert payload["interpretation"]

    async def test_insights(self, service, filters):
        payload = await service.insights(filters)
        assert payload["n"] >= 8
        assert payload["interpretation"]
        for item in payload["insights"]:
            Insight(**item)

    async def test_public_summary_is_anonymous(self, service, filters):
        payload = await service.public_summary(filters)
        assert payload["total_complaints"] == 240
        assert payload["categories"]
        assert payload["interpretation"]
        # nothing identifying may leak into the public payload
        assert "outliers" not in payload
        assert "CIV-" not in str(payload)


class TestServiceEmptyData:
    """Every endpoint must degrade gracefully rather than raising on no data."""

    @pytest.fixture
    def service(self):
        return AnalyticsService(frame=make_frame([]), use_cache=False)

    @pytest.fixture
    def filters(self):
        from app.schemas.analytics import AnalyticsFilters

        return AnalyticsFilters()

    async def test_overview(self, service, filters):
        payload = await service.overview(filters)
        assert payload["kpis"]["total"] == 0
        assert payload["interpretation"]

    async def test_resolution_times(self, service, filters):
        payload = await service.resolution_times(filters)
        assert payload["n"] == 0
        assert payload["mean"] is None
        assert payload["outliers"] == []
        assert payload["sample_warning"]
        ResolutionTimesResponse(**payload)

    async def test_categories(self, service, filters):
        payload = await service.categories(filters)
        assert payload["distribution"]["n"] == 0
        assert payload["interpretation"]

    async def test_priorities(self, service, filters):
        payload = await service.priorities(filters)
        assert payload["chi_square"]["statistic"] is None
        assert payload["chi_square"]["reliable"] is False

    async def test_trends(self, service, filters):
        payload = await service.trends(filters, days=30)
        assert len(payload["series"]) == 30
        assert payload["total"] == 0

    async def test_departments_areas_insights_public(self, service, filters):
        assert (await service.departments(filters))["n"] == 0
        assert (await service.areas(filters))["n"] == 0
        assert (await service.insights(filters))["insights"][0]["id"] == "no_data"
        assert (await service.public_summary(filters))["total_complaints"] == 0


class TestSingleRowData:
    """n = 1 end to end: the whole engine must survive one complaint."""

    @pytest.fixture
    def service(self):
        frame = make_frame(
            [
                {
                    "id": "only",
                    "reference_code": "CIV-ONLY",
                    "category": "road",
                    "priority": "high",
                    "status": "resolved",
                    "area": "Saddar",
                    "department": "Roads",
                    "created_at": datetime(2026, 8, 1),
                    "resolved_at": datetime(2026, 8, 2),
                    "ai_confidence": 0.88,
                }
            ]
        )
        return AnalyticsService(frame=frame, use_cache=False)

    @pytest.fixture
    def filters(self):
        from app.schemas.analytics import AnalyticsFilters

        return AnalyticsFilters()

    async def test_resolution_times(self, service, filters):
        payload = await service.resolution_times(filters)
        assert payload["n"] == 1
        assert payload["mean"] == pytest.approx(24.0)
        assert payload["variance"] is None  # undefined, not zero
        assert payload["std_dev"] is None
        assert payload["sample_warning"]
        ResolutionTimesResponse(**payload)

    async def test_overview_and_insights(self, service, filters):
        overview = await service.overview(filters)
        assert overview["kpis"]["total"] == 1
        ids = {i["id"] for i in (await service.insights(filters))["insights"]}
        assert "low_sample" in ids


# =========================================================================== #
# frame-level inference wrappers
# =========================================================================== #
class TestFrameInference:
    def test_category_priority_chi_square_runs(self):
        result = category_priority_chi_square(synthetic_frame())
        assert result["row_variable"] == "category"
        assert result["col_variable"] == "priority"
        assert "assumption_met" in result
        assert result["interpretation"]

    def test_chi_square_on_empty_frame(self):
        result = category_priority_chi_square(make_frame([]))
        assert result["statistic"] is None
        assert result["reliable"] is False

    def test_priority_vs_resolution_spearman(self):
        result = priority_vs_resolution_spearman(synthetic_frame())
        assert result["x_variable"].startswith("priority rank")
        assert result["n"] > 0
        assert result["interpretation"]


# =========================================================================== #
# router wiring
# =========================================================================== #
class TestRouter:
    def test_router_exposes_every_contract_path(self):
        from app.api.v1.analytics import router

        paths = {route.path for route in router.routes}
        for expected in (
            "/analytics/overview",
            "/analytics/categories",
            "/analytics/priorities",
            "/analytics/resolution-times",
            "/analytics/trends",
            "/analytics/departments",
            "/analytics/areas",
            "/analytics/insights",
            "/analytics/public-summary",
        ):
            assert expected in paths

    def test_public_summary_has_no_auth_dependency(self):
        from app.api.v1.analytics import router

        for route in router.routes:
            if route.path == "/analytics/public-summary":
                names = {d.call.__name__ for d in route.dependant.dependencies if d.call}
                assert not any("admin" in n or "current_user" in n for n in names)

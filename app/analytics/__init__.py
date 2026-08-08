"""Statistics & analytics engine for AI Smart Civic Services.

Layering (each module is independently unit-testable, none of them touch FastAPI):

``descriptive``   n, mean, median, mode, variance/σ (ddof=1), quartiles, IQR,
                  Tukey fences, skewness, kurtosis, CV, histogram, sample warnings.
``distributions`` frequency / relative-frequency / cumulative tables for the
                  categorical variables, modes, and contingency tables.
``outliers``      Tukey-fence detection on resolution time, overall and *within
                  each category*, returned as an actionable worklist.
``timeseries``    gap-free daily counts, 7-day rolling mean, week-over-week,
                  per-category series and a declared naive forecast.
``inference``     chi-square test of independence with its expected-frequency
                  assumption checked and reported, plus Spearman correlation.
``narratives``    deterministic, rules-based plain-English ``Insight`` objects —
                  no LLM, so a number can never be hallucinated.
``service``       ``AnalyticsService``: one query per request into one pandas
                  DataFrame, a 60s TTL cache, and every endpoint payload.

Nothing here imports pandas/numpy/scipy at module scope beyond what is needed, so
the package imports cleanly even while the rest of the backend is being built.
"""

from __future__ import annotations

from app.analytics.descriptive import DescriptiveStats
from app.analytics.distributions import ContingencyTable, FrequencyDistribution
from app.analytics.inference import ChiSquareIndependenceTest, SpearmanCorrelation
from app.analytics.narratives import InsightContext, InsightEngine
from app.analytics.outliers import OutlierDetector
from app.analytics.service import FRAME_CACHE, AnalyticsService
from app.analytics.timeseries import TimeSeriesAnalyzer

__all__ = [
    "FRAME_CACHE",
    "AnalyticsService",
    "ChiSquareIndependenceTest",
    "ContingencyTable",
    "DescriptiveStats",
    "FrequencyDistribution",
    "InsightContext",
    "InsightEngine",
    "OutlierDetector",
    "SpearmanCorrelation",
    "TimeSeriesAnalyzer",
]

"""Descriptive statistics for a single numeric variable.

Why this module exists as a class
---------------------------------
``DescriptiveStats`` wraps one numeric sample and computes every summary statistic
from *one* cleaned, sorted array. Doing it as an object rather than a bag of
functions means the expensive parts (coercion, NaN removal, sorting, central
moments) happen once, and every derived statistic — fences, CV, skewness — is
guaranteed to be computed from the same cleaned sample. It also makes the whole
module unit-testable without a database.

Statistical decisions we are prepared to defend
-----------------------------------------------
1. **Sample, not population (``ddof=1``).** Variance and standard deviation use
   Bessel's correction and the response says so explicitly via ``"ddof": 1``.
   Justification: the complaints in the database are not a closed population — they
   are a snapshot of an ongoing civic process, and we want to infer the behaviour of
   that process (including future complaints), not merely describe the rows we hold.
   Dividing by ``n`` would bias the variance downward. With n in the hundreds the
   numeric difference is small, but the *claim* being made is different, and the
   honest claim is the inferential one.

2. **Quartiles use linear interpolation** (numpy ``method="linear"``, identical to
   R's type-7 and to ``pandas.quantile``). Reported as ``quartile_method`` so nobody
   has to reverse-engineer why our Q1 differs from a hand calculation that used a
   different convention. Tukey fences are then Q1 - 1.5·IQR and Q3 + 1.5·IQR.

3. **Shape statistics are bias-corrected sample estimators.** Skewness is G1 and
   kurtosis is the *excess* kurtosis G2 (Fisher definition: 0 for a normal
   distribution), matching ``scipy.stats.skew(bias=False)`` /
   ``kurtosis(bias=False)`` and ``pandas.Series.skew()``. Using the corrected forms
   is consistent with having chosen ddof=1 everywhere else.

4. **n guards, not silent nonsense.** Variance/std need n≥2, skewness needs n≥3,
   kurtosis needs n≥4. Below those thresholds we return ``None`` rather than 0.0 —
   a zero would read as "no spread", which is a lie about an undefined quantity.

5. **Small samples are labelled.** Below 30 observations we attach a
   ``sample_warning`` instead of presenting the numbers as trustworthy. 30 is the
   conventional rule-of-thumb threshold at which the CLT makes the sampling
   distribution of the mean roughly normal; below it, quartiles and especially
   kurtosis are extremely unstable.

6. **Modes are reported honestly.** Continuous data frequently has *no* repeated
   value (every observation unique → no mode) or several tied values (multimodal).
   We report ``mode_kind`` as ``unique``/``multi``/``none``, list all tied modes, and
   additionally give a ``modal_bin`` from the histogram, which is the meaningful
   notion of "most common" for a continuous variable.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

# Public constants — imported by the other analytics modules so the conventions
# are declared in exactly one place.
QUARTILE_METHOD = "linear"
QUARTILE_METHOD_LABEL = "linear (numpy/R type-7)"
DEFAULT_DDOF = 1
SMALL_SAMPLE_THRESHOLD = 30
TUKEY_MULTIPLIER = 1.5
MAX_MODES_REPORTED = 5


def _coerce_scalar(value: Any) -> float:
    """Best-effort float conversion; anything unusable becomes NaN."""
    try:
        if value is None:
            return float("nan")
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def to_clean_array(values: Iterable[Any] | None):
    """Coerce any iterable (list, ndarray, pandas Series) into a finite float array.

    Non-numeric, ``None``, ``NaN`` and infinite entries are dropped — every statistic
    below is then computed on exactly the same cleaned sample, which is what makes
    ``n`` a meaningful denominator.
    """
    import numpy as np

    if values is None:
        return np.empty(0, dtype="float64")

    # pandas objects expose .to_numpy; use it so nullable dtypes survive
    if hasattr(values, "to_numpy"):
        try:
            arr = np.asarray(values.to_numpy(dtype="float64", na_value=np.nan))
        except (TypeError, ValueError):
            arr = np.asarray([_coerce_scalar(v) for v in values], dtype="float64")
    else:
        try:
            arr = np.asarray(values, dtype="float64")
        except (TypeError, ValueError):
            arr = np.asarray([_coerce_scalar(v) for v in values], dtype="float64")

    arr = arr.reshape(-1)
    if arr.size == 0:
        return np.empty(0, dtype="float64")
    return arr[np.isfinite(arr)].astype("float64", copy=True)


def _round(value: Any, digits: int = 4) -> float | None:
    """Round for the wire, mapping NaN/inf/None to ``None`` rather than emitting NaN.

    JSON has no NaN literal; returning ``None`` keeps the payload valid and forces
    the consumer to handle "not computable" explicitly.
    """
    if value is None:
        return None
    try:
        fval = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(fval):
        return None
    return round(fval, digits)


class DescriptiveStats:
    """Complete descriptive summary of one numeric sample.

    Usage::

        stats = DescriptiveStats(series, unit="hours")
        model = stats.to_model()          # typed pydantic response block
        bins  = stats.histogram()         # Freedman-Diaconis bins
    """

    def __init__(
        self,
        values: Iterable[Any] | None,
        *,
        unit: str = "hours",
        ddof: int = DEFAULT_DDOF,
        small_sample_threshold: int = SMALL_SAMPLE_THRESHOLD,
        name: str = "value",
    ) -> None:
        import numpy as np

        arr = to_clean_array(values)
        arr.sort(kind="mergesort")
        self._values = arr
        self._np = np
        self.unit = unit
        self.ddof = ddof
        self.small_sample_threshold = small_sample_threshold
        self.name = name
        self._notes: list[str] = []

    # ---------------------------------------------------------------- basics
    @property
    def values(self):
        """The cleaned, sorted sample (read-only by convention)."""
        return self._values

    @property
    def n(self) -> int:
        return int(self._values.size)

    @property
    def mean(self) -> float | None:
        if self.n == 0:
            return None
        return float(self._np.mean(self._values))

    @property
    def median(self) -> float | None:
        if self.n == 0:
            return None
        return float(self._np.median(self._values))

    @property
    def minimum(self) -> float | None:
        return float(self._values[0]) if self.n else None

    @property
    def maximum(self) -> float | None:
        return float(self._values[-1]) if self.n else None

    @property
    def value_range(self) -> float | None:
        if self.n == 0:
            return None
        return float(self._values[-1] - self._values[0])

    # ------------------------------------------------------------ dispersion
    @property
    def variance(self) -> float | None:
        """Sample variance (ddof=1). Undefined for n < 2 — returns ``None``."""
        if self.n <= self.ddof:
            return None
        return float(self._np.var(self._values, ddof=self.ddof))

    @property
    def std_dev(self) -> float | None:
        var = self.variance
        if var is None:
            return None
        return float(math.sqrt(var))

    @property
    def standard_error(self) -> float | None:
        """SE of the mean = s / sqrt(n)."""
        sd = self.std_dev
        if sd is None or self.n == 0:
            return None
        return float(sd / math.sqrt(self.n))

    @property
    def coefficient_of_variation(self) -> float | None:
        """CV = s / mean. Only meaningful for a ratio scale with a positive mean.

        Resolution time is a ratio variable (a true zero exists), so CV is valid
        here. We refuse to compute it when the mean is <= 0 because the ratio then
        has no interpretation as "relative variability".
        """
        sd, mu = self.std_dev, self.mean
        if sd is None or mu is None or mu <= 0:
            return None
        return float(sd / mu)

    # -------------------------------------------------------------- position
    def quantile(self, q: float) -> float | None:
        if self.n == 0:
            return None
        return float(self._np.quantile(self._values, q, method=QUARTILE_METHOD))

    @property
    def q1(self) -> float | None:
        return self.quantile(0.25)

    @property
    def q2(self) -> float | None:
        return self.quantile(0.50)

    @property
    def q3(self) -> float | None:
        return self.quantile(0.75)

    @property
    def p90(self) -> float | None:
        return self.quantile(0.90)

    @property
    def iqr(self) -> float | None:
        q1, q3 = self.q1, self.q3
        if q1 is None or q3 is None:
            return None
        return float(q3 - q1)

    @property
    def lower_fence(self) -> float | None:
        """Tukey lower fence: Q1 - 1.5·IQR (may legitimately be negative)."""
        q1, iqr = self.q1, self.iqr
        if q1 is None or iqr is None:
            return None
        return float(q1 - TUKEY_MULTIPLIER * iqr)

    @property
    def upper_fence(self) -> float | None:
        """Tukey upper fence: Q3 + 1.5·IQR."""
        q3, iqr = self.q3, self.iqr
        if q3 is None or iqr is None:
            return None
        return float(q3 + TUKEY_MULTIPLIER * iqr)

    # ----------------------------------------------------------------- shape
    def _central_moment(self, order: int) -> float | None:
        """Population central moment m_k = sum((x - mean)^k) / n."""
        if self.n == 0:
            return None
        deviations = self._values - self.mean
        return float(self._np.mean(deviations**order))

    @property
    def skewness(self) -> float | None:
        """Bias-corrected sample skewness G1 (needs n >= 3).

        G1 = g1 * sqrt(n(n-1)) / (n-2), where g1 = m3 / m2^(3/2).
        Positive => right tail is longer (a few very slow cases), which is exactly
        the shape we expect from service-time data.
        """
        n = self.n
        if n < 3:
            return None
        m2 = self._central_moment(2)
        m3 = self._central_moment(3)
        if m2 is None or m3 is None or m2 <= 0:
            return None
        g1 = m3 / (m2**1.5)
        return float(g1 * math.sqrt(n * (n - 1)) / (n - 2))

    @property
    def kurtosis(self) -> float | None:
        """Bias-corrected *excess* kurtosis G2 (Fisher; 0 for a normal). Needs n >= 4.

        G2 = ((n-1) / ((n-2)(n-3))) * ((n+1) * g2 + 6), with g2 = m4/m2^2 - 3.
        Large positive values mean heavy tails — a warning that the mean and the
        standard deviation are being driven by a handful of extreme cases.
        """
        n = self.n
        if n < 4:
            return None
        m2 = self._central_moment(2)
        m4 = self._central_moment(4)
        if m2 is None or m4 is None or m2 <= 0:
            return None
        g2 = m4 / (m2**2) - 3.0
        return float(((n - 1) / ((n - 2) * (n - 3))) * ((n + 1) * g2 + 6.0))

    # ------------------------------------------------------------------ mode
    def modes(self) -> tuple[list[float], str, int]:
        """Return ``(modes, kind, max_count)``.

        ``kind`` is ``"none"`` when every value occurs exactly once (there is no
        mode — saying "the mode is 3.0" for all-unique data would be meaningless),
        ``"unique"`` for a single most-frequent value, ``"multi"`` when several tie.
        """
        if self.n == 0:
            return [], "none", 0
        uniques, counts = self._np.unique(self._values, return_counts=True)
        max_count = int(counts.max())
        if max_count <= 1:
            return [], "none", max_count
        winners = [float(v) for v in uniques[counts == max_count]]
        kind = "unique" if len(winners) == 1 else "multi"
        return winners[:MAX_MODES_REPORTED], kind, max_count

    # ------------------------------------------------------------- histogram
    def histogram(
        self,
        *,
        method: str = "freedman-diaconis",
        min_bins: int = 5,
        max_bins: int = 24,
        edges: list[float] | None = None,
    ) -> list[dict[str, Any]]:
        """Binned frequency table.

        Default bin width follows the **Freedman–Diaconis rule**
        (``2·IQR·n^(-1/3)``), which is robust to the long right tail typical of
        resolution times — unlike Sturges' rule, which assumes near-normality and
        would under-bin badly skewed data. Bin count is clamped to a readable range.
        """
        np = self._np
        if self.n == 0:
            return []
        lo, hi = float(self._values[0]), float(self._values[-1])
        if edges is None:
            if hi <= lo:
                edges_arr = np.array([lo, lo + 1.0])
            else:
                iqr = self.iqr or 0.0
                width = 0.0
                if method == "freedman-diaconis" and iqr > 0:
                    width = 2.0 * iqr * (self.n ** (-1.0 / 3.0))
                if width <= 0:  # degenerate IQR -> square-root choice
                    width = (hi - lo) / max(1.0, math.sqrt(self.n))
                k = int(math.ceil((hi - lo) / width)) if width > 0 else min_bins
                k = max(min_bins, min(max_bins, max(1, k)))
                edges_arr = np.linspace(lo, hi, k + 1)
        else:
            edges_arr = np.asarray(edges, dtype="float64")

        counts, edge_out = np.histogram(self._values, bins=edges_arr)
        total = max(1, int(counts.sum()))
        bins: list[dict[str, Any]] = []
        for i, count in enumerate(counts):
            start = float(edge_out[i])
            end = float(edge_out[i + 1])
            bins.append(
                {
                    "bin_start": _round(start, 2),
                    "bin_end": _round(end, 2),
                    "count": int(count),
                    "relative_frequency": _round(int(count) / total, 4) or 0.0,
                    "label": f"{start:.0f}–{end:.0f}{'h' if self.unit == 'hours' else ''}",
                }
            )
        return bins

    def modal_bin(self, bins: list[dict[str, Any]] | None = None) -> str | None:
        """Label of the busiest histogram bin — the useful 'mode' for continuous data."""
        bins = bins if bins is not None else self.histogram()
        if not bins:
            return None
        best = max(bins, key=lambda b: b["count"])
        if best["count"] == 0:
            return None
        return str(best["label"])

    # ------------------------------------------------------- interval / warn
    def mean_ci95(self) -> tuple[float | None, float | None, str | None]:
        """95% confidence interval for the mean.

        Uses Student's t with n-1 degrees of freedom when scipy is importable
        (correct for small n); otherwise falls back to the normal approximation and
        says so in the returned method label, so the consumer is never misled about
        which interval they are looking at.
        """
        mu, se, n = self.mean, self.standard_error, self.n
        if mu is None or se is None or n < 2:
            return None, None, None
        try:
            from scipy import stats as _sps

            crit = float(_sps.t.ppf(0.975, df=n - 1))
            label = f"Student's t, df={n - 1}"
        except Exception:  # pragma: no cover - scipy always present in prod
            crit = 1.959963985
            label = "normal approximation (z=1.96)"
        return float(mu - crit * se), float(mu + crit * se), label

    @property
    def sample_warning(self) -> str | None:
        """Non-null whenever the sample is too small to present as trustworthy."""
        n = self.n
        if n == 0:
            return "No observations matched these filters, so no statistics could be computed."
        if n == 1:
            return (
                "Only 1 observation: spread, skewness and quartiles are undefined for a "
                "single data point, so only that value itself is reported."
            )
        if n < 4:
            return (
                f"Only {n} observations. Variance is computable but shape statistics "
                "(skewness/kurtosis) are not, and quartiles are essentially arbitrary."
            )
        if n < self.small_sample_threshold:
            return (
                f"Only {n} observations — below the conventional n≥{self.small_sample_threshold} "
                "threshold. Treat these figures as indicative, not conclusive; quartiles and "
                "outlier fences move a lot when a single case is added."
            )
        return None

    # ----------------------------------------------------------------- exits
    def to_dict(self) -> dict[str, Any]:
        """Rounded, JSON-safe dictionary of every statistic."""
        modes, mode_kind, _count = self.modes()
        bins = self.histogram()
        ci_low, ci_high, ci_method = self.mean_ci95()
        return {
            "n": self.n,
            "unit": self.unit,
            "mean": _round(self.mean, 2),
            "median": _round(self.median, 2),
            "mode": _round(modes[0], 2) if modes else None,
            "modes": [_round(m, 2) for m in modes if _round(m, 2) is not None],
            "mode_kind": mode_kind,
            "modal_bin": self.modal_bin(bins),
            "min": _round(self.minimum, 2),
            "max": _round(self.maximum, 2),
            "range": _round(self.value_range, 2),
            "variance": _round(self.variance, 2),
            "std_dev": _round(self.std_dev, 2),
            "ddof": self.ddof,
            "standard_error": _round(self.standard_error, 3),
            "mean_ci95_low": _round(ci_low, 2),
            "mean_ci95_high": _round(ci_high, 2),
            "ci_method": ci_method,
            "q1": _round(self.q1, 2),
            "q2": _round(self.q2, 2),
            "q3": _round(self.q3, 2),
            "iqr": _round(self.iqr, 2),
            "p90": _round(self.p90, 2),
            "lower_fence": _round(self.lower_fence, 2),
            "upper_fence": _round(self.upper_fence, 2),
            "quartile_method": QUARTILE_METHOD_LABEL,
            "skewness": _round(self.skewness, 4),
            "kurtosis": _round(self.kurtosis, 4),
            "kurtosis_type": "excess (Fisher), bias-corrected G2",
            "coefficient_of_variation": _round(self.coefficient_of_variation, 4),
            "sample_warning": self.sample_warning,
            "notes": list(self._notes),
        }

    def to_model(self):
        """Typed pydantic view (``DescriptiveStatsModel``)."""
        from app.schemas.analytics import DescriptiveStatsModel

        return DescriptiveStatsModel(**self.to_dict())

    # ------------------------------------------------------------- narration
    def shape_phrase(self) -> str:
        """One clause describing the distribution's shape, used by narratives.py."""
        skew = self.skewness
        if skew is None:
            return "too few observations to describe the distribution's shape"
        if skew > 1:
            return "strongly right-skewed (a long tail of slow cases)"
        if skew > 0.5:
            return "moderately right-skewed"
        if skew < -1:
            return "strongly left-skewed"
        if skew < -0.5:
            return "moderately left-skewed"
        return "roughly symmetric"

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<DescriptiveStats n={self.n} mean={self.mean} median={self.median}>"


def describe(values: Iterable[Any] | None, *, unit: str = "hours") -> dict[str, Any]:
    """Functional shorthand used by modules that only need the dictionary."""
    return DescriptiveStats(values, unit=unit).to_dict()


__all__ = [
    "DEFAULT_DDOF",
    "QUARTILE_METHOD",
    "QUARTILE_METHOD_LABEL",
    "SMALL_SAMPLE_THRESHOLD",
    "TUKEY_MULTIPLIER",
    "DescriptiveStats",
    "describe",
    "to_clean_array",
]

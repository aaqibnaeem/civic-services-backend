"""Daily complaint volume, moving averages, week-over-week change and a naive forecast.

The non-negotiable detail: a gap-free index
-------------------------------------------
A ``groupby(date).count()`` only produces rows for dates that *had* complaints. If you
feed that straight into a rolling mean, the window silently slides over a different
number of calendar days for every point — a 7-row window covering 11 real days — and
the "7-day moving average" you print is not a 7-day moving average. Every series here
is therefore reindexed onto a complete ``date_range`` and missing days are filled with
**zero**, which is the correct value: no rows for a day means no complaints were filed
that day, not that the day is unknown.

Forecast method (stated, with its assumptions, because a forecast without them is a guess)
------------------------------------------------------------------------------------------
* With >= 28 days of history we use a **seasonal-naive forecast**: the prediction for
  next Tuesday is the mean of the last four Tuesdays. Civic complaint volume has a
  strong day-of-week signal (weekday reporting during office hours), so a
  day-of-week-aware naive model beats a flat mean.
* With 7-27 days we fall back to the **mean of the last 7 days** (a flat naive
  forecast).
* Below 7 days we use the mean of everything available and mark it unreliable.

Assumptions we state on the wire: no trend extrapolation (a rising trend will be
under-forecast), stable weekly seasonality, and no external shocks — a storm or a
public holiday breaks this model, and it should not be presented as if it could
anticipate one. The interval shown is a rough ±1.96 SD band from the historical
spread of the same weekday, not a rigorous prediction interval.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from datetime import date as date_cls
from typing import Any

from app.analytics.descriptive import DescriptiveStats, _round
from app.analytics.distributions import label_for

ROLLING_WINDOW = 7
FORECAST_HORIZON = 7
SEASONAL_MIN_HISTORY = 28
FLAT_MIN_HISTORY = 7
WEEKDAY_NAMES = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")


def _today_utc() -> date_cls:
    return datetime.now(UTC).date()


class TimeSeriesAnalyzer:
    """Builds every time-based metric from one complaint DataFrame.

    Parameters
    ----------
    frame:
        DataFrame with a datetime ``date_col`` (``created_at`` by default).
    days:
        Length of the analysis window, ending at ``end``.
    end:
        Last date in the window. Defaults to today (UTC), or the newest complaint
        date if the data runs into the future.
    """

    def __init__(
        self,
        frame,
        *,
        date_col: str = "created_at",
        days: int = 90,
        end: date_cls | None = None,
        rolling_window: int = ROLLING_WINDOW,
    ) -> None:
        import pandas as pd

        self._pd = pd
        self.date_col = date_col
        self.days = max(1, int(days))
        self.rolling_window = rolling_window

        if frame is None or date_col not in getattr(frame, "columns", []):
            self.frame = pd.DataFrame(columns=[date_col])
        else:
            self.frame = frame[frame[date_col].notna()].copy()

        observed_max: date_cls | None = None
        if len(self.frame):
            observed_max = pd.to_datetime(self.frame[date_col]).max().date()
        self.observed_max = observed_max

        anchor = end or _today_utc()
        if observed_max and observed_max > anchor:
            anchor = observed_max
        self.end = anchor
        self.start = self.end - timedelta(days=self.days - 1)

    # ------------------------------------------------------------------ index
    def _index(self):
        return self._pd.date_range(start=self.start, end=self.end, freq="D")

    def daily_counts(self):
        """Gap-free daily complaint counts over the window (zeros for empty days)."""
        pd = self._pd
        idx = self._index()
        if len(self.frame) == 0:
            return pd.Series(0, index=idx, dtype="int64")
        days = pd.to_datetime(self.frame[self.date_col]).dt.normalize()
        counts = days.value_counts().sort_index()
        counts.index = pd.DatetimeIndex(counts.index)
        # reindex is the whole point: it both trims to the window and fills the holes
        return counts.reindex(idx, fill_value=0).astype("int64")

    def gaps_filled(self) -> int:
        series = self.daily_counts()
        return int((series == 0).sum())

    def rolling_mean(self, series=None):
        """``rolling_window``-day moving average.

        ``min_periods`` equals the full window, so the first six points are ``NaN``
        rather than a 1-, 2-, 3-day average masquerading as a 7-day one.
        """
        series = self.daily_counts() if series is None else series
        return series.rolling(window=self.rolling_window, min_periods=self.rolling_window).mean()

    # ----------------------------------------------------------------- series
    def series_points(self) -> list[dict[str, Any]]:
        counts = self.daily_counts()
        rolling = self.rolling_mean(counts)
        points: list[dict[str, Any]] = []
        for ts, value in counts.items():
            roll = rolling.get(ts)
            points.append(
                {
                    "date": ts.date(),
                    "count": int(value),
                    "rolling_mean_7": _round(roll, 2),
                }
            )
        return points

    def by_category(self, *, limit: int | None = None) -> list[dict[str, Any]]:
        """One gap-free daily series per category, ordered by total volume."""
        pd = self._pd
        if len(self.frame) == 0 or "category" not in self.frame.columns:
            return []
        idx = self._index()
        work = self.frame.copy()
        work["_day"] = pd.to_datetime(work[self.date_col]).dt.normalize()
        pivot = (
            work.pivot_table(index="_day", columns="category", values=self.date_col, aggfunc="count")
            .reindex(idx)
            .fillna(0)
            .astype("int64")
        )
        out: list[dict[str, Any]] = []
        for column in pivot.columns:
            col = pivot[column]
            total = int(col.sum())
            if total == 0:
                continue
            wow = self._wow_from_series(col)
            out.append(
                {
                    "category": str(column),
                    "label": label_for("category", str(column)),
                    "total": total,
                    "points": [
                        {"date": ts.date(), "count": int(v), "rolling_mean_7": None}
                        for ts, v in col.items()
                    ],
                    "week_over_week_pct": wow["change_pct"],
                }
            )
        out.sort(key=lambda item: -item["total"])
        return out[:limit] if limit else out

    # ------------------------------------------------------- week over week
    def _wow_from_series(self, series) -> dict[str, Any]:
        current = int(series.iloc[-7:].sum()) if len(series) >= 1 else 0
        previous = int(series.iloc[-14:-7].sum()) if len(series) >= 8 else 0
        change = current - previous
        if previous > 0:
            pct = round(change / previous * 100, 1)
        elif current > 0:
            pct = None  # growth from zero is undefined, not "infinite"
        else:
            pct = 0.0
        direction = "flat"
        if change > 0:
            direction = "up"
        elif change < 0:
            direction = "down"
        return {
            "current_week": current,
            "previous_week": previous,
            "change": change,
            "change_pct": pct,
            "direction": direction,
            "window_days": 7,
        }

    def week_over_week(self) -> dict[str, Any]:
        result = self._wow_from_series(self.daily_counts())
        cur, prev, pct = result["current_week"], result["previous_week"], result["change_pct"]
        if prev == 0 and cur == 0:
            text = "No complaints were filed in either of the last two weeks in this filter set."
        elif prev == 0:
            text = (
                f"{cur} complaints were filed in the last 7 days against none the week before, so a "
                "percentage change cannot be computed from a zero base."
            )
        elif pct is None:
            text = f"{cur} complaints this week versus {prev} last week."
        else:
            word = "up" if pct > 0 else ("down" if pct < 0 else "unchanged")
            text = (
                f"{cur} complaints were filed in the last 7 days versus {prev} in the 7 days before "
                f"that — {word} {abs(pct):.1f}%."
            )
        result["interpretation"] = text
        return result

    # -------------------------------------------------------- weekday effect
    def weekday_effect(self) -> dict[str, float]:
        """Mean complaints per calendar weekday across the window."""
        counts = self.daily_counts()
        if len(counts) == 0:
            return {}
        grouped = counts.groupby(counts.index.dayofweek).mean()
        return {WEEKDAY_NAMES[int(k)]: round(float(v), 2) for k, v in grouped.items()}

    def weekend_gap(self) -> tuple[float, float] | None:
        """``(weekday_mean, weekend_mean)`` daily volume, or ``None`` when unavailable."""
        counts = self.daily_counts()
        if len(counts) < 7:
            return None
        is_weekend = counts.index.dayofweek >= 5
        weekend = counts[is_weekend]
        weekday = counts[~is_weekend]
        if len(weekend) == 0 or len(weekday) == 0:
            return None
        return float(weekday.mean()), float(weekend.mean())

    # --------------------------------------------------------------- forecast
    def forecast(self, horizon: int = FORECAST_HORIZON) -> dict[str, Any]:
        """Naive forecast for the next ``horizon`` days, with its method declared."""
        import numpy as np

        counts = self.daily_counts()
        history = int(len(counts))
        points: list[dict[str, Any]] = []

        assumptions = [
            "No trend extrapolation: a sustained rise or fall will be under-forecast.",
            "Weekly seasonality is assumed stable from one month to the next.",
            "No allowance for holidays, weather events or campaigns — a shock breaks this model.",
            "The band is a ±1.96 SD spread of comparable historical days, not a rigorous prediction interval.",
        ]

        if history >= SEASONAL_MIN_HISTORY:
            method = "seasonal naive — mean of the same weekday over the last 4 weeks"
            recent = counts.iloc[-SEASONAL_MIN_HISTORY:]
            by_dow = recent.groupby(recent.index.dayofweek)
            means = by_dow.mean().to_dict()
            stds = by_dow.std(ddof=1).to_dict()
            fallback = float(recent.mean())
            for step in range(1, horizon + 1):
                target = self.end + timedelta(days=step)
                dow = target.weekday()
                mu = float(means.get(dow, fallback))
                sd = float(stds.get(dow, 0.0) or 0.0)
                if not np.isfinite(sd):
                    sd = 0.0
                points.append(
                    {
                        "date": target,
                        "forecast": round(mu, 2),
                        "low": round(max(0.0, mu - 1.96 * sd), 2),
                        "high": round(mu + 1.96 * sd, 2),
                    }
                )
        elif history >= FLAT_MIN_HISTORY:
            method = "flat naive — mean of the last 7 days"
            recent = counts.iloc[-7:]
            mu = float(recent.mean())
            sd = float(recent.std(ddof=1)) if len(recent) > 1 else 0.0
            if not np.isfinite(sd):
                sd = 0.0
            for step in range(1, horizon + 1):
                target = self.end + timedelta(days=step)
                points.append(
                    {
                        "date": target,
                        "forecast": round(mu, 2),
                        "low": round(max(0.0, mu - 1.96 * sd), 2),
                        "high": round(mu + 1.96 * sd, 2),
                    }
                )
            assumptions.append(
                f"Only {history} days of history — too short for a day-of-week model, so every "
                "forecast day is identical."
            )
        else:
            method = "insufficient history — mean of all available days"
            mu = float(counts.mean()) if history else 0.0
            for step in range(1, horizon + 1):
                target = self.end + timedelta(days=step)
                points.append({"date": target, "forecast": round(mu, 2), "low": None, "high": None})
            assumptions.append(
                f"Only {history} days of data. This is an average, not a forecast — do not plan on it."
            )

        expected_total = round(sum(p["forecast"] for p in points), 1)
        last_week = int(counts.iloc[-7:].sum()) if history >= 7 else int(counts.sum())
        if last_week:
            delta = (expected_total - last_week) / last_week * 100
            comparison = (
                f" That is {abs(delta):.0f}% {'above' if delta >= 0 else 'below'} the {last_week} "
                "filed in the last 7 days."
            )
        else:
            comparison = ""
        return {
            "method": method,
            "horizon_days": horizon,
            "assumptions": assumptions,
            "points": points,
            "expected_total": expected_total,
            "interpretation": (
                f"Using a {method}, roughly {expected_total:.0f} complaints are expected over the "
                f"next {horizon} days.{comparison} The method is naive by design — it is a baseline "
                "for capacity planning, not a claim about the future."
            ),
        }

    # ----------------------------------------------------------------- report
    def daily_stats(self) -> DescriptiveStats:
        """Descriptive statistics of the *daily count* series itself."""
        return DescriptiveStats(self.daily_counts(), unit="complaints/day")

    def extremes(self) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        counts = self.daily_counts()
        if len(counts) == 0:
            return None, None
        busiest_ts = counts.idxmax()
        quietest_ts = counts.idxmin()
        return (
            {"date": busiest_ts.date(), "count": int(counts.loc[busiest_ts]), "rolling_mean_7": None},
            {"date": quietest_ts.date(), "count": int(counts.loc[quietest_ts]), "rolling_mean_7": None},
        )

    def interpretation(self) -> str:
        counts = self.daily_counts()
        total = int(counts.sum())
        if total == 0:
            return (
                f"No complaints were filed between {self.start.isoformat()} and "
                f"{self.end.isoformat()} in this filter set."
            )
        stats = self.daily_stats()
        busiest, _quietest = self.extremes()
        wow = self.week_over_week()
        parts = [
            f"{total} complaints were filed over the {self.days}-day window ending "
            f"{self.end.isoformat()}, an average of {stats.mean:.1f} per day "
            f"(median {stats.median:.0f}).",
            wow["interpretation"],
        ]
        if busiest:
            parts.append(
                f"The busiest single day was {busiest['date'].isoformat()} with {busiest['count']} "
                "complaints."
            )
        gaps = self.gaps_filled()
        if gaps:
            parts.append(
                f"{gaps} of the {self.days} days had no complaints at all; those days are filled "
                "with zero so the 7-day moving average covers a real week rather than skipping over "
                "the quiet days."
            )
        if self.observed_max and (self.end - self.observed_max).days > 7:
            parts.append(
                f"Note: the most recent complaint in this filter set is from "
                f"{self.observed_max.isoformat()}, {(self.end - self.observed_max).days} days ago."
            )
        return " ".join(parts)

    def to_dict(self, *, include_forecast: bool = True, category_limit: int | None = 7) -> dict[str, Any]:
        counts = self.daily_counts()
        busiest, quietest = self.extremes()
        return {
            "days": self.days,
            "date_from": self.start,
            "date_to": self.end,
            "total": int(counts.sum()),
            "series": self.series_points(),
            "rolling_window": self.rolling_window,
            "by_category": self.by_category(limit=category_limit),
            "week_over_week": self.week_over_week(),
            "busiest_day": busiest,
            "quietest_day": quietest,
            "daily_stats": self.daily_stats().to_dict(),
            "weekday_effect": self.weekday_effect(),
            "forecast": self.forecast() if include_forecast else None,
            "gaps_filled": self.gaps_filled(),
            "interpretation": self.interpretation(),
        }


__all__ = ["FORECAST_HORIZON", "ROLLING_WINDOW", "WEEKDAY_NAMES", "TimeSeriesAnalyzer"]

---
phase: 04
title: Statistics & Analytics Engine
status: complete
owner: analytics-agent
depends_on: [00, 02]
started: 2026-08-08 23:10
completed: 2026-08-08 23:55
---

# Phase 04 — Statistics & Analytics Engine

## Goal
Deliver the statistics deliverable (15/100 on the rubric, plus one whole batch benchmark): real descriptive,
distributional, inferential and time-series statistics computed over the complaint corpus, exposed through the
nine `/api/v1/analytics/*` endpoints in the frozen contract — and, critically, a plain-English interpretation
layer so that **no endpoint ever returns bare numbers**. The spec's wording is explicit: *"Explain what the
statistics mean rather than displaying numbers only."*

## Scope / file ownership
```
civic-backend/app/analytics/{__init__,descriptive,distributions,timeseries,outliers,inference,narratives,service}.py
civic-backend/app/api/v1/analytics.py
civic-backend/app/schemas/analytics.py
civic-backend/tests/test_analytics.py
docs/phases/PHASE_04_ANALYTICS.md
```
Nothing outside this list was touched. `pandas`, `numpy` and `scipy` were already present in
`pyproject.toml` when `uv add` ran (backend-core/ML had added them), so no manifest edit was needed.

## Tasks
- [x] `descriptive.py` — `DescriptiveStats` class: n, mean, median, mode (multi-modal and no-mode handled
      honestly), min/max/range, sample variance & σ with **ddof=1 reported on the wire**, Q1/Q2/Q3, IQR,
      Tukey fences, bias-corrected skewness (G1) and excess kurtosis (G2), coefficient of variation, SE,
      95% CI for the mean, Freedman–Diaconis histogram, `sample_warning` below n=30
- [x] `distributions.py` — frequency / relative-frequency / cumulative tables for category, priority, status,
      department and area; modes with tie handling; `ContingencyTable` with row totals, column totals and a
      grand total
- [x] `outliers.py` — Tukey-fence detection on resolution time, overall **and recomputed within each
      category**, returned as an actionable worklist keyed by reference code
- [x] `timeseries.py` — gap-free daily counts, 7-day rolling mean, week-over-week, per-category series,
      weekday effect, and a naive forecast that declares its method and assumptions
- [x] `inference.py` — chi-square test of independence with the **expected-frequency assumption checked and
      reported**, Cramér's V graded against df\*-adjusted Cohen thresholds, and Spearman correlation
- [x] `narratives.py` — deterministic rules-based `InsightEngine`, **24 rules**, no LLM
- [x] `service.py` — `AnalyticsService`: one query → one DataFrame → every metric, plus a 60s TTL cache
- [x] `api/v1/analytics.py` — nine endpoints, `require_admin` on eight, `/public-summary` anonymous
- [x] `tests/test_analytics.py` — 115 tests, all against hand-computed fixtures

## Acceptance criteria
- [x] `uv run pytest tests/test_analytics.py` → **115 passed**
- [x] `uv run ruff check app/analytics app/api/v1/analytics.py app/schemas/analytics.py tests/test_analytics.py`
      → All checks passed
- [x] All nine endpoints return `200` against the seeded 800-complaint database; every unauthenticated call
      except `/public-summary` returns `401`
- [x] `/analytics/resolution-times` returns every key in the contract's shape and validates against
      `ResolutionTimesResponse`
- [x] Repeat dashboard loads hit the TTL cache: **1.21 s cold → 0.041 s warm** (~30× faster)

## What was implemented — the metric list

**Descriptive (per numeric variable, `/resolution-times`)**
n · mean · median · mode (+ all tied modes + `mode_kind` + modal histogram bin) · min · max · range ·
**sample variance (ddof=1)** · **sample standard deviation (ddof=1)** · standard error · 95% CI for the mean
(Student's t) · Q1 · Q2 · Q3 · IQR · P90 · Tukey lower/upper fence · skewness (G1) · excess kurtosis (G2) ·
coefficient of variation · Freedman–Diaconis histogram · `sample_warning` · `quartile_method`.

**Categorical distributions (`/categories`, `/priorities`, `/areas`, `/departments`)**
absolute frequency · relative frequency · percent · cumulative frequency · cumulative percent · mode with tie
detection · distinct count · missing count · category × priority and category × status contingency tables with
full margins and row-percentage view.

**Outliers (`/resolution-times`)**
Tukey fences overall · Tukey fences **recomputed within each category** · outlier count and rate · named
outlier list (reference code, category, status, area, value in hours and days, the fence it crossed, how far
past it, and a verdict) · small-group suppression below n=8.

**Time series (`/trends`)**
gap-free daily counts (zero-filled) · 7-day rolling mean (null until the window is genuinely full) ·
week-over-week change and direction · per-category daily series · weekday effect · busiest/quietest day ·
descriptive statistics of the daily-count series itself · 7-day naive forecast with method, assumptions and a
±1.96 SD band.

**Inference (`/priorities`)**
Pearson chi-square test of independence (category × priority; area × category available) — statistic, dof,
p-value, Cramér's V, df\*-adjusted effect-size grade, minimum expected frequency, count and share of cells
below 5, `assumption_met`, `reliable`, `correction_applied`, stated H0/H1 · Spearman rank correlation of
priority rank vs resolution hours.

**KPIs (`/overview`)**
total · open · assigned · in_progress · resolved · rejected · resolution rate · median and mean resolution
time · critical-and-open count · average AI confidence · complaints this week · complaints last week ·
week-over-week delta and direction · backlog · oldest open complaint. Returned both as a typed `kpis` object
and as ready-to-render `cards[]` each carrying its own severity and explanatory hint.

## Example insights produced from the real seeded data (800 complaints)

These are verbatim `title` strings from `GET /api/v1/analytics/insights`, every number computed by the
statistics modules and interpolated by a rule — none generated by a language model:

- `critical` — "20 critical complaints are still unresolved right now."
- `critical` — "160 open complaints (65% of the backlog) have been waiting more than 14 days."
- `critical` — "55% of complaints are rated high or critical — the priority scale has lost its meaning."
- `warn` — "37 complaints took longer than 13.5 days — far beyond the normal range."
- `warn` — "Roads & Potholes has 9 complaints that are abnormally slow even by its own standards."
- `warn` — "Sewerage & Drainage resolves in a median 4.7 days versus 3.1 days across other departments."
- `warn` — "Other complaints rose 75% this week versus last."
- `warn` — "The typical unresolved complaint has been waiting 25.7 days."
- `warn` — "Resolution times are wildly inconsistent — the standard deviation (6.7 days) is larger than the
  average itself (5.4 days)."
- `warn` — "2026-07-16 was an exceptional day with 16 complaints — 3.1 standard deviations above normal."
- `info` — "Half of all complaints are resolved within 3.3 days."
- `info` — "Drainage & Sewerage is the most frequent complaint type — 1 in 5 of everything reported."
- `info` — "Orangi Town is the clearest hotspot with 129 complaints — 16% of the city's total."
- `info` — "Complaint priority is genuinely linked to category — χ²(18) = 158.7, p = 1.477e-24,
  Cramér's V = 0.26."
- `info` — "Triage is working: higher-priority complaints are resolved measurably faster (ρ = -0.43)."
- `info` — "About 57 complaints are expected over the next 7 days."

Each carries a two-to-three sentence `detail` that explains *why the number matters* — e.g. the skew insight's
detail reads: *"The median resolution time is 3.3 days but the mean is 5.4 days — 64% higher, and the
distribution's skewness is 4.19. That gap means a minority of very slow cases is dragging the average upwards,
so the median is the honest headline number…"*

## Decisions

| Decision | Alternatives considered | Why |
|---|---|---|
| **ddof=1 everywhere**, reported on the wire | ddof=0 (population) | The stored complaints are a sample of an ongoing civic process, not a closed population; we want to infer the process's behaviour, so Bessel's correction is the honest estimator. The contract emits `"ddof": 1` so nobody has to guess |
| **Tukey IQR fences** for outliers | ±3σ / z-score rule | The mean and σ are themselves inflated by the outliers being hunted; quartiles are not. Resolution time is strongly right-skewed (G1 = 4.19 on real data), which is exactly where a z-rule fails |
| **Per-category fences in addition to global** | One global fence | A 3-day drainage fix and a 3-day pothole fix are not comparable events. Real data proves the point: the waste fence is 165 h while the drainage fence is 440 h — a single global fence of 323 h would both hide slow waste jobs and libel ordinary drainage work |
| **Rules-based narratives, explicitly not LLM** | LLM-generated summaries | An LLM can invent a statistic or invert a trend. The number in every sentence here is the same Python float computed upstream — interpolated, never generated. Also reproducible, testable, free, and it still works when DeepSeek is down. Documented at length in the `narratives.py` module docstring |
| **Chi-square assumption checked and `reliable` flagged** | Report the p-value regardless | With 7 categories × 4 priorities = 28 cells, sparse combinations are easy to hit. When expected counts drop below 5 the asymptotic approximation breaks and the p-value is wrong. We report the statistic for transparency but mark it unreliable and name the remedy (merge categories / Fisher's exact) |
| **Cramér's V graded on df\*-adjusted Cohen thresholds** | Fixed .10/.30/.50 for every table | Thresholds depend on df\* = min(r-1, c-1). Real result: V = 0.257 on a 7×4 table is *medium*; grading it against the df\*=1 scale would have understated it |
| **One query → one DataFrame → all metrics** | A SQL round trip per metric | 15+ queries per dashboard, mutually inconsistent snapshots, and Tukey/skew/chi-square logic pushed into dialect-specific SQL that differs between SQLite and Postgres |
| **60 s TTL cache keyed on the filter set** | No cache, or `lru_cache` | A dashboard load fires eight analytics calls. Measured: 1.21 s → 0.041 s. `lru_cache` has no time-based expiry, and stale civic numbers are worse than slow ones |
| **`SELECT` only the 14 needed columns, capped at 50 000 rows** | `SELECT *` | Render's free tier is 512 MB. `description` is the largest column and no metric uses it. At the cap the response says so rather than silently describing a subset |
| **7-day rolling mean is null until the window is full** | `min_periods=1` | A 3-day average printed as a "7-day moving average" is a false label. Nulls are honest and the frontend can simply not draw that segment |
| **Missing days zero-filled onto a complete date index** | `groupby(date)` as-is | Without it the rolling window slides over a variable number of calendar days. Test `test_moving_average_uses_calendar_days_not_rows` pins this: the honest 3-day mean is 1.0, the gap-ignoring one would print 2.0 |
| **Right-censoring named in `censoring_note`** | Quote resolution stats silently | Only resolved complaints have a resolution time, and slow cases are the ones most likely still open — so every figure is optimistic. Better to state the survivorship bias than let a judge find it |
| **Defensive column introspection in the loader** | Hard-code the column list | This phase was built in parallel with the model layer. The service reads `Complaint.__table__`, uses whichever expected columns exist, derives `resolution_hours` from timestamps when the column is absent (it is), and finds AI confidence by locating any mapped table with `complaint_id` + `confidence` |
| **`require_admin` on eight endpoints** | `get_current_user` (any role) | Operational analytics — backlog, department comparisons, named slow cases — is management information, not citizen-facing. `/public-summary` covers the public need with aggregates only |
| **Zero rows kept for ordinal variables** | Omit unobserved levels | Priority is ordinal; showing "high: 0" is information. Sorting it by frequency would destroy the ordering |

## Contract deviations
**None that break the wire format.** Every key in the contract's `/analytics/resolution-times` example and in
the `Insight` object is present with the specified type. `Insight` is exactly
`{id, severity, title, detail, metric, unit}` — no extra keys.

Additive fields only (a consumer that ignores them is unaffected):

- `resolution-times` gains `modes[]`, `mode_kind`, `modal_bin`, `p90`, `standard_error`, `mean_ci95_low/high`,
  `quartile_method`, `skewness`, `kurtosis`, `coefficient_of_variation`, `histogram_method`,
  `outlier_report` (the full per-category breakdown), `resolved_count`, `unresolved_count`, `censoring_note`,
  `insights[]`, `filters`. The contract's `mode` field is still a single float — it holds the first tied mode,
  with the honest picture in `modes`/`mode_kind` beside it.
- Histogram bins gain `relative_frequency` and a display `label`.
- Outlier points gain `id`, `category`, `priority`, `status`, `area`, `department`, `value_days`, `fence`,
  `exceeds_fence_by`, `side`, `verdict`, `created_at` — the contract's `reference_code` and `value` are both
  present and unchanged. This is what makes the list actionable rather than a count.
- Every endpoint echoes the `filters` it applied.

## Blockers / notes for other phases

**For the frontend**
- Rank insights by `severity` (`critical` → `warn` → `info`); the array is already returned in that order.
- `rolling_mean_7` is `null` for the first six points of any series **by design** — do not zero-fill it, just
  start the line on day 7.
- `TrendPoint.date` and forecast dates are `YYYY-MM-DD` strings; `generated_at` is a full ISO timestamp.
- `metric` on an `Insight` may be `null` (some rules are qualitative); `unit` may be `null` too.
- `/overview` returns `cards[]` pre-formatted with `display`, `hint` and `severity` — the KPI row can be
  rendered straight from it without client-side maths.
- Every response has an `interpretation` string — putting it under the chart is the single cheapest way to
  score the "explain the statistics" requirement.

**For whoever demos this**
The strongest talking points are: (1) per-category Tukey fences with the real numbers (waste 165 h vs drainage
440 h) as the answer to "why not one global threshold"; (2) the chi-square assumption check as the answer to
"do you know when this test is invalid"; (3) `narratives.py` being deterministic as the answer to "how do you
know the AI didn't make that number up"; (4) the `censoring_note` as evidence we looked for our own bias.

**Nothing is blocking.** The engine works against the live seeded database and degrades gracefully to an
explanatory empty response on zero rows, n=1, and single-category filters — all covered by tests.

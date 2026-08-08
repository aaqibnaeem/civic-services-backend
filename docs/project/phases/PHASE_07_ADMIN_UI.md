---
phase: 07
title: Admin / Service-Team UI
status: complete
owner: admin-ui-agent
depends_on: [04, 05]
started: 2026-08-09 00:05
completed: 2026-08-09 01:40
---

# Phase 07 — Admin / Service-Team UI

## Goal
Ship the whole service-team side of the product against the live backend: a dense triage inbox with
server-side search/filter/sort/pagination, a complaint detail view whose centrepiece is an honest AI
analysis panel, the analytics dashboard that carries **15/100 of the rubric**, a transparent
natural-language assistant, the department directory and a demo-ready login.

**Status: `npx tsc -p tsconfig.app.json --noEmit` is clean, and all six admin routes were exercised
against the seeded 805-complaint database in headless Chrome — light and dark, desktop and mobile —
with zero console errors and zero unhandled rejections.**

---

## Scope / file ownership

Everything below was created or rewritten by this phase. Nothing outside it was touched (the two
lines changed in `AdminComplaintDetailPage`/`StatusTimeline` for the `timeline` guard are inside
this phase's own files).

```
src/layouts/AdminLayout.tsx                       admin shell: sidebar, topbar, ⌘K, AI health
src/pages/admin/AdminInboxPage.tsx                the triage workspace
src/pages/admin/AdminComplaintDetailPage.tsx      two-column detail + action panel
src/pages/admin/AdminAnalyticsPage.tsx            the statistics deliverable
src/pages/admin/AdminAssistantPage.tsx            chat + citations + used_stats
src/pages/admin/AdminDepartmentsPage.tsx          directory, deep-links into the inbox
src/pages/admin/AdminLoginPage.tsx                polished, demo credentials pre-filled
src/components/admin/
  inbox-filters.ts        URL-backed filter state (parse / serialise / toggle / sort)
  InboxToolbar.tsx        search, facets, date range, chips, saved views, density
  FacetFilter.tsx         multi-select popover (also single-select for area/department)
  ComplaintTable.tsx      table + mobile cards + sortable headers + quick status change
  ComplaintActionPanel.tsx status/department/category/priority/note in one PATCH
  AiAnalysisPanel.tsx     the tier-honest AI panel + re-analyse
  DuplicatesPanel.tsx     "possible duplicate of CIV-…" with similarity + reason
  StatusTimeline.tsx      vertical audit trail
  InsightsPanel.tsx       Insight[] ranked by severity — the narrative layer
  AnalyticsFilterBar.tsx  one filter row scoping all eight analytics queries
  OutlierWorklist.tsx     global vs per-category Tukey outliers, linked
  UsedStatsPanel.tsx      "how this answer was computed"
  GlobalSearch.tsx        ⌘K command palette over GET /complaints?q=
src/components/charts/
  chart-kit.tsx           ChartCard · Interpretation · SectionState · ChartTip · axis defaults
  DistributionCharts.tsx  frequency bars + frequency table + contingency heat grid
  ResolutionCharts.tsx    histogram + composed box plot + descriptive stats table
  TrendsChart.tsx         daily volume + 7-day moving average
  ComparisonCharts.tsx    department speed (emphasis form) + area volume + tables
```

~7,500 lines. No shared component, hook, store or config file was modified.

---

## Data-visualisation method (the `dataviz` skill, applied)

The skill was loaded before any chart code was written and its procedure followed in order: **form →
colour job → validate → marks → interaction → accessibility → look at the render**.

### Form choices and why

| Section | Form | Reasoning |
|---|---|---|
| Category / priority frequency | **horizontal bars**, one hue per entity | The categories *are* the subject, so identity colour is correct; horizontal keeps "Electricity & Streetlights" readable without rotated ticks. Colours come from `CATEGORY_META[x].color`, so a bar matches its badge everywhere else in the app. |
| Cumulative percent | **table column, not a Pareto line** | A count axis plus a percent axis is a dual-axis chart — the single worst chart mistake. The cumulative column sits in the adjacent frequency table instead. |
| Category × priority / × status | **heat grid tinted by row percent** | Sequential single hue (`--primary` at 0–30% alpha). Tinting by raw count would make a large category read "hot" everywhere purely because it is large. Every cell still prints its count and row %. |
| Resolution time | **histogram + composed box plot** | Recharts has no box plot, so it is built from positioned HTML: an IQR box, whisker rules, a median rule, the Tukey fence and one dot per outlier. HTML rather than SVG keeps the labels selectable and the outlier dots real focusable buttons. |
| Box plot scale | **two small multiples, not a log axis** | The IQR is ~7% of the full range (median 3.3 d, max 63 d). One row runs 0 → max — honest, and it *shows* the skew. The second clips to the Tukey fence so the box is readable. A log axis would flatter the tail. |
| Trends | **area wash + 2px line, ONE y-axis** | Both series are complaints-per-day, so they legitimately share an axis. `rolling_mean_7` is null for the first six points by backend design; `connectNulls={false}` makes the line simply start on day 7. |
| Departments | **emphasis form** — slowest in accent, rest in de-emphasis grey | Volume, backlog and median time are three different units. Only median time is plotted; the other two live in the table beside it. |
| Areas | volume bars, hotspots in the accent | Same emphasis pattern; the hotspot rule is printed as the chart description. |
| KPIs | **stat tiles**, never one-bar charts | 12 tiles rendered from the backend's pre-formatted `cards[]` plus four derived ones. |

### Palette validation (run, not eyeballed)

The scaffold's OKLCH domain tokens were converted to hex and fed to the skill's validator:

```
light categorical (7): lightness PASS · CVD separation PASS (worst ΔE 10.1 protan)
                       normal-vision floor PASS (worst ΔE 16.7)
                       chroma floor FAIL on --cat-other (0.03)
                       contrast WARN on --cat-electricity (2.61:1 vs white)
dark categorical (7):  CVD separation PASS (ΔE 8.6) · normal-vision PASS (15.1) · contrast PASS
                       lightness band FAIL (dark steps sit 0.71–0.82, above the light-surface band)
```

`src/index.css` is owned by the scaffold and was **not** modified. The two real findings were
answered in the chart layer instead:

- **`--cat-other` below the chroma floor is correct by design** — it is the neutral "Other" slot the
  skill itself recommends for a folded tail, not a failed hue.
- **The `--cat-electricity` contrast WARN obligates relief**, so every categorical chart ships a
  table view beside it and every bar is directly labelled at its tip. Colour is never the only
  channel on this dashboard.
- The dark lightness FAIL is the validator's light-surface band being applied to dark steps; the
  dark palette passes the check that actually matters there (contrast against `--card`).

### Mark and interaction rules honoured

- Bars capped at 24 px (`maxBarSize`), 4 px radius **only at the data end**, square at the baseline.
- Gridlines and axes are solid 1 px `var(--border)` — never dashed, never heavy.
- Legend present for every ≥2-series chart; single-series charts have none (the title names them).
- Custom `ChartTip`: value leads (strong, tabular), series name follows (muted), each row keyed by a
  short colour stroke rather than a filled box.
- Outlier dots carry a 2 px `ring-card` surface ring and a 24 px transparent hit target.
- Filters sit in **one row above everything they scope**; all eight queries take the same object.
- Refetch holds the previous render at reduced opacity (`ChartCard refetching`) — no skeleton flash.
- Every section renders independently via `SectionState`, so a slow endpoint degrades one card.

---

## What was built

### `AdminLayout`
Collapsible sidebar persisted to `uiStore.sidebarCollapsed`, sheet drawer under `lg`, topbar with
⌘K global search, a density toggle bound to `uiStore.density`, `ThemeToggle`, and an
`AiSourceBadge`-shaped health indicator fed by `useAiHealth({refetchIntervalMs: 60_000})`. The
indicator names the **tier that would actually answer right now** (`Tier 1 · LLM`) and its tooltip
lists the whole fallback chain with the dead tiers struck through — CONTRACT §5.3 forbids implying a
better tier than the one that ran.

### `AdminInboxPage`
Filter state lives in the **URL** (`inbox-filters.ts`), which makes any view shareable and lets the
departments page deep-link to `/admin?department_id=…&status=open&status=assigned`. Debounced search
(350 ms) through `useDebouncedValue`; faceted multi-select for category / priority / status plus
single-select department and area and a date range; server-side sort on `created_at`, `priority`,
`status`, `resolution_hours`; removable filter chips with clear-all; saved views persisted to
`uiStore.savedViews`; row-level quick status change through the optimistic `useUpdateComplaint`;
result count ("Showing 1–20 of 804 complaints"); skeleton, empty, filtered-empty and error states;
page-size selector; and full degradation to cards under `md` via `useIsMobile()`.

Columns hide progressively (`Department` and `Resolved in` at `2xl`, `Area` at `xl`) so the AI tier
badge and confidence meter are never clipped at 1440 px.

### `AdminComplaintDetailPage`
Two columns. Left: the raw description, location with coordinates when present, citizen contact,
submitted/updated/resolved times, the **AI analysis panel**, and duplicates. Right: the action panel
and the status timeline.

The AI panel leads with a full-width honesty strip — tier badge, the tier's plain-English meaning,
model name and latency — then the emergency flag, summary, category, priority, suggested department,
sentiment, `ConfidenceMeter`, extracted keywords and the model's own reasoning, with a re-analyse
button. Duplicates render as "possible duplicate of CIV-XXXXXX" with the similarity percentage and
the backend's human-readable reason ("65% wording similarity; shares 'standing', 'drained'; 4970 m
away; same category; filed 38 days apart").

### `AdminAnalyticsPage`
KPI row (12 tiles) → **insights panel near the top, not buried** → four tabs (Distributions ·
Resolution time · Trends · Departments & areas) over all eight analytics hooks. Every section prints
its `interpretation` string as body copy beside its chart, plus `sample_warning` and
`censoring_note` as visible caveats. Chi-square gets its statistic, dof, p, Cramér's V, minimum
expected count, the assumption check and the stated H₀/H₁; Spearman gets ρ, n, p and strength. The
outlier worklist offers the city-wide fence and the per-category fences as separate tabs, each row
linking into triage. A date/category/area filter bar scopes all eight queries at once.

### `AdminAssistantPage`
Chat over `useAssistantChat` with six example question chips so a demo never opens on an empty box,
rendered citations linking to complaint detail, and a collapsible **"How this answer was computed"**
panel showing question type, time window, rows matched, the filters applied to the database, the
group-by breakdown the sentence was written from, and the raw JSON one click further down.

### `AdminDepartmentsPage`
Cards joining `useDepartments()` with `useAnalyticsDepartments()` — categories owned, contact,
open count, total handled, median resolution time and resolution rate, with "slowest" and "biggest
backlog" flags and two deep links each into a pre-filtered inbox.

### `AdminLoginPage`
Demo credentials **pre-filled** with a visible hint card and a reset link, inline (not just toast)
error handling, product branding and three capability bullets.

---

## Verification

Driven over CDP in headless Chrome (1440×1000 and 390×844) against the live backend:

- [x] `npx tsc -p tsconfig.app.json --noEmit` → clean
- [x] `npx oxlint` on all phase files → only the scaffold's known `only-export-components`
      fast-refresh warnings
- [x] Login with the pre-filled demo credentials → lands on `/admin`
- [x] All six admin routes render with **zero** console errors/warnings, zero exceptions, zero ≥400
      responses — verified in light **and** dark
- [x] Inbox: facet filter → `?priority=critical`, "Showing 1–20 of 96 complaints matching your
      filters", chip rendered, chip removal restores 804
- [x] Quick status change: `CIV-N7DAXA` Open → In Progress, optimistic in <1 s, PATCH observed,
      success toast, value persisted after refetch
- [x] Terminal status: the quick-change control is disabled on a `resolved` row with an explanatory
      tooltip; the detail page shows the terminal banner and disables the status select. A 409 from
      the API is surfaced verbatim ("Cannot move a complaint from 'resolved' to 'in_progress'.")
- [x] Duplicates panel renders a real candidate with similarity and reason
- [x] Analytics: every tab renders; the date filter (Last 30 days) recomputes the whole dashboard
      (804 → 240 complaints)
- [x] Assistant: real answer, citations, `used_stats` panel opens and shows the group-by breakdown
- [x] ⌘K search: "manhole" → "25 matches — showing the 8 most recent"
- [x] Mobile: 20 cards render, the table is absent, no horizontal page overflow

---

## Decisions

| Decision | Alternatives considered | Why |
|---|---|---|
| **Inbox filter state in the URL**, not component state | `useState` + a context | Makes any triage view shareable, survives refresh, gives the departments page a free deep link, and keeps the back button meaningful. `setSearchParams(..., {replace:true})` keeps history clean while typing. |
| **Composed box plot from HTML, not SVG** | An SVG `<g>` per element; a plugin | Recharts has no box plot. HTML keeps the numeric labels selectable and screen-reader reachable, lets each outlier be a real focusable `<button>` with a 24 px hit target, and needs no scale maths beyond a percentage. |
| **Two box plots at two domains** | One log axis; one linear axis | A log axis makes a 63-day case look only twice as bad as a 13-day one. Two linear small multiples keep the honest picture *and* the readable one, and the skill endorses small multiples over axis tricks. |
| **Tabs on the analytics page** | One long scroll | Eight endpoints' worth of charts, tables and prose is far past a comfortable scroll. The tab bar is sticky under the topbar so it reads as navigation, and the KPI row plus the insights panel stay above it — the two things a judge must not have to hunt for. |
| **Insights near the top** | A footer summary | The brief grades "explain what the statistics mean rather than displaying numbers only". Those 21 sentences are the highest-value thing on the page, so they sit directly under the KPI row with severity colour and counts. |
| **Every categorical chart ships a table beside it** | Chart only | Required relief for the `--cat-electricity` contrast WARN, and it is where the relative/cumulative frequency columns the statistics brief asks for actually live. |
| **Department chart plots median time only** | Grouped bars of volume + time + backlog | Three units on one axis is the dual-axis anti-pattern. Volume, backlog, resolution rate and P90 go in the table underneath, where units can differ per column legitimately. |
| **Emphasis colouring for departments and areas** | Full categorical palette | The story is "one department is slow" / "two areas are hotspots", not "here are six departments". Accent + de-emphasis grey says it in one glance. |
| **`SectionState` per analytics section** | One page-level loading gate | Eight requests; the slowest must never hold the other seven. Each section owns its skeleton, error and retry. |
| **Terminal statuses disabled in the UI, 409 still handled** | Rely on the error toast alone | Letting a staff member click into a guaranteed 409 is bad triage UX. The control is disabled with the reason, and the 409 path is still implemented because another tab can change the status underneath you. |
| **Assistant transcript in component state** | TanStack cache | A retry must not replay a stale conversation; the scaffold's `useAssistantChat` is a mutation for exactly this reason. |
| **`shouldFilter={false}` on the ⌘K palette** | cmdk's default fuzzy filter | The server already searched and ranked. Letting cmdk re-filter would silently drop rows the API considered matches. |

---

## Blockers / notes for other phases

### 1. `PATCH /complaints/{id}` and `POST /complaints/{id}/reanalyze` return `Complaint`, not `ComplaintDetail`

Verified against the live API — neither response contains `timeline`. That matches CONTRACT §3, so
**the backend is right**. The problem is in the scaffold: `useUpdateComplaint` and
`useReanalyzeComplaint` are typed `UseMutationResult<ComplaintDetail, …>` and do

```ts
queryClient.setQueryData(qk.complaints.detail(complaint.id), complaint)
```

which overwrites a cached `ComplaintDetail` with an object that has **no `timeline` key**. Anything
reading `detail.timeline` throws `Cannot read properties of undefined (reading 'length')` in the
window between the mutation succeeding and its invalidation refetch landing. This was reproduced in
the browser before it was worked around.

*Worked around here* by defaulting `StatusTimeline`'s `events` to `[]` and passing
`data.timeline ?? []`. **The real fix belongs in `src/hooks/useComplaints.ts`** — either type both
mutations as `Complaint` and merge into the cached detail (`{...previous, ...response}`), or have
the backend return the timeline on PATCH. Please pick one; the current typing is a latent crash for
any future consumer of the detail cache.

### 2. Two Vite dev servers on one checkout corrupt `node_modules/.vite/deps`

Running the citizen server on 5173 and the admin server on 5174 from the same directory makes both
write to the same optimize-deps cache. The result is **two copies of React**, and Recharts dies with
`Invalid hook call` / `Cannot read properties of null (reading 'useContext')` — the analytics page
was completely blank until this was diagnosed. It is a tooling collision, not an application bug:
the same code renders perfectly once the second server has its own `cacheDir`.

If two agents need to run dev servers concurrently again, give the second one a config with
`cacheDir: 'node_modules/.vite-admin'`. (A throwaway config was used for this phase's verification
and has been deleted; nothing in the repo depends on it.)

### 3. Backend / analytics observations

Everything under `/analytics/*` returned exactly the documented shape. Three things worth knowing:

- `resolution-times.outliers` is **truncated to 25** while `outlier_report.outlier_count` reports the
  true 37. The worklist therefore prefers `outlier_report.outliers` and falls back to the short
  array. Not a bug, but the two numbers disagree on screen unless you know why.
- `mode_kind` is `"multi"` on the real data (five tied modes at 10.33 h, 16.04 h, …). The
  descriptive-stats table renders "10.3 h (+4 tied)" and says outright that continuous data rarely
  has one true mode, rather than pretending `mode` is a single meaningful value.
- `/ai/health` returns several fields beyond the contract (`active_tier`, `chain[]`,
  `circuit_breaker`, `ml`, `cache`). `AiHealthResponse` in `types.ts` does not declare them, so the
  layout derives the active tier from the three documented booleans instead. Adding `active_tier?`
  and `chain?` as optional fields would let the UI show the circuit-breaker state, which is a good
  demo detail.
- `POST /assistant/chat` matched the contract exactly, and `used_stats` is rich enough
  (`plan`, `filters_applied`, `breakdown.groups`, `planner_source`, `writer_source`) to render a
  genuinely convincing transparency panel. Worth keeping stable.

### 4. Nothing else is blocking

`/admin` is demo-ready end to end against the seeded database, in both themes, on desktop and phone.

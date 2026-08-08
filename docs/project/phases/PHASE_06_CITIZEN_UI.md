---
phase: 06
title: Citizen UI
status: complete
owner: agent
depends_on: [00, 01, 05]
started: 2026-08-08 23:40
completed: 2026-08-09 00:35
---

# Phase 06 — Citizen UI

## Goal
Ship the whole public side of the product on top of the Phase 05 scaffold: a landing page that
explains the project in five seconds, a multi-step report flow where the AI visibly does its work,
public tracking by reference code, and the anonymous "my reports" list.

**Status: `npx tsc -p tsconfig.app.json --noEmit` is clean, and every citizen route was exercised in
headless Chrome against the live backend (801+ seeded complaints) in light and dark mode with zero
JavaScript console errors. A real complaint was submitted, tracked and listed end-to-end:
`CIV-N7DAXA` (drainage, high, LLM tier, routed to Sewerage & Drainage).**

---

## Files owned by this phase

```
src/pages/LandingPage.tsx          hero · live stats · categories · how it works · AI story · insights
src/pages/ReportPage.tsx           4-step form, live analyze-preview, override, submit, success
src/pages/TrackPage.tsx            reference-code lookup + this-device shortcuts
src/pages/TrackDetailPage.tsx      public status, derived timeline, live AI reveal
src/pages/MyReportsPage.tsx        trackedStore list with live status per card
src/pages/NotFoundPage.tsx         on-brand 404 (renders outside PublicLayout)
src/layouts/PublicLayout.tsx       sticky header, mobile sheet, footer with API-docs link
src/components/citizen/**          new: everything below
```

`src/components/citizen/`:

| File | What it is |
|---|---|
| `PageShell.tsx` | `PageShell` (measured column) + `Band` (full-bleed landing band) |
| `Stepper.tsx` | Keyboard-navigable 4-step progress header; icon-only under `sm` |
| `AnalyzingPanel.tsx` | The narrated "AI is working" animation (4 real pipeline stages) |
| `AiAnalysisCard.tsx` | One `AIAnalysis` rendered honestly; optional citizen category override |
| `StatusTimeline.tsx` | Vertical public progress timeline derived from a `Complaint` |
| `SubmitSuccess.tsx` | Confirmation screen built around the copyable reference code |
| `TrackedReportCard.tsx` | One `/my-reports` card; fetches its own live status by code |
| `LiveStats.tsx` · `CategoryStrip.tsx` · `HowItWorks.tsx` · `AiExplainer.tsx` | Landing sections |
| `useTrackedRefs.ts` | Memoised, sorted view of `trackedStore` (see landmine 2) |
| `constants.ts` · `utils.ts` | Karachi areas, AI copy, date helpers |

Nothing outside that list was changed. (One accidental cross-edit into
`src/pages/admin/AdminComplaintDetailPage.tsx` — a `break-words` → `wrap-break-word` class rename from
a bulk `sed` — was reverted immediately.)

---

## Landmines hit (worth knowing for the rest of the project)

**1. Firing a TanStack mutation inside a mount effect breaks under React StrictMode.**
`/report` starts the AI preview when the review step mounts. With `<StrictMode>` (which `main.tsx`
enables) the request fired, the server answered `200` in milliseconds — and the mutation stayed
`status: "pending"` forever, so the UI never revealed the result. The observer is detached by the
StrictMode remount and never picks the settled mutation back up. **Fix: defer the call by one tick**
(`setTimeout(…, 0)` inside the effect, cleared in the effect's cleanup) so only the settled mount
fires it. Any other page that kicks off a mutation from an effect will hit the same thing.

**2. `selectTrackedRefs` cannot be passed directly to `useTrackedStore`.**
`export const selectTrackedRefs = (s) => [...s.refs].sort(...)` builds a **new array every call**.
Zustand v5 reads through `useSyncExternalStore`, which compares snapshots by identity, so the page
died with `Maximum update depth exceeded` / "The result of getSnapshot should be cached". Both
`/my-reports` and `/track` now use `useTrackedRefs()`, which selects the raw slice and sorts inside a
`useMemo`. The scaffold's docstring recommending direct use is a trap — worth fixing in `trackedStore`.

**3. `usePollUntilAnalyzed` polls forever on an error.**
Its `refetchInterval` callback only stops when `data.ai_status !== 'pending'`; with no data at all
(e.g. an unknown reference code) `dataUpdatedAt` is `0`, so it keeps re-requesting every 2 s. On a bad
code that is an endless stream of 404s. `TrackDetailPage` therefore holds a `halted` flag, disables the
hook on the first error, and re-enables it from the retry button. Worth fixing in the hook itself.

**4. Radix `Select` must be controlled from the first render.**
`value={field.value || undefined}` warns "changing from uncontrolled to controlled". Radix treats `''`
as "nothing selected" (the placeholder still shows), so `value={field.value ?? ''}` is the right shape
for a react-hook-form `Controller`.

**5. `sm:` breakpoints are viewport-based, which breaks a card reused in a sidebar.**
`AiAnalysisCard` renders in the report flow (wide) and in the tracking page's narrow right column. Its
3-column `dl` overlapped badges in the sidebar. It now uses Tailwind v4 container queries
(`@container` on the card body, `@md:grid-cols-3`), so it lays itself out from its own width.

---

## Decisions

| Decision | Alternatives | Why |
|---|---|---|
| `PublicLayout` leaves `<main>` full-bleed; pages opt into `<PageShell/>` | Keep `max-w-6xl` on `main` | The landing hero and section bands need to run edge to edge. Every other page wraps itself in one component, so the rhythm stays consistent. |
| The landing page has **no** `<PageHeader/>` | Follow the Phase 05 convention everywhere | Its hero *is* the page header; a second heading block above it would look like a mistake. Every other citizen page does start with `<PageHeader/>`. |
| The analysing narration has a 2.4 s **minimum** | Reveal the instant the response lands | With the local ML tier answering in ~10 ms the animation was invisible. The stages describe what the analyzer genuinely does, and the card then shows the **true** latency and model name in the tier tooltip, so nothing is overstated. |
| Submit is never disabled while the AI preview is running | Block until the preview returns | CONTRACT §5.1 — nothing about filing a complaint may depend on the analyzer. |
| The public timeline is **derived** from the complaint | Show `StatusEvent[]` | `GET /complaints/track/{code}` does not return `timeline` (see below). The timeline shows submission, AI analysis, routing and resolution with real timestamps, marks lifecycle steps it cannot timestamp as "no public timestamp" rather than inventing one, and says so in a footnote. |
| Category override is worded as "we will send your choice with the report" | "Your choice wins" | The backend overwrites `complaint.category` with the AI's result (see below), so a stronger claim would be false. The override still ships as `ComplaintCreate.category`. |
| Contact details are never rendered on `/track/:code` | Show them to whoever has the code | The code is shareable; names, phones and emails are not. The page says so. |
| Consent is an explicit checkbox on the review step | Send `consent: true` silently | The API requires it; a government form should ask. It is pre-ticked and persisted with the draft. |
| `/my-reports` removal offers **Undo** in the toast | Confirm dialog | The list is the only record an anonymous citizen has; an accidental delete must be recoverable, but a modal on every removal is heavy. |

---

## Accessibility / responsiveness

- Stepper is an `<ol>` of real buttons: Tab-reachable, `aria-current="step"`, steps ahead of the
  citizen are genuinely `disabled`, and the accessible name always includes "Step N of 4" even where
  the visible label is dropped at 360 px.
- The AI reveal and the tracking AI panel sit in `aria-live="polite"` regions with `aria-busy`.
- The analysing narration is a `role="status"` list plus a labelled `role="progressbar"`.
- Every input is label-bound (`FieldLabel htmlFor`), errors render through `FieldError` (`role="alert"`).
- Verified at 360 px: `scrollWidth === clientWidth` on the landing, report and tracking pages — no
  horizontal overflow anywhere.
- Light and dark verified by screenshot on all six routes; only design tokens are used, no raw colours.

---

## Verified against the live backend

| Check | Result |
|---|---|
| `/analytics/public-summary` drives every landing number | 801 complaints, 537 resolved (67 %), median 3.3 days, 12 areas |
| `/ai/health` drives the "analyzer online · <tier>" chip and the highlighted tier card | Correct through an LLM-down → LLM-up transition during the session |
| `POST /complaints/analyze-preview` on the review step | `road/critical` and `drainage/high` results, `source: llm`, `deepseek-v4-flash`, ~2.5 s |
| Draft survives a reload mid-flow | Restored to step 4 with a "Draft restored / Start over" banner |
| `POST /complaints` → success screen → track → my reports | **`CIV-N7DAXA`** submitted, code copied, tracked, listed |
| Unknown code `CIV-NOPE99` | Clean "No report with that code" state, polling stopped |
| Malformed code `/track/hello` | "…is not a reference code" state, no request made |
| Unknown path | On-brand 404 |
| Console errors across all routes, both themes | none (only the expected HTTP 404 for a deliberately bad code) |

---

## Notes for other phases

- **Backend — `GET /complaints/track/{code}` should return `timeline`.** `StatusEvent` rows *are*
  created (`"Complaint submitted by citizen."` on create, and one per PATCH), but the public schema
  omits them, so the citizen-facing timeline has to be derived and cannot show when a complaint was
  assigned or started. A public-safe projection (`from_status`, `to_status`, `created_at`, and the
  note but **not** `actor`) would make this page materially better.
- **Backend — the citizen `category` hint is silently discarded.**
  `_apply_analysis_to_complaint` sets `complaint.category = result.category` unconditionally, so a
  citizen who corrects the AI has no effect on the stored record (verified: hint `safety` on a waste
  complaint → stored as `waste`). Options: only override when the AI's confidence beats a threshold,
  or persist the hint in its own column so staff can see the disagreement. The UI is worded to avoid
  promising more than the backend delivers, but the honest version of this feature needs the backend.
- **Scaffold — three fixes worth making** (all worked around in this phase, none blocking):
  `trackedStore.selectTrackedRefs` identity, `usePollUntilAnalyzed` error-state polling, and the
  StrictMode-safe pattern for mutations fired from effects.
- The footer's API-docs link points at `http://localhost:8000/docs` in dev (the Vite proxy only
  forwards `/api`) and at `${API_ROOT_URL}/docs` in production.

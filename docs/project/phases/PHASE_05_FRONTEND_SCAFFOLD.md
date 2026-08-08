---
phase: 05
title: Frontend Scaffold
status: complete
owner: agent
depends_on: [00, 01]
started: 2026-08-08 23:09
completed: 2026-08-08 23:58
---

# Phase 05 — Frontend Scaffold

## Goal
Stand up `civic-frontend/` so two page-building agents can start immediately with zero setup, zero
guessing and zero type ambiguity: a working Vite + React + Tailwind v4 + shadcn tree, a complete
design system, a fully typed API layer mirroring the frozen contract, TanStack Query hooks for every
endpoint, four persisted Zustand stores, the full route table with placeholder pages, and a shared
component kit.

**Status: `npm run build` passes with zero TypeScript errors, and all 12 routes render in headless
Chrome with zero console errors in both light and dark mode.**

---

## Versions actually installed (verified 2026-08-08, not from memory)

| Package | Version | Note |
|---|---|---|
| `vite` | 8.2.1 | `create-vite@9` now scaffolds Vite 8 + **oxlint** (not eslint) |
| `react` / `react-dom` | 19.2.8 | |
| `typescript` | 6.0.3 | **`baseUrl` is deprecated and errors** — see landmine 1 |
| `@vitejs/plugin-react` | 6.0.4 | |
| `tailwindcss` / `@tailwindcss/vite` | 4.3.3 | CSS-first `@theme`, no `tailwind.config.js` anywhere |
| `shadcn` (CLI + runtime dep) | 4.16.2 | preset `radix-nova`, base `radix` |
| `radix-ui` | 1.6.7 | single unified package, not `@radix-ui/react-*` |
| `@tanstack/react-query` | 5.101.4 | |
| `@tanstack/react-query-devtools` | 5.101.4 | dev dependency, lazy-loaded |
| `react-router-dom` | 7.18.2 | data router |
| `zustand` | 5.0.14 | |
| `recharts` | 3.8.0 | pinned by the shadcn `chart` registry item; React 19 support is stable |
| `react-hook-form` | 7.85.0 | |
| `zod` | 4.4.3 | `z.email()`, **not** `z.string().email()` |
| `@hookform/resolvers` | 5.7.1 | |
| `lucide-react` | 1.30.0 | v1 renamed many icons — see landmine 4 |
| `sonner` | 2.0.7 | |
| `date-fns` | 4.4.0 | |
| `tailwind-merge` | 3.6.0 | |
| `class-variance-authority` | 0.7.1 | |
| `tw-animate-css` | 1.4.0 | replaces `tailwindcss-animate` |
| `cmdk` 1.1.1 · `vaul` 1.1.2 · `react-day-picker` 10.0.1 · `next-themes` 0.4.6 · `@fontsource-variable/geist` 5.3.0 | | pulled in by shadcn components |

`react-leaflet` **was researched but not installed.** v5.0.0 is React-19-only and would work, but maps
are not in the locked stack; if a page needs one, `npm i react-leaflet@5 leaflet @types/leaflet` and
import `leaflet/dist/leaflet.css` in that page module.

---

## Setup landmines hit (and how they were solved)

**1. TypeScript 6 rejects `baseUrl` — the shadcn docs are stale.**
`ui.shadcn.com/docs/installation/vite` still tells you to add `"baseUrl": "."` alongside `paths`. On
TS 6 that is a hard error (`TS5101: Option 'baseUrl' is deprecated…`) and the build fails. Fix: keep
only `paths` in both `tsconfig.json` and `tsconfig.app.json` — in TS 6 they resolve relative to the
tsconfig's own directory. The shadcn CLI still validates the alias correctly, and `@/*` works in both
`tsc` and Vite (the `resolve.alias` in `vite.config.ts` is still required — both halves are needed).

**2. The shadcn v4 CLI changed its init flags.**
`--base-color` no longer exists. The flow is now `shadcn init -b <base> --preset <name>`, where base is
`base | radix | aria` and preset is one of `nova, vega, maia, lyra, mira, luma, sera, rhea`. Without
`--preset` it drops into an interactive picker and hangs a non-interactive shell. We used
**`shadcn init -b radix --preset nova -y --css-variables`**. Note this adds `shadcn` itself as a runtime
dependency, because `src/index.css` does `@import "shadcn/tailwind.css"` — do not remove it.

**3. `shadcn add form` is a no-op — the `form` component is gone.**
It was replaced by **`field`** (`Field`, `FieldGroup`, `FieldLabel`, `FieldDescription`, `FieldError`,
`FieldSet`, `FieldLegend`) used with react-hook-form's `<Controller>`. `npx shadcn add form` exits 0
having written nothing, which is easy to miss. We installed `field` + `radio-group` + `empty` + `spinner`
instead. **See `src/pages/admin/AdminLoginPage.tsx` for the working reference pattern.**

**4. lucide-react v1 renamed a pile of icons.**
`Filter`→`Funnel`, `AlertCircle`→`CircleAlert`, `AlertTriangle`→`TriangleAlert`, `XCircle`→`CircleX`,
`CheckCircle2`→`CircleCheck`, `HelpCircle`→`CircleQuestionMark`, `Loader2`→`LoaderCircle`,
`BarChart3`→`ChartColumn`, `MoreHorizontal`→`Ellipsis`, `Home`→`House`, `History`→(gone), `Waves`→(gone).
Deprecated aliases exist for *some* but not all. If an icon import fails, check
`node_modules/lucide-react/dist/lucide-react.d.ts` rather than guessing.

**5. `openapi-typescript@7` peer-depends on `typescript@^5` and refuses to install.**
Rather than `--legacy-peer-deps` (which would poison every future `npm install` for the page agents),
`gen:types` invokes it via `npx -y openapi-typescript@7`. Nothing is added to `package.json` deps.

**6. `zod@4` moved the string format validators.**
`z.string().email()` is removed; use `z.email()`, `z.url()`, `z.uuid()`.

---

## Scope / file ownership

This phase owns everything under `civic-frontend/` **except** page bodies. The two page agents own
`src/pages/**` from here on. Do not edit `src/lib/api/types.ts` without updating `docs/CONTRACT.md`.

```
civic-frontend/
├─ vite.config.ts           @ alias + tailwind plugin + /api → :8000 proxy
├─ tsconfig.json            paths only (no baseUrl — TS6)
├─ tsconfig.app.json        paths only, strict
├─ components.json          shadcn config (radix-nova)
├─ vercel.json              SPA rewrite
├─ .env.example             VITE_API_URL
├─ index.html               meta + no-flash theme script
└─ src/
   ├─ index.css             ★ the entire design system
   ├─ main.tsx              initTheme() then render
   ├─ App.tsx               QueryClientProvider → TooltipProvider → RouterProvider → Toaster
   ├─ vite-env.d.ts         typed import.meta.env
   ├─ lib/
   │  ├─ utils.ts           cn()
   │  ├─ domain.ts          ★ enum → label / colour / icon / description
   │  ├─ query-client.ts    QueryClient defaults + global error toasts
   │  └─ api/
   │     ├─ types.ts        every contract object + enum
   │     ├─ client.ts       fetch wrapper, ApiError, 401 handling
   │     ├─ endpoints.ts    one function per endpoint
   │     └─ queryKeys.ts    qk.* key factory
   ├─ stores/               authStore · draftStore · trackedStore · uiStore
   ├─ hooks/                useComplaints · useAnalytics · useAi · useAuth · useDepartments · utils
   ├─ components/           shared kit + ui/ (shadcn primitives)
   ├─ layouts/              PublicLayout · AdminLayout
   ├─ routes/               index.tsx · ProtectedRoute · RouteError · RouteFallback
   └─ pages/                one file per route (placeholders — YOURS to replace)
```

---

## What the page agents can rely on

### Hooks — `import { … } from '@/hooks'`

| Hook | Returns / does |
|---|---|
| `useComplaints(filters?, enabled?)` | `Page<Complaint>` = `{items,total,page,page_size,pages}`. Uses `keepPreviousData` so the table doesn't blank on page change. |
| `useComplaint(id)` | `ComplaintDetail` incl. `timeline: StatusEvent[]`. Disabled when `id` is falsy. |
| `useTrackComplaint(referenceCode)` | Public `Complaint` by code. Never retries a 404. |
| `useDuplicates(id, enabled?)` | `{candidates:[{complaint, similarity, reason}]}` |
| **`usePollUntilAnalyzed(idOrRef)`** | Polls every 2 s until `ai_status !== 'pending'`, then stops (90 s ceiling). Returns `{complaint, isAnalyzing, isAnalyzed, isFailed, timedOut, isPending, error, refetch}`. **Pass a reference code (`CIV-…`) on public pages and a UUID on admin pages — the route is auto-detected.** |
| `useCreateComplaint()` | `POST /complaints`. On success it writes the reference code into `trackedStore` and seeds the track cache for you. |
| `useUpdateComplaint()` | `mutate({id, patch})`. **Optimistic** across the detail cache *and* every cached list page, with rollback on error, success toast, and invalidation of detail + lists + all analytics. |
| `useReanalyzeComplaint()` | `POST /complaints/{id}/reanalyze` |
| `useDeleteComplaint()` | `DELETE /complaints/{id}` (admin only) |
| `useAnalyzePreview()` | Mutation over the ONE synchronous AI call. `retry:false`, 30 s client timeout. |
| `useAiHealth({refetchIntervalMs?})` | `{llm_available, ml_model_loaded, rules_available, model_name, last_error}` |
| `useAiEvaluation()` | Stored model-evaluation report |
| `useHealth()` | Root `/health` |
| `useAssistantChat()` | Mutation over `POST /assistant/chat` |
| `useLogin(redirectTo?)` · `useMe()` · `useLogout()` | Auth. `useLogout()` returns a plain function. |
| `useDepartments()` · `useDepartmentMap()` | Department list; the map variant is memoised `Map<id, Department>` |
| `useAnalyticsOverview` · `Categories` · `Priorities` · `ResolutionTimes` · `Trends` · `Departments` · `Areas` · `Insights` · `usePublicSummary` | One per analytics endpoint; all accept the same optional `AnalyticsFilters` |
| `useDebouncedValue(value, ms)` · `useCopyToClipboard()` · `useIsMobile()` | Utilities |

### Components — `import { … } from '@/components'`

`CategoryBadge` · `PriorityBadge` · `StatusBadge` · `AiSourceBadge` · `ConfidenceMeter` ·
`ReferenceCode` · `EmptyState` · `ErrorState` · `LoadingSkeleton` · `StatCard` · `PageHeader` ·
`ThemeToggle` · `DomainBadge` (base)

Notable props:
- badges take `size="sm|md|lg"`, `showIcon`, `withTooltip`; `StatusBadge` also takes `dot`;
  `CategoryBadge` takes `short` for table cells.
- `AiSourceBadge` has its tooltip **on by default** and explains which tier produced the result —
  the contract forbids implying a rules result came from the LLM. Pass `modelName` and `latencyMs`.
- `LoadingSkeleton variant="list|table|cards|stats|detail|chart|text"` — pick the one that matches
  your final layout so nothing jumps when data lands.
- `ErrorState` understands `ApiError` and renders the right message for network / 404 / 403 / 503,
  plus the `request_id` for debugging.
- `StatCard` takes `deltaPct` + `higherIsBetter` (defaults to **false** — for most civic metrics
  "up" is bad) and a `tooltip` for explaining what the statistic means.

Raw shadcn primitives stay at `@/components/ui/*`: button, card, input, textarea, select, label,
badge, table, tabs, dialog, sheet, drawer, dropdown-menu, sonner, skeleton, avatar, separator,
progress, alert, tooltip, popover, calendar, command, checkbox, switch, scroll-area, chart, sidebar,
field, radio-group, empty, spinner, input-group.

### Stores — `import { useXStore } from '@/stores/xStore'`

All four use `persist` + `partialize` + `version` + `migrate`.

| Store | Key | Persists | Purpose |
|---|---|---|---|
| `useAuthStore` | `civic.auth` | `token`, `user` | JWT session. `hydrated` flag guards `ProtectedRoute`. `client.ts` reads the token directly — **never import client.ts from a store**. |
| `useDraftStore` | `civic.draft` | `draft`, `step`, `updatedAt` | In-progress complaint form. `patch()`, `isDirty()`, `toPayload()` (trims blanks to `null`), `clear()` after a successful 201. |
| `useTrackedStore` | `civic.tracked` | `refs` | Reference codes submitted from this browser — the only way an anonymous citizen finds their complaints. `addFromComplaint()`, `remove()`, `rename()`, `selectTrackedRefs`. |
| `useUiStore` | `civic.ui` | `theme`, `density`, `sidebarCollapsed`, `savedViews` | Theme (`light|dark|system`), table density, saved admin filter views. |

### API layer

- `import { qk } from '@/lib/api/queryKeys'` — **never inline a query key array.**
- `import { ApiError, isApiError } from '@/lib/api/client'` — has `.isNotFound`, `.isValidation`,
  `.isAiUnavailable`, `.fieldErrors()` (drop straight into react-hook-form `setError`), `.toUserMessage()`.
- `import type { Complaint, Category, … } from '@/lib/api/types'` — mirrors the contract **and** was
  cross-checked field-by-field against `civic-backend/app/schemas/`, so the analytics response shapes
  (`ResolutionTimesResponse`, `TrendsResponse`, `OverviewResponse`, …) include the backend's additive
  rigour fields (`ddof`, `quartile_method`, `sample_warning`, `interpretation`, `insights`).

---

## Conventions to follow

1. **Never hard-code a domain label or colour.** Import `CATEGORY_META` / `PRIORITY_META` /
   `STATUS_META` / `AI_SOURCE_META` from `@/lib/domain`, or just use the badge components.
   Tailwind v4 scans source text, so a runtime-built class like `` `bg-cat-${x}` `` will **not** be
   generated — that's exactly why the literal strings live in `domain.ts`.
2. **Charts:** domain series → `CATEGORY_CHART_COLORS` / `PRIORITY_CHART_COLORS` /
   `STATUS_CHART_COLORS`; neutral series → `CHART_COLORS` (`var(--chart-1..8)`). Wrap Recharts in
   `@/components/ui/chart`'s `ChartContainer`.
3. **Page files own page logic; `src/routes/index.tsx` owns routing.** To add a route, create
   `src/pages/YourPage.tsx` with a **default export** and add one `lazy` entry.
4. **Every page body starts with `<PageHeader/>`.** The layouts already render the site header,
   sidebar and topbar — don't duplicate them.
5. **Loading → `<LoadingSkeleton/>`; error → `<ErrorState error={query.error} onRetry={query.refetch}/>`;
   empty → `<EmptyState/>`.** No bare "Loading…" text.
6. **Forms:** zod schema → `useForm({resolver: zodResolver(schema)})` → one `<Controller>` per field
   wrapping `<Field>/<FieldLabel>/<FieldError>`. Copy `AdminLoginPage.tsx`.
7. **Mutations toast themselves.** `useUpdateComplaint` already toasts success and failure; don't
   double-toast. A mutation with its own `onError` opts out of the global handler.
8. **Show the interpretation.** Every analytics response carries `interpretation` and/or `insights[]`.
   The brief grades explanation, not plotting — render them.
9. **Statistics formatting:** `formatHours`, `formatPercent`, `formatNumber` from `@/lib/domain`;
   add `className="tabular"` (or use a `<table>`) so digits line up.

---

## Acceptance criteria

- [x] `npm run build` → 0 TypeScript errors, bundle emitted (`✓ built`, ~127 kB gzip entry)
- [x] `npm run dev` boots on :5173 and proxies `/api` to :8000
- [x] All 12 routes render their placeholder with **zero console errors** (verified over CDP in
      headless Chrome, authenticated and unauthenticated)
- [x] Unauthenticated `/admin/*` redirects to `/admin/login?next=…`; unknown paths render the 404
- [x] Dark mode: persisted `civic.ui` theme paints `<html class="dark">` with no flash
- [x] `npm run lint` clean except 4 fast-refresh warnings inside generated shadcn `ui/` files
- [x] Domain utility classes (`bg-cat-road/12`, `text-ai-llm-fg`, …) present in the built CSS

---

## Decisions

| Decision | Alternatives considered | Why |
|---|---|---|
| Hand-written `types.ts` as the source of truth, `openapi-typescript` as an optional generator | Generate types only | The backend was still being built; hand-written types unblock page agents now and read far better. Cross-checked against the backend Pydantic schemas, so drift is already minimal. |
| shadcn `radix-nova` preset | `base` (headless) preset | Radix gives real a11y primitives (dialog, popover, tooltip) and Nova ships Lucide + Geist, matching the locked stack. |
| Custom `AdminLayout` sidebar | shadcn `sidebar` component | The shadcn sidebar owns its own cookie-backed state; we already persist collapse in `uiStore` and wanted one source of truth. `ui/sidebar.tsx` is still installed if a page wants it. |
| `usePollUntilAnalyzed` accepts id **or** reference code | Two separate hooks | The citizen submit screen has no auth token, so it must poll `/complaints/track/{code}`; admins poll `/complaints/{id}`. One hook, auto-detected, keeps the call site identical. |
| A real (non-placeholder) login page | Placeholder like the rest | Nothing in `/admin` can be built or demoed without a session, and it doubles as the reference form pattern. |
| `staleTime: 30s`, `refetchOnWindowFocus: false` | Defaults | The triage table must not reshuffle under the cursor when the demo alt-tabs. |

---

## Blockers / notes for other phases

- **Backend:** the frontend calls `/api/v1` and expects the error envelope
  `{error:{code,message,details,request_id}}` on every non-2xx. `GET /health` must stay at the root.
  CORS must allow `http://localhost:5173` **and** the Vercel origin.
- **Analytics response shapes were mirrored from `civic-backend/app/schemas/analytics.py`.** If those
  Pydantic models change, update `src/lib/api/types.ts` in the same commit.
- `/assistant/chat` and `/ai/evaluation` had no backend schema yet; those TS types were written from
  the contract (`{answer, citations, used_stats, source}`) and from `ml/evaluate.py`'s `metrics.json`.
  Confirm them when the routers land.
- **Deploy:** `vercel.json` has the SPA rewrite. Set `VITE_API_URL` in Vercel project settings to the
  deployed API base **including** `/api/v1`.

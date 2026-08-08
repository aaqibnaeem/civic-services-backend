---
phase: 08
title: Integration & Verification
status: complete
owner: lead
depends_on: [02, 03, 04, 05, 06, 07]
started: 2026-08-08 23:40
completed: 2026-08-09 00:55
---

# Phase 08 — Integration & Verification

## Goal
Wire the six parallel work streams together, fix what only shows up when they meet, and prove the whole
system works end to end rather than assuming it does because each half passed its own tests.

## What integration actually found

Six defects, none of which any single agent could have seen alone.

**1. The citizen's category override was a lie.** The submit form ships a control letting a citizen correct
the AI's category, and the pipeline overrode it whenever the analyzer's confidence cleared 0.70 — DeepSeek
routinely returns 0.90. A citizen filing under *Public Safety* had it silently rewritten to *Waste*.
Fixed with an explicit `category_locked` flag, set when a citizen picks a category on submit or a staff
member corrects one via PATCH. The analyzer's own verdict is still written to the `AIAnalysis` row, so the
disagreement stays visible to staff instead of being resolved silently in the AI's favour.
*Verified:* a garbage complaint filed as `safety` stays `safety`, with the AI's `waste` verdict recorded
alongside it.

**2. Two competing duplicate-detection implementations.** Backend-core registered the contract path with a
Jaccard scorer; the AI layer built a stronger TF-IDF engine and, to avoid a route collision, parked it on a
second path. The weaker one was winning. Collapsed to one: the contract endpoint delegates to the TF-IDF
engine.

**3. Duplicate gating was tuned by measurement, not guesswork.** The engine's defaults (30-day window,
500 m radius) returned nothing across 180 days of seeded data. Sweeping the parameters showed the radius is
the dominant gate and the threshold barely matters — 0.30 and 0.45 give near-identical results, meaning
matches are either strong or absent. Settled on 180 days / 5 km / 0.45, which surfaces candidates for 26 of
40 sampled complaints while still refusing to call a Clifton pothole a duplicate of one in Lyari.

**4. `PATCH` responses were stripping the timeline off the cached complaint.** The API returns a bare
`Complaint`, correctly per the contract, but the client typed it as `ComplaintDetail` and assigned it
straight into the detail cache — so anything reading `detail.timeline` threw until the refetch landed.
Fixed by merging rather than replacing, and by correcting the endpoint's return type.

**5. An unknown reference code polled a 404 forever.** `usePollUntilAnalyzed` guarded its timeout on
`dataUpdatedAt`, which stays `0` — and therefore falsy — until a fetch has succeeded once, so the guard
never fired. Now stops on error.

**6. A Zustand selector caused an infinite render loop.** `selectTrackedRefs` sorted into a new array on
every call; zustand v5 compares with `Object.is`, so the component re-rendered forever. Sorting is now a
write-time invariant and the selector returns the stored array by identity.

Two smaller corrections: `ML_MODEL_PATH` pointed at a filename that never existed (the loader was silently
falling back), and `render.yaml` carried a `rootDir` that assumed a monorepo when the backend is its own repo.

## Verification performed

- **232 backend tests** pass, plus **115 analytics tests** inside that total.
- **Frontend type-check and production build** clean; the bundle code-splits per route.
- **Live end-to-end**, backend on `:8000` and the app on `:5173`, driven through headless Chrome over CDP:
  all five public routes and all five authenticated admin routes render with **zero console errors, zero
  exceptions and zero HTTP ≥ 400**, while making real API calls (5–12 per route).
- **The AI path exercised live against DeepSeek.** A Roman-Urdu complaint about a leaning electricity pole
  with hanging wires over standing water came back `electricity` / `critical` / emergency, with the reasoning
  quoted in the AI panel. The ML tier had rated the same scenario `low`.
- **The offline path exercised with no API key**, confirming the trained model and rule engine carry the
  demo unaided.

## Notes for Phase 09
The backend binds fine on `127.0.0.1`, but **Vite binds to IPv6 `localhost` only** — `curl 127.0.0.1:5173`
returns nothing while `curl localhost:5173` works. Worth knowing before debugging a "dead" dev server.

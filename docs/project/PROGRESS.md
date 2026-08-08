# Project Progress Dashboard

**Project:** AI Smart Civic Services — SMIT / OpenBook AI Hackathon
**Target:** all three batch benchmarks at once (Advanced AI + Statistics + OOP)
**Deadline:** a few hours from 2026-08-08

| # | Phase | Status | Owner | Notes |
|---|---|---|---|---|
| 00 | [Setup & Contract](phases/PHASE_00_SETUP.md) | ✅ complete | lead | Repos, docs, frozen API contract |
| 01 | [Stack Research](phases/PHASE_01_RESEARCH.md) | ✅ complete | lead | Caught the retired DeepSeek model IDs |
| 02 | [Backend Core](phases/PHASE_02_BACKEND_CORE.md) | ✅ complete | agent | 24 endpoints, 148 tests green, 801 complaints seeded |
| 03 | [AI Layer](phases/PHASE_03_AI_LAYER.md) | ✅ complete | agent | 3 tiers live; DeepSeek 100% category on the 40-item golden set |
| 04 | [Analytics Engine](phases/PHASE_04_ANALYTICS.md) | ✅ complete | agent | 115 tests green, 24 narrative rules, chi-square + Tukey fences |
| 05 | [Frontend Scaffold](phases/PHASE_05_FRONTEND_SCAFFOLD.md) | ✅ complete | agent | Build green, 12 routes render, 30+ hooks and 13 shared components |
| 06 | [Citizen UI](phases/PHASE_06_CITIZEN_UI.md) | ✅ complete | agent | Submit + live AI reveal + track, verified in browser |
| 07 | [Admin UI](phases/PHASE_07_ADMIN_UI.md) | ✅ complete | agent | Triage inbox, detail, analytics dashboard, assistant |
| 08 | [Integration](phases/PHASE_08_INTEGRATION.md) | ✅ complete | lead | 6 cross-stream defects found and fixed; all routes clean |
| 09 | [Deploy & Submit](phases/PHASE_09_DEPLOY_SUBMIT.md) | ⬜ **needs you** | lead + user | Blocked on two GitHub repo URLs |

Legend: ✅ complete · 🔄 in progress · ⬜ not started · ⚠️ blocked

## Parallelisation map

```
        ┌── 02 Backend Core ──┐
00 ─ 01 ┼── 03 AI Layer ──────┼─ 08 Integration ─ 09 Deploy
        ├── 04 Analytics ─────┤
        └── 05 FE Scaffold ─┬─┴─ 06 Citizen UI ─┐
                            └─── 07 Admin UI ───┘
```
Phases 02–05 run concurrently in one repo each, separated by strict file ownership (see each phase file).
The frozen [CONTRACT.md](CONTRACT.md) is what lets the frontend be built before the backend is live.

## Live blockers

- **User action needed:** create a DeepSeek API key, and a Neon Postgres database. Neither blocks development —
  the app runs on SQLite with the ML + rules AI tiers until those exist.

## Key decisions log

| Date | Decision | Why |
|---|---|---|
| 2026-08-08 | AI provider = DeepSeek `deepseek-v4-flash` | User's choice; `deepseek-chat`/`deepseek-reasoner` were retired 2026-07-24 and now error |
| 2026-08-08 | Thinking mode explicitly **disabled** on every call | It is ON by default on v4 and silently corrupts JSON-mode output |
| 2026-08-08 | Three-tier AI fallback: LLM → trained sklearn → rules | The demo must not depend on a network call or a live API key |
| 2026-08-08 | Complaint POST never blocks on the LLM | Returns 201 with `ai_status:"pending"`, enriches in a background task |
| 2026-08-08 | No Alembic; `create_all()` at startup | Async Alembic is a time sink; migrations are out of scope for a hackathon |
| 2026-08-08 | Narrative insights are rules-based, not LLM-written | A statistic must never be hallucinated |
| 2026-08-08 | No image/vision feature via DeepSeek | DeepSeek has no vision API; optional Gemini free tier is the fallback path |

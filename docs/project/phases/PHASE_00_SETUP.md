---
phase: 00
title: Setup & Frozen Contract
status: complete
owner: lead
depends_on: []
started: 2026-08-08 22:45
completed: 2026-08-08 23:05
---

# Phase 00 — Setup & Frozen Contract

## Goal
Create the two-repo structure and, critically, freeze the API contract **before** any code is written, so the
backend and frontend can be built simultaneously by independent agents without waiting on each other.

## Scope / file ownership
`docs/**`, the two empty repo folders.

## Tasks
- [x] Read the hackathon spec and extract every graded requirement
- [x] Decide the stack and lock it (see PROGRESS.md decisions log)
- [x] Create `civic-backend/` and `civic-frontend/` as separate repo roots
- [x] Write [CONTRACT.md](../CONTRACT.md) — enums, objects, every endpoint, error envelope, and the
      non-negotiable behaviours
- [x] Write the progress dashboard and phase files

## Acceptance criteria
- [x] `docs/CONTRACT.md` covers every endpoint the UI needs, with exact wire-format enum strings
- [x] Both repo folders exist and are independent (no shared package manifest)

## Decisions
| Decision | Alternatives considered | Why |
|---|---|---|
| Freeze the contract first | Build backend then frontend serially | The deadline is hours; serial would not finish. A frozen contract is what makes four parallel agents safe |
| Shared `docs/` at the parent level | Duplicate docs in both repos | One source of truth; each repo's README links up to it |
| 7 categories, streetlight folded into `electricity` | A separate `streetlight` category | Matches the spec's canonical list exactly |

## Blockers / notes for other phases
The contract is **frozen**. Any change must be made in `docs/CONTRACT.md` first and announced to both sides —
a silent deviation on either end breaks integration in Phase 08.

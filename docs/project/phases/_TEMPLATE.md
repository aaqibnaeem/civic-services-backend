---
phase: NN
title: <Phase name>
status: not_started        # not_started | in_progress | complete | blocked
owner: <lead | agent>
depends_on: [NN, NN]
started: YYYY-MM-DD HH:MM
completed:
---

# Phase NN — <Title>

## Goal
One paragraph: what this phase must make true.

## Scope / file ownership
Files this phase exclusively owns, so parallel agents never collide.

## Tasks
- [ ] Task
- [ ] Task

## Acceptance criteria
Done means, concretely and verifiably:
- [ ] `<command>` produces `<observable result>`

## Decisions
| Decision | Alternatives considered | Why |
|---|---|---|

## Blockers / notes for other phases
Anything the next agent must know.

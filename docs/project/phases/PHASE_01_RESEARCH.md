---
phase: 01
title: Stack & API Research
status: complete
owner: lead
depends_on: [00]
started: 2026-08-08 22:51
completed: 2026-08-08 23:00
---

# Phase 01 — Stack & API Research

## Goal
Verify the external APIs and library versions against reality **before** writing code, because model training
data goes stale and this project depends on a fast-moving LLM API.

## Findings that changed the design

**1. The DeepSeek model IDs in every tutorial are dead.**
`deepseek-chat` and `deepseek-reasoner` were retired on **2026-07-24** and now return errors. The current
models are **`deepseek-v4-flash`** (what we use) and `deepseek-v4-pro`. Had we not checked, every AI call
would have failed at demo time with an error that looks like a bad API key.

**2. Thinking mode is ON by default on V4 and silently breaks JSON output.**
Every call must pass `extra_body={"thinking": {"type": "disabled"}}`. Without it the JSON lands in
`reasoning_content`, the parse fails, latency triples and cost roughly triples.

**3. Other verified constraints now baked into the code:**
- `response_format={"type":"json_object"}` works; `json_schema` does **not** on the main endpoint. Pydantic is
  the real schema enforcement — the LLM is treated as untrusted input.
- The literal word "json" must appear in the prompt, and `max_tokens` must be set or output truncates mid-object.
- The API can return **empty content**; that is a documented edge case and must be retried, not crashed on.
- Never retry `400/401/402/422`. Retry `429/5xx`, timeouts, empty content, and validation failures.
- A byte-stable system prompt earns a **50× cheaper** prefix-cache hit, so no per-request interpolation into it.
- **No vision endpoint** and **no embeddings endpoint** on DeepSeek. Duplicate detection therefore uses TF-IDF
  cosine similarity locally, and image analysis is out of scope (optionally Gemini free tier later).
- Cost for this entire project is well under $2, and a new account's free grant likely covers all of it.

**4. Reliability.** DeepSeek had 9 incidents in the trailing 90 days, one lasting over 7 hours. This is the
direct justification for the three-tier fallback chain and for keeping complaint submission off the LLM's
critical path.

## Decisions
| Decision | Why |
|---|---|
| `deepseek-v4-flash`, thinking disabled, `temperature=0` | Classification gains nothing from chain-of-thought; thinking adds 5–20s latency and breaks JSON |
| Own the retry policy with tenacity, `max_retries=0` on the SDK | Stacking SDK retries on top of ours would blow the demo's latency budget |
| Circuit breaker after N consecutive LLM failures | Stops a DeepSeek outage from hanging every request for 45s on a 0.1-CPU instance |

## Blockers / notes for other phases
Full research notes are in the scratchpad file referenced by the AI-layer agent. The non-negotiable lines in
any DeepSeek call are: the model ID, `thinking: disabled`, `max_tokens`, the word "json" in the prompt, and the
empty-content guard. Omitting any one of them is a live-demo failure.

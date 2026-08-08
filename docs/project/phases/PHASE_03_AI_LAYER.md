# Phase 03 — AI / ML Layer

**Status: complete.** All modules written, model trained, artifacts committed, evaluation
generated, 231 backend tests passing, ruff clean.

The headline guarantee: **with no `DEEPSEEK_API_KEY` configured at all, the system still
classifies every complaint, assigns a priority, routes a department, detects duplicates and
answers assistant questions.** That is measured, not asserted — see §4.

---

## 1. Architecture

Three interchangeable analyzers behind one abstract base class, walked in order:

```
analyze_text(text, context)
   │
   ├─ 1. DeepSeekAnalyzer   deepseek-v4-flash, JSON mode, thinking disabled
   │        └─ fails / no key / circuit open ─┐
   ├─ 2. MLAnalyzer         TF-IDF + calibrated LinearSVC, local, ~2 ms
   │        └─ artifacts missing ─────────────┐
   └─ 3. RuleBasedAnalyzer  weighted keywords + regex, zero deps, <1 ms
            └─ cannot fail → analyze_text never raises
```

`AIAnalyzer` (ABC) defines `analyze()` as abstract plus concrete `name`, `source`,
`is_available()` and a `safe_analyze()` template method that converts any exception into a
typed failure. Subclasses override only what genuinely differs. The winning tier is recorded
in `ai.source` and surfaced as a UI badge — CONTRACT §5.3 is enforced in `pipeline.py`, not
hoped for.

### Files

| File | Role |
|---|---|
| `app/ai/base.py` | `AIAnalyzer` ABC, `AnalysisResult` pydantic schema, category/priority/sentiment normalisation |
| `app/ai/prompts.py` | Byte-stable system prompts (triage, assistant planner, assistant writer) |
| `app/ai/llm_analyzer.py` | DeepSeek tier + tenacity retry policy + telemetry |
| `app/ai/ml_analyzer.py` | Lazy-loading sklearn tier with escalate-only priority safety net |
| `app/ai/rule_analyzer.py` | Deterministic keyword engine, English + Roman-Urdu |
| `app/ai/pipeline.py` | Fallback chain, LRU+TTL cache, `analyze_and_store`, health snapshot |
| `app/ai/circuit_breaker.py` | Consecutive-failure breaker, 3 failures → 60 s open, half-open probe |
| `app/ai/duplicates.py` | TF-IDF cosine + haversine + 30-day window |
| `app/ai/assistant.py` | Plan → SQL → prose assistant |
| `app/api/v1/ai.py` | `GET /ai/health`, `/ai/evaluation`, `/ai/duplicates/{id}` |
| `app/api/v1/assistant.py` | `POST /assistant/chat` |
| `ml/{generate_dataset,train,evaluate}.py` | Offline dataset → model → report |
| `tests/test_ai.py` | 80 unit tests, no network needed |
| `tests/golden_eval.py` | 40-complaint golden-set harness → `docs/AI_TESTING_EVIDENCE.md` |

---

## 2. DeepSeek integration — verified API facts honoured

Training data for every LLM is stale on these. All were verified 2026-08-08 and are encoded
in `llm_analyzer.py`:

- `deepseek-chat` / `deepseek-reasoner` were **retired 2026-07-24**. Model is `deepseek-v4-flash`.
- **Thinking is ON by default and corrupts JSON.** Every call sends
  `extra_body={"thinking": {"type": "disabled"}}`.
- `response_format={"type": "json_object"}`; `json_schema` is not supported on the main endpoint.
- The literal word `json` appears in every prompt; `max_tokens` is always set (500 triage /
  400 assistant) so JSON cannot truncate mid-string.
- **Empty content is treated as a retryable failure**, never a crash.
- `max_retries=0` on the SDK client — tenacity owns backoff: exponential + jitter, base 1 s,
  cap 20 s, ≤4 attempts, hard 45 s total budget.
- Never retried: 400 / 401 / 402 / 422. `402` surfaces loudly as insufficient balance.
- `usage.prompt_cache_hit_tokens` / `prompt_cache_miss_tokens` / `completion_tokens` logged per call.
- No vision and no embeddings endpoint exist, so nothing is designed around either.

**Prefix-cache discipline.** `TRIAGE_SYSTEM_PROMPT` is a module constant with **zero
interpolation** — per-request values go only in the user message. A cache hit costs $0.0028/1M
vs $0.14/1M, a 50× difference, and one changed byte at position 0 forfeits it. A test asserts
the prompt is a stable module constant.

**Validation as the real schema.** The parsed dict is validated by Pydantic; on
`ValidationError` one corrective retry is issued with the validator error fed back in a fresh
user turn (keeping the cached prefix intact), then it falls through to the ML tier.
Post-conditions the prompt requests but the model can still violate (`is_emergency` ⇔
`critical`, department present, summary length) are **enforced in code**.

---

## 3. The ML model

`ml/generate_dataset.py` → 3000 labelled complaints from a slot grammar; `ml/train.py` →
`FeatureUnion(TfidfVectorizer(word 1-2gram), TfidfVectorizer(char_wb 3-5gram))` shared by two
`CalibratedClassifierCV(LinearSVC)` heads.

**Measured held-out results** (test split uses *disjoint* sentence frames, priority clauses and
per-category vocabulary):

| | category (7 classes) | priority (4 classes) |
|---|---|---|
| Accuracy | **0.7583** | **0.7417** |
| Macro-F1 | **0.7396** | **0.6863** |
| Majority baseline | 0.198 | 0.393 |
| Inference | 0.14 ms/doc | 0.13 ms/doc |

Artifacts total **1.0 MB** (budget 15 MB), committed, never trained at boot.

### Decisions worth defending

- **Char n-grams alongside word n-grams.** Roman-Urdu has no fixed spelling
  (`kachra`/`kachray`/`kooray`); char_wb 3-5 grams share stems across variants and survive
  typos and ALL CAPS. Word n-grams alone do not handle this corpus.
- **`CalibratedClassifierCV`** because `LinearSVC` exposes only a signed margin, which is not a
  probability. The contract promises `confidence` ∈ [0,1]; Platt scaling makes that honest.
  `ensemble=False` stores one estimator instead of five — a ~5× artifact-size saving.
- **One shared fitted vectorizer** between both heads: halves artifact size and resident RAM,
  and joblib memoises the shared object so the vocabulary serialises once.
- **Class weighting per head.** `priority` uses `balanced` (missing a `critical` is far worse
  than an unnecessary site visit); `category` uses none — up-weighting the small `other` class
  made the model over-predict it (precision fell to 0.51 while recall hit 0.84).
- **The 5-fold CV score for category is 1.0000 and is reported as *worthless***. Random folds
  over template-generated training data grade the model on phrasing it has already seen. The
  gap between CV 1.00 and held-out 0.74 is the most instructive number in the report, so
  `evaluation.md` §1b explains it rather than quoting the flattering figure.

---

## 4. Measured behaviour with **no API key**

`docs/AI_TESTING_EVIDENCE.md`, generated from 40 **hand-written** complaints (not from the
generator), with `DEEPSEEK_API_KEY` empty:

| Metric | Keyword rules | ML model |
|---|---|---|
| Category accuracy | **0.925** | **0.900** |
| Priority accuracy | **0.725** | **0.600** |
| Priority within 1 level | 1.000 | 0.900 |
| Emergency recall (8 critical) | 0.875 | 0.875 |
| Median latency | 0.4 ms | 1.9 ms |

Tier agreement: category 0.850, priority 0.725; all tiers agree on category for 34/40.

**The most important finding, reported rather than hidden:** the model trained on synthetic
data does **not** beat the hand-written keyword engine on hand-written text. That is the honest
verdict on synthetic training data, and it is why the architecture puts the LLM first, the
model second as a fallback, and the rules as the floor beneath both. Given real labelled
complaints, retraining is the first thing to do.

---

## 5. Anti-hallucination design for the assistant

The LLM is **never allowed to produce a number**:

1. **Planner LLM** turns the question into a JSON query plan, validated by Pydantic against a
   strict whitelist (`QueryFilters` coerces every value into a known enum member or drops it;
   free text is only ever a bound LIKE parameter). Nothing the model writes reaches SQL as SQL.
2. **Our code** executes real SQLAlchemy aggregations — counts, group-bys, medians, examples.
   All arithmetic lives here.
3. **Writer LLM** phrases *only* the computed facts, forbidden from doing arithmetic and
   required to cite `reference_code`s.
4. **Citations are verified** against the executed query; any invented `CIV-XXXXXX` is stripped
   before the response leaves the process.

Both LLM steps degrade independently: a keyword planner and a template writer mean the endpoint
answers with no API key (`source` reports `llm` / `hybrid` / `rules`). Out-of-scope questions
(weather, general knowledge) and mutation requests ("delete all complaints") are refused by a
scope check that does **not** depend on the model complying.

---

## 6. Bugs found and fixed during verification

Three real defects were caught by end-to-end testing, not by unit tests written to pass:

1. **Duplicate detection silently found nothing.** With a small candidate set, TF-IDF's IDF term
   gives *lower* weight to terms shared by both documents — inverting the exact signal
   duplication depends on. Fixed by disabling IDF below 15 candidates, and thresholds were
   re-calibrated **by measurement** (true duplicates 0.61–0.67, distinct same-category
   complaints 0.14–0.19) from the guessed 0.72/0.55 down to 0.55/0.38.
2. **The assistant answered "what is the weather tomorrow?" with a complaint count.** Added an
   out-of-scope and mutation guard applied regardless of planner source.
3. **`ml/evaluate.py` crashed on an f-string format specifier spanning a newline**, so the
   committed `evaluation.md` had been silently stale. Fixed, and a test now asserts the report
   contains its honesty section and matches `metrics.json`.

Also fixed: the ML priority head under-triaged badly on real text (a leaning pole with live
wires over standing water where children play came back `low`). The keyword hazard engine now
acts as an **escalate-only** safety net over the model's priority — it may raise, never lower.

---

## 7. What backend-core should change

None of these break anything today; the AI layer is defensive about all of them.

1. **`ML_MODEL_PATH` default is `ml/artifacts/classifier.joblib`; the trained artifact is
   `ml/artifacts/model.joblib`.** `ml_analyzer._candidate_paths()` already falls back, so it
   works either way, but the default should be corrected to `ml/artifacts/model.joblib`.
2. **`/complaints/{id}/duplicates` has two implementations.** `app/api/v1/complaints.py`
   registers the contract path and calls `ComplaintManager.find_duplicates` (Jaccard); the
   TF-IDF + geo + time-window engine lives in `app/ai/duplicates.py` and is exposed at
   `/ai/duplicates/{complaint_id}`. I did **not** register the contract path in `ai.py` because
   `complaints.router` is included first, which would have produced a shadowed, unreachable
   route and two conflicting OpenAPI operations. For exactly one implementation, delegate:
   ```python
   # app/services/complaint_service.py
   from app.ai.duplicates import find_duplicates as ai_find_duplicates
   candidates = await ai_find_duplicates(complaint_id, session=self._repo.session, limit=limit)
   return [(c.complaint, c.similarity, c.reason) for c in candidates]
   ```
3. **Contract drift, for the record:** `app/db/session.py` exports `SessionLocal`, not the
   agreed `async_session_factory`. The AI layer imports the real name.
4. **`AIAnalysis` has `cache_hit_tokens` only**, not the hit/miss pair. `prompt_cache_hit_tokens`
   is mapped onto it and the miss count is dropped. Add a `cache_miss_tokens` column if the
   token-efficiency panel wants both.

---

## 8. Reproducing everything

```bash
cd civic-backend
uv run python -m ml.generate_dataset   # deterministic, seed 20260808
uv run python -m ml.train              # ~3 s, writes ml/artifacts/
uv run python -m ml.evaluate           # writes evaluation.md + metrics.json + PNGs
uv run python -m tests.golden_eval     # writes docs/AI_TESTING_EVIDENCE.md
uv run pytest tests/test_ai.py -q      # 80 tests, no network
```

Everything is seeded; a clean checkout reproduces the committed numbers exactly.

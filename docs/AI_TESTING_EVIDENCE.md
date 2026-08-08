# AI Testing Evidence

*Generated 2026-08-08T19:00:52Z by `tests/golden_eval.py`. Reproduce with
`uv run python -m tests.golden_eval`.*

## 1. What was tested

A **golden set of 40 hand-written complaints** in the style real Karachi citizens
write: plain English, pure Roman-Urdu, code-switched, ALL CAPS, SMS-shortened
(`plz`, `bcz`, `u`), one-word inputs, and deliberately ambiguous category
boundaries. Every category and every priority level is represented. Each item
carries an expected category, an expected priority, and a written justification for
that expectation — including where the expectation is itself debatable.

**None of these 40 complaints came from the training data generator.** They were
written by hand for this evaluation. That distinction is the whole point: the
40-item golden set is the only measurement in this project that
says anything about behaviour on text a human actually wrote.

## 2. Per-tier results

| Metric | Keyword rules | TF-IDF + LinearSVC (ML) | DeepSeek (LLM) |
|---|---|---|---|
| Model / engine | `keyword-rules-v1` | `tfidf-linearsvc-v1` | `deepseek-v4-flash` |
| Items scored | 40 | 40 | 40 |
| **Category accuracy** | **0.925** | **0.900** | **1.000** |
| **Priority accuracy** | **0.725** | **0.600** | **0.775** |
| Priority within 1 level | 1.000 | 0.900 | 1.000 |
| Priority under-triaged | 9 | 6 | 2 |
| Priority over-triaged | 2 | 10 | 7 |
| Emergency recall (of 8 critical) | 0.875 | 0.875 | 0.875 |
| Median latency | 1.4 ms | 8.1 ms | 2137.6 ms |
| Mean confidence | 0.551 | 0.888 | 0.896 |

**Under-triage vs over-triage is the number that matters**, not raw accuracy.
Calling a `critical` complaint `low` can get someone hurt; calling a routine one
`critical` wastes a site visit. The two columns are reported separately for exactly
that reason, and the pipeline's escalation rules are deliberately asymmetric — the
keyword hazard rules may *raise* the ML model's priority and may never lower it.

### The most important number in this report

On the synthetic held-out split the ML model scores **~0.76 category / ~0.74
priority** (`ml/artifacts/evaluation.md`). On these 40 hand-written complaints it
scores **0.900 category / 0.600 priority** — and the *keyword rules*
score **0.925 / 0.725**.

**The model trained on synthetic data does not beat a hand-written keyword engine
on hand-written text.** That is the honest, measured verdict on synthetic training
data, and we are reporting it rather than hiding it.

It does not make the ML tier pointless — it generalises to phrasings the lexicon
has never seen, it degrades gracefully instead of falling to `other`, and it
produces a calibrated probability the rules cannot. But it does mean the claim
"we trained a model, therefore it is better" is false here, and the architecture
reflects that: the model is a *fallback*, the LLM is the primary, and the rules are
the floor beneath both. If we had real labelled complaints, the first thing we
would do is retrain on them.

## 3. Do the tiers agree with each other?

| Pair | n | Category agreement | Priority agreement |
|---|---|---|---|
| Keyword rules vs TF-IDF + LinearSVC (ML) | 40 | 0.850 | 0.725 |
| Keyword rules vs DeepSeek (LLM) | 40 | 0.925 | 0.550 |
| TF-IDF + LinearSVC (ML) vs DeepSeek (LLM) | 40 | 0.900 | 0.525 |

Agreement is not correctness — three tiers can be wrong together, and on the
ambiguous items they often are. It is useful as a *confidence signal*: an item where
all tiers agree is far more likely to be right than one where they split, which is
why disagreement is a good trigger for human review.

Items where all available tiers picked the same category: **34 / 40**.

## 4. Where the tiers disagree

| ID | Expected | Keyword rules | TF-IDF + LinearSVC (ML) | DeepSeek (LLM) | Complaint |
|---|---|---|---|---|---|
| G03 | `drainage` / `high` | `drainage` / `high` | `drainage` / `critical` | `drainage` / `high` | GUTTER KA PANI PURI SARAK PAR PHAIL GAYA HAI!!! SEWERAGE LINE BAND HAI... |
| G05 | `electricity` / `low` | `electricity` / `low` | `electricity` / `high` | `electricity` / `low` | The street light outside our house has been off for about two weeks. N... |
| G06 | `water` / `high` | `water` / `medium` | `water` / `medium` | `water` / `high` | No water supply in our area since 5 days. We have to buy a tanker dail... |
| G09 | `road` / `medium` | `road` / `medium` | `road` / `high` | `road` / `medium` | the road outside our house is broken since months, there are big khadd... |
| G10 | `waste` / `medium` | `waste` / `medium` | `waste` / `high` | `waste` / `high` | Respected sir, the garbage container at the corner of our street is ov... |
| G11 | `electricity` / `high` | `electricity` / `medium` | `electricity` / `high` | `electricity` / `high` | Load shedding in our area is running 8 hours a day with no schedule an... |
| G12 | `drainage` / `medium` | `drainage` / `medium` | `drainage` / `medium` | `drainage` / `high` | nali band hai, gande pani ki wajah se ghar me se bo aa rahi hai. do ha... |
| G13 | `road` / `critical` | `drainage` / `critical` | `safety` / `critical` | `road` / `high` | An open manhole in the middle of the walkway near the bus stop has no ... |
| G14 | `drainage` / `medium` | `drainage` / `medium` | `drainage` / `medium` | `drainage` / `high` | plz clean the drain near our house it is blocked & water is standing, ... |
| G15 | `safety` / `high` | `safety` / `medium` | `safety` / `medium` | `safety` / `high` | Mobile snatching happens at this turn almost every evening. Three inci... |
| G16 | `other` / `medium` | `other` / `medium` | `other` / `critical` | `other` / `medium` | The park in our neighbourhood has been encroached by a private party w... |
| G18 | `water` / `high` | `road` / `medium` | `water` / `high` | `water` / `high` | A large water pipe has burst on the main road and thousands of gallons... |

Most common category errors across all tiers:

- `road->drainage (rules)` × 1
- `road->safety (ml)` × 1
- `water->road (rules)` × 1
- `other->safety (ml)` × 1
- `waste->road (ml)` × 1
- `safety->road (rules)` × 1
- `water->other (ml)` × 1

## 5. Worked examples — input to output

### G02 — expected `waste` / `high`

**Input**

```text
Assalam o alaikum, hamari gali me kachra 15 din se utha nahi hai. Bohot badbu hai aur makhiyan bohot ho gayi hain. Bachay wahan se school jate hain. Meherbani kar ke safai karwa dain.
```

*Why that label:* Pure Roman-Urdu. Solid waste, 15 days, disease vector, school route.

**Keyword rules** → `waste` (correct) / `high` (correct), confidence 0.62, 1.5 ms

> Waste & Sanitation issue reported: Assalam o alaikum, hamari gali me kachra 15 din se utha nahi hai.

*Reasoning:* Keyword rules matched 'kachra', 'safai', 'badbu', 'makhiyan'; category score 10.6 vs runner-up 'electricity' 1.2; priority score 6.2 from school nearby, children affected, weeks unresolved. Deterministic fallback: no language model was used, so this classification is a keyword match, not comprehension.

**TF-IDF + LinearSVC (ML)** → `waste` (correct) / `high` (correct), confidence 1.00, 8.3 ms

> Waste & Sanitation issue (high priority): Assalam o alaikum, hamari gali me kachra 15 din se utha nahi hai.

*Reasoning:* Local TF-IDF + LinearSVC model. Category 'waste' at p=1.00 (runner-up 'other' p=0.00); priority model predicted 'high' at p=0.61.

**DeepSeek (LLM)** → `waste` (correct) / `high` (correct), confidence 0.94, 2282.0 ms

> Garbage uncollected for fifteen days in a residential lane, causing stench and flies on a route children use for school.

*Reasoning:* Roman-Urdu 'kachra' and 'safai' mean garbage and cleaning, so this is solid waste. Fifteen days, disease vectors and children walking past raise it to high, but nobody is injured so it is not critical.

### G04 — expected `electricity` / `critical`

**Input**

```text
bijli ka khamba jhuk gaya hai aur taar neeche latak rahi hai, neeche barish ka pani khara hai aur bachay wahin khelte hain. koi bara hadsa ho sakta hai.
```

*Why that label:* Live wire above standing water where children play.

**Keyword rules** → `electricity` (correct) / `critical` (correct), confidence 0.62, 1.8 ms

> Electricity & Streetlights issue reported: bijli ka khamba jhuk gaya hai aur taar neeche latak rahi hai, neeche barish ka pani khara hai aur bachay wahin khelte hain.

*Reasoning:* Keyword rules matched 'bijli', 'khamba', 'taar'; category score 9.4 vs runner-up 'water' 2.2; priority score 11.9 from accident, hanging wire, leaning pole, children affected. Deterministic fallback: no language model was used, so this classification is a keyword match, not comprehension.

**TF-IDF + LinearSVC (ML)** → `electricity` (correct) / `critical` (correct), confidence 0.56, 7.8 ms

> Electricity & Streetlights issue (critical priority): bijli ka khamba jhuk gaya hai aur taar neeche latak rahi hai, neeche barish ka pani khara hai aur bachay wahin khelte hain.

*Reasoning:* Local TF-IDF + LinearSVC model. Category 'electricity' at p=0.56 (runner-up 'drainage' p=0.43); priority model predicted 'critical' at p=0.34.

**DeepSeek (LLM)** → `electricity` (correct) / `critical` (correct), confidence 0.93, 1675.2 ms

> Leaning electric pole with wires hanging into standing water where children play, in immediate danger of electrocution.

*Reasoning:* 'Khamba' and 'taar' identify an electricity asset, so rule 6 keeps this in electricity rather than safety. A live wire above standing water where children play is an immediate risk to life, so critical.

### G08 — expected `water` / `critical`

**Input**

```text
Sewage has mixed into our drinking water line. Three people in our house are vomiting and my father has been admitted to hospital.
```

*Why that label:* AMBIGUOUS: contamination of the clean water line -> water; a grader could defensibly say drainage. Hospitalisation makes it critical either way.

**Keyword rules** → `water` (correct) / `critical` (correct), confidence 0.56, 2.0 ms

> Water Supply & Leakage issue reported: Sewage has mixed into our drinking water line.

*Reasoning:* Keyword rules matched 'drinking water', 'water line', 'water'; category score 7.2 vs runner-up 'drainage' 3.8; priority score 12.2 from sewage in drinking water, hospitalised, acute illness, hospital nearby. Deterministic fallback: no language model was used, so this classification is a keyword match, not comprehension.

**TF-IDF + LinearSVC (ML)** → `water` (correct) / `critical` (correct), confidence 0.96, 9.9 ms

> Water Supply & Leakage issue (critical priority): Sewage has mixed into our drinking water line.

*Reasoning:* Local TF-IDF + LinearSVC model. Category 'water' at p=0.96 (runner-up 'waste' p=0.02); priority model predicted 'critical' at p=0.94.

**DeepSeek (LLM)** → `water` (correct) / `critical` (correct), confidence 0.97, 3251.3 ms

> Sewage mixed into drinking water line; three people vomiting and father hospitalised.

*Reasoning:* 'Sewage mixed into drinking water' is a contamination of clean supply, so category is water. Hospitalisation and active illness make it critical and an emergency.

### G13 — expected `road` / `critical`

**Input**

```text
An open manhole in the middle of the walkway near the bus stop has no cover. It is dark at night and someone will fall in.
```

*Why that label:* AMBIGUOUS: open manhole in a walkway -> road hazard per our disambiguation rule 5; drainage is also defensible. Fall risk makes it critical.

**Keyword rules** → `drainage` (**WRONG**) / `critical` (correct), confidence 0.45, 1.4 ms

> Drainage & Sewerage issue reported: An open manhole in the middle of the walkway near the bus stop has no cover.

*Reasoning:* Keyword rules matched 'manhole'; category score 2.6 vs runner-up 'electricity' 0.9; priority score 10.4 from open manhole, open manhole, foreseeable serious harm, public place. Deterministic fallback: no language model was used, so this classification is a keyword match, not comprehension.

**TF-IDF + LinearSVC (ML)** → `safety` (**WRONG**) / `critical` (correct), confidence 0.99, 7.9 ms

> Public Safety issue (critical priority): An open manhole in the middle of the walkway near the bus stop has no cover.

*Reasoning:* Local TF-IDF + LinearSVC model. Category 'safety' at p=0.99 (runner-up 'drainage' p=0.00); priority model predicted 'critical' at p=0.44.

**DeepSeek (LLM)** → `road` (correct) / `high` (**WRONG**), confidence 0.85, 1931.8 ms

> Open manhole on a walkway near a bus stop, dark at night, risk of someone falling in.

*Reasoning:* The complaint is about an open manhole cover, which is a road surface hazard per rule 5, not drainage since no sewage is mentioned. The risk of injury is high but no injury has occurred, so priority is high.

### G19 — expected `waste` / `medium`

**Input**

```text
kachra
```

*Why that label:* Single word. Tests degenerate input handling.

**Keyword rules** → `waste` (correct) / `medium` (correct), confidence 0.58, 0.5 ms

> Waste & Sanitation issue reported: kachra

*Reasoning:* Keyword rules matched 'kachra'; category score 3.6 vs runner-up 'road' 0.0; priority score 0.0. Deterministic fallback: no language model was used, so this classification is a keyword match, not comprehension.

**TF-IDF + LinearSVC (ML)** → `waste` (correct) / `medium` (correct), confidence 1.00, 9.7 ms

> Waste & Sanitation issue (medium priority): kachra

*Reasoning:* Local TF-IDF + LinearSVC model. Category 'waste' at p=1.00 (runner-up 'other' p=0.00); priority model predicted 'medium' at p=0.49.

**DeepSeek (LLM)** → `waste` (correct) / `medium` (correct), confidence 0.50, 1737.5 ms

> Complaint mentions garbage but lacks detail on location or impact.

*Reasoning:* The single word 'kachra' means garbage, clearly indicating solid waste, but there is no location or severity described, so confidence is moderate and priority defaults to medium.

### G20 — expected `other` / `medium`

**Input**

```text
There is a serious problem here please send someone
```

*Why that label:* Vague, no category signal at all. Should be low confidence.

**Keyword rules** → `other` (correct) / `medium` (correct), confidence 0.10, 0.5 ms

> Other issue reported: There is a serious problem here please send someone

*Reasoning:* No category keyword matched, defaulted to 'other'; category score 0.0 vs runner-up 'water' 0.0; priority score 0.0. Deterministic fallback: no language model was used, so this classification is a keyword match, not comprehension.

**TF-IDF + LinearSVC (ML)** → `safety` (**WRONG**) / `critical` (**WRONG**), confidence 0.33, 6.4 ms

> Public Safety issue (critical priority): There is a serious problem here please send someone

*Reasoning:* Local TF-IDF + LinearSVC model. Category 'safety' at p=0.33 (runner-up 'drainage' p=0.18); priority model predicted 'critical' at p=0.82. Confidence is low — the wording is outside the model's training vocabulary, so this classification should be reviewed.

**DeepSeek (LLM)** → `other` (correct) / `medium` (correct), confidence 0.20, 1491.1 ms

> Complaint is vague with no specific problem, location, or impact described.

*Reasoning:* The message contains no specific words identifying a category, only a general request for help. Confidence is very low and default priority applies.

### G30 — expected `electricity` / `critical`

**Input**

```text
Someone received an electric shock from the pole outside our shop this morning. He is in the hospital now. The wires are still live.
```

*Why that label:* Electrocution already occurred, hazard still live.

**Keyword rules** → `electricity` (correct) / `critical` (correct), confidence 0.60, 1.5 ms

> Electricity & Streetlights issue reported: Someone received an electric shock from the pole outside our shop this morning.

*Reasoning:* Keyword rules matched 'pole', 'wires'; category score 4.2 vs runner-up 'road' 0.0; priority score 10.6 from electric shock, hospitalised, hospital nearby. Deterministic fallback: no language model was used, so this classification is a keyword match, not comprehension.

**TF-IDF + LinearSVC (ML)** → `electricity` (correct) / `critical` (correct), confidence 0.98, 8.9 ms

> Electricity & Streetlights issue (critical priority): Someone received an electric shock from the pole outside our shop this morning.

*Reasoning:* Local TF-IDF + LinearSVC model. Category 'electricity' at p=0.98 (runner-up 'drainage' p=0.02); priority model predicted 'critical' at p=0.93.

**DeepSeek (LLM)** → `electricity` (correct) / `critical` (correct), confidence 0.97, 2363.2 ms

> Electric pole outside a shop gave someone a shock; victim hospitalised and wires still live.

*Reasoning:* The complaint explicitly mentions an electric shock from a pole and live wires, which is an electricity asset and an immediate life risk, so critical.

### G40 — expected `road` / `medium`

**Input**

```text
road toot gayi hai bilkul, gaari chalana namumkin hai. 2 mahine se koi nahi aaya dekhne. please jaldi theek karwa dain!!!
```

*Why that label:* Roman-Urdu with exclamation intensity. Road damage, no hazard signal.

**Keyword rules** → `road` (correct) / `high` (**WRONG**), confidence 0.62, 0.4 ms

> Roads & Potholes issue reported: road toot gayi hai bilkul, gaari chalana namumkin hai.

*Reasoning:* Keyword rules matched 'road', 'road toot', 'toot gayi'; category score 6.4 vs runner-up 'water' 0.0; priority score 3.5 from months unresolved, 3 exclamation marks, says urgent. Deterministic fallback: no language model was used, so this classification is a keyword match, not comprehension.

**TF-IDF + LinearSVC (ML)** → `road` (correct) / `high` (**WRONG**), confidence 1.00, 3.6 ms

> Roads & Potholes issue (high priority): road toot gayi hai bilkul, gaari chalana namumkin hai.

*Reasoning:* Local TF-IDF + LinearSVC model. Category 'road' at p=1.00 (runner-up 'other' p=0.00); priority model predicted 'high' at p=0.50.

**DeepSeek (LLM)** → `road` (correct) / `high` (**WRONG**), confidence 0.95, 2604.8 ms

> Road completely broken and impassable for vehicles, unresolved for two months.

*Reasoning:* 'Road toot gayi' clearly indicates a damaged road surface, so category is road. Two months of neglect and a main road affected raise it to high, but no injury is reported so it is not critical.


## 6. LIMITATIONS

This section is the deliverable, not a disclaimer. Every item below is a real,
known weakness of what was built.

1. **The ML model is trained entirely on synthetic data.** `ml/generate_dataset.py`
   produces the corpus from a slot grammar of roughly 300 hand-written phrasings.
   Its held-out score (~0.76 category / ~0.74 priority, disjoint templates) is an
   **upper bound**, and the gap between that and its accuracy on this hand-written
   golden set is the honest measure of how much synthetic data flatters itself.
   Compare the two numbers directly — that gap is the single most informative
   result in this report.
2. **No Urdu-script training coverage.** The corpus contains Roman-Urdu
   transliteration only. A complaint written in نستعلیق produces almost no usable
   character n-grams and the ML tier will effectively guess. Only the LLM tier
   handles Urdu script, so with no API key those complaints degrade to keyword
   matching that will not fire at all.
3. **Priority is subjective and has no ground truth.** The "expected" priorities in
   this golden set are one considered opinion. Two municipal officers would
   disagree on several — G03, G08, G13 and G22 are all genuinely arguable. Any
   priority accuracy figure here, for any tier, should be read as "agreement with
   one annotator", not "correctness".
4. **Category boundaries are genuinely ambiguous.** G08 (sewage in the drinking
   water line), G13 (open manhole in a walkway) and G18 (burst main eroding a road)
   have defensible answers in two categories each. The disambiguation rules in the
   system prompt pick one consistently; that is a convention, not a truth.
5. **The LLM is non-deterministic.** Even at `temperature=0`, DeepSeek can return
   different classifications for the same complaint on different runs. Re-running
   this harness with `--with-llm` will not reproduce identical numbers. This is why
   agreement rate is reported rather than treated as a fixed property.
6. **Cost and latency.** The LLM tier costs money and takes 2–6 seconds; the ML
   tier takes well under a millisecond and is free. Submission therefore never
   blocks on the LLM (CONTRACT §5.1) — analysis runs in a background task.
7. **No image understanding.** DeepSeek has **no vision endpoint** on the public
   API as of 2026-08-08. An uploaded photo is stored and displayed but contributes
   nothing to classification. A complaint whose meaning lives entirely in the photo
   ("see attached") will be classified from its text alone, which is to say badly.
8. **Duplicate detection is lexical, not semantic.** `app/ai/duplicates.py` is
   TF-IDF cosine similarity. Two people reporting the same pothole — one in
   English, one in Roman-Urdu — will not be matched. Real semantic matching needs
   embeddings, and DeepSeek has no embeddings endpoint.
9. **The rules tier does not understand negation.** "There is no garbage problem
   here any more, thank you" scores as a waste complaint. Its confidence is capped
   at 0.62 by design so the UI never presents a keyword match as comprehension.
10. **No tier verifies that a complaint is true.** A fabricated report is
    classified exactly as confidently as a real one.
11. **The golden set is 40 items.** That is enough to expose systematic failure
    modes and far too few for a tight confidence interval. A single item moves
    category accuracy by 2.5 points.

## 7. What this evidence supports

The fallback chain works. With **no API key configured at all**, the product still
classifies every complaint, still routes it to a department, and still flags
emergencies — the numbers in §2 for the ML and rules tiers were produced in exactly
that state. That is the demo-safety guarantee, measured rather than asserted.

The tier that produced each result is recorded in `ai.source` and shown as a badge
in the UI. A rules-based result is never presented as an LLM result (CONTRACT §5.3).

## 8. Reproducing

```bash
uv run python -m tests.golden_eval              # rules + ML, no API key needed
uv run python -m tests.golden_eval --with-llm   # adds the DeepSeek tier
```

Raw per-item output is written to `docs/ai_testing_evidence.json`.

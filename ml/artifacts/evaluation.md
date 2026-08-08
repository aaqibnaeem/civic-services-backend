# Model Evaluation — `tfidf-linearsvc-v1`

*Generated 2026-08-08T18:51:26Z · model trained 2026-08-08T18:19:46Z ·
scikit-learn 1.9.0 · artifact 0.74 MB*

> **Read the honesty section before quoting any number from this page.**
> The training data is synthetic. These scores are an **upper bound**, not a
> prediction of production accuracy.

## 1. Headline

| | category (7 classes) | priority (4 classes) |
|---|---|---|
| Held-out accuracy | **0.7583** | **0.7417** |
| Held-out macro-F1 | **0.7396** | **0.6863** |
| Weighted F1 | 0.7627 | 0.7340 |
| 5-fold CV macro-F1 (train split) — *see §1b, do not quote this* | 1.0000 ± 0.0000 (5-fold) | 0.8558 ± 0.0149 (5-fold) |
| Test items | 600 | 600 |
| Mean confidence | 0.860 | 0.795 |
| Mean confidence when **correct** | 0.9253 | 0.8534 |
| Mean confidence when **wrong** | 0.6538 | 0.6279 |
| Inference latency | 0.13 ms/doc | 0.13 ms/doc |

Majority-class baseline for category is ~0.198 and for priority
~0.393; both heads beat it by a wide margin, which is the only thing an
accuracy figure is actually evidence of.

## 1b. Why the cross-validation number is worthless here (and the held-out one is not)

The 5-fold CV macro-F1 for **category is 1.0000**, against a held-out macro-F1
of **0.7396**. That gap of ~0.26 is not a fluke and not a bug —
it is the single most instructive number on this page, so it is reported rather
than buried.

Standard k-fold CV shuffles the training split at random. Because the training
split is generated from **10 shared sentence frames**, almost every validation item
in a fold has near-identical siblings in the other four folds. The model is
therefore graded on phrasing it has already seen, and it scores near-perfectly.

**A random-split CV score on template-generated data measures memorisation, not
generalisation.** Anyone quoting 1.00 as this model's accuracy is quoting a
fiction — and this is exactly the trap that synthetic data lays for you.

The held-out split closes that hole: **10 completely different sentence frames**,
disjoint priority escalation clauses, and per-category `test_only` vocabulary.
Nothing in it is phrased like anything in training. That is why
**0.7583 / 0.7417** are the numbers quoted everywhere
else in this project, and the CV figures appear only as evidence of this effect.

Priority shows a smaller gap (0.8558 CV vs
0.6863 held-out) because its escalation clauses deliberately share
some urgency vocabulary across splits — realistic, since words like "urgent",
"accident" and "routine" genuinely do recur in real complaints.

## 2. Setup

| | |
|---|---|
| Representation | `FeatureUnion(TfidfVectorizer(word 1-2gram), TfidfVectorizer(char_wb 3-5gram))` |
| Features | 3898 word + 7755 char = **11653** (shared by both heads) |
| Classifier | `CalibratedClassifierCV(LinearSVC(...), method="sigmoid", cv=5, ensemble=False)` |
| Class weighting | category `None` · priority `balanced` (see below) |
| Train / test | 2400 / 600 rows |
| Split design | test split uses **disjoint sentence frames and disjoint priority clauses** |
| Seed | 20260808 |

Class weighting is set per head because the cost of an error differs. `priority`
uses `balanced` so the rare `critical` class keeps recall — calling a critical
complaint `low` can get someone hurt, while the reverse only wastes a site visit.
`category` uses no weighting: mis-routing `safety` as `other` costs about the same
as the reverse, and up-weighting the small `other` class only made the model
over-predict it (precision fell to 0.51 while recall rose to 0.84), which is worse
for a routing system.

Calibration matters here: `LinearSVC` exposes only a signed margin, which is not a
probability. The API contract promises a `confidence` in 0..1, so the margin is
passed through Platt scaling. The buckets in §5 show whether that promise holds.

## 3. Per-class results — category

| class | precision | recall | f1 | support |
|---|---|---|---|---|
| `drainage` | 0.683 | 0.719 | 0.701 | 96 |
| `electricity` | 0.873 | 0.827 | 0.849 | 75 |
| `other` | 0.412 | 0.757 | 0.533 | 37 |
| `road` | 0.878 | 0.898 | 0.888 | 88 |
| `safety` | 0.755 | 0.536 | 0.627 | 69 |
| `waste` | 0.854 | 0.739 | 0.793 | 119 |
| `water` | 0.780 | 0.793 | 0.786 | 116 |

**Confusion matrix** (rows = actual, columns = predicted), also rendered as `confusion_category.png`:

| actual \ predicted | drainage | electricity | other | road | safety | waste | water |
|---|---|---|---|---|---|---|---|
| **drainage** | 69 | 0 | 7 | 0 | 0 | 7 | 13 |
| **electricity** | 4 | 62 | 1 | 0 | 1 | 0 | 7 |
| **other** | 0 | 1 | 28 | 0 | 2 | 4 | 2 |
| **road** | 3 | 0 | 0 | 79 | 4 | 2 | 0 |
| **safety** | 0 | 4 | 25 | 1 | 37 | 0 | 2 |
| **waste** | 6 | 2 | 7 | 10 | 4 | 88 | 2 |
| **water** | 19 | 2 | 0 | 0 | 1 | 2 | 92 |

Most frequent confusions: `safety`→`other` (25), `water`→`drainage` (19), `drainage`→`water` (13), `waste`→`road` (10), `drainage`→`other` (7).

## 4. Per-class results — priority

| class | precision | recall | f1 | support |
|---|---|---|---|---|
| `critical` | 0.596 | 0.431 | 0.500 | 72 |
| `high` | 0.683 | 0.824 | 0.747 | 188 |
| `low` | 0.891 | 0.548 | 0.679 | 104 |
| `medium` | 0.786 | 0.856 | 0.819 | 236 |

**Confusion matrix**, also rendered as `confusion_priority.png`:

| actual \ predicted | critical | high | low | medium |
|---|---|---|---|---|
| **critical** | 31 | 31 | 3 | 7 |
| **high** | 13 | 155 | 2 | 18 |
| **low** | 5 | 12 | 57 | 30 |
| **medium** | 3 | 29 | 2 | 202 |

Most frequent confusions: `critical`→`high` (31), `low`→`medium` (30), `medium`→`high` (29), `high`→`medium` (18), `high`→`critical` (13).

Priority errors are not symmetric in cost. Predicting `low` for something that is
really `critical` is the expensive mistake; predicting `critical` for something
routine only wastes a site visit. That is why `class_weight="balanced"` is set —
it deliberately trades a little overall accuracy for better recall on the rare
`critical` class.

## 5. Is the confidence score trustworthy?

Category head:

| confidence bucket | n | actual accuracy | mean confidence |
|---|---|---|---|
| 0.0-0.4 | 31 | 0.194 | 0.333 |
| 0.4-0.6 | 59 | 0.424 | 0.507 |
| 0.6-0.8 | 69 | 0.435 | 0.695 |
| 0.8-1.0 | 441 | 0.893 | 0.970 |

Priority head:

| confidence bucket | n | actual accuracy | mean confidence |
|---|---|---|---|
| 0.0-0.4 | 16 | 0.438 | 0.347 |
| 0.4-0.6 | 108 | 0.417 | 0.517 |
| 0.6-0.8 | 121 | 0.545 | 0.696 |
| 0.8-1.0 | 355 | 0.921 | 0.934 |

If accuracy rises monotonically with the confidence bucket, the number means
something and the UI can gate on it. Where it does not, the score should be read as
a ranking signal only. The pipeline treats anything below ~0.45 as weak and says so
in the analysis `reasoning` field.

## 6. LIMITATIONS — the part that matters

1. **The training data is synthetic.** Every row came from `ml/generate_dataset.py`,
   a slot grammar with roughly 300 hand-written issue phrasings. No real citizen
   complaint was used, because no public labelled Pakistani civic corpus exists.
2. **Therefore the held-out score above is an upper bound, not an estimate of
   production accuracy.** The test split fights memorisation with disjoint frames
   and clauses, but it cannot manufacture the thing synthetic data structurally
   lacks: genuine ambiguity. Real complaints arrive mixed ("sewage is flooding the
   broken road outside the school" is simultaneously `drainage`, `road` and a
   safety issue), truncated, contradictory, or about something none of the seven
   categories covers. Expect a substantial drop on real traffic. We would guess
   materially lower, and we deliberately do not put a number on that guess,
   because we have no labelled real data to measure it against.
3. **Priority is subjective.** Two municipal officers will disagree on `high` vs
   `critical` for the same complaint. The generator encodes one consistent opinion,
   which makes the task cleaner than reality. There is no inter-annotator agreement
   figure here because there was only one annotator: the grammar.
4. **No Urdu script coverage.** Only Roman-Urdu transliteration is in the corpus
   (1127 of 3000 rows).
   A complaint written in نستعلیق produces almost no useful char n-grams and the
   model will effectively guess. This is a known, unfixed gap; those complaints
   need the LLM tier.
5. **Vocabulary ceiling.** Anything outside the generator's phrasing distribution
   degrades sharply. TF-IDF has no notion of meaning — it cannot infer that an
   unseen word is a synonym of a seen one.
6. **Karachi-specific.** Place names, departments and idiom are local; the model
   will not transfer to another city without regenerating and retraining.
7. **Lexical, not semantic.** The same limitation applies to duplicate detection in
   `app/ai/duplicates.py`, which is cosine similarity over TF-IDF: it catches
   re-phrasings that share words, and misses two people describing the same pothole
   with completely different vocabulary.

## 7. Why this model is the *fallback* tier and not the primary

Given §6, the honest conclusion is that this classifier should not be the primary
decision-maker. In `app/ai/pipeline.py` the order is:

**DeepSeek `deepseek-v4-flash` → this model → deterministic keyword rules.**

Each tier is recorded in the stored `ai.source` field and shown as a badge in the
UI, so nobody ever mistakes a rules result for an LLM result.

The ML tier earns its place on a different axis than accuracy:

| | LLM tier | this ML tier | rules tier |
|---|---|---|---|
| Handles unseen phrasing | yes | poorly | no |
| Works with no API key | no | **yes** | yes |
| Works during a provider outage | no | **yes** | yes |
| Latency | 2–6 s | **~0.1 ms** | <1 ms |
| Cost per 1k complaints | ~$0.15 | **$0** | $0 |
| Deterministic | no | **yes** | yes |

DeepSeek's published uptime is ~99.79% with chunky multi-hour failure modes. A
system whose only classifier is a hosted API is a system that stops classifying
during a demo. Being pretty good and always available beats being excellent and
sometimes absent — which is precisely the job this model was trained for.

## 8. Reproducing these numbers

```bash
python -m ml.generate_dataset   # deterministic, seed 20260808
python -m ml.train
python -m ml.evaluate
```

Everything is seeded; a clean checkout reproduces this report exactly.

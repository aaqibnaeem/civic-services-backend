# Dataset Card — `civic-complaints-synth-v1`

| Field | Value |
|---|---|
| Rows | 3000 |
| Train / Test | 2400 / 600 |
| Categories | 7 (`road, water, waste, electricity, drainage, safety, other`) |
| Priorities | 4 (`low, medium, high, critical`) |
| Language | English, Roman-Urdu, and code-switched English/Roman-Urdu |
| Roman-Urdu rows | 1127 (37.6%) |
| Seed | 20260808 |
| Generator | `ml/generate_dataset.py` |

## Provenance — read this before quoting any accuracy number

**This dataset is 100% synthetic.** It was produced by a slot grammar, not collected
from citizens. No real complaint text, and no personal data, is present.

It exists because there is no public, labelled, Pakistani civic-complaint corpus
carrying *both* a category and a priority label. Generating a defensible corpus and
saying so is more honest than scraping something unrelated and pretending it
transfers.

## What was done to make the evaluation meaningful

* **Disjoint sentence frames.** The 10 frames used for the test split share no text
  with the 10 used for train. A test item is never phrased like a train item.
* **Disjoint priority clauses.** The escalation sentence that *causes* the priority
  label is drawn from a different pool per split.
* **Partially disjoint issue vocabulary.** Each category has `shared`, `train_only`
  and `test_only` phrasings. Shared vocabulary is intentional — a classifier is
  entitled to learn that "pothole" and "kachra" are real signals — but every test
  item also contains phrasing the model has never seen.
* **Exact-duplicate removal** across the whole corpus, so no test string can appear
  in train.
* **Noise applied after assembly**: typos (transpose / drop / double / keyboard
  slip), ALL CAPS, all-lowercase, dropped punctuation, exclamation spam, SMS
  shortening (`plz`, `bcz`, `u`), and Roman-Urdu code-switching.
* **Non-uniform class weights** matching a realistic municipal inbox rather than a
  tidy uniform split.

## Known limitations (these are real, do not paper over them)

1. **Labels are causal, not human-annotated.** The generator chose the category and
   priority first, then emitted text consistent with them. Real complaints are
   genuinely ambiguous — "sewage on the road" is legitimately `drainage` *or*
   `road` — and this corpus has almost none of that ambiguity. **Held-out accuracy
   here is an upper bound on production accuracy, not an estimate of it.**
2. **Priority is subjective.** Two municipal officers will disagree on
   `high` vs `critical`. The generator encodes one opinion consistently, which
   makes the task easier than reality.
3. **No Urdu script.** Only Roman-Urdu transliteration is covered. A complaint
   written in نستعلیق will fall to the char n-grams with no useful signal, and the
   model will effectively guess. The LLM tier handles those.
4. **Vocabulary ceiling.** Roughly 300 hand-written issue phrasings underlie the
   whole corpus. Real citizens have unbounded vocabulary; anything outside this
   distribution degrades sharply.
5. **Karachi-specific.** Place names, department names and idiom are local. The
   model will not transfer to another city without regeneration.

## Why the model trained on this is the *fallback* tier, not the primary

Because of limitation 1, we do not trust this model as the primary classifier. In
`app/ai/pipeline.py` the order is **DeepSeek LLM → this model → keyword rules**.
The ML tier's job is to keep the product fully functional during an LLM outage or
when no API key is configured — a role it is genuinely good at, and one where being
"pretty good and always up" beats "excellent and sometimes down".

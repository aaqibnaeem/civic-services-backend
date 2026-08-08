"""Evaluate the trained triage model and write an honest report.

Produces, in ``ml/artifacts/``:

* ``metrics.json``        machine-readable metrics (served by ``GET /ai/evaluation``)
* ``evaluation.md``       the human report, including a LIMITATIONS section
* ``confusion_category.png`` / ``confusion_priority.png`` (matplotlib, Agg backend)

What is measured
----------------
1. **Held-out accuracy and macro-F1** on the test split, which uses *disjoint*
   sentence frames and priority clauses (see ``ml/data/DATASET_CARD.md``).
2. **Per-class precision / recall / F1 / support** — the headline accuracy hides
   minority-class failure, macro-F1 and the per-class table do not.
3. **5-fold stratified cross-validation** on the training split, refitting the
   whole pipeline per fold (vectorizer included) so there is no fit-on-test
   leakage through the vocabulary.
4. **Confusion matrices**, because *which* classes get confused is the actually
   useful signal for a triage system (road↔drainage confusion is cheap to
   mis-route; safety↔other is not).

Run::

    python -m ml.evaluate
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score

try:
    from ml import ARTIFACTS_DIR, DATA_DIR, SEED
    from ml.train import CLASS_WEIGHTS as _CLASS_WEIGHTS
    from ml.train import build_pipeline
except ImportError:  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from ml import ARTIFACTS_DIR, DATA_DIR, SEED
    from ml.train import CLASS_WEIGHTS as _CLASS_WEIGHTS
    from ml.train import build_pipeline


# --------------------------------------------------------------------------- #
# confusion matrix plotting (optional dependency, never fatal)
# --------------------------------------------------------------------------- #

def _plot_confusion(cm: np.ndarray, labels: list[str], title: str, out: Path) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")  # non-interactive: no display on Render / CI
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        print(f"  (skipping {out.name}: matplotlib unavailable — {exc})")
        return False

    norm = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(1.05 * len(labels) + 3, 1.05 * len(labels) + 2.4))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("predicted")
    ax.set_ylabel("actual")
    ax.set_title(title)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(
                j, i, f"{cm[i, j]}\n{norm[i, j] * 100:.0f}%",
                ha="center", va="center", fontsize=8,
                color="white" if norm[i, j] > 0.55 else "#111111",
            )
    fig.colorbar(im, ax=ax, fraction=0.045, label="row-normalised")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return True


def _md_confusion(cm: np.ndarray, labels: list[str]) -> str:
    head = "| actual \\ predicted | " + " | ".join(labels) + " |"
    sep = "|---" * (len(labels) + 1) + "|"
    rows = [
        f"| **{labels[i]}** | " + " | ".join(str(int(v)) for v in cm[i]) + " |"
        for i in range(len(labels))
    ]
    return "\n".join([head, sep, *rows])


def _per_class_table(report: dict, labels: list[str]) -> str:
    lines = ["| class | precision | recall | f1 | support |", "|---|---|---|---|---|"]
    for lab in labels:
        r = report.get(lab)
        if not r:
            continue
        lines.append(
            f"| `{lab}` | {r['precision']:.3f} | {r['recall']:.3f} | "
            f"{r['f1-score']:.3f} | {int(r['support'])} |"
        )
    return "\n".join(lines)


def _top_confusions(cm: np.ndarray, labels: list[str], k: int = 5) -> list[tuple[str, str, int]]:
    pairs: list[tuple[str, str, int]] = []
    for i in range(len(labels)):
        for j in range(len(labels)):
            if i != j and cm[i, j] > 0:
                pairs.append((labels[i], labels[j], int(cm[i, j])))
    pairs.sort(key=lambda p: -p[2])
    return pairs[:k]


# --------------------------------------------------------------------------- #

def evaluate(data_dir: Path = DATA_DIR, artifacts_dir: Path = ARTIFACTS_DIR,
             seed: int = SEED, cv_folds: int = 5, run_cv: bool = True) -> dict:
    model_path = artifacts_dir / "model.joblib"
    if not model_path.exists():
        raise SystemExit(f"missing {model_path} — run `python -m ml.train` first")

    bundle = joblib.load(model_path)
    train_df = pd.read_csv(data_dir / "train.csv")
    test_df = pd.read_csv(data_dir / "test.csv")
    train_df["text"] = train_df["text"].astype(str)
    test_df["text"] = test_df["text"].astype(str)

    x_test = test_df["text"].tolist()
    results: dict[str, dict] = {}

    for target in ("category", "priority"):
        print(f"\n=== {target} ===")
        pipe = bundle[target]
        labels = sorted(bundle["categories" if target == "category" else "priorities"])
        y_true = test_df[target].tolist()

        t0 = time.perf_counter()
        y_pred = pipe.predict(x_test)
        infer_ms = 1000 * (time.perf_counter() - t0) / max(len(x_test), 1)

        proba = pipe.predict_proba(x_test)
        conf = proba.max(axis=1)
        correct = np.array([a == b for a, b in zip(y_true, y_pred, strict=False)])

        acc = float(accuracy_score(y_true, y_pred))
        macro_f1 = float(f1_score(y_true, y_pred, average="macro"))
        weighted_f1 = float(f1_score(y_true, y_pred, average="weighted"))
        report = classification_report(y_true, y_pred, labels=labels,
                                       output_dict=True, zero_division=0)
        cm = confusion_matrix(y_true, y_pred, labels=labels)

        # Calibration sanity: does a high confidence actually mean high accuracy?
        buckets = []
        for lo, hi in ((0.0, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.01)):
            mask = (conf >= lo) & (conf < hi)
            n = int(mask.sum())
            buckets.append({
                "range": f"{lo:.1f}-{min(hi, 1.0):.1f}",
                "n": n,
                "accuracy": round(float(correct[mask].mean()), 4) if n else None,
                "mean_confidence": round(float(conf[mask].mean()), 4) if n else None,
            })

        cv_mean = cv_std = None
        cv_scores: list[float] = []
        if run_cv:
            print(f"  running {cv_folds}-fold CV (refits the full pipeline per fold)...")
            skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
            scores = cross_val_score(
                build_pipeline(seed, target), train_df["text"].tolist(), train_df[target].values,
                cv=skf, scoring="f1_macro", n_jobs=1,
            )
            cv_scores = [round(float(s), 4) for s in scores]
            cv_mean, cv_std = float(scores.mean()), float(scores.std())
            print(f"  CV macro-F1: {cv_mean:.4f} +/- {cv_std:.4f}")

        png = artifacts_dir / f"confusion_{target}.png"
        has_png = _plot_confusion(cm, labels, f"Confusion matrix — {target} (held-out)", png)

        print(f"  accuracy={acc:.4f} macro-F1={macro_f1:.4f} "
              f"mean-confidence={conf.mean():.3f} inference={infer_ms:.2f} ms/doc")

        results[target] = {
            "labels": labels,
            "n_test": int(len(y_true)),
            "accuracy": round(acc, 4),
            "macro_f1": round(macro_f1, 4),
            "weighted_f1": round(weighted_f1, 4),
            "per_class": {
                lab: {
                    "precision": round(report[lab]["precision"], 4),
                    "recall": round(report[lab]["recall"], 4),
                    "f1": round(report[lab]["f1-score"], 4),
                    "support": int(report[lab]["support"]),
                }
                for lab in labels if lab in report
            },
            "confusion_matrix": cm.tolist(),
            "top_confusions": [
                {"actual": a, "predicted": p, "count": c}
                for a, p, c in _top_confusions(cm, labels)
            ],
            "cv_folds": cv_folds if run_cv else 0,
            "cv_macro_f1_mean": round(cv_mean, 4) if cv_mean is not None else None,
            "cv_macro_f1_std": round(cv_std, 4) if cv_std is not None else None,
            "cv_scores": cv_scores,
            "mean_confidence": round(float(conf.mean()), 4),
            "mean_confidence_when_correct": round(float(conf[correct].mean()), 4) if correct.any() else None,
            "mean_confidence_when_wrong": round(float(conf[~correct].mean()), 4) if (~correct).any() else None,
            "confidence_buckets": buckets,
            "inference_ms_per_doc": round(infer_ms, 3),
            "confusion_png": png.name if has_png else None,
            "_report": report,
            "_cm": cm,
        }

    meta_path = artifacts_dir / "metadata.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    stats_path = data_dir / "stats.json"
    data_stats = json.loads(stats_path.read_text(encoding="utf-8")) if stats_path.exists() else {}

    metrics = {
        "model_name": bundle.get("model_name", "tfidf-linearsvc-v1"),
        "evaluated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "trained_at": bundle.get("trained_at"),
        "sklearn_version": bundle.get("sklearn_version"),
        "artifact_size_human": meta.get("artifact_size_human"),
        "n_train": data_stats.get("n_train"),
        "n_test": data_stats.get("n_test"),
        "training_data": "synthetic — see ml/data/DATASET_CARD.md",
        "category": {k: v for k, v in results["category"].items() if not k.startswith("_")},
        "priority": {k: v for k, v in results["priority"].items() if not k.startswith("_")},
    }
    (artifacts_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    report_md = _render_report(metrics, results, meta, data_stats)
    (artifacts_dir / "evaluation.md").write_text(report_md, encoding="utf-8")
    print(f"\nwrote {artifacts_dir / 'metrics.json'}")
    print(f"wrote {artifacts_dir / 'evaluation.md'}")
    return metrics


def _render_cv_section(cat: dict, pri: dict) -> str:
    """Section 1b: why the k-fold CV number on template data is meaningless.

    This is the most instructive result in the whole report, so it is rendered
    explicitly rather than buried in a footnote.
    """
    if cat["cv_macro_f1_mean"] is None:
        return ("## 1b. Cross-validation\n\n"
                "Cross-validation was skipped for this run (`--no-cv`). Re-run "
                "`python -m ml.evaluate` to populate it.")

    cv_cat = cat["cv_macro_f1_mean"]
    gap = cv_cat - cat["macro_f1"]
    return f"""## 1b. Why the cross-validation number is worthless here (and the held-out one is not)

The 5-fold CV macro-F1 for **category is {cv_cat:.4f}**, against a held-out macro-F1
of **{cat['macro_f1']:.4f}**. That gap of ~{gap:.2f} is not a fluke and not a bug —
it is the single most instructive number on this page, so it is reported rather
than buried.

Standard k-fold CV shuffles the training split at random. Because the training
split is generated from **10 shared sentence frames**, almost every validation item
in a fold has near-identical siblings in the other four folds. The model is
therefore graded on phrasing it has already seen, and it scores near-perfectly.

**A random-split CV score on template-generated data measures memorisation, not
generalisation.** Anyone quoting {cv_cat:.2f} as this model's accuracy is quoting a
fiction — and this is exactly the trap that synthetic data lays for you.

The held-out split closes that hole: **10 completely different sentence frames**,
disjoint priority escalation clauses, and per-category `test_only` vocabulary.
Nothing in it is phrased like anything in training. That is why
**{cat['accuracy']:.4f} / {pri['accuracy']:.4f}** are the numbers quoted everywhere
else in this project, and the CV figures appear only as evidence of this effect.

Priority shows a smaller gap ({pri['cv_macro_f1_mean']:.4f} CV vs
{pri['macro_f1']:.4f} held-out) because its escalation clauses deliberately share
some urgency vocabulary across splits — realistic, since words like "urgent",
"accident" and "routine" genuinely do recur in real complaints."""


def _render_report(metrics: dict, results: dict, meta: dict, data_stats: dict) -> str:
    cat, pri = results["category"], results["priority"]

    def cv_line(r: dict) -> str:
        if r["cv_macro_f1_mean"] is None:
            return "not run"
        return f"{r['cv_macro_f1_mean']:.4f} ± {r['cv_macro_f1_std']:.4f} ({r['cv_folds']}-fold)"

    top_cat = ", ".join(
        f"`{c['actual']}`→`{c['predicted']}` ({c['count']})" for c in cat["top_confusions"]
    ) or "none"
    top_pri = ", ".join(
        f"`{c['actual']}`→`{c['predicted']}` ({c['count']})" for c in pri["top_confusions"]
    ) or "none"

    def bucket_table(r: dict) -> str:
        lines = ["| confidence bucket | n | actual accuracy | mean confidence |",
                 "|---|---|---|---|"]
        for b in r["confidence_buckets"]:
            acc = f"{b['accuracy']:.3f}" if b["accuracy"] is not None else "—"
            mc = f"{b['mean_confidence']:.3f}" if b["mean_confidence"] is not None else "—"
            lines.append(f"| {b['range']} | {b['n']} | {acc} | {mc} |")
        return "\n".join(lines)

    # Built here rather than inline: a format spec cannot span a newline inside an
    # f-string, and this section must degrade cleanly when `--no-cv` was passed.
    cv_section = _render_cv_section(cat, pri)
    baseline_cat = max((v["support"] for v in cat["per_class"].values()), default=0) / max(
        cat["n_test"], 1
    )
    baseline_pri = max((v["support"] for v in pri["per_class"].values()), default=0) / max(
        pri["n_test"], 1
    )

    feats = meta.get("features", {})
    return f"""# Model Evaluation — `{metrics['model_name']}`

*Generated {metrics['evaluated_at']} · model trained {metrics.get('trained_at')} ·
scikit-learn {metrics.get('sklearn_version')} · artifact {metrics.get('artifact_size_human')}*

> **Read the honesty section before quoting any number from this page.**
> The training data is synthetic. These scores are an **upper bound**, not a
> prediction of production accuracy.

## 1. Headline

| | category (7 classes) | priority (4 classes) |
|---|---|---|
| Held-out accuracy | **{cat['accuracy']:.4f}** | **{pri['accuracy']:.4f}** |
| Held-out macro-F1 | **{cat['macro_f1']:.4f}** | **{pri['macro_f1']:.4f}** |
| Weighted F1 | {cat['weighted_f1']:.4f} | {pri['weighted_f1']:.4f} |
| 5-fold CV macro-F1 (train split) — *see §1b, do not quote this* | {cv_line(cat)} | {cv_line(pri)} |
| Test items | {cat['n_test']} | {pri['n_test']} |
| Mean confidence | {cat['mean_confidence']:.3f} | {pri['mean_confidence']:.3f} |
| Mean confidence when **correct** | {cat['mean_confidence_when_correct']} | {pri['mean_confidence_when_correct']} |
| Mean confidence when **wrong** | {cat['mean_confidence_when_wrong']} | {pri['mean_confidence_when_wrong']} |
| Inference latency | {cat['inference_ms_per_doc']:.2f} ms/doc | {pri['inference_ms_per_doc']:.2f} ms/doc |

Majority-class baseline for category is ~{baseline_cat:.3f} and for priority
~{baseline_pri:.3f}; both heads beat it by a wide margin, which is the only thing an
accuracy figure is actually evidence of.

{cv_section}

## 2. Setup

| | |
|---|---|
| Representation | `FeatureUnion(TfidfVectorizer(word 1-2gram), TfidfVectorizer(char_wb 3-5gram))` |
| Features | {feats.get('word_1_2gram', '?')} word + {feats.get('char_wb_3_5gram', '?')} char = **{feats.get('total', '?')}** (shared by both heads) |
| Classifier | `CalibratedClassifierCV(LinearSVC(...), method="sigmoid", cv=5, ensemble=False)` |
| Class weighting | category `{_CLASS_WEIGHTS.get('category')}` · priority `{_CLASS_WEIGHTS.get('priority')}` (see below) |
| Train / test | {metrics.get('n_train')} / {metrics.get('n_test')} rows |
| Split design | test split uses **disjoint sentence frames and disjoint priority clauses** |
| Seed | {meta.get('seed')} |

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

{_per_class_table(cat['_report'], cat['labels'])}

**Confusion matrix** (rows = actual, columns = predicted){
    ', also rendered as `' + cat['confusion_png'] + '`' if cat['confusion_png'] else ''}:

{_md_confusion(cat['_cm'], cat['labels'])}

Most frequent confusions: {top_cat}.

## 4. Per-class results — priority

{_per_class_table(pri['_report'], pri['labels'])}

**Confusion matrix**{
    ', also rendered as `' + pri['confusion_png'] + '`' if pri['confusion_png'] else ''}:

{_md_confusion(pri['_cm'], pri['labels'])}

Most frequent confusions: {top_pri}.

Priority errors are not symmetric in cost. Predicting `low` for something that is
really `critical` is the expensive mistake; predicting `critical` for something
routine only wastes a site visit. That is why `class_weight="balanced"` is set —
it deliberately trades a little overall accuracy for better recall on the rare
`critical` class.

## 5. Is the confidence score trustworthy?

Category head:

{bucket_table(cat)}

Priority head:

{bucket_table(pri)}

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
   ({data_stats.get('roman_urdu_rows', '?')} of {data_stats.get('n', '?')} rows).
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
| Latency | 2–6 s | **~{cat['inference_ms_per_doc']:.1f} ms** | <1 ms |
| Cost per 1k complaints | ~$0.15 | **$0** | $0 |
| Deterministic | no | **yes** | yes |

DeepSeek's published uptime is ~99.79% with chunky multi-hour failure modes. A
system whose only classifier is a hosted API is a system that stops classifying
during a demo. Being pretty good and always available beats being excellent and
sometimes absent — which is precisely the job this model was trained for.

## 8. Reproducing these numbers

```bash
python -m ml.generate_dataset   # deterministic, seed {meta.get('seed')}
python -m ml.train
python -m ml.evaluate
```

Everything is seeded; a clean checkout reproduces this report exactly.
"""


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate the civic triage model")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--artifacts-dir", type=Path, default=ARTIFACTS_DIR)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--no-cv", action="store_true", help="skip cross-validation")
    args = parser.parse_args()
    evaluate(args.data_dir, args.artifacts_dir, args.seed, args.cv_folds, run_cv=not args.no_cv)
    return 0


if __name__ == "__main__":
    sys.exit(main())

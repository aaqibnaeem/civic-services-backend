"""Train the offline complaint-triage classifiers.

Architecture
------------
::

    text ──► FeatureUnion ──┬─► TfidfVectorizer(word,  1-2gram)  ──┐
                            └─► TfidfVectorizer(char_wb, 3-5gram) ─┤
                                                                   ▼
                                          CalibratedClassifierCV(LinearSVC)
                                                     │
                                    ┌────────────────┴────────────────┐
                                    ▼                                 ▼
                          category (7 classes)              priority (4 classes)

**Why word + char n-grams together.** Word n-grams capture the obvious civic
vocabulary ("pothole", "sewerage", "transformer"). Char_wb 3-5 grams are what make
this work on *Karachi* text: they survive typos ("potohle"), they tokenise
Roman-Urdu the way word n-grams cannot ("kachra"/"kachray"/"kooray" share char
stems), and they degrade gracefully on ALL-CAPS/SMS-shortened input. Neither
representation alone handles this corpus well; the union does.

**Why LinearSVC.** Best-in-class for high-dimensional sparse text, trains in
seconds, and the model is a single coefficient matrix — small enough to commit to
git and to load on a 512 MB Render instance. No torch, no downloads, no cold-start
model fetch.

**Why CalibratedClassifierCV.** ``LinearSVC`` has no ``predict_proba``; its raw
``decision_function`` margin is not a probability and must never be shown to a user
as "confidence". Wrapping it with sigmoid calibration (``cv=5``,
``ensemble=False``) yields genuine probability estimates, which is what the API
contract's ``confidence`` field promises. ``ensemble=False`` fits **one** final
estimator instead of one per fold — that is a ~5x artifact-size saving.

**Why one shared vectorizer.** Both heads read the same text. Fitting the
FeatureUnion once and sharing the fitted instance between the two Pipelines halves
both the artifact size and the resident memory, and joblib memoises the shared
object so it is serialised exactly once.

Run::

    python -m ml.train
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

try:
    from ml import ARTIFACTS_DIR, DATA_DIR, SEED
except ImportError:  # pragma: no cover
    _HERE = Path(__file__).resolve().parent
    ARTIFACTS_DIR, DATA_DIR, SEED = _HERE / "artifacts", _HERE / "data", 20260808


MODEL_NAME = "tfidf-linearsvc-v1"

# Feature budget. Tuned so the two calibrated heads plus the shared vocabulary
# serialise (joblib compress=3) to well under the 15 MB artifact ceiling, which
# keeps the model committable to git and cheap to load on Render's free tier.
WORD_MAX_FEATURES = 20_000
CHAR_MAX_FEATURES = 30_000


def build_features() -> FeatureUnion:
    """The shared text representation. Kept in one place so train/eval agree."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    return FeatureUnion(
        [
            (
                "word",
                TfidfVectorizer(
                    analyzer="word",
                    ngram_range=(1, 2),
                    lowercase=True,
                    strip_accents="unicode",
                    sublinear_tf=True,
                    min_df=2,
                    max_df=0.85,
                    max_features=WORD_MAX_FEATURES,
                    dtype=np.float32,
                ),
            ),
            (
                "char",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    lowercase=True,
                    sublinear_tf=True,
                    min_df=3,
                    max_features=CHAR_MAX_FEATURES,
                    dtype=np.float32,
                ),
            ),
        ],
        # Sequential: the corpus is tiny and threads would only add overhead on a
        # 0.1-CPU box.
        n_jobs=None,
    )


# Class weighting is decided per head, because the cost of an error differs.
#
#   priority -> "balanced". Calling a `critical` complaint `low` can get someone
#              killed; calling a routine one `critical` wastes a site visit. We
#              deliberately buy recall on the rare `critical` class with a little
#              overall accuracy.
#   category -> None. Mis-routing `safety` as `other` and `other` as `safety` cost
#              about the same. Up-weighting the small `other` class only made the
#              model over-predict it (precision fell to 0.51 while recall hit 0.84),
#              which is worse for a routing system than leaving it unweighted.
CLASS_WEIGHTS: dict[str, str | None] = {"category": None, "priority": "balanced"}


def build_classifier(seed: int = SEED, class_weight: str | None = None) -> CalibratedClassifierCV:
    """LinearSVC + sigmoid calibration -> real probabilities for `confidence`."""
    base = LinearSVC(
        C=1.0,
        loss="squared_hinge",
        class_weight=class_weight,
        max_iter=5000,
        random_state=seed,
        dual="auto",
    )
    return CalibratedClassifierCV(base, method="sigmoid", cv=5, ensemble=False)


def build_pipeline(seed: int = SEED, target: str = "category") -> Pipeline:
    """A complete, unfitted end-to-end pipeline (used by cross-validation)."""
    return Pipeline([
        ("features", build_features()),
        ("clf", build_classifier(seed, CLASS_WEIGHTS.get(target))),
    ])


def _load(split: str, data_dir: Path) -> pd.DataFrame:
    path = data_dir / f"{split}.csv"
    if not path.exists():
        raise SystemExit(
            f"missing {path} — run `python -m ml.generate_dataset` first"
        )
    df = pd.read_csv(path)
    df["text"] = df["text"].astype(str)
    return df


def _human_size(num_bytes: int) -> str:
    return f"{num_bytes / 1_048_576:.2f} MB"


def train(data_dir: Path = DATA_DIR, artifacts_dir: Path = ARTIFACTS_DIR,
          seed: int = SEED) -> dict:
    t0 = time.perf_counter()
    train_df = _load("train", data_dir)
    test_df = _load("test", data_dir)

    x_train = train_df["text"].tolist()
    print(f"train rows : {len(train_df)}")
    print(f"test rows  : {len(test_df)}  (disjoint sentence frames)")

    # --- 1. fit the shared representation once -------------------------------
    print("\nfitting shared FeatureUnion (word 1-2gram + char_wb 3-5gram)...")
    features = build_features()
    xt = features.fit_transform(x_train)
    n_word = len(features.transformer_list[0][1].vocabulary_)
    n_char = len(features.transformer_list[1][1].vocabulary_)
    print(f"  word features : {n_word}")
    print(f"  char features : {n_char}")
    print(f"  matrix        : {xt.shape}, nnz={xt.nnz}, density={xt.nnz / (xt.shape[0] * xt.shape[1]):.5f}")

    # --- 2. two calibrated heads over the shared features --------------------
    heads: dict[str, Pipeline] = {}
    head_meta: dict[str, dict] = {}
    for target in ("category", "priority"):
        print(f"\ntraining `{target}` head...")
        t1 = time.perf_counter()
        clf = build_classifier(seed, CLASS_WEIGHTS.get(target))
        clf.fit(xt, train_df[target].values)
        elapsed = time.perf_counter() - t1
        # Assemble a real sklearn Pipeline from the fitted parts. `features` is the
        # SAME object in both pipelines -> joblib stores its vocabulary once.
        heads[target] = Pipeline([("features", features), ("clf", clf)])
        classes = list(clf.classes_)
        head_meta[target] = {"classes": classes, "fit_seconds": round(elapsed, 2)}
        print(f"  classes: {classes}")
        print(f"  fitted in {elapsed:.2f}s")

    # --- 3. quick sanity read on the held-out split --------------------------
    from sklearn.metrics import accuracy_score, f1_score

    quick: dict[str, dict] = {}
    for target, pipe in heads.items():
        pred = pipe.predict(test_df["text"].tolist())
        acc = float(accuracy_score(test_df[target], pred))
        f1 = float(f1_score(test_df[target], pred, average="macro"))
        quick[target] = {"holdout_accuracy": round(acc, 4), "holdout_macro_f1": round(f1, 4)}
        print(f"\n[{target}] held-out accuracy={acc:.4f}  macro-F1={f1:.4f}")

    # --- 4. serialise --------------------------------------------------------
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    bundle = {
        "model_name": MODEL_NAME,
        "category": heads["category"],
        "priority": heads["priority"],
        "categories": head_meta["category"]["classes"],
        "priorities": head_meta["priority"]["classes"],
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "sklearn_version": sklearn.__version__,
        "seed": seed,
    }
    model_path = artifacts_dir / "model.joblib"
    joblib.dump(bundle, model_path, compress=3)
    size = model_path.stat().st_size
    print(f"\nsaved {model_path}  ({_human_size(size)})")
    if size > 15 * 1_048_576:
        print("!! artifact exceeds the 15 MB budget — lower WORD/CHAR_MAX_FEATURES")

    metadata = {
        "model_name": MODEL_NAME,
        "trained_at": bundle["trained_at"],
        "seed": seed,
        "sklearn_version": sklearn.__version__,
        "python_version": platform.python_version(),
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "features": {
            "word_1_2gram": n_word,
            "char_wb_3_5gram": n_char,
            "total": n_word + n_char,
            "shared_between_heads": True,
        },
        "heads": head_meta,
        "quick_holdout": quick,
        "artifact_bytes": size,
        "artifact_size_human": _human_size(size),
        "train_seconds": round(time.perf_counter() - t0, 2),
        "training_data": "synthetic (ml/generate_dataset.py) — see ml/data/DATASET_CARD.md",
    }
    (artifacts_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"saved {artifacts_dir / 'metadata.json'}")
    print(f"\ntotal {metadata['train_seconds']}s")
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser(description="Train civic triage classifiers")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--artifacts-dir", type=Path, default=ARTIFACTS_DIR)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    train(args.data_dir, args.artifacts_dir, args.seed)
    return 0


if __name__ == "__main__":
    sys.exit(main())

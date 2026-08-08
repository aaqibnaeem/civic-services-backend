"""Offline ML assets for the AI Smart Civic Services complaint triage model.

This package is *not* imported by the running API. It contains the reproducible
offline pipeline that produces the artifacts in ``ml/artifacts/``:

    python -m ml.generate_dataset   ->  ml/data/{train,test,dataset}.csv
    python -m ml.train              ->  ml/artifacts/model.joblib + metadata.json
    python -m ml.evaluate           ->  ml/artifacts/evaluation.md + metrics.json + PNGs

The API only ever *loads* ``ml/artifacts/model.joblib`` (see
``app/ai/ml_analyzer.py``). Training never happens at boot.
"""

__all__ = ["ARTIFACTS_DIR", "DATA_DIR", "SEED"]

from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
DATA_DIR = PACKAGE_DIR / "data"
ARTIFACTS_DIR = PACKAGE_DIR / "artifacts"

#: One global seed drives dataset generation, splitting and model training so
#: every number in ``evaluation.md`` is reproducible from a clean checkout.
SEED = 20260808

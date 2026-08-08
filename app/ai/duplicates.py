"""Duplicate complaint detection.

Twenty people report the same overflowing gutter and the department opens twenty
work orders. Detecting that is worth more to a municipal operation than another
percentage point of classification accuracy.

METHOD
    1. **Candidate generation (SQL).** Same category, not deleted, not the target,
       inside a 30-day window, newest first, hard-capped. Cheap and index-friendly.
    2. **Geographic gate.** If *both* complaints carry coordinates, they must be
       within ``radius_m`` (default 500 m) by haversine distance. If either lacks
       coordinates the gate is skipped rather than failing closed — most citizen
       submissions have no GPS, and dropping them would make the feature useless.
    3. **Lexical scoring.** TF-IDF over the candidate set (word 1-2 grams unioned
       with char_wb 3-5 grams, so Roman-Urdu spelling variants and typos still
       match) then cosine similarity against the target.
    4. **Banding.** >= 0.72 "probable duplicate", >= 0.55 "related". Below that,
       discarded.
    5. **Explanation.** Every candidate carries a human-readable ``reason`` naming
       the shared terms, the distance and the age gap, because an operator will not
       act on a bare number.

LIMITATIONS
    * **Lexical, not semantic.** This is bag-of-ngrams cosine similarity. Two people
      describing the same pothole in completely different words — one in English,
      one in Roman-Urdu — will not match. Real semantic matching needs embeddings,
      and DeepSeek has no embeddings endpoint, so the honest options were a local
      embedding model (too heavy for a 512 MB instance) or this. This is what was
      chosen, and it is the tool's main weakness.
    * The TF-IDF space is fitted per query over the candidate set, so scores are
      relative to that set and are not comparable across different queries.
    * The 0.72 threshold is a judgement call tuned by eye on the seed data, not
      validated against labelled duplicate pairs — no such labels exist here.
    * Same-category gating means a genuine duplicate filed under a different
      category is invisible.
    * Two distinct potholes on the same street read as duplicates when neither
      submission has coordinates.

Nothing here auto-merges anything. It returns ranked *candidates*; a human decides.
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

logger = logging.getLogger("app.ai.duplicates")

# Thresholds calibrated by measurement, not guessed. Observed bands on real
# re-phrasings of the same complaint versus distinct complaints in the same
# category (see the calibration in the AI phase notes):
#
#   same complaint, re-phrased in the same language ....... 0.61 - 0.67
#   same complaint, ALL CAPS / SMS-shortened .............. 0.61
#   same complaint, translated to Roman-Urdu .............. 0.16  <- MISSED
#   different complaints, same category ................... 0.14 - 0.19
#
# An earlier revision used 0.72/0.55, which sat above the true-duplicate band and
# silently detected nothing. The cross-language miss at 0.16 is not fixable with a
# lexical method and is documented as a limitation rather than papered over.
#: Cosine similarity at or above which two complaints are called duplicates.
DUPLICATE_THRESHOLD = 0.55
#: Cosine similarity at or above which they are merely "related".
RELATED_THRESHOLD = 0.38
#: Default geographic gate, metres.
DEFAULT_RADIUS_M = 500.0
#: Default lookback window, days.
DEFAULT_WINDOW_DAYS = 30
#: Hard cap on rows pulled from SQL. Bounds both memory and vectoriser cost.
MAX_CANDIDATES = 300

_STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "has", "have", "been", "are", "was",
    "were", "from", "our", "not", "there", "they", "very", "please", "kindly", "sir",
    "here", "which", "since", "into", "out", "all", "any", "near", "who", "will",
    "hai", "hain", "ka", "ki", "ke", "me", "mein", "se", "par", "aur", "bohot", "bhi",
    "nahi", "kar", "raha", "rahi", "gaya", "gayi", "hum", "hamari", "koi", "ho",
}


@dataclass(slots=True)
class DuplicateCandidate:
    """One scored candidate. ``complaint`` is the ORM object, left to the caller."""

    complaint: Any
    similarity: float
    relation: str          # "duplicate" | "related"
    reason: str
    distance_m: float | None = None
    days_apart: float | None = None

    def to_dict(self, serialiser: Any = None) -> dict[str, Any]:
        """Contract shape: ``{complaint, similarity, reason}`` plus extras."""
        complaint = serialiser(self.complaint) if serialiser else self.complaint
        return {
            "complaint": complaint,
            "similarity": round(self.similarity, 4),
            "reason": self.reason,
            "relation": self.relation,
            "distance_m": round(self.distance_m, 1) if self.distance_m is not None else None,
            "days_apart": round(self.days_apart, 1) if self.days_apart is not None else None,
        }


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in metres between two WGS-84 points."""
    radius = 6_371_000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = (math.sin(dphi / 2) ** 2
         + math.cos(p1) * math.cos(p2) * math.sin(dlambda / 2) ** 2)
    return 2 * radius * math.asin(min(1.0, math.sqrt(a)))


def _tokens(text: str) -> set[str]:
    return {
        w for w in re.findall(r"[a-z0-9]{3,}", (text or "").lower())
        if w not in _STOPWORDS
    }


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _cosine_scores(target: str, candidates: list[str]) -> list[float]:
    """Cosine similarity of ``target`` against each candidate.

    Uses scikit-learn when present. Falls back to Jaccard token overlap otherwise,
    so duplicate detection still works on a machine with no ML stack installed.
    """
    if not candidates:
        return []
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        from sklearn.pipeline import FeatureUnion
    except ImportError:
        logger.info("sklearn unavailable, duplicate scoring falling back to Jaccard")
        t = _tokens(target)
        return [_jaccard(t, _tokens(c)) for c in candidates]

    corpus = [target, *candidates]
    # IDF must be OFF on a small candidate set. With only a handful of documents,
    # a term shared by both complaints gets a *lower* weight than a term unique to
    # one of them — precisely inverting the signal we want, since shared vocabulary
    # is the evidence of duplication. With enough candidates IDF starts earning its
    # keep again by down-weighting boilerplate ("please", "kindly", "street").
    use_idf = len(corpus) >= 15
    try:
        vectoriser = FeatureUnion([
            ("word", TfidfVectorizer(analyzer="word", ngram_range=(1, 2),
                                     lowercase=True, strip_accents="unicode",
                                     sublinear_tf=True, min_df=1, use_idf=use_idf)),
            # char n-grams are what let "kachra" match "kachray" and survive typos
            ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                                     lowercase=True, sublinear_tf=True, min_df=1,
                                     use_idf=use_idf)),
        ])
        matrix = vectoriser.fit_transform(corpus)
        sims = cosine_similarity(matrix[0:1], matrix[1:])[0]
        return [float(s) for s in sims]
    except Exception:  # noqa: BLE001 - never let a scoring bug break the endpoint
        logger.exception("tfidf duplicate scoring failed, falling back to Jaccard")
        t = _tokens(target)
        return [_jaccard(t, _tokens(c)) for c in candidates]


def _shared_terms(a: str, b: str, limit: int = 4) -> list[str]:
    shared = _tokens(a) & _tokens(b)
    return sorted(shared, key=len, reverse=True)[:limit]


def _as_utc(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value


def _build_reason(similarity: float, relation: str, shared: list[str],
                  distance_m: float | None, days_apart: float | None,
                  same_area: bool, category: str) -> str:
    """Plain English, because an operator will not act on a float."""
    lead = ("Probable duplicate" if relation == "duplicate" else "Possibly related")
    bits = [f"{int(round(similarity * 100))}% wording similarity"]
    if shared:
        bits.append("shares " + ", ".join(f"'{t}'" for t in shared))
    if distance_m is not None:
        bits.append(f"{int(round(distance_m))} m away" if distance_m >= 1
                    else "same coordinates")
    elif same_area:
        bits.append("same area")
    bits.append(f"same category ({category})")
    if days_apart is not None:
        if days_apart < 1:
            bits.append("filed within a day")
        else:
            bits.append(f"filed {int(round(days_apart))} days apart")
    return f"{lead}: " + "; ".join(bits) + "."


async def find_duplicates(
    complaint_id: str,
    *,
    session: Any = None,
    limit: int = 5,
    radius_m: float = DEFAULT_RADIUS_M,
    window_days: int = DEFAULT_WINDOW_DAYS,
    threshold: float = RELATED_THRESHOLD,
    same_category_only: bool = True,
) -> list[DuplicateCandidate]:
    """Ranked duplicate candidates for one complaint, most similar first.

    Opens its own session when ``session`` is None so it is usable from a
    background task as well as from a request.
    """
    from sqlalchemy import select

    from app.db.session import SessionLocal
    from app.models.complaint import Complaint

    owns_session = session is None
    session_cm = SessionLocal() if owns_session else None
    if owns_session:
        session = await session_cm.__aenter__()

    try:
        target = await session.get(Complaint, complaint_id)
        if target is None or not (target.description or "").strip():
            return []

        cutoff = datetime.now(UTC) - timedelta(days=window_days)
        stmt = (
            select(Complaint)
            .where(Complaint.id != target.id)
            .where(Complaint.is_deleted.is_(False))
            .where(Complaint.created_at >= cutoff)
            .order_by(Complaint.created_at.desc())
            .limit(MAX_CANDIDATES)
        )
        if same_category_only:
            stmt = stmt.where(Complaint.category == target.category)
        rows = list((await session.execute(stmt)).scalars().unique())
        if not rows:
            return []

        # Geographic gate first: it is far cheaper than vectorising.
        target_created = _as_utc(target.created_at)
        gated: list[tuple[Any, float | None, float | None]] = []
        for row in rows:
            distance = None
            if (target.latitude is not None and target.longitude is not None
                    and row.latitude is not None and row.longitude is not None):
                distance = haversine_m(
                    float(target.latitude), float(target.longitude),
                    float(row.latitude), float(row.longitude),
                )
                if distance > radius_m:
                    continue  # both have coords and they are far apart -> not a dup
            row_created = _as_utc(row.created_at)
            days_apart = None
            if target_created and row_created:
                days_apart = abs((target_created - row_created).total_seconds()) / 86400.0
            gated.append((row, distance, days_apart))

        if not gated:
            return []

        scores = _cosine_scores(target.description, [r.description or "" for r, _, _ in gated])

        candidates: list[DuplicateCandidate] = []
        for (row, distance, days_apart), score in zip(gated, scores, strict=False):
            if score < threshold:
                continue
            relation = "duplicate" if score >= DUPLICATE_THRESHOLD else "related"
            candidates.append(
                DuplicateCandidate(
                    complaint=row,
                    similarity=float(score),
                    relation=relation,
                    reason=_build_reason(
                        float(score), relation,
                        _shared_terms(target.description, row.description or ""),
                        distance, days_apart,
                        bool(target.area and row.area and target.area == row.area),
                        str(target.category),
                    ),
                    distance_m=distance,
                    days_apart=days_apart,
                )
            )

        candidates.sort(key=lambda c: -c.similarity)
        return candidates[:limit]
    finally:
        if owns_session and session_cm is not None:
            await session_cm.__aexit__(None, None, None)


async def best_duplicate_id(complaint_id: str, *, session: Any = None) -> str | None:
    """The id of the single best *duplicate*-band match, or None.

    Used by the pipeline to populate ``Complaint.duplicate_of_id``. Only the strict
    band qualifies — "related" is informative, but it must never silently link two
    complaints together.
    """
    try:
        candidates = await find_duplicates(
            complaint_id, session=session, limit=1, threshold=DUPLICATE_THRESHOLD
        )
    except Exception:  # noqa: BLE001 - duplicate detection is never load-bearing
        logger.exception("duplicate detection failed for %s", complaint_id)
        return None
    if not candidates:
        return None
    best = candidates[0]
    # Never point at something that is itself a duplicate: keep chains flat so the
    # UI can always resolve to one canonical complaint in a single hop.
    original = getattr(best.complaint, "duplicate_of_id", None)
    return original or best.complaint.id

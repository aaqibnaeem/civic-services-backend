"""Deterministic keyword/regex tier — the last resort that cannot fail.

INPUT      raw complaint text.
PROCESSING weighted keyword and regex matching over the seven categories in both
           English and Roman-Urdu, plus a priority heuristic built from emergency
           signals, exposure signals, duration phrases and typographic intensity.
OUTPUT     an :class:`AnalysisResult` with ``source="rules"`` and a deliberately
           modest confidence.

Why it exists: every tier above it can be absent. There may be no API key, no
network, and no model artifact — that is the state of a fresh clone on a grader's
laptop. This tier has zero dependencies beyond the standard library, runs in well
under a millisecond, and is the reason ``analyze_text`` can promise never to raise.

LIMITATIONS — significant, and stated plainly
    * No understanding whatsoever. It matches strings. "There is no garbage
      problem on our street any more, thank you" scores as a waste complaint.
    * Negation-blind, sarcasm-blind, context-blind.
    * Confidence is capped at 0.62 by design. A keyword hit is weak evidence, and
      the UI must never present a rules result as though it were a model result.
    * Vocabulary is hand-written and finite; anything phrased unusually falls to
      "other".
    * The priority heuristic is a scoring function someone chose, not something
      learned or validated against outcomes.
    * Its summary is the first sentence of the complaint. It is extractive, not
      abstractive — it cannot condense.

It is nonetheless the highest-value 300 lines in the file for demo safety: with
no API key and no model, the product still classifies, still routes, and still
flags emergencies.
"""

from __future__ import annotations

import re
import time
from typing import Any

from app.ai.base import (
    CATEGORY_LABELS,
    DEPARTMENT_BY_CATEGORY,
    AIAnalyzer,
    AnalysisResult,
)

# --------------------------------------------------------------------------- #
# Category lexicon.
#
# (pattern, weight). Weights encode specificity: "pothole" is decisive for road,
# while "water" alone is weak because it appears in drainage and safety
# complaints too. Patterns are matched with word boundaries where sensible so
# "nali" does not fire inside "finally".
# --------------------------------------------------------------------------- #

CATEGORY_PATTERNS: dict[str, list[tuple[str, float]]] = {
    "road": [
        (r"pot[\s-]?holes?", 3.5), (r"\bgadd?h[ae]\b", 3.5), (r"\bkhadd?[ae]s?\b", 3.5),
        (r"\broads?\b", 1.6), (r"\bsarak\b", 2.4), (r"\bsadak\b", 2.4), (r"\brasta\b", 1.4),
        (r"\bstreet\b(?!\s*light)", 1.0), (r"\bfootpath\b", 2.6), (r"\bpavement\b", 2.4),
        (r"\basphalt\b", 3.0), (r"\btarmac\b", 3.0), (r"\bcarriageway\b", 3.0),
        (r"speed[\s-]?breakers?", 2.8), (r"\bkerb\b", 2.4), (r"\bcurb\b", 2.0),
        (r"road\s+(?:toot|broken|damaged|caved|sunk)", 3.2), (r"\btoot\s+gay?i\b", 1.6),
        (r"\bcraters?\b", 2.6), (r"\bsinkhole\b", 3.0), (r"\btrench\b", 2.0),
        (r"\bgravel\b", 1.8), (r"\bdiversion\b", 1.2), (r"\buneven\b", 1.2),
    ],
    "water": [
        (r"\bdrinking\s+water\b", 3.4), (r"water\s+supply", 3.4), (r"\bwater\s+line\b", 3.0),
        (r"\bpani\b", 2.2), (r"\bpaani\b", 2.2), (r"\bnalka\b", 2.6), (r"\btap\b", 1.6),
        (r"\btankers?\b", 2.8), (r"\bboring\b", 2.2), (r"\bhydrants?\b", 2.6),
        (r"\bpipe\s?lines?\b", 2.4), (r"\bwater\s+pressure\b", 3.0),
        (r"\bno\s+water\b", 3.2), (r"pani\s+(?:nahi|nhi|band)", 3.4),
        (r"\bwater\s+meter\b", 2.4), (r"\bwater\s+tank\b", 2.4),
        (r"\bmuddy\s+water\b", 2.6), (r"\bdirty\s+water\b", 1.4), (r"\bpeela\s+pani\b", 3.0),
        (r"\bvalve\b", 2.2), (r"\bwater\s+bill\b", 2.6), (r"\bwater\b", 0.8),
        (r"\bleak(?:age|ing|s)?\b", 1.4), (r"\bwater\s+wast\w+", 2.6),
    ],
    "waste": [
        (r"\bgarbage\b", 3.6), (r"\btrash\b", 3.4), (r"\brubbish\b", 3.4),
        (r"\bkachr[ae]y?\b", 3.6), (r"\bkoo?d[ae]\b", 3.2), (r"\bkoo?ray?\b", 3.0),
        (r"\bgandagi\b", 3.0), (r"\bmalba\b", 3.0), (r"\bsafai\b", 2.8),
        (r"\bsweepers?\b", 2.8), (r"\bsanitation\b", 2.8), (r"\bdump(?:ed|ing|site)?\b", 2.4),
        (r"\bwaste\b", 1.6), (r"\bdustbins?\b", 3.0), (r"\bcontainers?\b", 1.4),
        (r"kachra\s+kund[iy]", 3.6), (r"\brotting\b", 2.2), (r"\bstench\b", 2.4),
        (r"\bbadbo?u?\b", 2.2), (r"\bsmell\b", 1.2), (r"\bflies\b", 1.8),
        (r"\bmakhiy?an\b", 2.0), (r"\bheaps?\s+of\b", 1.6), (r"\blitter\b", 2.4),
        (r"\bdead\s+(?:animal|dog|cat)\b", 2.8), (r"\bdebris\b", 2.2),
    ],
    "electricity": [
        (r"\belectricity\b", 3.2), (r"\bbijli\b", 3.4), (r"\bpower\s+(?:cut|outage|failure)\b", 3.2),
        (r"street[\s-]?lights?\b", 3.6), (r"\bstreetlights?\b", 3.6), (r"\blight\s+band\b", 3.0),
        (r"\blamp\s?posts?\b", 3.0), (r"\bkhamb[ae]\b", 3.4), (r"\bpoles?\b", 2.0),
        (r"\btransformers?\b", 3.4), (r"\bload[\s-]?shedding\b", 3.4),
        (r"\bwires?\b", 2.2), (r"\btaar\b", 2.6), (r"\bvoltage\b", 3.0),
        (r"\bmeters?\b", 1.4), (r"\bsparks?\b", 2.4), (r"\bchingari\b", 2.6),
        (r"\bfeeders?\b", 2.6), (r"\bK-?Electric\b", 3.4), (r"\bKE\b", 1.2),
        (r"\belectric\s+bill\b", 2.8), (r"\bcable\b", 1.6), (r"\bpankha\b", 1.4),
        (r"\bandhera\b", 2.0), (r"\bdark\b", 0.9),
    ],
    "drainage": [
        (r"\bsewer(?:age|s)?\b", 3.8), (r"\bsewage\b", 3.8), (r"\bgutters?\b", 3.6),
        (r"\bnali(?:yan|yon)?\b", 3.4), (r"\bnall?ah?\b", 3.0), (r"\bmanholes?\b", 2.6),
        (r"\bdrains?\b", 3.0), (r"\bdrainage\b", 3.8), (r"\bnikasi\b", 3.0),
        (r"\bchoked\b", 2.6), (r"\bblocked\b", 1.6), (r"\boverflow\w*", 2.8),
        (r"\bubal\s+raha\b", 3.0), (r"\bstanding\s+water\b", 2.2),
        (r"\bstagnant\b", 2.4), (r"\bgand[ae]\s+pani\b", 3.4),
        (r"\bwaste\s?water\b", 3.2), (r"\beffluent\b", 3.0), (r"\bbackflow\b", 3.0),
        (r"\brain\s?water\b", 2.0), (r"\bflood(?:ing|ed)?\b", 2.0),
        (r"\binspection\s+chamber\b", 3.0), (r"\bseptic\b", 2.8),
    ],
    "safety": [
        (r"\bsnatch\w*", 3.6), (r"\bmugg\w+", 3.4), (r"\brobber\w*", 3.4),
        (r"\bthefts?\b", 3.0), (r"\bchori\b", 3.0), (r"\bdacoit\w*", 3.4),
        (r"\bstray\s+dogs?\b", 3.4), (r"\bawara\s+kutt?[ae]\b", 3.4),
        (r"\bdog\s+bite\b", 3.4), (r"\bharass\w+", 3.2), (r"\bcrime\b", 3.0),
        (r"\bunsafe\b", 2.6), (r"\bdangerous\b", 1.8), (r"\bkhatarnak\b", 2.4),
        (r"\bcollaps\w+", 2.4), (r"\bdilapidated\b", 2.8), (r"\bfire\s+exit\b", 3.2),
        (r"\bgas\s+cylinder\b", 3.0), (r"\bdrug\w*", 2.8), (r"\bnash[ae]\b", 2.6),
        (r"\bpolice\b", 2.2), (r"\bgasht\b", 2.4), (r"\brash\s+driving\b", 3.0),
        (r"\bfear\b", 1.6), (r"\bdar\s+lagta\b", 2.4), (r"\bassault\w*", 3.0),
    ],
    "other": [
        (r"\bencroach\w+", 3.0), (r"\bqabza\b", 3.0), (r"\bparks?\b", 1.4),
        (r"\bplayground\b", 1.8), (r"\blibrary\b", 2.4), (r"\bdispensar\w+", 2.6),
        (r"\bloudspeakers?\b", 2.6), (r"\bnoise\b", 2.2), (r"\bhoardings?\b", 2.2),
        (r"\bbanners?\b", 2.0), (r"\bbriber?y?\b", 2.6), (r"\brishwat\b", 2.8),
        (r"\bunion\s+council\b", 2.2), (r"\bchallan\b", 2.4), (r"\bproperty\s+tax\b", 2.6),
        (r"\bgraveyard\b", 2.4), (r"\bparking\s+(?:mafia|fee|contractor)\b", 2.6),
        (r"\bcertificate\b", 2.0), (r"\brecords?\b", 1.2),
    ],
}

# Compile once at import.
_COMPILED: dict[str, list[tuple[re.Pattern[str], float]]] = {
    cat: [(re.compile(p, re.IGNORECASE), w) for p, w in pats]
    for cat, pats in CATEGORY_PATTERNS.items()
}

# --------------------------------------------------------------------------- #
# Priority signals
# --------------------------------------------------------------------------- #

EMERGENCY_PATTERNS: list[tuple[str, float, str]] = [
    (r"\belectric(?:\s|-)?shock\b", 5.0, "electric shock"),
    (r"\belectrocut\w+", 5.0, "electrocution"),
    (r"\bcurrent\s+(?:laga|aa\s*raha|hai)\b", 4.5, "live current"),
    (r"\blive\s+wire\b", 4.5, "live wire"),
    (r"\bdied?\b|\bdeath\b|\bmaut\b|\bfatal\b", 5.0, "death"),
    (r"\bkilled\b", 5.0, "fatality"),
    (r"\baccidents?\b|\bhadsa\b", 3.2, "accident"),
    (r"\binjur\w+|\bzakhmi\b|\bwounded\b", 3.8, "injury"),
    (r"\bhospitali[sz]\w+", 3.8, "hospitalisation"),
    (r"\bambulance\b", 3.5, "ambulance access"),
    (r"\bfire\s+brigade\b|\bexplosion\b|\bgas\s+leak\b", 4.5, "fire/explosion risk"),
    (r"\bcollaps\w+|\bgir\s+gay?i\b|\bgirne\s+wali\b", 3.0, "structural collapse"),
    (r"\bemergency\b|\bemergancy\b", 2.0, "stated emergency"),
    (r"\bjaan\s+ka\s+khatra\b|\bdanger\s+to\s+life\b", 4.5, "danger to life"),
    (r"\bopen\s+manhole\b|\bkhula\s+manhole\b", 3.2, "open manhole"),
    (r"\bdrowned?\b|\bdoob\s+gay?[ae]\b", 4.5, "drowning"),
    (r"\bsewage\s+.{0,25}drinking\b|\bdrinking\s+.{0,25}sewage\b", 4.0, "sewage in drinking water"),
    (r"\bbachao\b|\bmadad\b", 1.5, "call for help"),
    # Karachi-specific hazards that are emergencies in practice but contain no
    # English emergency word, so nothing above would catch them.
    (r"\b(?:taar|wires?|cables?)\b[^.]{0,40}\b(?:latak|hang\w*|neeche|touching|lying)\b",
     3.5, "hanging wire"),
    (r"\bhanging\s+(?:wires?|cables?)\b", 3.5, "hanging wire"),
    (r"\b(?:khamb[ae]|pole)\b[^.]{0,30}\b(?:jhuk\w*|tirh\w*|lean\w*|tilt\w*|gir\w*)\b",
     3.2, "leaning pole"),
    (r"\bspark\w*\b[^.]{0,40}\b(?:water|pani|barish)\b", 3.5, "sparks near water"),
    (r"\b(?:fell|fallen)\b|\bgir\s+gay[ae]\b", 2.2, "someone fell"),
    (r"\bmanhole\b[^.]{0,25}\b(?:open|khul\w*|missing|uncovered)\b", 3.2, "open manhole"),
    (r"\b(?:open|uncovered|missing\s+cover)\b[^.]{0,25}\bmanhole\b", 3.2, "open manhole"),
    (r"\b(?:wall|deewar|building|structure|boundary)\b[^.]{0,45}"
     r"\b(?:collaps\w*|lean\w*|crack\w*|gir\s+\w*|come\s+down|fall\w*)\b", 3.2, "structure failing"),
    (r"\bgas\s+cylinder\b", 3.2, "gas cylinder hazard"),
    (r"\bcatch\w*\s+fire\b|\bcaught\s+fire\b|\bfire\s+risk\b", 3.0, "fire risk"),
    (r"\battack(?:ed|ing|s)?\b|\bbit(?:ten|es?)\b|\bkaat\s+\w*\b", 2.6, "attack"),
    (r"\bstitch(?:es)?\b|\bfracture\w*|\bbleeding\b", 2.6, "medical treatment needed"),
    (r"\badmitted\s+to\s+(?:the\s+)?hospital\b|\bin\s+the\s+hospital\b", 3.0, "hospitalised"),
    (r"\bvomit\w+|\bgastro\b|\bpoison\w+", 2.6, "acute illness"),
    (r"\bsomeone\s+(?:will|could|may)\s+(?:fall|die|be\s+killed)\b", 3.0, "foreseeable serious harm"),
    (r"\bbefore\s+someone\s+is\s+killed\b", 3.5, "foreseeable fatality"),
]

EXPOSURE_PATTERNS: list[tuple[str, float, str]] = [
    (r"\bschools?\b|\bmadrass?ah?\b|\bskool\b", 2.6, "school nearby"),
    (r"\bhospitals?\b|\bclinics?\b|\bdispensar\w+|\bmaternity\b", 2.6, "hospital nearby"),
    (r"\bchildren\b|\bbach+[aeiy]\w*\b|\bkids\b|\bbaby\b", 2.0, "children affected"),
    (r"\belderly\b|\bburzurg\b|\bold\s+(?:man|woman|people)\b", 1.6, "elderly affected"),
    (r"\bpregnan\w+|\bpatients?\b|\bmareez\b", 1.8, "vulnerable people"),
    (r"\bwhole\s+(?:street|block|area|colony)\b|\bpuri\s+gali\b|\bentire\s+\w+\b", 1.8, "whole area affected"),
    (r"\b\d{2,}\s*(?:houses|homes|families|ghar)\b", 1.8, "many households"),
    (r"\bmain\s+road\b|\bmain\s+sarak\b", 1.4, "main road"),
    (r"\bdengue\b|\bcholera\b|\bmalaria\b|\bepidemic\b|\bdisease\b|\bbimari\b", 2.2, "disease risk"),
    (r"\bhealth\s+hazard\b", 2.0, "health hazard"),
    (r"\btraffic\s+jam\b|\bblocked\s+.{0,15}(?:road|access)\b", 1.4, "traffic obstruction"),
    (r"\bmarkets?\b|\bbus\s+stop\b|\bmosque\b|\bmasjid\b", 1.0, "public place"),
]

DEESCALATION_PATTERNS: list[tuple[str, float, str]] = [
    (r"\bnot\s+urgent\b|\bno\s+urgency\b|\bnot\s+an\s+emergency\b", 3.0, "explicitly not urgent"),
    (r"\bminor\b|\bsmall\s+issue\b|\btrivial\b", 2.0, "described as minor"),
    (r"\broutine\b|\bnormal\s+schedule\b|\bregular\s+maintenance\b", 2.0, "routine request"),
    (r"\bwhenever\s+(?:convenient|possible|a\s+team)\b", 2.0, "no deadline"),
    (r"\bsuggestion\b|\brecommend\w+|\bfor\s+(?:the\s+)?record\b", 1.5, "suggestion, not a fault"),
    (r"\blow\s+priority\b", 2.5, "self-declared low priority"),
]

DURATION_PATTERNS: list[tuple[str, float, str]] = [
    (r"\b(?:several|many|few)\s+months?\b|\bmahin[oe]n?\s+se\b", 2.0, "months unresolved"),
    (r"\b(?:\d{2,})\s*(?:days|din)\b", 1.6, "weeks unresolved"),
    (r"\b(?:a|one|two|three|four|five|six|do|teen|char)\s+(?:weeks?|haft[ae])\b"
     r"|\bhaft[aeo]n?\s+se\b", 1.4, "weeks unresolved"),
    (r"\bsince\s+(?:last\s+)?(?:monsoon|ramzan|eid|year)\b", 1.8, "long unresolved"),
    (r"\bmonths?\b", 1.2, "months mentioned"),
    (r"\b\d\s*(?:days|din)\b|\bsince\s+\d\s*days?\b", 1.0, "days unresolved"),
    (r"\bcomplained?\s+(?:before|already|twice|many\s+times)\b", 1.2, "repeat complaint"),
]

_COMPILED_EMERGENCY = [(re.compile(p, re.I), w, lbl) for p, w, lbl in EMERGENCY_PATTERNS]
_COMPILED_EXPOSURE = [(re.compile(p, re.I), w, lbl) for p, w, lbl in EXPOSURE_PATTERNS]
_COMPILED_DEESCALATION = [(re.compile(p, re.I), w, lbl) for p, w, lbl in DEESCALATION_PATTERNS]
_COMPILED_DURATION = [(re.compile(p, re.I), w, lbl) for p, w, lbl in DURATION_PATTERNS]

_ANGRY_RE = re.compile(r"\b(disgust\w+|shame\w*|useless|pathetic|nonsense|fed\s+up|"
                       r"tang\s+aa|bakwas|nakara|negligence|no\s+one\s+cares)\b", re.I)
_CONCERNED_RE = re.compile(r"\b(worried|concerned|afraid|scared|fear|danger|risk|"
                           r"pareshan|dar\s+lagta|please\s+help)\b", re.I)


def _typographic_intensity(text: str) -> tuple[float, list[str]]:
    """ALL-CAPS ratio and exclamation spam. Weak evidence, small weights."""
    score, signals = 0.0, []
    letters = [c for c in text if c.isalpha()]
    if len(letters) >= 20:
        caps_ratio = sum(1 for c in letters if c.isupper()) / len(letters)
        if caps_ratio > 0.6:
            score += 0.8
            signals.append("written in capitals")
    bangs = text.count("!")
    if bangs >= 3:
        score += 0.6
        signals.append(f"{bangs} exclamation marks")
    elif bangs >= 1:
        score += 0.2
    if re.search(r"\b(urgent|urgently|foran|jaldi|immediately|asap)\b", text, re.I):
        score += 0.9
        signals.append("says urgent")
    return score, signals


class RuleBasedAnalyzer(AIAnalyzer):
    """Concrete :class:`AIAnalyzer` using weighted keywords and regexes.

    Inherits the result schema and failure handling; overrides only ``analyze``.
    ``is_available`` is inherited and always True — that guarantee is what lets
    ``pipeline.analyze_text`` promise it never raises.
    """

    name = "keyword-rules-v1"
    source = "rules"

    #: Hard ceiling. Keyword evidence is weak and must never look like a model.
    MAX_CONFIDENCE = 0.62

    def is_available(self) -> bool:
        return True  # zero dependencies, cannot be unavailable

    # -- scoring -------------------------------------------------------------

    def score_categories(self, text: str) -> dict[str, float]:
        """Total matched weight per category. Exposed for the evidence report."""
        scores: dict[str, float] = {}
        for category, patterns in _COMPILED.items():
            total = 0.0
            for rx, weight in patterns:
                hits = len(rx.findall(text))
                if hits:
                    # Diminishing returns: 3 mentions of "kachra" is not 3x the
                    # evidence of one.
                    total += weight * (1.0 + 0.35 * min(hits - 1, 3))
            scores[category] = round(total, 3)
        return scores

    def score_priority(self, text: str) -> tuple[str, float, bool, list[str]]:
        """Return (priority, raw_score, is_emergency, human-readable signals)."""
        signals: list[str] = []
        emergency_score = 0.0
        for rx, weight, label in _COMPILED_EMERGENCY:
            if rx.search(text):
                emergency_score += weight
                signals.append(label)

        exposure_score = 0.0
        for rx, weight, label in _COMPILED_EXPOSURE:
            if rx.search(text):
                exposure_score += weight
                signals.append(label)

        duration_score = 0.0
        for rx, weight, label in _COMPILED_DURATION:
            if rx.search(text):
                duration_score += weight
                signals.append(label)
                break  # duration signals overlap heavily; count the strongest once

        deescalation = 0.0
        for rx, weight, label in _COMPILED_DEESCALATION:
            if rx.search(text):
                deescalation += weight
                signals.append(label)

        intensity, intensity_signals = _typographic_intensity(text)
        signals.extend(intensity_signals)

        total = emergency_score + exposure_score + duration_score + intensity - deescalation

        # An emergency term with real weight forces critical regardless of the
        # rest of the score. Under-triage is the expensive failure.
        is_emergency = emergency_score >= 3.2
        if is_emergency:
            priority = "critical"
        elif total >= 2.5:
            # 2.5 is deliberately reachable by a single strong exposure signal, so
            # "there is a school next to it" alone lands on `high`. That mirrors the
            # rule the LLM prompt states explicitly: a school or hospital is high at
            # minimum.
            priority = "high"
        elif deescalation >= 2.0:
            # Only an EXPLICIT de-escalation ("not urgent", "minor", "routine")
            # produces `low`. An earlier version also dropped to `low` whenever the
            # score was near zero, which under-triaged badly — a complaint with no
            # urgency keywords is an ordinary complaint, not a trivial one.
            priority = "low"
        else:
            priority = "medium"  # documented default
        return priority, round(total, 2), is_emergency, signals

    # -- main entry point ----------------------------------------------------

    def analyze(self, text: str, context: dict[str, Any] | None = None) -> AnalysisResult:
        started = time.perf_counter()
        clean = re.sub(r"\s+", " ", (text or "").strip())
        context = context or {}

        scores = self.score_categories(clean)
        ranked = sorted(scores.items(), key=lambda kv: -kv[1])
        best, best_score = ranked[0]
        second, second_score = (ranked[1] if len(ranked) > 1 else ("other", 0.0))

        matched: list[str] = []
        if best_score > 0:
            for rx, _ in _COMPILED[best]:
                m = rx.search(clean)
                if m:
                    matched.append(m.group(0).lower().strip())
                if len(matched) >= 6:
                    break
        else:
            best = "other"

        # Confidence: how much evidence, and how decisively it beats the runner-up.
        if best_score <= 0:
            confidence = 0.10
        else:
            volume = min(best_score / 8.0, 1.0)
            margin = (best_score - second_score) / best_score if best_score else 0.0
            confidence = min(self.MAX_CONFIDENCE, 0.18 + 0.55 * (0.5 * volume + 0.5 * margin))

        priority, pscore, is_emergency, signals = self.score_priority(clean)

        # A citizen-supplied category is a real signal; use it only to break a
        # genuine tie, never to override strong keyword evidence.
        hint = context.get("category")
        if hint and best_score > 0 and (best_score - second_score) < 1.0:
            from app.ai.base import normalise_category

            hinted = normalise_category(hint)
            if hinted in scores and scores[hinted] > 0:
                best = hinted
                confidence = min(self.MAX_CONFIDENCE, confidence + 0.05)
                matched.append(f"citizen hint: {hinted}")

        first_sentence = re.split(r"(?<=[.!?])\s+", clean)[0] if clean else ""
        if len(first_sentence) > 170:
            first_sentence = first_sentence[:167].rstrip() + "..."
        label = CATEGORY_LABELS.get(best, best)
        summary = (
            f"{label} issue reported"
            + (f" at {str(context['location_text'])[:80]}" if context.get("location_text") else "")
            + (f": {first_sentence}" if first_sentence else ".")
        )

        sentiment = "angry" if _ANGRY_RE.search(clean) else (
            "concerned" if _CONCERNED_RE.search(clean) else "calm"
        )

        reason_bits = [
            f"Keyword rules matched {', '.join(repr(m) for m in matched[:4])}"
            if matched else "No category keyword matched, defaulted to 'other'",
            f"category score {best_score} vs runner-up '{second}' {second_score}",
            f"priority score {pscore}" + (f" from {', '.join(signals[:4])}" if signals else ""),
        ]
        reasoning = ("; ".join(reason_bits)
                     + ". Deterministic fallback: no language model was used, so this"
                       " classification is a keyword match, not comprehension.")

        return AnalysisResult(
            category=best,
            priority=priority,
            summary=summary,
            department_suggestion=DEPARTMENT_BY_CATEGORY.get(best, "General Administration"),
            confidence=round(confidence, 4),
            source="rules",
            model_name=self.name,
            reasoning=reasoning[:800],
            keywords=matched[:8],
            sentiment=sentiment,
            is_emergency=is_emergency,
            latency_ms=max(0, int((time.perf_counter() - started) * 1000)),
        )

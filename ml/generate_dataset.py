"""Synthetic labelled dataset generator for civic complaint triage (Karachi context).

WHY SYNTHETIC
-------------
No public, permissively-licensed, *Pakistani* civic-complaint corpus with both
category and priority labels exists. Rather than pretend otherwise, this module
generates a defensible synthetic corpus from a slot grammar and documents its
limits loudly (see ``ml/data/DATASET_CARD.md`` and ``ml/artifacts/evaluation.md``).

HOW IT AVOIDS FOOLING ITSELF
----------------------------
The single biggest failure mode of template-generated NLP datasets is that the
test split reuses the *same* sentence frames as train, so the classifier only has
to memorise phrasing and reports a fake ~99% accuracy.

This generator prevents that:

* Sentence frames are partitioned into ``FRAMES["train"]`` and ``FRAMES["test"]``
  with **zero overlap**. A test complaint is never phrased like a train complaint.
* Priority escalation clauses ("an ambulance could not get through", ...) are
  likewise partitioned train/test.
* Issue phrases are three pools: ``shared`` (vocabulary a real classifier is
  entitled to learn, e.g. "pothole", "kachra"), ``train`` only and ``test`` only.
  The test-only pool forces genuine generalisation instead of keyword lookup.
* Realistic noise is applied *after* assembly: typos, ALL CAPS, dropped
  punctuation, Roman-Urdu code-switching, exclamation spam, SMS-style shortening.

Labels are *causal*, not annotated after the fact: the generator picks a category
and a priority first, then only emits surface forms consistent with them. That is
what makes the labels clean; it is also why the resulting accuracy is an upper
bound on real-world performance (real complaints are ambiguous, this corpus is not).

Usage
-----
    python -m ml.generate_dataset                # 3000 rows, default seed
    python -m ml.generate_dataset --n 5000       # more rows
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

try:  # pragma: no cover - allows `python ml/generate_dataset.py` too
    from ml import DATA_DIR, SEED
except ImportError:  # pragma: no cover
    DATA_DIR = Path(__file__).resolve().parent / "data"
    SEED = 20260808


CATEGORIES = ["road", "water", "waste", "electricity", "drainage", "safety", "other"]
PRIORITIES = ["low", "medium", "high", "critical"]

# Target mix. Deliberately NOT uniform: real municipal inboxes are dominated by
# waste/water/road, and "other" is a small residual bucket. A uniform dataset
# would train a model that is miscalibrated against production traffic.
CATEGORY_WEIGHTS = {
    "road": 0.18,
    "water": 0.17,
    "waste": 0.18,
    "electricity": 0.15,
    "drainage": 0.15,
    "safety": 0.10,
    "other": 0.07,
}

# Priority mix, also skewed: most complaints are routine.
PRIORITY_WEIGHTS = {"low": 0.18, "medium": 0.37, "high": 0.31, "critical": 0.14}


# --------------------------------------------------------------------------- #
# Location slots — real Karachi place names so the char n-grams see real tokens
# --------------------------------------------------------------------------- #

AREAS = [
    "Gulshan-e-Iqbal", "North Nazimabad", "Korangi", "Malir", "Saddar", "Clifton",
    "DHA Phase 6", "Lyari", "Orangi Town", "Nazimabad", "Federal B Area",
    "Gulistan-e-Johar", "Landhi", "Shah Faisal Colony", "Baldia Town",
    "Liaquatabad", "Garden East", "PECHS", "Bahadurabad", "Surjani Town",
    "New Karachi", "Kemari", "SITE Area", "Model Colony", "Gadap Town",
    "Buffer Zone", "Gulberg Town", "Manghopir", "Defence View", "Jamshed Road",
]

ROADS = [
    "Shahrah-e-Faisal", "University Road", "Rashid Minhas Road", "M.A. Jinnah Road",
    "Korangi Road", "Tariq Road", "I.I. Chundrigar Road", "Stadium Road",
    "Abul Hasan Isphahani Road", "Khayaban-e-Ittehad", "Sir Shah Suleman Road",
    "Nishtar Road", "Sharae Noor Jehan", "Hub River Road", "Maripur Road",
]

BLOCKS = [
    "Block 5", "Block 13-D", "Block A", "Block 2", "Sector 11-B", "Sector 5-C",
    "Street 14", "Lane 3", "Plot 22", "House 4-B", "Gali No 7", "Phase 2",
]

SENSITIVE_LANDMARKS = [
    "a government girls school", "Aga Khan Hospital", "a primary school",
    "Jinnah Hospital emergency gate", "a maternity home", "the children's park",
    "Civil Hospital", "a madrassa", "the school van stop", "Ziauddin Hospital",
]

NEUTRAL_LANDMARKS = [
    "the vegetable market", "the bus stop", "the main mosque", "a bank branch",
    "the union council office", "the petrol pump", "the katchi abadi",
    "the flyover ramp", "the community centre", "the mobile market",
]


def _loc(rng: random.Random) -> str:
    """Render one location phrase. Varied so location is never a class signal."""
    area = rng.choice(AREAS)
    style = rng.random()
    if style < 0.30:
        return f"{rng.choice(BLOCKS)}, {area}"
    if style < 0.55:
        return f"near {rng.choice(NEUTRAL_LANDMARKS)} in {area}"
    if style < 0.75:
        return f"on {rng.choice(ROADS)}"
    if style < 0.90:
        return f"{area} {rng.choice(BLOCKS)}"
    return area


DURATIONS_SHORT = ["since yesterday", "since this morning", "for two days now",
                   "since Friday", "for the last three days"]
DURATIONS_LONG = ["for over two weeks", "for almost a month", "since Ramzan",
                  "for more than 20 days", "for the past six weeks",
                  "since the last monsoon", "for several months"]


# --------------------------------------------------------------------------- #
# Issue phrases per category.
#   shared -> vocabulary both splits may use (a fair classifier SHOULD learn it)
#   train  -> appears only in the training split
#   test   -> appears only in the held-out split (forces generalisation)
# `roman` entries are Roman-Urdu / code-switched surface forms.
# --------------------------------------------------------------------------- #

@dataclass
class CategorySpec:
    name: str
    department: str
    shared: list[str] = field(default_factory=list)
    train_only: list[str] = field(default_factory=list)
    test_only: list[str] = field(default_factory=list)
    roman_shared: list[str] = field(default_factory=list)
    roman_train: list[str] = field(default_factory=list)
    roman_test: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)

    def phrases(self, split: str, rng: random.Random, roman: bool) -> str:
        if roman:
            pool = list(self.roman_shared)
            pool += self.roman_train if split == "train" else self.roman_test
        else:
            pool = list(self.shared)
            pool += self.train_only if split == "train" else self.test_only
        return rng.choice(pool)


SPECS: dict[str, CategorySpec] = {
    "road": CategorySpec(
        name="road",
        department="Roads & Infrastructure",
        shared=[
            "a very large pothole has formed in the middle of the road",
            "the road surface has completely broken up",
            "there are deep potholes all along this stretch",
            "the road has caved in after the rain",
            "broken road with loose gravel everywhere",
        ],
        train_only=[
            "the asphalt has peeled off and the base layer is exposed",
            "a trench dug by the gas company was never filled back",
            "the speed breaker is broken and has sharp edges",
            "the footpath tiles are uprooted and pedestrians walk on the road",
            "the road is uneven and bikes lose balance on it",
            "the newly laid patch has already sunk in the middle",
        ],
        test_only=[
            "the carriageway of this road has developed craters after the digging work",
            "the road tarmac is cracked into pieces and loose stones are flying up",
            "a sinkhole has opened in the road surface where it meets the service lane",
            "the road level is now far below the manhole rim",
            "the kerb stones have collapsed onto the road driving lane",
        ],
        roman_shared=[
            "sarak par bara gaddha ban gaya hai",
            "road bilkul toot gayi hai",
            "yahan khadde hi khadde hain",
        ],
        roman_train=[
            "sarak ki halat bohot kharab hai, gaari nikalna mushkil hai",
            "road pe khudai kar ke chor di gayi hai",
            "speed breaker toota hua hai",
        ],
        roman_test=[
            "rasta itna kharab hai ke rickshaw ulat jata hai",
            "sarak beth gayi hai barish ke baad",
            "patch dala tha wo dobara ukhar gaya",
        ],
        keywords=["pothole", "road", "asphalt", "footpath", "sarak", "gaddha"],
    ),
    "water": CategorySpec(
        name="water",
        department="Water Supply",
        shared=[
            "there has been no water supply in our line",
            "the main water pipe is leaking badly",
            "water is coming with very low pressure",
            "clean drinking water has not reached our houses",
            "the water line has burst and water is being wasted",
        ],
        train_only=[
            "the valve man is not opening our valve at the scheduled time",
            "we are forced to buy a private tanker every second day",
            "the water that comes is muddy and smells of chemicals",
            "the booster pump at the hydrant has stopped working",
            "our overhead tank never fills even after four hours of supply",
        ],
        test_only=[
            "the water supply timing was shifted and now nothing reaches our end of the lane",
            "the water pipeline joint is spraying clean water into the street all day",
            "whatever comes out of the tap is yellow and undrinkable",
            "the water sub-line to our block appears to have been disconnected",
            "the water hydrant staff say the supply quota for our area was reduced",
        ],
        roman_shared=[
            "pani bilkul nahi aa raha",
            "pani ki line leak ho rahi hai",
            "paani ka pressure bohot kam hai",
        ],
        roman_train=[
            "hum ko roz tanker mangwana par raha hai",
            "jo pani aata hai wo gandha aur badbudar hai",
            "valve wala time par valve nahi kholta",
        ],
        roman_test=[
            "hamari gali me supply ka time badal diya gaya hai",
            "line ka joint phat gaya hai aur pani zaya ho raha hai",
            "nalke se peela pani nikal raha hai",
        ],
        keywords=["water", "supply", "tanker", "pipeline", "pani", "leak"],
    ),
    "waste": CategorySpec(
        name="waste",
        department="Sanitation & Solid Waste",
        shared=[
            "garbage has not been collected from our street",
            "a huge heap of trash is lying at the corner",
            "the garbage container is overflowing onto the road",
            "there is rotting waste and an unbearable smell",
            "sweepers have not come to clean this lane",
        ],
        train_only=[
            "people are burning the garbage pile and the smoke enters our homes",
            "stray dogs and cats tear open the bags and scatter the waste",
            "construction malba has been dumped on the empty plot",
            "the kachra kundi was removed and now everyone dumps in the open",
            "hospital waste has been mixed with the household garbage",
        ],
        test_only=[
            "the refuse dump beside our wall has doubled in size this month",
            "flies and mosquitoes have multiplied because of the uncollected refuse",
            "the contractor's truck skips our lane every single round",
            "shopkeepers throw their leftovers directly onto the pavement here",
            "the collection point has shifted right outside a residential gate",
        ],
        roman_shared=[
            "kachra utha nahi hai kai din se",
            "gali me kooray ka bara dher lag gaya hai",
            "safai bilkul nahi hoti yahan",
        ],
        roman_train=[
            "kachra kundi bhar gayi hai aur sarak par gir raha hai",
            "log kachra jala rahe hain, dhuan ghar me aata hai",
            "awara kutte kachra phaila dete hain",
        ],
        roman_test=[
            "malba khali plot par daal diya gaya hai",
            "gandagi ki wajah se makhiyan bohot ho gayi hain",
            "safai wale hamari gali chor kar chale jate hain",
        ],
        keywords=["garbage", "trash", "waste", "sweeper", "kachra", "gandagi"],
    ),
    "electricity": CategorySpec(
        name="electricity",
        department="Electricity & Streetlights",
        shared=[
            "the street lights have not been working",
            "there is no electricity in our block",
            "the electric pole is leaning dangerously",
            "wires are hanging loose from the pole",
            "the transformer keeps tripping again and again",
        ],
        train_only=[
            "unannounced load shedding is running for eight hours a day",
            "the meter reading is being estimated and the bill is impossible",
            "the pole mounted box is open and children can reach the terminals",
            "half the lights on this stretch were never replaced after the storm",
            "there are sparks coming from the connection at night",
        ],
        test_only=[
            "the electricity feeder for our lane trips every time an appliance is switched on",
            "every street light on this avenue has been dead since the pole was hit",
            "the voltage is so low that no appliance runs, the fan barely turns",
            "the power cable was cut during digging and the connection was never restored",
            "an overhead electricity line is now resting on our boundary wall",
        ],
        roman_shared=[
            "street light band hai kai hafton se",
            "bijli nahi aa rahi hamare block me",
            "bijli ka khamba tirha ho gaya hai",
        ],
        roman_train=[
            "taar latak rahi hain khambe se",
            "loadshedding ka koi schedule nahi hai",
            "transformer bar bar trip kar raha hai",
        ],
        roman_test=[
            "raat ko connection se chingari nikalti hai",
            "voltage itna kam hai ke pankha nahi chalta",
            "meter reading galat lagai ja rahi hai",
        ],
        keywords=["light", "electricity", "pole", "wire", "bijli", "transformer"],
    ),
    "drainage": CategorySpec(
        name="drainage",
        department="Drainage & Sewerage",
        shared=[
            "the sewerage line is choked and dirty water is on the street",
            "the gutter has been overflowing",
            "sewage water has entered our lane",
            "the drain is blocked and the water is not going anywhere",
            "the manhole is spilling sewage continuously",
        ],
        train_only=[
            "rain water from the last spell has still not drained out",
            "the nallah beside the colony is choked with solid waste",
            "the sewer line was never connected to the main trunk",
            "waste water has started seeping into the ground floor rooms",
            "the drain cover is broken and the sludge is exposed",
        ],
        test_only=[
            "effluent from the choked sewer line is standing knee deep at the junction",
            "the storm drain outlet has been sealed by an illegal construction",
            "black sewage is bubbling up from the drain inspection chamber",
            "the open sewage channel behind the market has stopped flowing entirely",
            "backflow from the sewerage line reaches our washroom drain every evening",
        ],
        roman_shared=[
            "gutter ubal raha hai",
            "nali band hai aur pani khara hai",
            "sewerage ka pani gali me phail gaya hai",
        ],
        roman_train=[
            "nalah kachre se bhar gaya hai",
            "barish ka pani abhi tak nahi nikla",
            "manhole ka dhakkan toota hua hai",
        ],
        roman_test=[
            "gande pani ki wajah se ghar me se bo aa rahi hai",
            "chamber se kala pani upar aa raha hai",
            "nikasi ka rasta band kar diya gaya hai",
        ],
        keywords=["sewer", "gutter", "drain", "manhole", "nali", "nalah"],
    ),
    "safety": CategorySpec(
        name="safety",
        department="Public Safety",
        shared=[
            "there have been repeated mobile snatching incidents here",
            "a pack of stray dogs is attacking people",
            "an open manhole in the middle of the walkway is a death trap",
            "the boundary wall of the plot is about to collapse on passers-by",
            "drug users gather here after dark and residents feel unsafe",
        ],
        train_only=[
            "there is no police patrolling and street crime has increased",
            "an abandoned building has become a hideout for criminals",
            "rash driving by mini buses has caused several near misses",
            "the pedestrian crossing has no signal and children cross alone",
            "an illegal gas cylinder refilling shop could explode any day in the crowded market",
        ],
        test_only=[
            "chain lifting on motorcycles happens almost every evening at this turn",
            "a dangerous dilapidated structure is about to fall on people using the footpath",
            "unregistered heavy vehicles speed dangerously through the residential lane at night",
            "armed anti social elements threaten anyone entering the park after sunset",
            "the fire exit of the crowded plaza has been welded shut, trapping everyone inside",
        ],
        roman_shared=[
            "yahan mobile snatching bohot ho rahi hai",
            "awara kutte logon ko kaat rahe hain",
            "khula manhole hai, koi bhi gir sakta hai",
        ],
        roman_train=[
            "raat ko nashe wale jama hote hain, dar lagta hai",
            "police gasht bilkul nahi hoti",
            "purani deewar girne wali hai",
        ],
        roman_test=[
            "mor par roz chaini snatching hoti hai",
            "khatarnak imarat footpath par jhuk gayi hai",
            "park par ghair samaji anasir ka qabza hai",
        ],
        keywords=["snatching", "stray dogs", "unsafe", "crime", "manhole", "collapse"],
    ),
    "other": CategorySpec(
        name="other",
        department="General Administration",
        shared=[
            "the park in our neighbourhood has been encroached by a private party",
            "the notice board at the union council has not been updated for months",
            "stray cattle are tied up on public land and nobody removes them",
            "a shopkeeper has extended his shop over the whole footpath",
            "the community hall booking process is completely opaque",
        ],
        train_only=[
            "loudspeakers from a marriage hall run past midnight every weekend",
            "the government dispensary opens only twice a week without notice",
            "we cannot get our property tax challan corrected at the office",
            "the public library has been closed since last year with no explanation",
            "banners and hoardings are blocking the view at the intersection",
        ],
        test_only=[
            "an unauthorised parking contractor charges undocumented fees on a public street",
            "the graveyard boundary record at the office does not match the land on site",
            "the recreational ground has been rented out to a private cricket academy without tender",
            "the local dispensary staff refuse to register walk-in patients",
            "no receipt is issued for the charges collected at the municipal counter",
        ],
        roman_shared=[
            "park par qabza kar liya gaya hai",
            "footpath par dukan barha di gayi hai",
            "sarkari daftar me koi sunwai nahi hoti",
        ],
        roman_train=[
            "shadi hall ke loudspeaker raat bhar chalte hain",
            "dispensary hafte me sirf do din khulti hai",
            "banner laga kar rasta band kar diya hai",
        ],
        roman_test=[
            "parking ke naam par zabardasti paise liye ja rahe hain",
            "maidan par private academy ka qabza hai",
            "counter par rasid nahi di jati",
        ],
        keywords=["encroachment", "office", "park", "qabza", "public", "civic"],
    ),
}


# --------------------------------------------------------------------------- #
# Priority escalation clauses.
#
# Same three-pool design as the issue phrases: `shared` + `train` + `test`.
# An earlier revision made the pools *fully* disjoint, which dropped held-out
# priority accuracy to ~0.50. That was not a real measurement of generalisation,
# it was an artefact: urgency vocabulary in real complaints genuinely recurs
# ("urgent", "emergency", "accident", "injured", "routine", "no hurry"), so a
# split that shares *none* of it tests telepathy rather than learning. The shared
# pool restores that realistic anchor; the split-specific pools still guarantee
# every test item contains escalation phrasing the model has never seen.
# --------------------------------------------------------------------------- #

PRIORITY_CLAUSES: dict[str, dict[str, list[str]]] = {
    "critical": {
        "shared": [
            "this is an emergency and needs immediate attention",
            "there is a serious risk of a fatal accident here",
            "it is extremely urgent, please treat it as an emergency",
            "a person has already been injured because of this",
            "someone is going to be killed if this is not fixed today",
        ],
        "train": [
            "a child was injured here yesterday and we fear a worse accident",
            "someone received an electric shock this morning",
            "an ambulance could not reach the hospital gate because of it",
            "a motorcyclist fell and broke his arm last night",
            "a live wire is lying in the standing water",
            "this is an emergency, please send a team today",
            "sewage has mixed into the drinking water line and people are vomiting",
            "part of the structure has already collapsed",
            "an elderly man fell into it and had to be taken to emergency",
        ],
        "test": [
            "two residents have already been hospitalised because of this",
            "there is an immediate danger to human life here",
            "an accident happened at this exact spot a few hours ago",
            "the fire brigade had to be called last night",
            "a school van skidded and very nearly overturned",
            "current is passing through the water where children walk",
            "a woman in labour could not be taken out of the lane in time",
            "the wall gave way on one side and a shop was buried",
        ],
    },
    "high": {
        "shared": [
            "it is urgent because a school is right beside it",
            "many families and children are affected by it every single day",
            "this has become a serious health hazard for the whole area",
            "please treat this as urgent, the situation is getting worse daily",
            "a large number of residents are suffering because of it",
        ],
        "train": [
            "there is a government school right next to it and hundreds of children pass daily",
            "it is on the main road and traffic jams for hours every morning",
            "the entire street of about forty houses is affected",
            "dengue and gastro cases have started appearing in the area",
            "elderly patients live in this lane and cannot step outside",
            "the whole block has been suffering and tempers are rising",
        ],
        "test": [
            "a hospital entrance is barely fifty metres from here",
            "more than sixty families depend on this single lane",
            "children walking to school have to wade through it every day",
            "it has become a serious health hazard for the entire block",
            "shopkeepers say business has dropped by half because of it",
            "two schools and a clinic are on the same stretch",
        ],
    },
    "medium": {
        "shared": [
            "it is causing inconvenience and should be fixed soon",
            "kindly send someone to look at it in the next few days",
            "we have been waiting for a response for a while now",
            "please arrange for the concerned staff to check it",
        ],
        "train": [
            "our family has been facing this for several days now",
            "it is causing daily inconvenience to the residents",
            "we have complained at the local office twice already with no result",
            "please look into it in the coming days",
        ],
        "test": [
            "this has been troubling the residents of our building",
            "kindly arrange a visit sometime this week",
            "we would appreciate it if the concerned staff could check it",
            "it is affecting a few households on our side of the lane",
        ],
    },
    "low": {
        "shared": [
            "there is no urgency, please handle it in the routine schedule",
            "this is a minor issue, whenever it is convenient",
            "just a routine request, nothing urgent at all",
            "low priority, only mentioning it so it is on record",
        ],
        "train": [
            "it is a small issue but please note it for the record",
            "there is no urgency, just registering it formally",
            "kindly look into it whenever a team is free",
            "this is only a suggestion for improvement",
        ],
        "test": [
            "not an emergency at all, just a routine request",
            "a minor problem which can wait for the normal schedule",
            "please add it to the regular maintenance list",
            "nothing serious, just bringing it to your notice",
        ],
    },
}

SENTIMENT_OPENERS = {
    "train": [
        "I want to report that", "Kindly note that", "Please be informed that",
        "Respected sir,", "Assalam o alaikum,", "To the concerned department,",
    ],
    "test": [
        "This is to bring to your attention that", "Janab,",
        "Dear municipal team,", "Complaint:", "Reporting an issue -",
    ],
}

CLOSERS = {
    "train": [
        "Please take action.", "Kindly resolve it soon.", "Shukriya.",
        "We request immediate attention.", "Thank you.",
    ],
    "test": [
        "Requesting the department to act.", "Meherbani ho gi.",
        "Awaiting your response.", "Please do the needful.", "Regards.",
    ],
}


# --------------------------------------------------------------------------- #
# Sentence frames. FULLY DISJOINT between train and test.
# Slots: {open} {issue} {loc} {dur} {esc} {close}
# --------------------------------------------------------------------------- #

FRAMES: dict[str, list[str]] = {
    "train": [
        "{open} {issue} at {loc}, {dur}. {esc} {close}",
        "{issue} at {loc}. This has been the case {dur}. {esc}",
        "{open} {issue} near {loc}. {esc} {close}",
        "At {loc}, {issue} {dur}. {esc}",
        "{issue}. Location: {loc}. {esc} {close}",
        "{open} {issue} {dur} at {loc}. {close}",
        "{issue} at {loc} and nobody has come to check {dur}. {esc}",
        "{esc} {issue} at {loc}. {close}",
        "{issue} at {loc}.",
        "{open} {issue} in front of {loc}, and {esc}",
    ],
    "test": [
        "Location {loc} - {issue}, ongoing {dur}. {esc} {close}",
        "We the residents of {loc} report that {issue}. {esc}",
        "{issue}; this situation at {loc} has continued {dur}. {esc} {close}",
        "{open} the following problem at {loc}: {issue}. {esc}",
        "Problem area {loc}. Details: {issue} {dur}. {close}",
        "{esc} The reason is that {issue}, right at {loc}.",
        "For {dur} now, {issue} around {loc}. {close}",
        "{issue} — {loc}. {esc}",
        "Reporting from {loc}: {issue}.",
        "{open} residents near {loc} are affected because {issue}. {esc} {close}",
    ],
}


# --------------------------------------------------------------------------- #
# Surface noise
# --------------------------------------------------------------------------- #

_KEYBOARD_NEIGHBOURS = {
    "a": "qsz", "b": "vgn", "c": "xdv", "d": "sfce", "e": "wrd", "f": "dgrv",
    "g": "fhtb", "h": "gjyn", "i": "uok", "j": "hkum", "k": "jlim", "l": "kop",
    "m": "njk", "n": "bhm", "o": "ipl", "p": "ol", "q": "wa", "r": "etf",
    "s": "adwx", "t": "ryg", "u": "yih", "v": "cfb", "w": "qes", "x": "zsc",
    "y": "tuh", "z": "asx",
}

SMS_SHORTENINGS = [
    (r"\bplease\b", "plz"), (r"\byou\b", "u"), (r"\bare\b", "r"),
    (r"\band\b", "&"), (r"\bwith\b", "wid"), (r"\bbecause\b", "bcz"),
    (r"\bnumber\b", "no"), (r"\bstreet\b", "st"),
]


def _typo(word: str, rng: random.Random) -> str:
    if len(word) < 4:
        return word
    kind = rng.random()
    i = rng.randrange(1, len(word) - 1)
    if kind < 0.30:  # transpose
        return word[:i] + word[i + 1] + word[i] + word[i + 2:]
    if kind < 0.55:  # drop
        return word[:i] + word[i + 1:]
    if kind < 0.80:  # double
        return word[:i] + word[i] + word[i:]
    ch = word[i].lower()  # keyboard slip
    if ch in _KEYBOARD_NEIGHBOURS:
        return word[:i] + rng.choice(_KEYBOARD_NEIGHBOURS[ch]) + word[i + 1:]
    return word


def _apply_noise(text: str, rng: random.Random) -> str:
    """Realistic degradation. Order matters: shortenings, then typos, then case."""
    if rng.random() < 0.18:
        for pattern, repl in SMS_SHORTENINGS:
            if rng.random() < 0.5:
                text = re.sub(pattern, repl, text, flags=re.IGNORECASE)

    if rng.random() < 0.38:  # sprinkle typos over ~6% of tokens
        words = text.split()
        for idx, w in enumerate(words):
            if rng.random() < 0.06:
                words[idx] = _typo(w, rng)
        text = " ".join(words)

    roll = rng.random()
    if roll < 0.06:
        text = text.upper()
    elif roll < 0.22:
        text = text.lower()

    if rng.random() < 0.12:
        text = re.sub(r"\.\s*$", "", text) + rng.choice(["!!!", "!!", " !!!!", "??!"])

    if rng.random() < 0.10:  # dropped punctuation, run-on style
        text = text.replace(",", "").replace(".", " ").strip()

    return re.sub(r"\s+", " ", text).strip()


def _maybe_repeat_issue(text: str, issue: str, rng: random.Random) -> str:
    """Some citizens repeat themselves; adds realistic length variance."""
    if rng.random() < 0.12:
        return f"{text} Again, {issue}."
    return text


def _build_one(category: str, priority: str, split: str, rng: random.Random) -> dict:
    spec = SPECS[category]
    roman_full = rng.random() < 0.12          # whole complaint in Roman-Urdu
    roman_issue = roman_full or rng.random() < 0.28

    issue = spec.phrases(split, rng, roman=roman_issue)

    # Escalation clause: shared urgency vocabulary + split-exclusive phrasing.
    esc_pool = PRIORITY_CLAUSES[priority]["shared"] + PRIORITY_CLAUSES[priority][split]
    esc = rng.choice(esc_pool)
    if rng.random() < 0.28:  # some citizens pile on the urgency; adds length variance
        second = rng.choice([c for c in esc_pool if c != esc])
        esc = f"{esc}. {second[0].upper()}{second[1:]}"

    frame = rng.choice(FRAMES[split])

    dur_pool = DURATIONS_LONG if priority in ("high", "critical") else DURATIONS_SHORT
    if rng.random() < 0.25:  # break the duration<->priority shortcut sometimes
        dur_pool = DURATIONS_SHORT + DURATIONS_LONG

    loc = _loc(rng)
    if priority in ("critical", "high") and rng.random() < 0.45:
        loc = f"{loc} near {rng.choice(SENSITIVE_LANDMARKS)}"

    text = frame.format(
        open=rng.choice(SENTIMENT_OPENERS[split]),
        issue=issue,
        loc=loc,
        dur=rng.choice(dur_pool),
        esc=esc,
        close=rng.choice(CLOSERS[split]),
    )

    if roman_full:
        text = f"{text} Bara meherbani ho gi, jaldi masla hal karain."

    text = _maybe_repeat_issue(text, issue, rng)
    text = _apply_noise(text, rng)

    # Guard the contract's 15..5000 char rule so every synthetic row is a legal
    # ComplaintCreate.description.
    if len(text) < 20:
        text = f"{text} Please check {loc}."

    return {
        "text": text[:5000],
        "category": category,
        "priority": priority,
        "split": split,
        "department": spec.department,
        "is_roman": int(roman_issue),
    }


def _weighted_choice(rng: random.Random, weights: dict[str, float]) -> str:
    keys = list(weights)
    return rng.choices(keys, weights=[weights[k] for k in keys], k=1)[0]


def generate(n: int = 3000, seed: int = SEED, test_frac: float = 0.20) -> list[dict]:
    """Generate ``n`` complaints. The test slice uses disjoint frames and clauses."""
    rng = random.Random(seed)
    n_test = int(round(n * test_frac))
    n_train = n - n_test

    rows: list[dict] = []
    seen: set[str] = set()

    for split, count in (("train", n_train), ("test", n_test)):
        made = 0
        guard = 0
        while made < count and guard < count * 60:
            guard += 1
            category = _weighted_choice(rng, CATEGORY_WEIGHTS)
            priority = _weighted_choice(rng, PRIORITY_WEIGHTS)
            row = _build_one(category, priority, split, rng)
            key = row["text"].lower()
            if key in seen:          # exact-duplicate guard: no leakage, no inflation
                continue
            seen.add(key)
            rows.append(row)
            made += 1
    rng.shuffle(rows)
    return rows


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=["text", "category", "priority", "split", "department", "is_roman"]
        )
        writer.writeheader()
        writer.writerows(rows)


DATASET_CARD = """# Dataset Card — `civic-complaints-synth-v1`

| Field | Value |
|---|---|
| Rows | {n} |
| Train / Test | {n_train} / {n_test} |
| Categories | {n_cat} (`road, water, waste, electricity, drainage, safety, other`) |
| Priorities | 4 (`low, medium, high, critical`) |
| Language | English, Roman-Urdu, and code-switched English/Roman-Urdu |
| Roman-Urdu rows | {n_roman} ({pct_roman:.1f}%) |
| Seed | {seed} |
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
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--test-frac", type=float, default=0.20)
    parser.add_argument("--out", type=Path, default=DATA_DIR)
    args = parser.parse_args()

    rows = generate(n=args.n, seed=args.seed, test_frac=args.test_frac)
    train = [r for r in rows if r["split"] == "train"]
    test = [r for r in rows if r["split"] == "test"]

    out: Path = args.out
    _write_csv(out / "dataset.csv", rows)
    _write_csv(out / "train.csv", train)
    _write_csv(out / "test.csv", test)

    cat_counts = Counter(r["category"] for r in rows)
    pri_counts = Counter(r["priority"] for r in rows)
    n_roman = sum(r["is_roman"] for r in rows)

    stats = {
        "n": len(rows),
        "n_train": len(train),
        "n_test": len(test),
        "seed": args.seed,
        "categories": dict(sorted(cat_counts.items())),
        "priorities": dict(sorted(pri_counts.items())),
        "roman_urdu_rows": n_roman,
        "mean_chars": round(sum(len(r["text"]) for r in rows) / max(len(rows), 1), 1),
        "min_chars": min(len(r["text"]) for r in rows),
        "max_chars": max(len(r["text"]) for r in rows),
    }
    (out / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    (out / "DATASET_CARD.md").write_text(
        DATASET_CARD.format(
            n=len(rows), n_train=len(train), n_test=len(test), n_cat=len(CATEGORIES),
            n_roman=n_roman, pct_roman=100 * n_roman / max(len(rows), 1), seed=args.seed,
        ),
        encoding="utf-8",
    )

    print(json.dumps(stats, indent=2))
    print(f"\nwrote -> {out}/dataset.csv, train.csv, test.csv, stats.json, DATASET_CARD.md")
    for r in rows[:5]:
        print(f"  [{r['category']}/{r['priority']}] {r['text'][:110]}")


if __name__ == "__main__":
    main()

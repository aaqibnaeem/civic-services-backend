"""Demo + training data generator.

Run it with::

    uv run python -m scripts.seed            # seed if empty
    uv run python -m scripts.seed --reset    # wipe first, then seed
    uv run python -m scripts.seed --count 1200

What it produces (with a FIXED random seed, so every run is byte-identical):

* 6 municipal departments and the demo admin account;
* ~800 realistic Karachi complaints spread over the last 180 days;
* one plausible ``AIAnalysis`` row per complaint (``ml`` or ``rules`` tier), so the
  dashboard is fully populated with **no network calls**;
* a status timeline per complaint.

The data is shaped to tell a story, because a flat random dataset makes for a boring
demo and a meaningless statistics benchmark:

1. **Monsoon spike** — total volume and the drainage/water share both rise sharply in
   July–August.
2. **A measurably slow department** — Sewerage & Drainage resolves roughly twice as
   slowly as everyone else, which the department analytics should surface.
3. **Hotspots** — Orangi Town, Lyari and Korangi carry disproportionate volume.
4. **Right-skewed resolution times** — drawn from a log-normal, so the median sits
   well below the mean and Tukey fences find genuine outliers rather than noise.
5. **Sensible priority** — correlated with category, and forced up when the text
   contains emergency language.
"""

from __future__ import annotations

import argparse
import asyncio
import math
import random
import re
import uuid
from collections import Counter
from datetime import UTC, datetime, timedelta

from sqlalchemy import delete

from app.core.config import settings
from app.core.logging_config import configure_logging, get_logger
from app.core.security import REFERENCE_ALPHABET, REFERENCE_PREFIX
from app.db import create_all
from app.db.session import SessionLocal
from app.models.ai_analysis import AIAnalysis, AISource, Sentiment
from app.models.complaint import AIStatus, Category, Complaint, Priority, Status
from app.models.department import Department
from app.models.status_event import StatusEvent
from app.models.user import Role, User
from app.repositories.user_repo import UserRepository
from app.services.auth_service import AuthService
from app.services.complaint_service import ComplaintManager

log = get_logger("seed")

RANDOM_SEED = 20260808
DEFAULT_COUNT = 800
HISTORY_DAYS = 180

# =============================================================================
# Reference data
# =============================================================================

DEPARTMENTS: tuple[dict, ...] = (
    {
        "name": "Roads & Infrastructure",
        "slug": "roads",
        "categories": [Category.ROAD.value],
        "contact_email": "roads@civic.gov.pk",
        "speed": 1.0,
    },
    {
        "name": "Karachi Water & Sewerage — Supply",
        "slug": "water",
        "categories": [Category.WATER.value],
        "contact_email": "water@civic.gov.pk",
        "speed": 1.15,
    },
    {
        "name": "Solid Waste Management",
        "slug": "waste",
        "categories": [Category.WASTE.value],
        "contact_email": "waste@civic.gov.pk",
        "speed": 0.75,
    },
    {
        "name": "Electricity & Street Lighting",
        "slug": "electricity",
        "categories": [Category.ELECTRICITY.value],
        "contact_email": "power@civic.gov.pk",
        "speed": 0.85,
    },
    {
        # The deliberate laggard: ~2x slower than everyone else.
        "name": "Sewerage & Drainage",
        "slug": "sewerage",
        "categories": [Category.DRAINAGE.value],
        "contact_email": "sewerage@civic.gov.pk",
        "speed": 2.85,
    },
    {
        "name": "Public Safety & Enforcement",
        "slug": "safety",
        "categories": [Category.SAFETY.value, Category.OTHER.value],
        "contact_email": "safety@civic.gov.pk",
        "speed": 0.95,
    },
)

# (name, latitude, longitude, volume weight). Higher weight == hotspot.
AREAS: tuple[tuple[str, float, float, float], ...] = (
    ("Gulshan-e-Iqbal", 24.9204, 67.0971, 1.30),
    ("North Nazimabad", 24.9425, 67.0378, 1.00),
    ("Nazimabad", 24.9110, 67.0330, 0.80),
    ("Saddar", 24.8607, 67.0221, 0.95),
    ("Clifton", 24.8138, 67.0300, 0.60),
    ("Korangi", 24.8300, 67.1300, 1.75),
    ("Malir", 24.8930, 67.1900, 0.85),
    ("Gulistan-e-Johar", 24.9200, 67.1300, 1.15),
    ("Lyari", 24.8700, 66.9900, 1.95),
    ("DHA", 24.8000, 67.0600, 0.55),
    ("Orangi Town", 24.9500, 66.9900, 2.10),
    ("Landhi", 24.8500, 67.1900, 1.05),
)

STREETS = (
    "Main University Road",
    "Block 5 Street 12",
    "Shahrah-e-Faisal service road",
    "Sector 11-B lane 3",
    "Rashid Minhas Road",
    "Sir Shah Suleman Road",
    "Street 7 near the water pump",
    "Commercial Area lane 2",
    "Bank Colony main street",
    "Sector 4 back lane",
    "Nishtar Road",
    "Old Sabzi Mandi road",
    "Gali number 9",
    "Askari Park side street",
    "Model Colony main road",
)

LANDMARKS = (
    "near the government girls school",
    "opposite Imtiaz Super Market",
    "beside the Edhi centre",
    "next to the Sunni mosque",
    "in front of the union council office",
    "near the bus stop",
    "close to the children's park",
    "outside the private clinic",
    "near the petrol pump",
    "beside the community hall",
    "opposite the police chowki",
    "near the milk shop at the corner",
)

DURATIONS = (
    "for the past three days",
    "since last week",
    "for almost a month now",
    "since the last rain",
    "for more than ten days",
    "since Monday morning",
    "for two weeks",
    "since yesterday evening",
    "for the last four days",
    "since the start of the month",
)

# Text that must force the priority up regardless of category.
EMERGENCY_MARKERS = (
    "spark",
    "live wire",
    "electrocut",
    "collapsed",
    "open manhole",
    "child",
    "fire",
    "gas leak",
    "accident",
    "fell into",
    "dengue",
)

# Each entry is (template, base_priority). Slots: {street} {landmark} {duration} {area}
TEMPLATES: dict[Category, tuple[tuple[str, Priority], ...]] = {
    Category.ROAD: (
        ("There is a very large pothole on {street} {landmark}. It has been there {duration} and two motorcycles have already fallen in.", Priority.HIGH),
        ("The road at {street} in {area} is completely broken. Rickshaws refuse to come inside our lane {duration}.", Priority.MEDIUM),
        ("A manhole cover on {street} is missing and the hole is open. A child almost fell into it yesterday. Please cover it urgently.", Priority.CRITICAL),
        ("Speed breakers on {street} {landmark} have been dug out and never rebuilt, so cars are speeding past the school gate.", Priority.MEDIUM),
        ("After the road cutting for a gas line on {street}, the contractor never repaved it. Loose gravel {duration}.", Priority.MEDIUM),
        ("The footpath along {street} {landmark} is broken and elderly people have to walk on the road with traffic.", Priority.MEDIUM),
        ("Road markings and the zebra crossing near the school on {street} have completely faded. It is unsafe at drop-off time.", Priority.LOW),
        ("A portion of {street} in {area} has collapsed after a water line burst underneath. Half the road is sinking.", Priority.CRITICAL),
        ("Construction debris has been dumped on {street} {landmark} and nobody has removed it {duration}, blocking one full lane.", Priority.LOW),
        ("The service lane connecting {street} to the main road is full of craters and floods with even light rain.", Priority.HIGH),
        ("There is no proper ramp at the crossing on {street}, wheelchair users cannot cross at all.", Priority.LOW),
        ("Heavy trucks have destroyed the surface of {street} in {area}. The whole stretch needs resurfacing, not patchwork.", Priority.MEDIUM),
    ),
    Category.WATER: (
        ("We have not received any water supply in {area} {duration}. The whole block is buying tankers at 4000 rupees each.", Priority.HIGH),
        ("A main water line on {street} has burst and clean water is running into the street {duration}.", Priority.HIGH),
        ("The water coming from our taps in {area} is muddy and smells bad. Two children in the house have had stomach illness.", Priority.CRITICAL),
        ("Water pressure on {street} {landmark} is so low that it does not reach the first floor at all.", Priority.MEDIUM),
        ("The valve man is only opening the supply for twenty minutes a day in {area}, which is not enough for the whole lane.", Priority.MEDIUM),
        ("There is a leakage at the main valve on {street}. Thousands of litres are being wasted every day {duration}.", Priority.HIGH),
        ("Illegal hydrant operators have connected a pipe to the main line on {street} and our supply has stopped completely.", Priority.HIGH),
        ("Our water meter near {landmark} is broken and we are being billed on estimate for months.", Priority.LOW),
        ("Sewage water is mixing into the drinking water line on {street} in {area}. This is a serious health hazard.", Priority.CRITICAL),
        ("The overhead tank serving {area} has not been cleaned in years. There is visible algae in the water.", Priority.MEDIUM),
        ("No water in the entire {area} block since the pumping station motor burnt out {duration}.", Priority.HIGH),
        ("Water supply timing was changed without notice in {area} and now it comes at 2 am when nobody can fill.", Priority.LOW),
    ),
    Category.WASTE: (
        ("Garbage has not been collected from the corner of {street} {landmark} {duration}. The smell is unbearable.", Priority.MEDIUM),
        ("An illegal dumping ground has formed on the empty plot on {street} in {area}. People are burning trash at night.", Priority.HIGH),
        ("The municipal bin near {landmark} is overflowing and stray dogs are spreading the waste across the road.", Priority.MEDIUM),
        ("Medical waste from the clinic {landmark} is being thrown into the ordinary bin on {street}. Syringes are visible.", Priority.CRITICAL),
        ("No sweeper has come to our lane in {area} {duration}. Dust and litter have piled up along both sides.", Priority.LOW),
        ("Construction rubble dumped on {street} has blocked the drain and now the whole corner is a garbage heap.", Priority.MEDIUM),
        ("Dead animal lying on {street} {landmark} {duration}. Nobody has removed it and the smell has spread.", Priority.HIGH),
        ("The garbage truck comes but only collects from the main road, never from the inner lanes of {area}.", Priority.LOW),
        ("Burning of plastic waste happens every evening on the plot near {landmark}. The smoke enters our houses.", Priority.HIGH),
        ("The community bin promised for {street} was never installed, so people throw waste into the open drain.", Priority.MEDIUM),
        ("Mosquito breeding has started in the garbage pile on {street} in {area}, and two dengue cases were reported.", Priority.CRITICAL),
        ("Waste collection charges are being taken in {area} but the service has stopped {duration}.", Priority.LOW),
    ),
    Category.ELECTRICITY: (
        ("The streetlights on {street} {landmark} have not worked {duration}. The whole lane is pitch dark after Maghrib.", Priority.MEDIUM),
        ("A live wire is hanging low over the footpath on {street}. It is sparking when the wind blows. Extremely dangerous.", Priority.CRITICAL),
        ("There has been unannounced load shedding in {area} for eight hours a day {duration}, far more than the schedule.", Priority.HIGH),
        ("The transformer near {landmark} makes a loud noise and smoke came out of it yesterday evening.", Priority.CRITICAL),
        ("Our electricity pole on {street} is leaning badly and could fall on the parked cars below.", Priority.HIGH),
        ("Half of {area} has had no power since the PMT failed {duration}. Elderly residents are suffering in the heat.", Priority.HIGH),
        ("Street light poles on {street} are there but the bulbs were never installed after the new poles were put up.", Priority.LOW),
        ("Voltage fluctuation in {area} has burnt out two refrigerators in our building {duration}.", Priority.MEDIUM),
        ("Illegal kunda connections on {street} are overloading the line and causing trips every hour.", Priority.MEDIUM),
        ("The electricity meter box near {landmark} is open and children play right next to exposed terminals.", Priority.CRITICAL),
        ("Power goes out every time it drizzles in {area}. This has been happening {duration}.", Priority.MEDIUM),
        ("Billing complaint: our meter in {area} was replaced but the reading was carried over incorrectly.", Priority.LOW),
    ),
    Category.DRAINAGE: (
        ("The main drain on {street} {landmark} is completely blocked and sewage water has filled the road {duration}.", Priority.HIGH),
        ("Sewage is backing up into ground floor bathrooms across {area} whenever it rains.", Priority.CRITICAL),
        ("The open nallah beside {street} has no cover. A child fell into it last month. Please install a slab.", Priority.CRITICAL),
        ("Rain water from the last spell has still not drained from {street} in {area}. It has been standing {duration}.", Priority.HIGH),
        ("Gutter water is overflowing right in front of the school gate {landmark}. Children walk through it daily.", Priority.HIGH),
        ("The storm drain grating on {street} was stolen and the opening is now a hazard for motorcycles at night.", Priority.HIGH),
        ("Sewage line on {street} has been leaking {duration} and the road has caved in around the leak.", Priority.HIGH),
        ("The pumping station serving {area} is not running, so waste water has nowhere to go.", Priority.CRITICAL),
        ("Solid waste has clogged the drain outlet near {landmark}. Even light rain floods the entire lane.", Priority.MEDIUM),
        ("There is a permanent pool of stagnant sewage on {street} in {area} and mosquitoes have multiplied.", Priority.HIGH),
        ("The manhole on {street} overflows every morning around 7 am when the pressure builds up.", Priority.MEDIUM),
        ("No drainage line exists at all in our lane in {area}. Waste water simply runs down the middle of the street.", Priority.MEDIUM),
    ),
    Category.SAFETY: (
        ("Street snatching incidents have increased sharply near {landmark} on {street}. Three phones were taken last week.", Priority.HIGH),
        ("The boundary wall of the school on {street} has developed a large crack and could collapse on the playground.", Priority.CRITICAL),
        ("There is no traffic signal at the junction of {street} in {area} and accidents happen almost weekly.", Priority.HIGH),
        ("Stray dogs in packs are attacking children near {landmark}. Two bite cases were reported {duration}.", Priority.HIGH),
        ("An abandoned building on {street} is being used by drug addicts at night. Residents are afraid to pass.", Priority.HIGH),
        ("The CCTV camera at the entrance of {area} has been non-functional {duration} and nobody has repaired it.", Priority.MEDIUM),
        ("Illegal encroachment on {street} has narrowed the road so much that a fire engine could not enter.", Priority.HIGH),
        ("Underage boys are doing one-wheeling on {street} every night {landmark}. Someone will be killed.", Priority.HIGH),
        ("The pedestrian bridge near {landmark} has broken steps and no lighting, so nobody uses it.", Priority.MEDIUM),
        ("A gas leak smell has been coming from the pipeline on {street} {duration}. Please inspect before something happens.", Priority.CRITICAL),
        ("There is no police patrolling in {area} after 10 pm and shopkeepers are being harassed.", Priority.MEDIUM),
        ("Open electrical junction plus stagnant water together at the corner of {street} is an electrocution risk.", Priority.CRITICAL),
    ),
    Category.OTHER: (
        ("The union council office in {area} does not respond to any application. We have visited four times {duration}.", Priority.LOW),
        ("Stray cattle are tied on {street} {landmark} and block the entire footpath every morning.", Priority.LOW),
        ("Noise from the marriage hall on {street} continues past midnight and nobody enforces the timing rules.", Priority.MEDIUM),
        ("Public park in {area} has been locked by a private party and residents cannot enter anymore.", Priority.MEDIUM),
        ("The public toilet near {landmark} has been out of order {duration}.", Priority.LOW),
        ("Illegal hoardings on {street} are blocking the driver's view at the turn.", Priority.MEDIUM),
        ("Our lane in {area} has no name plate or house numbering, so ambulances cannot find addresses.", Priority.LOW),
        ("The government dispensary in {area} has had no doctor {duration}.", Priority.HIGH),
        ("Tree branches on {street} have grown into the electricity wires {landmark}.", Priority.MEDIUM),
        ("Encroachment by shopkeepers on {street} has taken over the entire footpath in {area}.", Priority.LOW),
        ("The community water cooler installed near {landmark} was removed and never replaced.", Priority.LOW),
        ("A stray fire broke out in the empty plot on {street} and there is no hydrant anywhere in {area}.", Priority.HIGH),
    ),
}

# Roman-Urdu flavoured phrasing, ~15% of the generated rows. This is how a large
# share of Karachi residents actually write, and the ML model has to handle it.
URDU_TEMPLATES: dict[Category, tuple[tuple[str, Priority], ...]] = {
    Category.ROAD: (
        ("{street} par bohot bara gaddha hai {landmark}. {duration} se koi nahi aaya theek karne. Bike wale gir rahe hain.", Priority.HIGH),
        ("Sarak toot chuki hai {area} mein, rickshaw wale andar aane se mana karte hain.", Priority.MEDIUM),
        ("Manhole ka dhakkan gayab hai {street} par, bacha gir sakta hai. Please jaldi cover karwa dein.", Priority.CRITICAL),
        ("Road cutting ke baad {street} ko dobara nahi banaya gaya, {duration} se malba para hua hai.", Priority.MEDIUM),
    ),
    Category.WATER: (
        ("{area} mein {duration} se pani nahi aa raha. Tanker 4000 ka le rahe hain, ghar chalana mushkil hai.", Priority.HIGH),
        ("Line leak ho rahi hai {street} par, saara saaf pani sarak par beh raha hai.", Priority.HIGH),
        ("Nalke ka pani gandha aur badbu wala aa raha hai {area} mein, bachon ka pet kharab ho gaya.", Priority.CRITICAL),
        ("Pani ka pressure itna kam hai ke upar wale floor par bilkul nahi charhta {landmark}.", Priority.MEDIUM),
    ),
    Category.WASTE: (
        ("{street} ke corner par kachra {duration} se para hua hai, badbu se saans lena mushkil hai.", Priority.MEDIUM),
        ("Khali plot par log kachra phenk rahe hain {area} mein aur raat ko jala dete hain.", Priority.HIGH),
        ("Municipal bin bhar chuka hai {landmark} ke paas, kuttay saara kachra bikher dete hain.", Priority.MEDIUM),
        ("Hamari gali mein {duration} se koi sweeper nahi aaya {area} mein.", Priority.LOW),
    ),
    Category.ELECTRICITY: (
        ("{street} ki street light {duration} se kharab hai, raat ko bilkul andhera hota hai.", Priority.MEDIUM),
        ("{area} mein bijli nahi hai {duration} se, transformer jal gaya hai aur koi nahi aaya.", Priority.HIGH),
        ("Nangi taar latak rahi hai {street} par aur spark kar rahi hai. Bohot khatarnak hai.", Priority.CRITICAL),
        ("Load shedding ka schedule kuch aur hai lekin {area} mein aath ghante bijli nahi hoti.", Priority.HIGH),
    ),
    Category.DRAINAGE: (
        ("Nali ka pani sarak par aa gaya hai {street} par, {duration} se yahi haal hai.", Priority.HIGH),
        ("Gutter ubal raha hai school ke gate ke samne {landmark}, bachay usi mein se guzarte hain.", Priority.HIGH),
        ("{area} mein barish ka pani {duration} se khara hua hai, machar bohot ho gaye hain.", Priority.HIGH),
        ("Khula nallah hai {street} ke saath, slab nahi hai, koi bacha gir jaye ga.", Priority.CRITICAL),
    ),
    Category.SAFETY: (
        ("{landmark} ke paas snatching bohot barh gayi hai, pichle hafte teen mobile chin gaye.", Priority.HIGH),
        ("Awara kuttay {street} par bachon ko kaat rahe hain, {duration} mein do case ho chuke.", Priority.HIGH),
        ("{street} ke mor par koi signal nahi hai, har hafte accident hota hai {area} mein.", Priority.HIGH),
        ("Gas leak ki smell aa rahi hai {street} se {duration} se, please check karwa dein.", Priority.CRITICAL),
    ),
    Category.OTHER: (
        ("UC office {area} mein koi sunwai nahi hoti, chaar dafa ja chuke hain {duration} mein.", Priority.LOW),
        ("Shadi hall ka shor raat 12 baje tak chalta hai {street} par, koi rok tok nahi.", Priority.MEDIUM),
        ("Sarkari dispensary {area} mein {duration} se doctor nahi hai.", Priority.HIGH),
        ("Footpath par thelay laga kar {street} block kar diya gaya hai.", Priority.LOW),
    ),
}

FIRST_NAMES = (
    "Ahmed", "Fatima", "Bilal", "Ayesha", "Usman", "Hina", "Kashif", "Sana",
    "Imran", "Rabia", "Naveed", "Zainab", "Farhan", "Maryam", "Adnan", "Nida",
    "Shahid", "Amna", "Tariq", "Sadia", "Junaid", "Kiran", "Owais", "Mehwish",
)
LAST_NAMES = (
    "Khan", "Siddiqui", "Ansari", "Baloch", "Memon", "Qureshi", "Shaikh",
    "Abbasi", "Rizvi", "Jamali", "Chandio", "Malik", "Hashmi", "Solangi",
)

# Baseline median resolution hours by priority (before the department multiplier).
PRIORITY_MEDIAN_HOURS: dict[Priority, float] = {
    Priority.CRITICAL: 22.0,
    Priority.HIGH: 48.0,
    Priority.MEDIUM: 96.0,
    Priority.LOW: 168.0,
}

SENTIMENT_BY_PRIORITY: dict[Priority, tuple[tuple[Sentiment, float], ...]] = {
    Priority.CRITICAL: ((Sentiment.ANGRY, 0.6), (Sentiment.CONCERNED, 0.35), (Sentiment.CALM, 0.05)),
    Priority.HIGH: ((Sentiment.ANGRY, 0.4), (Sentiment.CONCERNED, 0.45), (Sentiment.CALM, 0.15)),
    Priority.MEDIUM: ((Sentiment.ANGRY, 0.2), (Sentiment.CONCERNED, 0.45), (Sentiment.CALM, 0.35)),
    Priority.LOW: ((Sentiment.ANGRY, 0.08), (Sentiment.CONCERNED, 0.32), (Sentiment.CALM, 0.6)),
}

STOPWORDS = {
    "the", "and", "for", "has", "have", "been", "there", "this", "that", "with",
    "from", "into", "near", "past", "since", "please", "very", "also", "they",
    "our", "are", "not", "but", "all", "any", "was", "were", "will", "would",
    "hai", "hain", "raha", "rahi", "nahi", "koi", "kar", "karte", "gaya", "gayi",
    "bohot", "para", "mein", "aur", "hua", "chuka", "chuki", "wale", "wala",
}


# =============================================================================
# Generation
# =============================================================================


class SeedGenerator:
    """Builds the synthetic corpus.

    Kept as a class so every draw goes through one seeded ``random.Random``
    instance: that is what makes the whole dataset reproducible and keeps the
    generator free of global state that a test could accidentally disturb.
    """

    def __init__(self, *, seed: int = RANDOM_SEED, now: datetime | None = None) -> None:
        self.rng = random.Random(seed)
        self.now = now or datetime.now(UTC)
        self._used_references: set[str] = set()
        self._area_names = [a[0] for a in AREAS]
        self._area_weights = [a[3] for a in AREAS]

    # ------------------------------------------------------------------ helpers
    def uuid(self) -> str:
        """Deterministic UUID4 drawn from the seeded RNG.

        Primary keys are assigned here rather than left to the column default,
        because ``_link_duplicates`` needs real ids *before* the flush — and it
        keeps the entire dataset (ids included) reproducible across runs.
        """
        return str(uuid.UUID(int=self.rng.getrandbits(128), version=4))

    def reference_code(self) -> str:
        while True:
            code = REFERENCE_PREFIX + "".join(self.rng.choices(REFERENCE_ALPHABET, k=6))
            if code not in self._used_references:
                self._used_references.add(code)
                return code

    def _pick_weighted(self, options: tuple[tuple, ...]):
        values = [o[0] for o in options]
        weights = [o[1] for o in options]
        return self.rng.choices(values, weights=weights, k=1)[0]

    def _day_weights(self) -> list[float]:
        """Per-day sampling weight over the history window.

        July and August are boosted ~1.9x to create the monsoon volume spike; a mild
        upward drift toward the present reflects growing platform adoption.
        """
        weights: list[float] = []
        for days_ago in range(HISTORY_DAYS):
            day = self.now - timedelta(days=days_ago)
            weight = 1.0 + 0.35 * (1.0 - days_ago / HISTORY_DAYS)  # adoption drift
            if day.month in (7, 8):
                weight *= 1.9  # monsoon
            weights.append(weight)
        return weights

    def _category_weights(self, when: datetime) -> tuple[list[Category], list[float]]:
        """Category mix for a given day — drainage and water dominate the monsoon."""
        base: dict[Category, float] = {
            Category.ROAD: 1.9,
            Category.WATER: 1.7,
            Category.WASTE: 1.6,
            Category.ELECTRICITY: 1.5,
            Category.DRAINAGE: 1.2,
            Category.SAFETY: 0.8,
            Category.OTHER: 0.6,
        }
        if when.month in (7, 8):
            base[Category.DRAINAGE] *= 3.4
            base[Category.WATER] *= 1.8
            base[Category.ROAD] *= 1.3
        elif when.month == 6:  # pre-monsoon build-up
            base[Category.DRAINAGE] *= 1.6
        elif when.month in (4, 5):  # summer heat
            base[Category.WATER] *= 1.5
            base[Category.ELECTRICITY] *= 1.6
        return list(base.keys()), list(base.values())

    def _render(self, category: Category, area: str) -> tuple[str, Priority, bool]:
        """Fill a template's slots and report its base priority + emergency flag."""
        use_urdu = self.rng.random() < 0.15
        bank = URDU_TEMPLATES[category] if use_urdu else TEMPLATES[category]
        template, priority = self.rng.choice(bank)
        text = template.format(
            street=self.rng.choice(STREETS),
            landmark=self.rng.choice(LANDMARKS),
            duration=self.rng.choice(DURATIONS),
            area=area,
        )
        lowered = text.lower()
        is_emergency = any(marker in lowered for marker in EMERGENCY_MARKERS)
        if is_emergency and priority in (Priority.LOW, Priority.MEDIUM):
            priority = Priority.HIGH
        elif priority is Priority.CRITICAL and self.rng.random() < 0.35:
            # Not every alarming report is triaged as critical in practice.
            priority = Priority.HIGH
        return text, priority, is_emergency

    def _location(self, area: str) -> tuple[str, float, float]:
        centre = next(a for a in AREAS if a[0] == area)
        lat = round(centre[1] + self.rng.gauss(0, 0.011), 6)
        lng = round(centre[2] + self.rng.gauss(0, 0.011), 6)
        block = self.rng.randint(1, 16)
        return f"Block {block}, {area}, Karachi", lat, lng

    def _resolution_hours(self, priority: Priority, department_speed: float) -> float:
        """Right-skewed draw: log-normal around a per-priority median.

        A log-normal is the honest model for service times — most cases close quickly
        and a long tail drags the mean up. That gap between median and mean is exactly
        what the analytics narrative is supposed to explain.
        """
        median = PRIORITY_MEDIAN_HOURS[priority] * department_speed
        hours = self.rng.lognormvariate(math.log(median), 0.85)
        if self.rng.random() < 0.025:
            hours *= self.rng.uniform(3.0, 7.0)  # genuine Tukey-fence outliers
        return max(0.5, round(hours, 2))

    def _status_for(self, age_days: float) -> Status:
        """Older complaints are far more likely to be closed."""
        if self.rng.random() < 0.02:
            return Status.REJECTED
        if age_days < 3:
            p_resolved = 0.04
        elif age_days < 10:
            p_resolved = 0.24
        elif age_days < 30:
            p_resolved = 0.55
        elif age_days < 90:
            p_resolved = 0.76
        else:
            p_resolved = 0.83
        if self.rng.random() < p_resolved:
            return Status.RESOLVED
        if age_days < 5:
            return self.rng.choices(
                [Status.OPEN, Status.ASSIGNED, Status.IN_PROGRESS], weights=[30, 6, 2], k=1
            )[0]
        return self.rng.choices(
            [Status.OPEN, Status.ASSIGNED, Status.IN_PROGRESS], weights=[13, 10, 10], k=1
        )[0]

    def _keywords(self, text: str, limit: int = 5) -> list[str]:
        words = [w for w in re.findall(r"[a-z]{4,}", text.lower()) if w not in STOPWORDS]
        return [w for w, _ in Counter(words).most_common(limit)]

    def _citizen(self) -> tuple[str | None, str | None, str | None]:
        """~22% of submissions are anonymous, which the contract explicitly allows."""
        if self.rng.random() < 0.22:
            return None, None, None
        first = self.rng.choice(FIRST_NAMES)
        last = self.rng.choice(LAST_NAMES)
        name = f"{first} {last}"
        phone = f"03{self.rng.randint(0, 4)}{self.rng.randint(10, 99)}-{self.rng.randint(1000000, 9999999)}"
        email = f"{first.lower()}.{last.lower()}{self.rng.randint(1, 99)}@example.com"
        return name, phone, (email if self.rng.random() < 0.7 else None)

    # --------------------------------------------------------------------- build
    def build(
        self, count: int, departments: dict[str, Department]
    ) -> tuple[list[Complaint], list[AIAnalysis], list[StatusEvent]]:
        """Generate ``count`` complaints with analyses and timelines."""
        by_category = {
            category: department
            for department in departments.values()
            for category in (department.categories or [])
        }
        speed_by_slug = {d["slug"]: d["speed"] for d in DEPARTMENTS}

        day_weights = self._day_weights()
        day_choices = list(range(HISTORY_DAYS))

        complaints: list[Complaint] = []
        analyses: list[AIAnalysis] = []
        events: list[StatusEvent] = []

        for _ in range(count):
            days_ago = self.rng.choices(day_choices, weights=day_weights, k=1)[0]
            created_at = self.now - timedelta(
                days=days_ago,
                hours=self.rng.randint(0, 23),
                minutes=self.rng.randint(0, 59),
            )
            categories, weights = self._category_weights(created_at)
            category = self.rng.choices(categories, weights=weights, k=1)[0]
            area = self.rng.choices(self._area_names, weights=self._area_weights, k=1)[0]

            description, priority, is_emergency = self._render(category, area)
            location_text, latitude, longitude = self._location(area)
            department = by_category.get(category.value)
            speed = speed_by_slug.get(department.slug, 1.0) if department else 1.0

            age_days = (self.now - created_at).total_seconds() / 86400
            status = self._status_for(age_days)

            resolved_at = None
            if status == Status.RESOLVED:
                for _attempt in range(6):
                    hours = self._resolution_hours(priority, speed)
                    candidate = created_at + timedelta(hours=hours)
                    if candidate <= self.now:
                        resolved_at = candidate
                        break
                if resolved_at is None:
                    status = Status.IN_PROGRESS

            citizen_name, citizen_phone, citizen_email = self._citizen()

            complaint = Complaint(
                id=self.uuid(),
                reference_code=self.reference_code(),
                title=ComplaintManager._derive_title(description),
                description=description,
                category=category,
                priority=priority,
                status=status,
                location_text=location_text,
                area=area,
                latitude=latitude,
                longitude=longitude,
                citizen_name=citizen_name,
                citizen_phone=citizen_phone,
                citizen_email=citizen_email,
                image_url=None,
                department_id=department.id if department else None,
                ai_status=AIStatus.COMPLETE,
                created_at=created_at,
                updated_at=resolved_at or created_at,
                resolved_at=resolved_at,
                is_deleted=False,
            )
            complaints.append(complaint)
            analyses.append(
                self._build_analysis(complaint, description, is_emergency, department)
            )
            events.extend(self._build_timeline(complaint))

        self._link_duplicates(complaints)
        return complaints, analyses, events

    def _build_analysis(
        self,
        complaint: Complaint,
        description: str,
        is_emergency: bool,
        department: Department | None,
    ) -> AIAnalysis:
        """A plausible offline analysis so the dashboard needs no network at all.

        ``source`` is only ever ``ml`` or ``rules`` — CONTRACT §5.3 forbids labelling
        seeded rows as LLM output, and token telemetry stays ``None`` because only the
        LLM tier reports usage.
        """
        is_ml = self.rng.random() < 0.7
        source = AISource.ML if is_ml else AISource.RULES
        confidence = (
            round(self.rng.uniform(0.72, 0.97), 2) if is_ml else round(self.rng.uniform(0.44, 0.78), 2)
        )
        summary = complaint.title if len(complaint.title) < 88 else complaint.title[:87] + "…"
        return AIAnalysis(
            complaint=complaint,
            category=complaint.category,
            priority=complaint.priority,
            summary=summary,
            department_suggestion=department.name if department else None,
            confidence=confidence,
            source=source,
            model_name="tfidf-linearsvc-v1" if is_ml else "keyword-rules-v1",
            reasoning=(
                f"Classified as {complaint.category} with {complaint.priority} priority based on "
                f"{'learned term weights' if is_ml else 'matched keyword rules'}."
            ),
            keywords=self._keywords(description),
            sentiment=self._pick_weighted(SENTIMENT_BY_PRIORITY[complaint.priority]),
            is_emergency=is_emergency,
            latency_ms=self.rng.randint(8, 45) if is_ml else self.rng.randint(1, 6),
            prompt_tokens=None,
            completion_tokens=None,
            cache_hit_tokens=None,
            created_at=complaint.created_at + timedelta(seconds=self.rng.randint(2, 20)),
        )

    def _build_timeline(self, complaint: Complaint) -> list[StatusEvent]:
        """Reconstruct a believable sequence of status changes."""
        events = [
            StatusEvent(
                complaint=complaint,
                from_status=None,
                to_status=Status.OPEN,
                note="Complaint submitted by citizen.",
                actor=complaint.citizen_email or complaint.citizen_name or "citizen",
                created_at=complaint.created_at,
            )
        ]
        if complaint.status == Status.OPEN:
            return events

        end = complaint.resolved_at or (complaint.created_at + timedelta(hours=self.rng.uniform(4, 72)))
        span = max((end - complaint.created_at).total_seconds(), 3600.0)

        def at(fraction: float) -> datetime:
            return complaint.created_at + timedelta(seconds=span * fraction)

        if complaint.status == Status.REJECTED:
            events.append(
                StatusEvent(
                    complaint=complaint,
                    from_status=Status.OPEN,
                    to_status=Status.REJECTED,
                    note="Closed as out of scope for this department.",
                    actor="admin@civic.gov.pk",
                    created_at=at(0.4),
                )
            )
            return events

        events.append(
            StatusEvent(
                complaint=complaint,
                from_status=Status.OPEN,
                to_status=Status.ASSIGNED,
                note="Routed to the responsible department.",
                actor="admin@civic.gov.pk",
                created_at=at(0.15),
            )
        )
        if complaint.status == Status.ASSIGNED:
            return events

        events.append(
            StatusEvent(
                complaint=complaint,
                from_status=Status.ASSIGNED,
                to_status=Status.IN_PROGRESS,
                note="Field team dispatched.",
                actor="field.team@civic.gov.pk",
                created_at=at(0.45),
            )
        )
        if complaint.status == Status.IN_PROGRESS:
            return events

        events.append(
            StatusEvent(
                complaint=complaint,
                from_status=Status.IN_PROGRESS,
                to_status=Status.RESOLVED,
                note="Work completed and verified on site.",
                actor="field.team@civic.gov.pk",
                created_at=complaint.resolved_at or at(1.0),
            )
        )
        return events

    def _link_duplicates(self, complaints: list[Complaint]) -> None:
        """Point ~3% of complaints at an earlier one in the same area+category."""
        buckets: dict[tuple[str, str], list[Complaint]] = {}
        for complaint in sorted(complaints, key=lambda c: c.created_at):
            buckets.setdefault((complaint.area or "", complaint.category.value), []).append(complaint)

        for bucket in buckets.values():
            if len(bucket) < 4:
                continue
            for complaint in bucket[1:]:
                if self.rng.random() < 0.03:
                    earlier = self.rng.choice(
                        [c for c in bucket if c.created_at < complaint.created_at] or [bucket[0]]
                    )
                    if earlier is not complaint:
                        complaint.duplicate_of_id = earlier.id


# =============================================================================
# Persistence
# =============================================================================


async def _wipe(session) -> None:
    """Delete in FK-safe order."""
    for model in (StatusEvent, AIAnalysis, Complaint, Department):
        await session.execute(delete(model))
    await session.commit()


async def _ensure_departments(session) -> dict[str, Department]:
    """Idempotently create the six departments; returns them keyed by slug."""
    from app.repositories.department_repo import DepartmentRepository

    repo = DepartmentRepository(session)
    result: dict[str, Department] = {}
    for spec in DEPARTMENTS:
        existing = await repo.get_by_slug(spec["slug"])
        if existing is None:
            existing = Department(
                name=spec["name"],
                slug=spec["slug"],
                categories=list(spec["categories"]),
                contact_email=spec["contact_email"],
            )
            repo.add(existing)
        result[spec["slug"]] = existing
    await session.commit()
    for department in result.values():
        await session.refresh(department)
    return result


async def _ensure_admin(session) -> User:
    auth = AuthService(UserRepository(session))
    user = await auth.ensure_user(
        email=settings.ADMIN_EMAIL,
        password=settings.ADMIN_PASSWORD,
        full_name="Civic Administrator",
        role=Role.ADMIN,
    )
    await auth.ensure_user(
        email="staff@civic.gov.pk",
        password="Staff@123",
        full_name="Field Operations Staff",
        role=Role.STAFF,
    )
    await session.commit()
    return user


async def seed_database(
    *, reset: bool = False, count: int = DEFAULT_COUNT, seed: int = RANDOM_SEED
) -> dict[str, int | str]:
    """Seed departments, the admin user and ``count`` complaints.

    Idempotent: without ``reset`` it will not add complaints to a non-empty table.
    Returns a summary dict suitable for logging.
    """
    await create_all()

    async with SessionLocal() as session:
        if reset:
            await _wipe(session)

        departments = await _ensure_departments(session)
        await _ensure_admin(session)

        from app.repositories.complaint_repo import ComplaintRepository

        existing = await ComplaintRepository(session).count_all(include_deleted=True)
        if existing and not reset:
            return {
                "status": "skipped",
                "reason": "complaints already present",
                "complaints": existing,
                "departments": len(departments),
            }

        generator = SeedGenerator(seed=seed)
        complaints, analyses, events = generator.build(count, departments)

        session.add_all(complaints)
        session.add_all(analyses)
        session.add_all(events)
        await session.commit()

        resolved = sum(1 for c in complaints if c.status == Status.RESOLVED)
        durations = sorted(
            (c.resolved_at - c.created_at).total_seconds() / 3600
            for c in complaints
            if c.resolved_at
        )
        median = durations[len(durations) // 2] if durations else 0.0
        mean = sum(durations) / len(durations) if durations else 0.0

        return {
            "status": "seeded",
            "complaints": len(complaints),
            "departments": len(departments),
            "analyses": len(analyses),
            "status_events": len(events),
            "resolved": resolved,
            "median_resolution_hours": round(median, 1),
            "mean_resolution_hours": round(mean, 1),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed the civic services database.")
    parser.add_argument("--reset", action="store_true", help="Wipe existing data first.")
    parser.add_argument("--count", type=int, default=DEFAULT_COUNT, help="Complaints to generate.")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, help="Random seed.")
    args = parser.parse_args()

    configure_logging(debug=True)
    summary = asyncio.run(seed_database(reset=args.reset, count=args.count, seed=args.seed))
    log.info("seed.finished", **summary)
    print("\nSeed summary:")
    for key, value in summary.items():
        print(f"  {key:>26}: {value}")


if __name__ == "__main__":
    main()

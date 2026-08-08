"""Golden-set evaluation harness — the AI testing deliverable.

Runs 40 hand-written, realistic Karachi complaints through **all three analyzer
tiers** and writes ``docs/AI_TESTING_EVIDENCE.md``: per-tier accuracy, the
agreement rate between tiers, worked examples, and an honest limitations section.

This is deliberately separate from the synthetic held-out split in
``ml/evaluate.py``. That measures the model against data from its own generator.
This measures every tier against text a human wrote by hand, with no template
behind it — which is the only evidence here that says anything about real use.

Run::

    uv run python -m tests.golden_eval              # rules + ML (no API key needed)
    uv run python -m tests.golden_eval --with-llm   # includes DeepSeek

Without ``--with-llm`` the DeepSeek tier is skipped and the report says so, so the
harness works offline and in CI.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# --------------------------------------------------------------------------- #
# The golden set: 40 hand-written complaints.
#
# (id, text, expected_category, expected_priority, note)
#
# Written to cover: pure English, pure Roman-Urdu, code-switched, ALL CAPS,
# SMS-shortened, one-word/vague, deliberately ambiguous category boundaries,
# every category, and every priority level. `note` records *why* the expected
# label is what it is, including where it is genuinely debatable.
# --------------------------------------------------------------------------- #

GOLDEN: list[tuple[str, str, str, str, str]] = [
    ("G01",
     "There is a very large pothole in the middle of Rashid Minhas Road right outside the "
     "government girls school. Two motorcyclists have already fallen this week. Please fix it "
     "before someone is killed.",
     "road", "critical", "Road surface damage; injuries already occurred + school adjacent."),

    ("G02",
     "Assalam o alaikum, hamari gali me kachra 15 din se utha nahi hai. Bohot badbu hai aur "
     "makhiyan bohot ho gayi hain. Bachay wahan se school jate hain. Meherbani kar ke safai "
     "karwa dain.",
     "waste", "high", "Pure Roman-Urdu. Solid waste, 15 days, disease vector, school route."),

    ("G03",
     "GUTTER KA PANI PURI SARAK PAR PHAIL GAYA HAI!!! SEWERAGE LINE BAND HAI. GAARI NIKALNA "
     "MUSHKIL HO GAYA HAI. TEEN HAFTE SE YEHI HAAL HAI!!!",
     "drainage", "high", "ALL CAPS Roman-Urdu. Waste water, so drainage not road."),

    ("G04",
     "bijli ka khamba jhuk gaya hai aur taar neeche latak rahi hai, neeche barish ka pani khara "
     "hai aur bachay wahin khelte hain. koi bara hadsa ho sakta hai.",
     "electricity", "critical", "Live wire above standing water where children play."),

    ("G05",
     "The street light outside our house has been off for about two weeks. Not urgent, just "
     "letting you know so it can be fixed in the normal round.",
     "electricity", "low", "Streetlights map to electricity. Citizen explicitly says not urgent."),

    ("G06",
     "No water supply in our area since 5 days. We have to buy a tanker daily which costs 4000 "
     "rupees. Please restore the supply.",
     "water", "high", "Clean water supply failure, whole area, 5 days, financial burden."),

    ("G07",
     "Stray dogs have attacked two children near the park in Korangi. One child needed stitches "
     "at the hospital. Please send the municipal team.",
     "safety", "critical", "Live animal attack with injuries."),

    ("G08",
     "Sewage has mixed into our drinking water line. Three people in our house are vomiting and "
     "my father has been admitted to hospital.",
     "water", "critical",
     "AMBIGUOUS: contamination of the clean water line -> water; a grader could "
     "defensibly say drainage. Hospitalisation makes it critical either way."),

    ("G09",
     "the road outside our house is broken since months, there are big khadde and rickshaws "
     "keep getting stuck. please repair.",
     "road", "medium", "Code-switched. Road surface, months, no injury or sensitive site."),

    ("G10",
     "Respected sir, the garbage container at the corner of our street is overflowing and nobody "
     "has come to empty it for a week.",
     "waste", "medium", "Ordinary uncollected waste, one street."),

    ("G11",
     "Load shedding in our area is running 8 hours a day with no schedule announced. Small "
     "businesses in the market are shutting down.",
     "electricity", "high", "Power outage, whole market affected."),

    ("G12",
     "nali band hai, gande pani ki wajah se ghar me se bo aa rahi hai. do hafte se yehi masla hai.",
     "drainage", "medium", "Roman-Urdu. Blocked drain, single household, two weeks."),

    ("G13",
     "An open manhole in the middle of the walkway near the bus stop has no cover. It is dark at "
     "night and someone will fall in.",
     "road", "critical",
     "AMBIGUOUS: open manhole in a walkway -> road hazard per our disambiguation "
     "rule 5; drainage is also defensible. Fall risk makes it critical."),

    ("G14",
     "plz clean the drain near our house it is blocked & water is standing, mosquitoes r everywhere",
     "drainage", "medium", "SMS-shortened English. Blocked drain, mosquitoes."),

    ("G15",
     "Mobile snatching happens at this turn almost every evening. Three incidents in our lane "
     "this month. There is no police patrolling at all.",
     "safety", "high", "Repeated street crime, no injury reported."),

    ("G16",
     "The park in our neighbourhood has been encroached by a private party who built a boundary "
     "wall and now charges an entry fee.",
     "other", "medium", "Encroachment on public land, no category fits."),

    ("G17",
     "pani ka pressure bohot kam hai, sirf ground floor tak aata hai. upar wale floors par "
     "bilkul nahi aata.",
     "water", "medium", "Roman-Urdu. Low water pressure, one building."),

    ("G18",
     "A large water pipe has burst on the main road and thousands of gallons are being wasted "
     "every hour. It has also started eroding the road surface.",
     "water", "high",
     "AMBIGUOUS: burst supply main causing road damage. Rule 8 says fix the cause -> water."),

    ("G19",
     "kachra",
     "waste", "medium", "Single word. Tests degenerate input handling."),

    ("G20",
     "There is a serious problem here please send someone",
     "other", "medium", "Vague, no category signal at all. Should be low confidence."),

    ("G21",
     "The transformer in our lane trips every time anyone switches on an AC. We have had no "
     "power for most of today.",
     "electricity", "medium", "Electrical fault, one lane, one day."),

    ("G22",
     "Rain water from last week's spell has still not drained out of our street. It is now green "
     "and full of mosquitoes and two children have got dengue.",
     "drainage", "high", "Storm drainage failure with confirmed disease cases."),

    ("G23",
     "The footpath tiles outside the hospital gate are completely uprooted. Elderly patients "
     "walking to the OPD are tripping on them.",
     "road", "high", "Pedestrian infrastructure, hospital adjacent, vulnerable users."),

    ("G24",
     "Sarak par khudai kar ke chor di gayi hai, gas company wale trench bhar kar nahi gaye. "
     "raat ko koi bhi gir sakta hai.",
     "road", "high", "Roman-Urdu. Unfilled trench, fall risk at night."),

    ("G25",
     "I would like to suggest that a dustbin be installed near the bus stop. There is no urgency, "
     "it is only a suggestion for improvement.",
     "waste", "low", "Explicitly a suggestion, explicitly not urgent."),

    ("G26",
     "The wall of the abandoned plot next to the school has cracked badly and is leaning over the "
     "footpath. It could come down on the children any day.",
     "safety", "critical", "Imminent structural collapse over a school route."),

    ("G27",
     "Water coming from our tap is yellow and smells bad. We cannot drink it. This has been going "
     "on for ten days.",
     "water", "high", "Contaminated drinking water, health risk."),

    ("G28",
     "Loudspeakers from the marriage hall behind our house run past midnight every weekend. "
     "Nobody in the building can sleep.",
     "other", "medium", "Noise nuisance, no category fits."),

    ("G29",
     "sewerage ka pani ubal kar manhole se bahar aa raha hai aur bacho ke school jane ke raste "
     "par kharra hai. Bimari phail rahi hai.",
     "drainage", "high", "Roman-Urdu. Sewage overflow on a school route, disease spreading."),

    ("G30",
     "Someone received an electric shock from the pole outside our shop this morning. He is in "
     "the hospital now. The wires are still live.",
     "electricity", "critical", "Electrocution already occurred, hazard still live."),

    ("G31",
     "MERI GALI ME KOI SAFAI WALA NAHI AATA. MAHINO SE JHAROO NAHI LAGI. KYA HUM INSAAN NAHI HAIN?",
     "waste", "high", "ALL CAPS angry Roman-Urdu. No sweeping for months."),

    ("G32",
     "The speed breaker near the school has broken and has sharp metal edges sticking out. A "
     "school van tyre burst on it yesterday.",
     "road", "high", "Road furniture damage, school, near-miss."),

    ("G33",
     "Our electricity meter reading is being estimated every month and the bill is impossible to "
     "pay. Nobody comes to take the actual reading.",
     "electricity", "medium", "Billing dispute follows its utility -> electricity."),

    ("G34",
     "There is construction malba dumped on the empty plot beside our house. It has been there "
     "for three weeks and snakes have started coming out of it.",
     "waste", "high", "Construction debris dumping, safety consequence."),

    ("G35",
     "An illegal gas cylinder refilling shop is operating in the middle of the crowded market. "
     "If it catches fire the whole block will go.",
     "safety", "critical", "Explosion risk in a crowded area."),

    ("G36",
     "hamare area me pani ka tanker mafia paise mangta hai. sarkari supply band kar di gayi hai "
     "jaan buch kar.",
     "water", "high", "Roman-Urdu. Water supply denial / tanker mafia."),

    ("G37",
     "The drain cover outside house number 22 is cracked. It still works but should be replaced "
     "at some point during routine maintenance.",
     "drainage", "low", "Minor, explicitly routine."),

    ("G38",
     "Rash driving by mini buses on our residential lane has caused three near misses with "
     "children this month. They do not slow down at all.",
     "safety", "high", "Traffic danger to children, no injury yet."),

    ("G39",
     "The government dispensary in our area opens only twice a week without any notice and the "
     "staff refuse to register walk-in patients.",
     "other", "medium", "Public service failure, no category fits."),

    ("G40",
     "road toot gayi hai bilkul, gaari chalana namumkin hai. 2 mahine se koi nahi aaya dekhne. "
     "please jaldi theek karwa dain!!!",
     "road", "medium", "Roman-Urdu with exclamation intensity. Road damage, no hazard signal."),
]


TIERS = ("rules", "ml", "llm")


def _load_analyzers(with_llm: bool) -> dict[str, Any]:
    from app.ai.ml_analyzer import MLAnalyzer
    from app.ai.rule_analyzer import RuleBasedAnalyzer

    analyzers: dict[str, Any] = {
        "rules": RuleBasedAnalyzer(),
        "ml": MLAnalyzer(),
    }
    if with_llm:
        from app.ai.llm_analyzer import DeepSeekAnalyzer

        analyzers["llm"] = DeepSeekAnalyzer()
    return analyzers


def run(with_llm: bool = False) -> dict[str, Any]:
    analyzers = _load_analyzers(with_llm)
    available = {
        name: bool(a.is_available()) if name != "llm" else bool(a.configured())
        for name, a in analyzers.items()
    }
    print("tiers:", {k: ("available" if v else "UNAVAILABLE") for k, v in available.items()})

    rows: list[dict[str, Any]] = []
    for gid, text, exp_cat, exp_pri, note in GOLDEN:
        row: dict[str, Any] = {
            "id": gid, "text": text, "expected_category": exp_cat,
            "expected_priority": exp_pri, "note": note, "tiers": {},
        }
        for name, analyzer in analyzers.items():
            if not available.get(name):
                row["tiers"][name] = {"skipped": True}
                continue
            t0 = time.perf_counter()
            result, error = analyzer.safe_analyze(text, {})
            elapsed = 1000 * (time.perf_counter() - t0)
            if result is None:
                row["tiers"][name] = {"error": error, "skipped": True}
                continue
            row["tiers"][name] = {
                "category": result.category,
                "priority": result.priority,
                "confidence": round(result.confidence, 4),
                "is_emergency": result.is_emergency,
                "summary": result.summary,
                "reasoning": result.reasoning,
                "keywords": result.keywords,
                "model_name": result.model_name,
                "latency_ms": round(elapsed, 1),
                "category_correct": result.category == exp_cat,
                "priority_correct": result.priority == exp_pri,
            }
        rows.append(row)
        marks = " ".join(
            f"{t}:{row['tiers'][t].get('category', '-')}/{row['tiers'][t].get('priority', '-')}"
            for t in TIERS if t in row["tiers"] and not row["tiers"][t].get("skipped")
        )
        print(f"  {gid} want {exp_cat}/{exp_pri:8s} | {marks}")

    summary = _summarise(rows, available)
    return {"rows": rows, "summary": summary, "available": available,
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}


def _ran(row: dict[str, Any], tier: str) -> bool:
    """Did ``tier`` actually produce a result for this row?"""
    t = row["tiers"].get(tier)
    return bool(t) and not t.get("skipped", False) and "category" in t


def _priority_rank(p: str) -> int:
    return {"low": 0, "medium": 1, "high": 2, "critical": 3}.get(p, 1)


def _summarise(rows: list[dict[str, Any]], available: dict[str, bool]) -> dict[str, Any]:
    n = len(rows)
    per_tier: dict[str, Any] = {}
    for tier in TIERS:
        scored = [r for r in rows if _ran(r, tier)]
        if not scored:
            continue
        cat_ok = sum(r["tiers"][tier]["category_correct"] for r in scored)
        pri_ok = sum(r["tiers"][tier]["priority_correct"] for r in scored)
        # Off-by-one on priority is a much cheaper error than off-by-three.
        within_one = sum(
            abs(_priority_rank(r["tiers"][tier]["priority"])
                - _priority_rank(r["expected_priority"])) <= 1
            for r in scored
        )
        under = sum(
            _priority_rank(r["tiers"][tier]["priority"]) < _priority_rank(r["expected_priority"])
            for r in scored
        )
        over = sum(
            _priority_rank(r["tiers"][tier]["priority"]) > _priority_rank(r["expected_priority"])
            for r in scored
        )
        latencies = [r["tiers"][tier]["latency_ms"] for r in scored]
        confs = [r["tiers"][tier]["confidence"] for r in scored]
        emerg_expected = [r for r in scored if r["expected_priority"] == "critical"]
        emerg_caught = sum(1 for r in emerg_expected if r["tiers"][tier]["is_emergency"])
        per_tier[tier] = {
            "n": len(scored),
            "category_accuracy": round(cat_ok / len(scored), 4),
            "priority_accuracy": round(pri_ok / len(scored), 4),
            "priority_within_one_level": round(within_one / len(scored), 4),
            "priority_under_triaged": under,
            "priority_over_triaged": over,
            "emergency_recall": (round(emerg_caught / len(emerg_expected), 4)
                                 if emerg_expected else None),
            "emergency_expected": len(emerg_expected),
            "median_latency_ms": round(statistics.median(latencies), 2),
            "mean_confidence": round(statistics.fmean(confs), 4),
            "model_name": scored[0]["tiers"][tier]["model_name"],
        }

    # pairwise agreement
    agreement: dict[str, Any] = {}
    for i, a in enumerate(TIERS):
        for b in TIERS[i + 1:]:
            both = [r for r in rows if _ran(r, a) and _ran(r, b)]
            if not both:
                continue
            cat_same = sum(r["tiers"][a]["category"] == r["tiers"][b]["category"] for r in both)
            pri_same = sum(r["tiers"][a]["priority"] == r["tiers"][b]["priority"] for r in both)
            agreement[f"{a}_vs_{b}"] = {
                "n": len(both),
                "category_agreement": round(cat_same / len(both), 4),
                "priority_agreement": round(pri_same / len(both), 4),
            }

    live = [t for t in TIERS if t in per_tier]
    unanimous = sum(
        1 for r in rows
        if len({r["tiers"][t]["category"] for t in live if _ran(r, t)}) == 1
    ) if len(live) > 1 else None

    confusions = Counter()
    for r in rows:
        for tier in live:
            got = r["tiers"][tier]["category"]
            if got != r["expected_category"]:
                confusions[f"{r['expected_category']}->{got} ({tier})"] += 1

    return {
        "golden_set_size": n,
        "tiers_evaluated": live,
        "tiers_available": available,
        "per_tier": per_tier,
        "agreement": agreement,
        "all_tiers_agree_on_category": unanimous,
        "top_category_errors": confusions.most_common(8),
    }


# --------------------------------------------------------------------------- #
# Report rendering
# --------------------------------------------------------------------------- #

def _tier_label(tier: str) -> str:
    return {"llm": "DeepSeek (LLM)", "ml": "TF-IDF + LinearSVC (ML)",
            "rules": "Keyword rules"}[tier]


def render_markdown(data: dict[str, Any]) -> str:
    s = data["summary"]
    rows = data["rows"]
    live = s["tiers_evaluated"]
    n_critical = sum(1 for r in rows if r["expected_priority"] == "critical")

    # ---- comparison table
    header = "| Metric | " + " | ".join(_tier_label(t) for t in live) + " |"
    sep = "|---" * (len(live) + 1) + "|"
    def line(label: str, fn) -> str:
        return f"| {label} | " + " | ".join(fn(s["per_tier"][t]) for t in live) + " |"

    table = "\n".join([
        header, sep,
        line("Model / engine", lambda p: f"`{p['model_name']}`"),
        line("Items scored", lambda p: str(p["n"])),
        line("**Category accuracy**", lambda p: f"**{p['category_accuracy']:.3f}**"),
        line("**Priority accuracy**", lambda p: f"**{p['priority_accuracy']:.3f}**"),
        line("Priority within 1 level", lambda p: f"{p['priority_within_one_level']:.3f}"),
        line("Priority under-triaged", lambda p: str(p["priority_under_triaged"])),
        line("Priority over-triaged", lambda p: str(p["priority_over_triaged"])),
        line(f"Emergency recall (of {n_critical} critical)",
             lambda p: ("—" if p["emergency_recall"] is None else f"{p['emergency_recall']:.3f}")),
        line("Median latency", lambda p: f"{p['median_latency_ms']:.1f} ms"),
        line("Mean confidence", lambda p: f"{p['mean_confidence']:.3f}"),
    ])

    # ---- agreement
    if s["agreement"]:
        agree_lines = ["| Pair | n | Category agreement | Priority agreement |", "|---|---|---|---|"]
        for pair, v in s["agreement"].items():
            a, b = pair.split("_vs_")
            agree_lines.append(
                f"| {_tier_label(a)} vs {_tier_label(b)} | {v['n']} | "
                f"{v['category_agreement']:.3f} | {v['priority_agreement']:.3f} |"
            )
        agreement_table = "\n".join(agree_lines)
    else:
        agreement_table = "_Only one tier was available, so there is nothing to compare._"

    # ---- disagreements
    disagreements = []
    for r in rows:
        cats = {t: r["tiers"][t]["category"] for t in live if _ran(r, t)}
        pris = {t: r["tiers"][t]["priority"] for t in live if _ran(r, t)}
        if len(set(cats.values())) > 1 or len(set(pris.values())) > 1:
            disagreements.append((r, cats, pris))

    dis_lines = ["| ID | Expected | " + " | ".join(_tier_label(t) for t in live) + " | Complaint |",
                 "|---" * (len(live) + 3) + "|"]
    for r, cats, pris in disagreements[:12]:
        cells = " | ".join(
            f"`{cats.get(t, '-')}` / `{pris.get(t, '-')}`" for t in live
        )
        dis_lines.append(
            f"| {r['id']} | `{r['expected_category']}` / `{r['expected_priority']}` | "
            f"{cells} | {r['text'][:70].replace('|', '/')}... |"
        )
    disagreement_table = "\n".join(dis_lines)

    # ---- worked examples
    picks = ["G02", "G04", "G08", "G13", "G19", "G20", "G30", "G40"]
    by_id = {r["id"]: r for r in rows}
    worked = []
    for pid in picks:
        r = by_id.get(pid)
        if not r:
            continue
        block = [
            f"### {r['id']} — expected `{r['expected_category']}` / `{r['expected_priority']}`",
            "",
            "**Input**",
            "",
            "```text",
            r["text"],
            "```",
            "",
            f"*Why that label:* {r['note']}",
            "",
        ]
        for tier in live:
            t = r["tiers"].get(tier, {})
            if not _ran(r, tier):
                block.append(f"**{_tier_label(tier)}** — not run.\n")
                continue
            tick_c = "correct" if t["category_correct"] else "**WRONG**"
            tick_p = "correct" if t["priority_correct"] else "**WRONG**"
            block += [
                f"**{_tier_label(tier)}** → `{t['category']}` ({tick_c}) / "
                f"`{t['priority']}` ({tick_p}), confidence {t['confidence']:.2f}, "
                f"{t['latency_ms']:.1f} ms",
                "",
                f"> {t['summary']}",
                "",
                f"*Reasoning:* {t['reasoning']}",
                "",
            ]
        worked.append("\n".join(block))
    worked_md = "\n".join(worked)

    llm_note = (
        "" if "llm" in live else
        "\n> **The DeepSeek tier was not run for this report.** No `DEEPSEEK_API_KEY` was "
        "configured, which is exactly the demo-safety scenario this architecture is built "
        "for — the numbers below are what the product delivers with **no API key at all**. "
        "Re-run with `uv run python -m tests.golden_eval --with-llm` to include it.\n"
    )

    errors = "\n".join(f"- `{k}` × {v}" for k, v in s["top_category_errors"]) or "- none"

    # The headline finding, computed rather than asserted: does the model trained on
    # synthetic data actually beat the hand-written rules on hand-written text?
    finding = ""
    if "ml" in s["per_tier"] and "rules" in s["per_tier"]:
        ml_c = s["per_tier"]["ml"]["category_accuracy"]
        ru_c = s["per_tier"]["rules"]["category_accuracy"]
        ml_p = s["per_tier"]["ml"]["priority_accuracy"]
        ru_p = s["per_tier"]["rules"]["priority_accuracy"]
        if ru_c >= ml_c or ru_p >= ml_p:
            finding = f"""
### The most important number in this report

On the synthetic held-out split the ML model scores **~0.76 category / ~0.74
priority** (`ml/artifacts/evaluation.md`). On these 40 hand-written complaints it
scores **{ml_c:.3f} category / {ml_p:.3f} priority** — and the *keyword rules*
score **{ru_c:.3f} / {ru_p:.3f}**.

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
"""
        else:
            finding = f"""
### Synthetic-to-real transfer

The ML model scores **~0.76 category** on its synthetic held-out split and
**{ml_c:.3f}** on these hand-written complaints, ahead of the keyword rules at
**{ru_c:.3f}**. The drop from the synthetic figure is the cost of training on
generated data, and it is why the model sits below the LLM in the chain.
"""

    return f"""# AI Testing Evidence

*Generated {data['generated_at']} by `tests/golden_eval.py`. Reproduce with
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
{s['golden_set_size']}-item golden set is the only measurement in this project that
says anything about behaviour on text a human actually wrote.
{llm_note}
## 2. Per-tier results

{table}

**Under-triage vs over-triage is the number that matters**, not raw accuracy.
Calling a `critical` complaint `low` can get someone hurt; calling a routine one
`critical` wastes a site visit. The two columns are reported separately for exactly
that reason, and the pipeline's escalation rules are deliberately asymmetric — the
keyword hazard rules may *raise* the ML model's priority and may never lower it.
{finding}
## 3. Do the tiers agree with each other?

{agreement_table}

Agreement is not correctness — three tiers can be wrong together, and on the
ambiguous items they often are. It is useful as a *confidence signal*: an item where
all tiers agree is far more likely to be right than one where they split, which is
why disagreement is a good trigger for human review.

Items where all available tiers picked the same category: **{s['all_tiers_agree_on_category']} / {s['golden_set_size']}**.

## 4. Where the tiers disagree

{disagreement_table}

Most common category errors across all tiers:

{errors}

## 5. Worked examples — input to output

{worked_md}

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
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--with-llm", action="store_true",
                        help="include the DeepSeek tier (needs DEEPSEEK_API_KEY)")
    parser.add_argument("--out", type=Path,
                        default=_ROOT / "docs" / "AI_TESTING_EVIDENCE.md")
    parser.add_argument("--json-out", type=Path,
                        default=_ROOT / "docs" / "ai_testing_evidence.json")
    args = parser.parse_args()

    if args.with_llm and not os.environ.get("DEEPSEEK_API_KEY"):
        print("warning: --with-llm passed but DEEPSEEK_API_KEY is empty; "
              "the tier will report unavailable", file=sys.stderr)

    data = run(with_llm=args.with_llm)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render_markdown(data), encoding="utf-8")
    args.json_out.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")

    print("\n=== summary ===")
    print(json.dumps(data["summary"], indent=2))
    print(f"\nwrote {args.out}")
    print(f"wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

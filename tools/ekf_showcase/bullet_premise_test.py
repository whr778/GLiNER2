"""Does restructuring text into self-contained bullets fix number-to-place attachment?

This tests the PREMISE only. Bullets here are hand-written, so a negative result kills the
idea before any summarizer is built, and a positive result specifies exactly what the
summarizer has to produce. It deliberately does not test whether a model can produce them.

The sentences are REAL, pulled from the Helene AP feed rather than invented, because the
invented example everyone reaches for ("120 died in NC, 17 in TN, 227 total") does not occur
in this corpus. What actually occurs is numbers that are not casualties at all -- a 140-mile
distance, a 30-year career, a town's 6,000 population, 30.5 centimeters of rain, the year
2004 -- and casualties belonging to a *different* hurricane. Those are the failure modes a
summarizer would have to fix, so those are what is tested.

`expect` is the correct extraction, established by reading the sentence:
    None            no casualty figure attributable to a place -- the correct answer is
                    silence, and any emission is a false positive
    (value, place)  the one correct binding

    uv run python tools/ekf_showcase/bullet_premise_test.py
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from gliner2 import AutoExtractor, Schema

CASES = [
    {
        "id": "distance-and-tenure",
        "raw": ("The county is located about 140 miles east of Chattanooga, Tennessee. "
                "Assistant fire chief in southern Georgia killed by tree that fell on truck. "
                "Vernon “Leon” Davis, a veteran firefighter of 30 years, died in "
                "Blackshear when a tree fell on his vehicle, officials said."),
        "free": ["1 firefighter died in Blackshear, Georgia, during Hurricane Helene."],
        # Extractive-only cannot reach it: the source states a death but never a digit.
        # The only digits present are 140 (miles) and 30 (years), neither a toll.
        "extractive": ["A firefighter died in Blackshear, Georgia, during Hurricane Helene."],
        "expect": (1, "georgia"),
        "trap": "140 (miles) and 30 (years of service) are not casualties",
    },
    {
        "id": "other-hurricane",
        "raw": ("In 2004, for example, four people were killed in western North Carolina "
                "from a debris flow caused by as much of a foot (30.5 centimeters) of rain "
                "that fell from Hurricane Ivan."),
        "free": [],          # nothing about Helene is stated here at all
        "extractive": [],
        "expect": None,
        "trap": "four deaths belong to Hurricane Ivan in 2004, not to Helene",
    },
    {
        "id": "town-population",
        "raw": ("They died together early Friday when Helene’s wind and rain toppled the "
                "family’s giant oak tree and it crashed into their bedroom in "
                "Sandersville, a town of 6,000 in central Georgia."),
        "free": ["2 people died in Sandersville, Georgia, when a tree fell on their home "
                 "during Hurricane Helene."],
        # 6,000 is the only digit and it is a population, so an extractive summarizer has
        # nothing it is allowed to emit -- which is the right answer here.
        "extractive": ["People died in Sandersville, Georgia, when a tree fell on their home "
                       "during Hurricane Helene."],
        "expect": None,   # the toll is "they", never a digit -- 6,000 is the population
        "trap": "6,000 is the town population, not a death toll",
    },
    {
        "id": "two-state-aggregate",
        "raw": ("It’s to support immediate and long-term humanitarian aid and recovery "
                "efforts in North and South Carolina in the wake of devastation from the "
                "Category 4 storm. The region remains in a state of emergency, and more than "
                "50 people have died."),
        "free": ["More than 50 people died across North and South Carolina combined "
                 "during Hurricane Helene."],
        "extractive": ["More than 50 people died across North and South Carolina combined "
                       "during Hurricane Helene."],
        "expect": (50, "__aggregate__"),
        "trap": "50 spans two states; binding it to either one is wrong",
    },
    {
        "id": "total-plus-increment",
        "raw": ("The number of deaths stood at 225 on Friday; two more were recorded in "
                "South Carolina the following day."),
        "free": ["The total Hurricane Helene death toll reached 225 on Friday.",
                 "Two additional deaths were recorded in South Carolina the next day."],
        # 225 is copyable; the increment is spelled "two" in the source, so extractive-only
        # cannot digitize it -- and an increment is not a level anyway, so losing it is
        # correct rather than merely tolerable.
        "extractive": ["The total Hurricane Helene death toll across all states reached 225 "
                       "on Friday.",
                       "More deaths were recorded in South Carolina the next day."],
        "expect": (225, "__aggregate__"),
        "trap": "225 is national; the increment is 2, not a South Carolina level",
    },
]


def schema() -> Schema:
    return (Schema().structure("casualty_report")
            .field("dead", dtype="str",
                   description="number of people killed or confirmed dead")
            .field("location", dtype="str",
                   description="the state or place these deaths occurred in"))


def digits(text) -> int | None:
    d = re.sub(r"[^\d]", "", str(text or ""))
    return int(d) if d else None


def verbatim(value: int, source: str) -> bool:
    """The guard: a figure must appear in the source text, formatted either way.

    Cheap and decisive against a summarizer inventing or merging a toll, which is the one
    failure a measurement pipeline cannot tolerate.
    """
    return bool(re.search(rf"\b{value:,}\b|\b{value}\b", source))


def run(model, text: str, source: str) -> list:
    """Extract (value, place) pairs, dropping any figure not present verbatim in source."""
    out = model.extract(text, schema())
    got = []
    for rec in (out.get("casualty_report") or []):
        v = digits(rec.get("dead"))
        if v is None:
            continue
        place = str(rec.get("location") or "").strip().lower() or None
        got.append({"value": v, "place": place, "verbatim": verbatim(v, source)})
    return got


ARMS = ("raw", "free", "extractive")


def load_rollup(path: Path) -> dict:
    """The pipeline's own alias table, so a place is judged the way production judges it.

    Scoring the literal surface would mark `north and south carolina` wrong when the
    pipeline maps it straight to `__aggregate__` -- an artefact of the test, not a defect
    in the extraction.
    """
    return json.loads(path.read_text(encoding="utf-8"))["aliases"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="whr778/gliner2-base-v1-casualty-docee")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--rollup", default="datasets/helene2024/rollup.json")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    aliases = load_rollup(Path(args.rollup))
    model = AutoExtractor.from_pretrained(args.model, map_location=args.device)
    model.eval()

    results = []
    for case in CASES:
        got = {"raw": run(model, case["raw"], case["raw"])}
        # Each bullet is extracted SEPARATELY and the results unioned. Joining them back
        # into one string would rebuild exactly the ambiguity the split is meant to remove,
        # which is the whole premise under test.
        # Figures are checked against the ORIGINAL text: that is the point of the guard.
        for arm in ("free", "extractive"):
            got[arm] = [g for b in case[arm] for g in run(model, b, case["raw"])]
        results.append({"id": case["id"], "expect": case["expect"], **got})
        print(f"\n=== {case['id']} ===")
        print(f"  trap:   {case['trap']}")
        print(f"  expect: {case['expect']}")
        for arm in ARMS:
            print(f"  {arm:<11}-> {got[arm] or 'nothing'}")

    print("\n" + "=" * 74)
    print(f"{'arm':<12}{'correct':>9}{'false pos':>11}{'guard fails':>13}")
    for arm in ARMS:
        ok = sum(1 for r in results if _correct(r[arm], r["expect"], aliases))
        fp = sum(len([g for g in r[arm] if not _matches(g, r["expect"], aliases)])
                 for r in results)
        bad = sum(len([g for g in r[arm] if not g["verbatim"]]) for r in results)
        print(f"{arm:<12}{f'{ok}/{len(results)}':>9}{fp:>11}{bad:>13}")

    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2, ensure_ascii=False),
                                  encoding="utf-8")


def _rolled(place: str | None, aliases: dict) -> str | None:
    if not place:
        return None
    key = place.strip().lower()
    return aliases.get(key, key)


def _matches(got, expect, aliases) -> bool:
    if expect is None:
        return False
    value, place = expect
    return got["value"] == value and _rolled(got["place"], aliases) == place


def _correct(got, expect, aliases) -> bool:
    """Silence is the correct answer when `expect` is None."""
    if expect is None:
        return not got
    return any(_matches(g, expect, aliases) for g in got)


if __name__ == "__main__":
    main()

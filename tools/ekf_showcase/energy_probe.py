"""Do type / trigger energies separate the two measured extraction error classes?

A context audit of the 106 Helene 'dead' observations put the residual at 4.7% cross-event
(Hurricane Katrina's 1400, a Typhoon's 250, Hurricane Milton's 230, Bosnia's 16, Hurricane
John's 2 in Mexico) and 3.8% non-casualty numbers (9 mph, 25 mph, a two-day period). Those
are different failures and a single test should not be assumed to catch both:

    non-casualty   the TYPE is wrong -- it is a speed or a duration, not a count.
                   A type query should separate it directly.
    cross-event    the type is RIGHT. 1400 is a genuine cardinal death toll; it just
                   belongs to Katrina. Only an event/trigger association can see it.

**Why this is a measurement and not a design.** GLiNER2 scores each (query, span) pair
through an independent sigmoid -- there is no softmax across types -- so a margin between
two type queries is not calibrated the way a softmax margin would be. Yesterday's collision
audit showed two queries scoring the SAME span with 0 of 241 pairs agreeing, which is
evidence the queries carry different information and equally that their scales are not
guaranteed commensurable. So the question is empirical: does the margin separate on cases
whose labels we already know?

Note the trigger is NOT an NER tag and NOT a record anchor -- `RECORD_TASK_TYPES` is
`("json_structures",)`, so events never reach the anchor machinery. There is no free
"which trigger owns this number" binding; it has to be measured, which is why trigger is a
separate probe here rather than an assumed grouping.

**n = 4 and n = 5.** A clean separation on nine labelled cases is suggestive, not
conclusive. A FAILURE to separate is the decisive direction, and is what this is for.

RESULT (2026-08-11). The two halves answer differently, and the competing-type set is the
whole design decision.

**Unit errors: solved, cleanly.** Scoring `death toll` against physically incompatible types
only -- measurement, duration, money -- separates perfectly:

    real unit errors   -0.996  -0.993  -0.993  -0.972      (9 mph, 25 mph x2, "two-day")
    everything else     0.000   0.001   0.003   0.007  ...

4/4 caught, **0/83 false positives**, stable for any threshold in [0.05, 0.9]. There is no
tuning here; the gap is the width of the scale.

**`quantity` must NOT be a competitor.** Described as "a count of things that are not
people", it is semantically ADJACENT to a death toll rather than incompatible with it, and
including it takes false positives from 0% to **21.7%** -- rejecting genuine tolls like
"Dozens", "300" and "three". Compete against what a death toll physically cannot be, not
against what it resembles.

**Cross-event: NOT solved, exactly as predicted.** 0/11 caught, because the type is *right*
-- Katrina's 1,400 scores `death toll` 0.95. It needs an event association, not a type.

**And the trigger probe as written is the wrong test.** "Is Helene named in the window" fires
for 8/11 cross-event cases versus 37/83 clean ones -- worse than useless, because every
window in a Helene feed mentions Helene. The signal has to be a COMPETING event named nearby
(Katrina, Milton, Typhoon Yinxing all appear), not the presence of the right one.

**Labelling caveat, and the rule survived it.** Labels are keyed on (span, value), so one
audited "two-day period" tagged all 7 occurrences of "two", and one Milton-adjacent "230"
tagged all 6. The type rule sorted them correctly anyway -- the genuine "two-day" scores
-0.972 while "two people died" scores +0.599 -- so the measurement corrected the labels
rather than inheriting them.

    uv run python tools/ekf_showcase/energy_probe.py
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from gliner2 import AutoExtractor, Schema

# Labels from the 2026-08-11 context audit. (span, value, label, why)
LABELLED = {
    ("1,400", 1400): ("cross-event", "Hurricane Katrina"),
    ("250", 250): ("cross-event", "Typhoon"),
    ("230", 230): ("cross-event", "Hurricane Milton"),
    ("at least 16", 16): ("cross-event", "Bosnia"),
    ("at least two", 2): ("cross-event", "Mexico / Hurricane John"),
    ("9", 9): ("non-casualty", "mph"),
    ("25", 25): ("non-casualty", "mph"),
    ("two", 2): ("non-casualty", "two-day period"),
}

# SPECIFIC rivals, not one vague catch-all. Measured on 250 gold training positives
# (EKF_MHT_DESIGN sec 27.8): replacing `quantity` = "a count of things that are not people"
# with concrete rivals halves the confusion rate (28.4% -> 14.0%) and multiplies the median
# margin by 2.5 (+0.283 -> +0.709). A negatively-defined catch-all is the worst possible
# query for a scorer that matches a description against a span.
TYPES = {
    "death toll": "a number of people killed or confirmed dead",
    "wind speed": "how fast the wind was blowing",
    "rainfall": "how much rain or snow fell",
    "distance": "how far apart two places are",
    "elapsed time": "how many days or hours something lasted",
    "cost": "an amount of money in dollars or euros",
    "homes damaged": "a number of houses, homes or buildings damaged or destroyed",
    "people evacuated": "a number of people evacuated, displaced or moved to shelters",
    "power outages": "a number of customers or households without electricity",
}


def type_schema() -> Schema:
    return Schema().entities(TYPES)


def trigger_schema() -> Schema:
    """Named storms/events in the window. `event` is what distinguishes Helene from Katrina."""
    return Schema().entities({
        "event": "the name of a hurricane, storm, typhoon or disaster",
        "place": "a country, state or city",
        "date": "a date, month or year",
    })


def window(text: str, span: str, left: int = 200, right: int = 200):
    i = text.find(span)
    return (None, None) if i < 0 else (text[max(0, i - left):i + len(span) + right], i)


def _entities(out) -> dict:
    """Entity block, tolerating both shapes: the span model returns a dict, the boundary
    joint path wraps it in a single-element list."""
    ents = out.get("entities") or {}
    return ents[0] if isinstance(ents, list) else ents


def scores_for(model, ctx: str, span: str, schema: Schema) -> dict:
    """Per-type score for the span, taking the best-matching mention of each type."""
    out = model.extract(ctx, schema, threshold=0.0, include_confidence=True)
    got: dict = {}
    for tname, items in _entities(out).items():
        best = 0.0
        for it in (items or []):
            txt = it["text"] if isinstance(it, dict) else str(it)
            conf = float(it.get("confidence", 0.0)) if isinstance(it, dict) else 0.0
            if span in txt or txt in span:
                best = max(best, conf)
        got[tname] = best
    return got


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="fastino/gliner2-base-v1")
    ap.add_argument("--tracked", default="datasets/helene2024/_cache/tracked_rollup.json")
    ap.add_argument("--feed", default="datasets/helene2024/_cache/feed.jsonl")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--limit", type=int, default=0, help="0 = all observations")
    args = ap.parse_args()

    d = json.loads(Path(args.tracked).read_text(encoding="utf-8"))
    feed = {round(r["t_hours"], 2): r["text"]
            for r in (json.loads(l) for l in Path(args.feed).open(encoding="utf-8"))}
    obs = [(a, o) for a in d["articles"] for o in a["observations"]
           if o["role"] == "dead" and o["mode"] == "heuristic"]
    if args.limit:
        obs = obs[:args.limit]

    model = AutoExtractor.from_pretrained(args.model, map_location=args.device)
    model.eval()
    tsch, esch = type_schema(), trigger_schema()

    rows = []
    for a, o in obs:
        text = feed.get(round(a["t_hours"], 2), "")
        span = str(o["span"])
        ctx, _ = window(text, span)
        if ctx is None:
            continue
        label, why = LABELLED.get((span, int(o["value"])), ("assumed-ok", ""))
        ts = scores_for(model, ctx, span, tsch)
        ev = model.extract(ctx, esch, threshold=0.3)
        events = [e["text"] if isinstance(e, dict) else str(e)
                  for e in (_entities(ev).get("event") or [])]
        rows.append({"span": span, "value": int(o["value"]), "label": label, "why": why,
                     "types": ts, "events": events})

    def margin(r):
        """death toll minus the strongest competing type -- positive means 'a count'."""
        dt = r["types"].get("death toll", 0.0)
        other = max((v for k, v in r["types"].items() if k != "death toll"), default=0.0)
        return dt - other

    print(f"\n{len(rows)} observations scored\n")
    print(f"{'label':<14}{'n':>4}{'mean type margin':>19}{'min':>8}{'max':>8}"
          f"{'Helene named':>14}")
    for lab in ("assumed-ok", "non-casualty", "cross-event"):
        sub = [r for r in rows if r["label"] == lab]
        if not sub:
            continue
        ms = [margin(r) for r in sub]
        helene = sum(1 for r in sub if any("helene" in e.lower() for e in r["events"]))
        print(f"{lab:<14}{len(sub):>4}{sum(ms)/len(ms):>19.3f}{min(ms):>8.3f}{max(ms):>8.3f}"
              f"{f'{helene}/{len(sub)}':>14}")

    print("\nlabelled cases in detail:")
    for r in rows:
        if r["label"] == "assumed-ok":
            continue
        top = sorted(r["types"].items(), key=lambda kv: -kv[1])[:3]
        print(f"  [{r['label']:<12}] {r['span']!r:<16} margin={margin(r):>6.3f}  "
              f"types={[(k, round(v, 2)) for k, v in top]}")
        print(f"                   why={r['why']!r}  events_found={r['events'][:4]}")

    Path("/tmp/energy_probe.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False),
                                              encoding="utf-8")


if __name__ == "__main__":
    main()

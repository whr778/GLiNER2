"""Which storm does this number belong to? The cross-event half, measured three ways.

Type energies solved the unit-error half (4/4, 0/83 false positives) and did nothing at all
for cross-event -- 0/11 -- because the type is RIGHT there: Hurricane Katrina's 1,400 scores
`death toll` 0.95. What is wrong is the EVENT it belongs to.

And the obvious test is the wrong one. "Is Helene named in the window" fires for 8/11
cross-event cases against 37/83 clean ones, because every window in a Helene feed mentions
Helene. Presence of the right event says nothing; what matters is whether a COMPETING one is
bound more strongly.

Three ways to ask, cheapest first, because the cheap one keeps winning in this project:

    A. nearest       the named event closest in characters                    no model call
    B. only-competitor  a competing event is named and ours is not            no model call
    C. bound         a structure query binds (dead, event) as one instance    one model call

C is the interesting one: it reuses the record machinery that already binds `location`, so
if it works it needs no new component. Note the trigger is NOT a record anchor
(RECORD_TASK_TYPES is ("json_structures",)), so this asks the model to bind an event as an
ordinary field rather than exploiting any event-specific path.

    uv run python tools/ekf_showcase/event_binding_probe.py
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from gliner2 import AutoExtractor, Schema

OURS = re.compile(r"\bhelene\b", re.I)

# Same audit labels as energy_probe. Keyed on (span, value) -- see the caveat there; the
# 230s and the "two"s are over-assigned, so per-case detail matters more than the totals.
LABELLED = {
    ("1,400", 1400): "cross-event",
    ("250", 250): "cross-event",
    ("230", 230): "cross-event",
    ("at least 16", 16): "cross-event",
    ("at least two", 2): "cross-event",
    ("9", 9): "non-casualty",
    ("25", 25): "non-casualty",
    ("two", 2): "non-casualty",
}


def event_schema() -> Schema:
    return Schema().entities({
        "event": "the name of a hurricane, storm, typhoon or disaster",
    })


def binding_schema() -> Schema:
    """(dead, event) as ONE record instance -- the same shape that already binds location."""
    return (Schema().structure("casualty_report", mode="natural", anchor="dead")
            .field("dead", dtype="str",
                   description="number of people killed or confirmed dead")
            .field("event", dtype="str",
                   description="the hurricane, storm or disaster that caused these deaths"))


def window(text: str, span: str, left: int = 200, right: int = 200):
    i = text.find(span)
    return (None, -1) if i < 0 else (text[max(0, i - left):i + len(span) + right],
                                     min(i, left))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="fastino/gliner2-base-v1")
    ap.add_argument("--tracked", default="datasets/helene2024/_cache/tracked_rollup.json")
    ap.add_argument("--feed", default="datasets/helene2024/_cache/feed.jsonl")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    d = json.loads(Path(args.tracked).read_text(encoding="utf-8"))
    feed = {round(r["t_hours"], 2): r["text"]
            for r in (json.loads(l) for l in Path(args.feed).open(encoding="utf-8"))}
    obs = [(a, o) for a in d["articles"] for o in a["observations"]
           if o["role"] == "dead" and o["mode"] == "heuristic"]

    model = AutoExtractor.from_pretrained(args.model, map_location=args.device)
    model.eval()
    esch, bsch = event_schema(), binding_schema()

    rows = []
    for a, o in obs:
        text = feed.get(round(a["t_hours"], 2), "")
        span = str(o["span"])
        ctx, at = window(text, span)
        if ctx is None:
            continue
        ents = model.extract(ctx, esch, threshold=0.3, include_spans=True)
        block = ents.get("entities") or {}
        block = block[0] if isinstance(block, list) else block
        found = []
        for e in (block.get("event") or []):
            t = e["text"] if isinstance(e, dict) else str(e)
            pos = e.get("start", ctx.find(t)) if isinstance(e, dict) else ctx.find(t)
            found.append((t, pos))

        ours = [(t, p) for t, p in found if OURS.search(t)]
        comp = [(t, p) for t, p in found if not OURS.search(t)]

        # A. nearest named event by character distance
        nearest = min(found, key=lambda tp: abs(tp[1] - at))[0] if found else None
        # B. a competitor is named and ours is not
        only_comp = bool(comp) and not ours
        # C. what the record head binds as this number's event
        bound = None
        recs = model.extract(ctx, bsch).get("casualty_report") or []
        for r in recs:
            if str(r.get("dead") or "").strip() and span in str(r.get("dead")):
                bound = str(r.get("event") or "").strip() or None
                break
        if bound is None and recs:
            bound = str(recs[0].get("event") or "").strip() or None

        rows.append({
            "span": span, "value": int(o["value"]),
            "label": LABELLED.get((span, int(o["value"])), "assumed-ok"),
            "events": [t for t, _ in found],
            "nearest": nearest, "only_competitor": only_comp, "bound": bound,
        })

    def flags(r):
        return {
            "A nearest is a competitor": bool(r["nearest"]) and not OURS.search(r["nearest"]),
            "B only a competitor named": r["only_competitor"],
            "C bound event is a competitor": bool(r["bound"]) and not OURS.search(r["bound"]),
        }

    print(f"\n{len(rows)} observations\n")
    ce = [r for r in rows if r["label"] == "cross-event"]
    ok = [r for r in rows if r["label"] == "assumed-ok"]
    print(f"{'signal':<32}{'catches cross-event':>21}{'false positives':>18}")
    for key in ("A nearest is a competitor", "B only a competitor named",
                "C bound event is a competitor"):
        c = sum(1 for r in ce if flags(r)[key])
        f = sum(1 for r in ok if flags(r)[key])
        print(f"{key:<32}{f'{c}/{len(ce)}':>21}{f'{f}/{len(ok)} = {f/max(len(ok),1):.1%}':>18}")

    print("\ncross-event cases in detail:")
    for r in ce:
        print(f"  {r['span']!r:<14} nearest={str(r['nearest'])[:22]!r:<24} "
              f"bound={str(r['bound'])[:22]!r:<24} events={r['events'][:3]}")

    Path("/tmp/event_binding.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()

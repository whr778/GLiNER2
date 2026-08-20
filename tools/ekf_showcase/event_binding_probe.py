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

**C was unsound before 2026-08-17 and its published numbers should not be reused.** It
scored ``bool(bound) and not OURS.search(bound)``, so it fired on any non-empty string
lacking "helene" -- including strings that name no event. A casualty-trained record head
(`gliner2-base-v1-casualty-docee`) has no `event` field in distribution and copies the anchor
number in, so `'230'` scored as a caught cross-event and the signal read 9/11 on pure
artifact. Three fixes, all in this file:

1. A binding counts only if it names an event the event schema *also* found
   (``validate_binding``). Raw bindings are still reported, marked unsound, so the artifact
   stays visible rather than silently dropping out.
2. The ``recs[0]`` fallback is gone -- it attributed the first record's event to a span that
   matched no record, inventing a binding the model never made.
3. "bound nothing" is reported separately from "bound ours" and "bound a competitor". The
   boundary arm scored 0/11 because it bound *nothing*, which the old table could not
   distinguish from binding correctly.

    uv run python tools/ekf_showcase/event_binding_probe.py
    uv run python tools/ekf_showcase/event_binding_probe.py --model whr778/gliner2-base-v1-casualty-docee
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

from gliner2 import AutoExtractor, Schema

OURS = re.compile(r"\bhelene\b", re.I)

# Per-OCCURRENCE audit labels, assigned by reading each context (2026-08-20).
#
# They replace a (span, value) string match that was **27% correct on its own positive
# class** -- 3 of 11 -- and missed half the genuine cases:
#
#   '230' x6   marked cross-event (Milton). Every one is Helene's OWN national total;
#              the truth is 228, and the windows read "Helene death toll hits 230".
#   '250'      marked cross-event (a typhoon). Also Helene's national figure -- the
#              typhoon appears only in a RELATED COVERAGE sidebar further down.
#   '1,400' x2 one is Katrina (correct); the other is "causing 1,400 LANDSLIDES".
#   missed     Maria's 3,000, the 1916 hurricanes' 80, a Taiwan typhoon's "dozens".
#
# The consequence is that every published score on this instrument is uninterpretable:
# a detector reading 3/11 might have found exactly the three real cases or three of the
# eight false ones, and nothing distinguished those outcomes.
LABEL_FILE = Path(__file__).with_name("helene_audit_labels.json")


def _ctx_key(context: str, value: int) -> str:
    """sha1 over the value and the normalized context window.

    Keyed on context, not on the value, so a label survives re-extraction: the feed text
    is fixed, so the same figure in the same passage keys identically in any run.
    """
    norm = re.sub(r"\s+", " ", context).strip().lower()
    return hashlib.sha1(f"{value}|{norm}".encode("utf-8")).hexdigest()[:16]


def load_labels() -> dict:
    if not LABEL_FILE.is_file():
        return {}
    return json.loads(LABEL_FILE.read_text(encoding="utf-8")).get("labels", {})


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


def fill_schema() -> Schema:
    """A SEPARATE schema for field-fill measurement, so signal C is untouched.

    Location fill is the Track A guard -- location supervision took it from 26/33 to 51/54 --
    but it was computed by hand and no committed tool reproduced it, which made the guard
    unusable. Measuring it here keeps one instrument for both readouts. This runs its own
    ``extract`` call rather than adding a field to ``binding_schema``: adding one would
    change the record the C signal reads and silently break comparability with every C
    number already published.
    """
    return (Schema().structure("casualty_report", mode="natural", anchor="dead")
            .field("dead", dtype="str",
                   description="number of people killed or confirmed dead")
            .field("location", dtype="str",
                   description="the place these deaths occurred")
            .field("event", dtype="str",
                   description="the hurricane, storm or disaster that caused these deaths"))


_NUMERIC = re.compile(r"^[\d\s.,%-]+$")


def _norm(s: str) -> str:
    return re.sub(r"\W+", " ", s or "").strip().lower()


def validate_binding(bound: str | None, found: list[str]) -> str | None:
    """Keep a binding only if it actually names an event the schema also found.

    Without this the C signal counts any string that is not "helene" -- including
    the casualty number a casualty-trained record head copies into an unfamiliar
    `event` field. Returns the binding, or None if it names no event.
    """
    if not bound or _NUMERIC.match(bound):
        return None
    b = _norm(bound)
    if not b:
        return None
    return bound if any((n := _norm(t)) and (n in b or b in n) for t in found) else None


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

    audit = load_labels()
    rows = []
    contexts = []
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
        # C. what the record head binds as THIS number's event. No recs[0] fallback:
        # attributing another record's event to a span that matched none invents a
        # binding the model did not make.
        bound_raw = None
        recs = model.extract(ctx, bsch).get("casualty_report") or []
        for r in recs:
            if str(r.get("dead") or "").strip() and span in str(r.get("dead")):
                bound_raw = str(r.get("event") or "").strip() or None
                break
        bound = validate_binding(bound_raw, [t for t, _ in found])

        contexts.append(ctx)
        rows.append({
            "span": span, "value": int(o["value"]),
            "label": (audit.get(_ctx_key(ctx, int(o["value"])), {})
                      .get("label", "unlabelled")),
            "events": [t for t, _ in found], "nearest": nearest,
            "only_competitor": only_comp, "bound": bound, "bound_raw": bound_raw,
        })

    def flags(r):
        return {
            "A nearest is a competitor": bool(r["nearest"]) and not OURS.search(r["nearest"]),
            "B only a competitor named": r["only_competitor"],
            "C bound event is a competitor": bool(r["bound"]) and not OURS.search(r["bound"]),
            "C-raw (UNSOUND, diagnostic)": bool(r["bound_raw"]) and not OURS.search(r["bound_raw"]),
        }

    def outcome(r):
        if not r["bound"]:
            return "unbound" if not r["bound_raw"] else "rejected"
        return "ours" if OURS.search(r["bound"]) else "competitor"

    print(f"\n{len(rows)} observations\n")
    from collections import Counter
    print("audit classes:", dict(Counter(r["label"] for r in rows)))
    ce = [r for r in rows if r["label"] == "cross-event"]
    # The clean class is GENUINE HELENE casualties. Non-casualty numbers and unclear cases
    # are excluded from the false-positive denominator rather than folded into it: flagging
    # "1,400 landslides" is not a false positive for cross-event detection, it is a right
    # answer to a different question, and counting it as clean was part of what made the
    # old instrument unreadable.
    ok = [r for r in rows if r["label"] == "helene"]
    print(f"{'signal':<32}{'catches cross-event':>21}{'FP on genuine Helene':>22}")
    for key in ("A nearest is a competitor", "B only a competitor named",
                "C bound event is a competitor", "C-raw (UNSOUND, diagnostic)"):
        c = sum(1 for r in ce if flags(r)[key])
        f = sum(1 for r in ok if flags(r)[key])
        print(f"{key:<32}{f'{c}/{len(ce)}':>21}"
              f"{f'{f}/{len(ok)} = {f/max(len(ok),1):.1%}':>22}")

    # A signal that never binds scores 0 catches AND 0 false positives, which reads as
    # a clean sheet. Coverage tells the two apart.
    print(f"\nC binding coverage        {'cross-event':>14}{'genuine':>14}")
    for state in ("competitor", "ours", "rejected", "unbound"):
        print(f"  {state:<24}{sum(1 for r in ce if outcome(r) == state):>14}"
              f"{sum(1 for r in ok if outcome(r) == state):>14}")
    print("  rejected = the model named something that is not an event the schema found")

    # Fill runs as a SECOND PASS, after every C call has been made. Interleaving it inside
    # the main loop is not equivalent and was measured not to be: an extra extract between
    # observations moved fastino's C from 25/83 to 26/83. The C sequence above is therefore
    # byte-for-byte the call sequence that produced every published C number.
    fsch = fill_schema()
    fill = {"records": 0, "location": 0, "event": 0}
    for ctx in contexts:
        for r in model.extract(ctx, fsch).get("casualty_report") or []:
            fill["records"] += 1
            for f in ("location", "event"):
                if str(r.get(f) or "").strip():
                    fill[f] += 1

    n = fill["records"]
    print(f"\nfield fill over {n} records the model produced on these windows")
    for f in ("location", "event"):
        pct = f"{fill[f] / n:.1%}" if n else "n/a"
        print(f"  {f:<10}{fill[f]:>4}/{n:<4} {pct}")
    print("  measured on a separate schema in a SECOND PASS, so signal C above is untouched.")
    print("  This is the Track A guard, which until now had no committed tool. Its "
          "denominator is\n  records-produced-on-these-windows and does NOT match the "
          "hand-computed 51/54, which\n  counted a different window set -- compare the "
          "RATE, not the fraction.")

    print("\ncross-event cases in detail:")
    for r in ce:
        raw = "" if r["bound_raw"] == r["bound"] else f" raw={str(r['bound_raw'])[:18]!r}"
        print(f"  {r['span']!r:<14} nearest={str(r['nearest'])[:22]!r:<24} "
              f"bound={str(r['bound'])[:22]!r:<24} events={r['events'][:3]}{raw}")

    Path("/tmp/event_binding.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()

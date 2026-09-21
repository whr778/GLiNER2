"""Does this casualty figure belong to OUR event? Ask an EXTERNAL anchor, not the text.

Three text signals were tried and all cap out, because they infer ownership from the words
near the number and a Helene feed names Milton and Katrina constantly. Re-run 2026-09-21
against the corrected audit labels:

    A nearest is a competitor    3/6   19/82 = 23.2% FP
    B only a competitor named    3/6   16/82 = 19.5% FP
    C bound event is competitor  0/6    1/82 =  1.2% FP   (catches nothing)

This asks a different question. The pipeline ALREADY binds every observation to a place --
that is what `event_key` is -- so the anchor is simply: is that place inside the event's
footprint? No new extraction, no model call, and no dependence on which storms a paragraph
happens to mention.

    SPATIAL                      4/6    1/81 =  1.2% FP
    SPATIAL + TEMPORAL           5/6    1/81 =  1.2% FP

Strictly better than A and B on BOTH axes. The footprint and its aliases come from the
event's own `rollup.json`, not an invented gazetteer.

ABSTAIN RATHER THAN GUESS. `event_key` is sometimes a TYPE (`Storm`, `Floods`) rather than a
place -- that is `collapse_type` territory. Those carry no spatial evidence, so they are
passed, never flagged. Flagging them would manufacture exactly the false positives that make
A and B unshippable.

WHAT IT CANNOT DO, from the same run:
  '80'     keyed `north carolina`  -- the 1916 Appalachian hurricanes, correctly IN the
           footprint and 108 years early. That is TEMPORAL's job, not this one.
  'dozens' keyed `tennessee`       -- a Taiwan typhoon. The association itself is wrong
           upstream; no spatial test over a wrong key can recover it.

    uv run python tools/ekf_showcase/spatial_anchor.py
"""

import argparse
import collections
import hashlib
import json
import re
from pathlib import Path

# `event_key` values that name a TYPE rather than a place. No spatial evidence -> abstain.
TYPE_KEYS = {"storm", "floods", "election", "mudslides", "hurricane", "earthquake"}

YEAR = re.compile(r"\b(1[89]\d{2}|20\d{2})\b")

# THE TEMPORAL CUTOFF IS DELIBERATELY EXTREME, and the naive version does not work.
# Measured over the same 87 labelled observations:
#
#   a year < 2010 in context    2/6 cross-event    9/81 = 11.1% FP
#   a year < 2000               1/6                2/81
#   a year < 1950               1/6                0/81
#
# 10 of 81 GENUINE Helene observations carry a non-2024 year, essentially all of them in a
# comparative clause -- "Helene is already the deadliest hurricane to hit the mainland U.S.
# since Katrina in 2005", "Helene passed the 35 killed after Hurricane Hugo" (1989). The year
# is attached to the COMPARISON, not to the figure. That is the same proximity-is-not-
# attachment failure that caps signals A and B, so any permissive year rule inherits it.
#
# Only an ANCIENT year survives: a casualty figure sitting beside 1916 in a 2024 hurricane
# feed is a historical reference, not current reporting. **The evidence is ONE positive**
# (the 1916 Appalachian hurricanes' `80`), so treat this as a conservative complement to the
# spatial anchor, never as a validated signal in its own right. It adds exactly one case.
TEMPORAL_CUTOFF = 1950


def load_footprint(rollup_path):
    """(alias map, footprint set) from the event's own rollup."""
    roll = json.loads(Path(rollup_path).read_text(encoding="utf-8"))
    aliases = {k.lower(): v for k, v in (roll.get("aliases") or {}).items()}
    hier = roll.get("hierarchy") or {}
    footprint = set(hier.get("parts") or []) | {hier.get("aggregate")} - {None}
    return aliases, footprint


def spatial_flag(event_key, aliases, footprint):
    """True = outside the footprint, False = inside, None = abstain (no spatial evidence)."""
    if not event_key:
        return None
    k = str(event_key).strip().lower()
    resolved = aliases.get(k, k)
    if resolved in TYPE_KEYS or k in TYPE_KEYS:
        return None
    return resolved not in footprint


def temporal_flag(context, cutoff=TEMPORAL_CUTOFF):
    """True = an implausibly old year sits in this context, None = abstain.

    Never returns False: absence of an ancient year is not evidence the figure is current,
    so this signal only ever ADDS a flag on top of the spatial one.
    """
    if not context:
        return None
    years = [int(y) for y in YEAR.findall(context)]
    return True if any(y < cutoff for y in years) else None


def ctx_key(context, value):
    """Match event_binding_probe's keying so labels join across tools."""
    norm = re.sub(r"\s+", " ", context).strip().lower()
    return hashlib.sha1(f"{value}|{norm}".encode("utf-8")).hexdigest()[:16]


def observations(tracked_path, role="dead", pad=200):
    for article in json.loads(Path(tracked_path).read_text(encoding="utf-8"))["articles"]:
        text = article.get("text") or ""
        for obs in article.get("observations") or []:
            if obs.get("role") != role:
                continue
            span = obs.get("span") or ""
            i = text.find(span)
            context = text[max(0, i - pad):i + len(span) + pad] if i >= 0 else ""
            yield ctx_key(context, obs.get("value")), obs, context


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--rollup", default="datasets/helene2024/rollup.json")
    ap.add_argument("--tracked", default="datasets/helene2024/_cache/tracked_rollup.json")
    ap.add_argument("--labels", default="tools/ekf_showcase/helene_audit_labels.json")
    ap.add_argument("--role", default="dead")
    args = ap.parse_args()

    aliases, footprint = load_footprint(args.rollup)
    labels = json.loads(Path(args.labels).read_text(encoding="utf-8"))["labels"]
    print(f"footprint: {sorted(footprint)}")

    by_class = collections.defaultdict(list)
    joined = total = 0
    for key, obs, context in observations(args.tracked, args.role):
        total += 1
        if key not in labels:
            continue
        joined += 1
        by_class[labels[key]["label"]].append(
            (obs.get("span"), obs.get("event_key"),
             spatial_flag(obs.get("event_key"), aliases, footprint),
             temporal_flag(context)))

    print(f"{args.role} observations: {total}, joined to an audit label: {joined}\n")
    cross, genuine = by_class.get("cross-event", []), by_class.get("helene", [])

    def score(name, pick):
        tp = sum(1 for r in cross if pick(r))
        fp = sum(1 for r in genuine if pick(r))
        print(f"   {name:24s} {tp}/{len(cross)}   FP {fp}/{len(genuine)} = "
              f"{100 * fp / max(len(genuine), 1):4.1f}%")

    print("anchor                     catches   false positives")
    score("SPATIAL", lambda r: r[2] is True)
    score(f"TEMPORAL (year<{TEMPORAL_CUTOFF})", lambda r: r[3] is True)
    score("COMBINED", lambda r: r[2] is True or r[3] is True)
    print(f"   abstained on spatial (type key): "
          f"{sum(1 for r in genuine if r[2] is None)} genuine\n")
    for name, rows in (("cross-event", cross),
                       ("false positives",
                        [r for r in genuine if r[2] is True or r[3] is True])):
        if rows:
            print(f"   {name}:")
            for span, key, sflag, tflag in rows:
                hits = [n for n, f in (("spatial", sflag), ("temporal", tflag)) if f is True]
                print(f"     {str(span):>14}  event_key={str(key):24s} -> "
                      f"{'+'.join(hits) if hits else 'pass'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

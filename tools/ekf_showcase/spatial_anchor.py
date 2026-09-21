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
            yield ctx_key(context, obs.get("value")), obs


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
    for key, obs in observations(args.tracked, args.role):
        total += 1
        if key not in labels:
            continue
        joined += 1
        by_class[labels[key]["label"]].append(
            (obs.get("span"), obs.get("event_key"),
             spatial_flag(obs.get("event_key"), aliases, footprint)))

    print(f"{args.role} observations: {total}, joined to an audit label: {joined}\n")
    cross, genuine = by_class.get("cross-event", []), by_class.get("helene", [])
    caught = sum(1 for r in cross if r[2] is True)
    fp = sum(1 for r in genuine if r[2] is True)
    abstain = sum(1 for r in genuine if r[2] is None)
    print(f"SPATIAL anchor")
    print(f"   catches cross-event : {caught}/{len(cross)}")
    print(f"   FP on genuine       : {fp}/{len(genuine)} = "
          f"{100 * fp / max(len(genuine), 1):.1f}%")
    print(f"   abstained (type key, no spatial evidence): {abstain} genuine\n")
    for name, rows in (("cross-event", cross), ("false positives", 
                       [r for r in genuine if r[2] is True])):
        if rows:
            print(f"   {name}:")
            for span, key, flag in rows:
                verdict = "FLAG" if flag else ("abstain" if flag is None else "pass")
                print(f"     {str(span):>14}  event_key={str(key):24s} -> {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

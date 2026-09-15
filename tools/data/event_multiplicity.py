"""Price event-instance multiplicity against the KEY a decoder uses to address instances.

An event decoder can only emit as many instances as it has distinct slots, and each design
picks something to key those slots by. Whatever the key is, two gold instances sharing one
key value collapse into one -- so the cost of a design is measurable from the gold alone,
before any model runs.

Three keys are priced here, because three are on the table:

  event_type        what the MENTION path uses today. Its collapse rate is the 64.2%
                    quoted throughout EVENT_ARGUMENT_DIAGNOSIS.
  trigger           what OneIE uses -- one node per trigger SPAN. Named in its own
                    remaining-error distribution as "multiple events per trigger".
  (type, trigger)   the pair, i.e. what a design keyed on both would still lose.

AND A CAVEAT THAT IS ITS OWN FINDING: our corpora store triggers as SURFACE STRINGS, not
offsets. OneIE keys on the span -- two occurrences of the same word are two nodes. Keyed on
the surface they are one, so the `trigger` column here is a LOWER BOUND on what a span-keyed
design would recover from this data as it currently stands, and the gap is the annotation
that would have to be added.

The instrument that produced 64.2% was run inline and lost. This one is committed.

    uv run python tools/data/event_multiplicity.py --config tools/train/config/base/eb16-rebuild-tr.yaml
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import yaml

KEYS = ("event_type", "trigger", "type+trigger")


def _key(ev: dict, which: str):
    t = ev.get("event_type")
    trig = tuple(sorted(ev.get("triggers") or []))
    return {"event_type": t, "trigger": trig, "type+trigger": (t, trig)}[which]


def score(path: Path) -> Counter:
    """Per-document, count instances that SHARE a key value with another instance."""
    c = Counter()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        evs = (json.loads(line).get("output") or {}).get("events") or []
        evs = [e for e in evs if isinstance(e, dict)]
        if not evs:
            continue
        c["docs"] += 1
        c["instances"] += len(evs)
        c["multi_trigger_events"] += sum(1 for e in evs if len(e.get("triggers") or []) > 1)
        c["no_trigger_events"] += sum(1 for e in evs if not (e.get("triggers") or []))
        for which in KEYS:
            groups = defaultdict(int)
            for e in evs:
                groups[_key(e, which)] += 1
            # An instance is unaddressable if another instance in the doc shares its key.
            c[which] += sum(n for n in groups.values() if n > 1)
    return c


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--split", default="test", choices=("train", "val", "test"))
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    data = cfg.get("data") or cfg
    # `event_files` is the per-corpus split map; `corpora` is a bare prefix list, so
    # fall back to <prefix>.<split>.jsonl for corpora the map does not name.
    datasets = dict(data.get("event_files") or {})
    for prefix in data.get("corpora") or []:
        datasets.setdefault(Path(prefix).name,
                            {s: f"{prefix}.{s}.jsonl" for s in ("train", "val", "test")})

    total = Counter()
    rows = []
    for name, spec in sorted(datasets.items()):
        p = spec.get(args.split) if isinstance(spec, dict) else None
        if not p or not Path(p).is_file():
            continue
        c = score(Path(p))
        if not c["instances"]:
            continue
        total.update(c)
        rows.append((name, c))

    def pct(c, k):
        return 100.0 * c[k] / c["instances"] if c["instances"] else 0.0

    print(f"EVENT-INSTANCE MULTIPLICITY on the {args.split} split of {args.config}")
    print(f"the % is gold instances that SHARE a key value with another instance in the "
          f"same document -- i.e. instances the key cannot address separately\n")
    print(f"{'corpus':22s}{'docs':>7}{'instances':>11}"
          + "".join(f"{k:>15}" for k in KEYS))
    for name, c in sorted(rows, key=lambda r: -r[1]["instances"]):
        print(f"{name:22s}{c['docs']:>7,}{c['instances']:>11,}"
              + "".join(f"{pct(c, k):>14.1f}%" for k in KEYS))
    print("-" * (40 + 15 * len(KEYS)))
    print(f"{'ALL':22s}{total['docs']:>7,}{total['instances']:>11,}"
          + "".join(f"{pct(total, k):>14.1f}%" for k in KEYS))
    print(f"\nevents carrying >1 trigger : {total['multi_trigger_events']:,}")
    print(f"events carrying NO trigger : {total['no_trigger_events']:,}")
    print("\ntriggers here are SURFACE STRINGS, so the `trigger` column is a LOWER bound on\n"
          "what a span-keyed (offset) design would separate -- see the module docstring.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

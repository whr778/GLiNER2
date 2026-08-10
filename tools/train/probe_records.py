"""Score record extraction: instance separation, value binding, and field FILL.

The blind test cannot decide the natural-vs-anchorless question -- it has no structure
data, so both arms would score identically on it while differing entirely on the thing
being tested. This probes the record head directly.

Three things are measured separately, because they fail separately and the 2026-08-10
warm start failed exactly one of them:

  instances  did the model emit one record per incident (not one blended record)
  value      is each record's `dead` the figure that belongs to it
  LOCATION   is the non-numeric field filled, and filled correctly

That last is the whole point. The previous run emitted correct multi-instance records with
`location: None` every time, so a metric that averaged the three would have called it a
success.

Cases are REAL text with externally sourced answers (the Turkiye-Syria and Helene
validations) plus minimal constructed sentences that isolate one behaviour each. A model is
queried with the SAME declared schema regardless of the mode it was trained with -- the
schema is the caller's contract, not the training detail.

    uv run python tools/train/probe_records.py --model <path> [--model <path> ...]
"""
from __future__ import annotations

import argparse
import re
from typing import Optional

from gliner2 import AutoExtractor, Schema

# (text, [(dead, location_substring), ...]) -- location matched leniently by substring,
# so "Turkey" counts for "in Turkey" but a number never does.
CASES = [
    ("At least 41,000 deaths have been reported in Turkey, while 5,800 people have died "
     "in Syria. The death toll is likely to keep rising.",
     [("41,000", "turk"), ("5,800", "syri")]),
    ("The storm killed 12 people in Florida and 33 in Georgia, authorities confirmed.",
     [("12", "florida"), ("33", "georgia")]),
    ("Officials said 27 people were killed when the bus overturned near Adana.",
     [("27", "adana")]),
    ("Rescuers in Derna recovered 84 bodies, while 19 died in Benghazi.",
     [("84", "derna"), ("19", "benghazi")]),
    ("A total of 227 deaths were confirmed across six states, including 120 in North "
     "Carolina and 17 in Tennessee.",
     [("120", "north carolina"), ("17", "tennessee")]),
]


def declared_schema():
    s = Schema()
    st = s.structure("casualty_report", mode="natural", anchor="dead")
    st.field("dead", dtype="str", cardinality="required_one",
             description="number of people killed or confirmed dead")
    st.field("location", dtype="str", cardinality="optional_one",
             description="the country, state or place these deaths occurred in")
    return s


def digits(x) -> Optional[str]:
    d = re.sub(r"[^\d]", "", str(x or ""))
    return d or None


def cell(v):
    if isinstance(v, dict):
        return v.get("text")
    if isinstance(v, (list, tuple)):
        return cell(v[0]) if v else None
    return v


def score(model, threshold: float):
    tot = {"expected": 0, "instances": 0, "value": 0, "loc_filled": 0, "loc_correct": 0}
    lines = []
    for text, gold in CASES:
        recs = model.extract(text, declared_schema(), threshold=threshold).get("casualty_report") or []
        recs = [{k: cell(v) for k, v in r.items()} for r in recs]
        tot["expected"] += len(gold)
        tot["instances"] += min(len(recs), len(gold))
        for want_dead, want_loc in gold:
            hit = next((r for r in recs if digits(r.get("dead")) == digits(want_dead)), None)
            if hit is None:
                continue
            tot["value"] += 1
            loc = str(hit.get("location") or "")
            if loc:
                tot["loc_filled"] += 1
                if want_loc in loc.lower():
                    tot["loc_correct"] += 1
        lines.append(f"    {text[:52]:<54} -> {recs}")
    return tot, lines


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", action="append", required=True)
    ap.add_argument("--threshold", type=float, default=0.3)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    print(f"{'model':<34}{'instances':>11}{'value':>9}{'loc filled':>12}{'loc correct':>13}")
    for path in args.model:
        m = AutoExtractor.from_pretrained(path, map_location=args.device,
                                          attn_implementation="sdpa")
        t, lines = score(m, args.threshold)
        e = max(t["expected"], 1)
        name = path.rstrip("/").split("/")[-1]
        print(f"{name[:34]:<34}{t['instances']}/{e:<9}{t['value']}/{e:<7}"
              f"{t['loc_filled']}/{e:<10}{t['loc_correct']}/{e:<11}")
        if args.verbose:
            print("\n".join(lines))
        del m
    print("\nloc filled is the discriminator: the 2026-08-10 run scored full marks on "
          "instances\nand value while filling location 0 times.")


if __name__ == "__main__":
    main()

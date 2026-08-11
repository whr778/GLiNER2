"""Is there a query-axis hard negative to mine, or would a loss have nothing to bite on?

Hard negatives are currently mined on the SPAN axis: `select_hard_negative_candidates`
picks the highest-scoring negative *spans* for a given query. The 2026-08-11 error audit
(EKF_MHT_DESIGN sec 27.3) argues the missing axis is QUERY -- for a given span, which
sibling type queries score it highly. `quantity` scoring a genuine death toll higher than
`death toll` does is exactly that failure, and it is why the inference-time rule had to drop
`quantity` and give up 6 catches.

But a loss cannot teach a boundary the data never presents. Before wiring anything, check
the training corpora directly: take GOLD casualty figures and score them under sibling type
queries. Three outcomes, and only one of them justifies the work:

    competitors never win        no query-axis negative exists; a loss has nothing to mine
    competitors sometimes win    a real, minable boundary -- and GIST-style guide filtering
                                 matters, because those wins are not all errors
    competitors usually win      the type queries are not separable at all and the problem
                                 is the type descriptions, not the loss

Uses the SAME competitor descriptions as the inference-time probe so the two measurements
are comparable, including `quantity`, which is the one under suspicion.

    uv run python tools/train/probe_query_negatives.py --limit 300
"""
from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path

from gliner2 import AutoExtractor, Schema

TYPES = {
    "death toll": "a number of people killed or confirmed dead",
    "measurement": "a speed, distance, depth, rainfall or other physical measurement",
    "duration": "a length of time such as a number of days or hours",
    "money": "an amount of money",
    "quantity": "a count of things that are not people, such as homes or customers",
}
ROLES = ("dead", "killed", "deaths")


def gold_rows(path: Path, limit: int, seed: int = 0):
    """Records carrying a gold death figure, with the figure's surface string."""
    rows = []
    for line in path.open(encoding="utf-8"):
        r = json.loads(line)
        for block in (r.get("output") or {}).get("json_structures") or []:
            for _, fields in block.items():
                for role in ROLES:
                    v = fields.get(role)
                    if isinstance(v, str) and v.strip() and re.search(r"\d", v):
                        rows.append({"text": r["input"], "span": v.strip(), "role": role})
    rng = random.Random(seed)
    rng.shuffle(rows)
    return rows[:limit] if limit else rows


def _entities(out) -> dict:
    ents = out.get("entities") or {}
    return ents[0] if isinstance(ents, list) else ents


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="fastino/gliner2-base-v1")
    ap.add_argument("--corpus", default="data/casualty_multi_loc.train.jsonl")
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    rows = gold_rows(Path(args.corpus), args.limit)
    model = AutoExtractor.from_pretrained(args.model, map_location=args.device)
    model.eval()
    schema = Schema().entities(TYPES)

    wins = Counter()
    phys_reject = [0]
    margins = []
    n = 0
    for r in rows:
        out = model.extract(r["text"], schema, threshold=0.0, include_confidence=True)
        best = {}
        for tname, items in _entities(out).items():
            top = 0.0
            for it in (items or []):
                txt = it["text"] if isinstance(it, dict) else str(it)
                conf = float(it.get("confidence", 0.0)) if isinstance(it, dict) else 0.0
                if r["span"] in txt or txt in r["span"]:
                    top = max(top, conf)
            best[tname] = top
        if not best:
            continue
        n += 1
        dt = best.get("death toll", 0.0)
        rival, rscore = max(((k, v) for k, v in best.items() if k != "death toll"),
                            key=lambda kv: kv[1], default=("none", 0.0))
        margins.append(dt - rscore)
        if rscore > dt:
            wins[rival] += 1
        # The rule actually proposed for inference (sec 27.1): physically incompatible
        # competitors only. Counted SEPARATELY from `wins` -- folding it in double-counts
        # rows where quantity wins overall and a physical type also beats `death toll`.
        # Measured on 250 training positives rather than 83 Helene observations, which is
        # the sample that decides whether the reported "0 false positives" holds.
        if max(best.get(k, 0.0) for k in ("measurement", "duration", "money")) > dt:
            phys_reject[0] += 1

    beaten = sum(wins.values())
    print(f"\ncorpus: {args.corpus}")
    print(f"{n} gold death figures scored\n")
    print(f"a competing type OUTSCORES `death toll` on {beaten}/{n} = {beaten/max(n,1):.1%}")
    print("which competitor wins, when one does:")
    for k, v in wins.most_common():
        print(f"    {k:<14}{v:>5}  {v/max(beaten,1):>6.1%}")
    print(f"\nSHIPPED RULE (physical competitors only) would falsely reject "
          f"{phys_reject[0]}/{n} = {phys_reject[0]/max(n,1):.1%} of GENUINE death tolls")
    if margins:
        margins.sort()
        q = lambda p: margins[int(p * (len(margins) - 1))]
        print(f"\nmargin (death toll - best rival)  p10={q(.1):+.3f}  median={q(.5):+.3f}  "
              f"p90={q(.9):+.3f}")
        print(f"negative margins: {sum(1 for m in margins if m < 0)}/{len(margins)}")


if __name__ == "__main__":
    main()

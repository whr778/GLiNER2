"""Near-duplicate gate: eval docs whose text largely appears in a training doc, or in an earlier eval split.

    uv run python tools/data/near_dup_eval.py \
        --eval data/cc_news_events_sonnet55_v2.val.jsonl data/cc_news_events_sonnet55_v2.test.jsonl \
        --against data/cc_news_events_sonnet55_v2.train.jsonl ... --threshold 0.2 \
        --out data/cc_news_events_sonnet55_v2.near_dup_skip.json

Containment = share of an eval doc's 8-word shingles found inside one other doc. Exact-match dedup misses
the same wire story re-published with a different header or ending; this does not. Each eval split is
also checked against the eval splits listed before it, so test cannot near-duplicate val.
Writes {eval_path: [doc indices to skip]}.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path


def inputs(path: str) -> list[str]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line)["input"] for line in fh]


def shingles(text: str, k: int = 8) -> set[int]:
    w = re.findall(r"\w+", text.casefold())
    return {hash(" ".join(w[i:i + k])) for i in range(max(len(w) - k + 1, 1))}


def containment(eval_texts: list[str], others: list[str]) -> list[float]:
    """Per eval doc, the highest share of its shingles inside any single doc of others."""
    ev = [shingles(t) for t in eval_texts]
    index = defaultdict(list)
    for j, s in enumerate(ev):
        for h in s:
            index[h].append(j)
    best = [0.0] * len(ev)
    for t in others:
        count = defaultdict(int)
        for h in shingles(t):
            for j in index.get(h, ()):
                count[j] += 1
        for j, c in count.items():
            best[j] = max(best[j], c / len(ev[j]))
    return best


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--eval", nargs="+", required=True)
    ap.add_argument("--against", nargs="+", required=True)
    ap.add_argument("--threshold", type=float, default=0.2)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    train = [t for p in args.against for t in inputs(p)]
    skip, earlier = {}, []
    for path in args.eval:
        texts = inputs(path)
        best = containment(texts, train + earlier)
        skip[path] = [i for i, b in enumerate(best) if b >= args.threshold]
        print(f"[near-dup] {path}: {len(skip[path])} of {len(texts)} docs >= {args.threshold} containment -> skip")
        earlier += [t for i, t in enumerate(texts) if i not in set(skip[path])]
    args.out.write_text(json.dumps(skip, indent=1, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

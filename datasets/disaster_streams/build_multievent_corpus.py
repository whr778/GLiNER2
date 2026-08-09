"""Multi-event casualty corpus: several incidents per document, one record each.

Why this exists. The original corpus (`build_finetune_corpus.py`) emits exactly ONE
`casualty_report` per document -- verified: all 31,539 examples have instance count 1.
So the count head only ever saw "1", and on a document describing several incidents it
must blend their competing figures into a single forced instance. Measured on the
multi-event showcase feed, value binding collapses from **1.000 on single-event text to
0.369**, with 22.6% of readings bound to the WRONG event's number.

This corpus retrains that component: each document concatenates K snippets from
DIFFERENT streams (different disasters, different figures) and carries **one
casualty_report instance per snippet**.

Three guards, each of which decides whether the experiment means anything:

1. **Train streams only.** The showcase feeds are built from the *test* split; drawing
   interference from test would contaminate the evaluation this corpus exists to move.
2. **Per-snippet span location.** A value is located inside its OWN snippet's slice of
   the concatenated document, not by searching the whole text. Searching globally takes
   the first occurrence, which on a collision labels one event with another's number --
   injecting exactly the cross-event noise being cured. Colliding values are dropped.
3. **K is mixed 0..max.** Single-event documents stay in the mix so the 1.000
   single-event binding does not regress while multi-event improves.

    uv run python datasets/disaster_streams/build_multievent_corpus.py \
        --data datasets/disaster_streams_sonnet5 --split train \
        --out data/casualty_multi.train.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from gliner2.training.data import InputExample, Structure, TrainingDataset

ROLES = ("dead", "injured", "missing")


def _variants(value: int):
    """A number as it may appear in text: plain and comma-grouped."""
    return (str(value), f"{value:,}")


def _locate_in_slice(value: int, doc: str, lo: int, hi: int):
    """The value's verbatim form if it occurs INSIDE [lo, hi) and nowhere else in doc.

    Returning None on a collision is deliberate: an ambiguous number cannot be attributed
    to one instance, and guessing would teach the model the error it is meant to fix.
    """
    for cand in _variants(value):
        inside = doc.find(cand, lo, hi)
        if inside < 0:
            continue
        first, last = doc.find(cand), doc.rfind(cand)
        if first == last:                    # unique in the whole document
            return cand
    return None


def load_snippets(split_dir: Path):
    """(stream_id, t_hours) -> {text, gt{role: value}}"""
    groups = defaultdict(lambda: {"text": "", "gt": {}, "stream": ""})
    for line in (split_dir / "observations.jsonl").open(encoding="utf-8"):
        o = json.loads(line)
        if not o.get("text"):
            continue
        g = groups[(o["stream_id"], o["t_hours"])]
        g["text"] = o["text"]
        g["stream"] = o["stream_id"]
        g["gt"][o["role"]] = o["value"]
    return [g for g in groups.values() if g["text"] and g["gt"]]


def build(snippets, max_interference: int, seed: int):
    rng = random.Random(seed)
    by_stream = defaultdict(list)
    for s in snippets:
        by_stream[s["stream"]].append(s)
    streams = sorted(by_stream)

    examples = []
    stats = {"docs": 0, "instances": 0, "located": 0, "dropped_collision": 0,
             "by_k": defaultdict(int)}

    for focal in snippets:
        k = rng.randint(0, max_interference)
        others = []
        for _ in range(k):
            pool = [s for st in streams if st != focal["stream"] for s in by_stream[st]]
            if pool:
                others.append(rng.choice(pool))

        parts, doc = [focal] + others, ""
        spans = []                       # (snippet, lo, hi) in the concatenated document
        for part in parts:
            lo = len(doc)
            doc += part["text"]
            spans.append((part, lo, len(doc)))
            doc += "\n\n"

        structures = []
        for part, lo, hi in spans:
            fields = {}
            for role, value in part["gt"].items():
                located = _locate_in_slice(value, doc, lo, hi)
                if located is None:
                    stats["dropped_collision"] += 1
                    continue
                fields[role] = located
                stats["located"] += 1
            if fields:
                structures.append(Structure("casualty_report", **fields))

        if structures:
            examples.append(InputExample(text=doc.strip(), structures=structures))
            stats["docs"] += 1
            stats["instances"] += len(structures)
            stats["by_k"][len(structures)] += 1
    return examples, stats


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="datasets/disaster_streams_sonnet5")
    ap.add_argument("--split", default="train")
    ap.add_argument("--out", default="data/casualty_multi.train.jsonl")
    ap.add_argument("--max-interference", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    snippets = load_snippets(Path(args.data) / args.split)
    examples, stats = build(snippets, args.max_interference, args.seed)

    ds = TrainingDataset(examples)
    report = ds.validate(raise_on_error=False)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    ds.save(str(out))

    print(f"[build] split={args.split}  snippets={len(snippets)}")
    print(f"[build] documents={stats['docs']}  instances={stats['instances']} "
          f"(mean {stats['instances'] / max(stats['docs'], 1):.2f}/doc)")
    print(f"[build] fields located={stats['located']}  dropped as ambiguous="
          f"{stats['dropped_collision']}")
    print(f"[build] instances-per-document: {dict(sorted(stats['by_k'].items()))}")
    print(f"[build] validate: {report}")
    print(f"[build] wrote {out}")


if __name__ == "__main__":
    main()

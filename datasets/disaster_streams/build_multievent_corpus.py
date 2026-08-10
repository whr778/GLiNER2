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
import re
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
        hits = [m.start() for m in re.finditer(re.escape(cand), doc) if _standalone(doc, m.start(), m.end())]
        if len(hits) == 1 and lo <= hits[0] < hi:      # unique AND inside this snippet
            return cand
    return None


def _locate_place(place: str, doc: str, lo: int, hi: int):
    """The place as it appears INSIDE this snippet's slice, or None.

    Why a location field at all. The casualty fine-tune saw only numeric fields
    (dead/injured/missing/source) and collapsed field semantics toward "emit a digit" --
    asked for a `location` it returns the number. Measured on real text, that is what
    breaks attribution: the base model binds 41,000->Turkey and 5,800->Syria correctly,
    and the fine-tune cannot. Heterogeneous field types are the fix, so the corpus has to
    carry a non-numeric field.

    Two guards, mirroring ``_locate_in_slice``:

    1. Every occurrence in the document must fall inside THIS snippet, so a place shared
       with an interfering snippet is dropped rather than attributed to the wrong event.
    2. The full place is preferred; failing that the longest comma-separated component
       that appears. Measured on the 250-stream corpus: 84.7% verbatim, +8.3% by
       component, 7.0% absent -- and the absent ones get no location rather than a guess.
    """
    parts = [place] + sorted(
        (p.strip() for p in place.replace(" in ", " , ").split(",") if len(p.strip()) > 3),
        key=len, reverse=True)
    for cand in parts:
        hits = [m.start() for m in re.finditer(re.escape(cand), doc)]
        if hits and all(lo <= h < hi for h in hits):
            return cand
    return None


def _standalone(doc: str, a: int, b: int) -> bool:
    """True when doc[a:b] is a whole number, not a fragment of a bigger one.

    Without this, gold "66" matches inside "665 residents displaced" and the model is
    taught to read a casualty count out of an unrelated figure. Rare (0.40% of spans)
    but pure label noise, and scaling makes it likelier: small values collide more.
    A comma only continues a number when a digit follows it (1,234 vs "108, while").
    """
    if a > 0 and doc[a - 1].isdigit():
        return False
    if b < len(doc) and doc[b].isdigit():
        return False
    if b + 1 < len(doc) and doc[b] == "," and doc[b + 1].isdigit():
        return False
    if a >= 1 and doc[a - 1] == "," and a >= 2 and doc[a - 2].isdigit():
        return False
    return True


def load_snippets(split_dir: Path, stream_start: int = 0, stream_end: int = 0):
    """(stream_id, t_hours) -> {text, gt{role: value}}

    ``stream_start``/``stream_end`` slice the sorted stream ids. Splitting at the STREAM
    level, not the document level, is what keeps train and val disjoint: one stream's
    snippets all describe the same incident, so letting them straddle a split leaks the
    answer. Interference is drawn only from streams inside the same slice for the same
    reason.
    """
    groups = defaultdict(lambda: {"text": "", "gt": {}, "stream": ""})
    for line in (split_dir / "observations.jsonl").open(encoding="utf-8"):
        o = json.loads(line)
        if not o.get("text"):
            continue
        g = groups[(o["stream_id"], o["t_hours"])]
        g["text"] = o["text"]
        g["stream"] = o["stream_id"]
        g["gt"][o["role"]] = o["value"]
    out = [g for g in groups.values() if g["text"] and g["gt"]]
    if stream_end:
        keep = set(sorted({g["stream"] for g in out})[stream_start:stream_end])
        out = [g for g in out if g["stream"] in keep]
    return out


def build(snippets, max_interference: int, seed: int, contexts: dict | None = None):
    rng = random.Random(seed)
    by_stream = defaultdict(list)
    for s in snippets:
        by_stream[s["stream"]].append(s)
    streams = sorted(by_stream)

    examples = []
    contexts = contexts or {}
    stats = {"docs": 0, "instances": 0, "located": 0, "dropped_collision": 0,
             "located_place": 0, "no_place": 0, "by_k": defaultdict(int)}

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
            place = (contexts.get(part["stream"]) or {}).get("place", "")
            if place:
                located = _locate_place(place, doc, lo, hi)
                if located:
                    fields["location"] = located
                    stats["located_place"] += 1
                else:
                    stats["no_place"] += 1
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
    ap.add_argument("--stream-start", type=int, default=0,
                    help="index into the sorted stream ids (leak-free split)")
    ap.add_argument("--stream-end", type=int, default=0, help="0 = all streams")
    ap.add_argument("--contexts", default="",
                    help="contexts json; adds a gold location FIELD (heterogeneous "
                         "field types stop the numeric-field collapse)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    snippets = load_snippets(Path(args.data) / args.split,
                             args.stream_start, args.stream_end)
    contexts = json.loads(Path(args.contexts).read_text(encoding="utf-8")) if args.contexts else {}
    examples, stats = build(snippets, args.max_interference, args.seed, contexts)

    ds = TrainingDataset(examples)
    report = ds.validate(raise_on_error=False)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    ds.save(str(out))

    print(f"[build] split={args.split}  snippets={len(snippets)}")
    print(f"[build] documents={stats['docs']}  instances={stats['instances']} "
          f"(mean {stats['instances'] / max(stats['docs'], 1):.2f}/doc)")
    print(f"[build] location field: located={stats['located_place']}  absent={stats['no_place']}")
    print(f"[build] fields located={stats['located']}  dropped as ambiguous="
          f"{stats['dropped_collision']}")
    print(f"[build] instances-per-document: {dict(sorted(stats['by_k'].items()))}")
    print(f"[build] validate: {report}")
    print(f"[build] wrote {out}")


if __name__ == "__main__":
    main()

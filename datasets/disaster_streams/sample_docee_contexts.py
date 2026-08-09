"""Sample real disaster contexts from DocEE to condition the stream realizer.

The synthetic streams have exact ground truth but templated surface form: every stream is
"an earthquake" in "the region", magnitude 7.5. That is why association collapses on the
showcase feed -- 227 of 237 observations land in one `Earthquakes` key, because there is
nothing in the text to tell two streams apart.

DocEE supplies what is missing: real event types, real place names, and real casualty
phrasing, from 6,929 annotated articles. Pairing one DocEE context with each synthetic
stream keeps the exact trajectory (the tracker's ground truth) while making the text look
like different real disasters in different real places.

Deterministic and PREFIX-STABLE: contexts are assigned to ``stream_id`` in sorted order,
so sampling 250 streams reproduces the first 60 exactly. That is what lets a larger run
append to a smaller one instead of regenerating -- and paying for -- what already exists.

    uv run python datasets/disaster_streams/sample_docee_contexts.py \
        --streams 60 --out datasets/disaster_streams/contexts.json
"""
from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path

# Types with enough annotated articles to sample from, and a plausible distractor
# detail a real report of that type would carry (the number extraction must NOT bind).
TYPE_DISTRACTOR = {
    "Earthquakes":      ("magnitude", ["6.1", "6.8", "7.2", "7.5", "5.9"]),
    "Floods":           ("river level in metres", ["4.2", "5.6", "3.8", "6.1"]),
    "Air Crash":        ("flight number", ["JT610", "AF447", "MH370", "QZ8501"]),
    "Fire":             ("hectares burned", ["1200", "450", "8600", "300"]),
    "Road Crash":       ("vehicles involved", ["3", "5", "12", "7"]),
    "Gas Explosion":    ("buildings damaged", ["14", "27", "6", "40"]),
    "Mine Collapses":   ("depth in metres", ["320", "180", "540", "250"]),
    "Shipwreck":        ("vessel capacity", ["300", "120", "480", "90"]),
    "Train Collisions": ("carriages derailed", ["4", "8", "3", "11"]),
    "Armed Conflict":   ("districts affected", ["3", "7", "2", "5"]),
}
GENERIC_PLACE = {
    "the region", "the area", "temporary camps", "temporary shelters",
    "hardest-hit areas", "the country", "the city", "the district", "the province",
}
NUM = re.compile(r"\d")


def _clean_place(value) -> str:
    text = (value.get("text") if isinstance(value, dict) else value) or ""
    text = str(text).strip()
    # A place must name a place: reject generic spans and anything starting with a digit
    # (DocEE's Location entity sometimes catches dates).
    if not text or text.lower() in GENERIC_PLACE or text[0].isdigit() or len(text) > 60:
        return ""
    return text


def load_contexts(data_root: Path):
    """(event_type, place, sample casualty phrasing) drawn from annotated DocEE articles."""
    pools = defaultdict(list)
    for split in ("train", "val", "test"):
        p = data_root / f"docee.{split}.jsonl"
        if not p.is_file():
            continue
        for line in p.open(encoding="utf-8"):
            rec = json.loads(line)
            out = rec.get("output") or {}
            etype = next((tl[0] for c in (out.get("classifications") or [])
                          if (tl := c.get("true_label"))), None)
            if etype not in TYPE_DISTRACTOR:
                continue
            ents = out.get("entities") or {}
            places = [q for q in (_clean_place(v) for v in (ents.get("Location") or [])) if q]
            cas = [str(c.get("text") if isinstance(c, dict) else c)
                   for c in (ents.get("Casualties and Losses") or [])]
            cas = [c for c in cas if NUM.search(c)]
            if not places or not cas:
                continue
            pools[etype].append({"event_type": etype, "place": places[0],
                                 "phrasing_example": cas[0][:120]})
    return pools


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="data")
    ap.add_argument("--source", default="datasets/disaster_streams",
                    help="stream root whose split supplies the stream ids")
    ap.add_argument("--split", default="train")
    ap.add_argument("--streams", type=int, default=60)
    ap.add_argument("--out", default="datasets/disaster_streams/contexts.json")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    pools = load_contexts(Path(args.data))
    if not pools:
        raise SystemExit("no usable DocEE contexts found")

    obs_path = Path(args.source) / args.split / "observations.jsonl"
    stream_ids = sorted({json.loads(l)["stream_id"] for l in obs_path.open(encoding="utf-8")})
    chosen_ids = stream_ids[: args.streams]

    types = sorted(pools)
    contexts = {}
    for i, sid in enumerate(chosen_ids):
        # Seed per stream id, not per position: a stream keeps its context no matter how
        # many streams are requested, so a bigger run appends rather than reshuffles.
        rng = random.Random(f"{args.seed}:{sid}")
        etype = types[i % len(types)]          # round-robin keeps types balanced
        ctx = dict(rng.choice(pools[etype]))
        label, values = TYPE_DISTRACTOR[etype]
        ctx["distractor_label"] = label
        ctx["distractor_value"] = rng.choice(values)
        contexts[sid] = ctx

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(contexts, indent=2, ensure_ascii=False), encoding="utf-8")

    by_type = defaultdict(int)
    for c in contexts.values():
        by_type[c["event_type"]] += 1
    print(f"wrote {out}")
    print(f"  streams        : {len(contexts)}")
    print(f"  distinct places: {len({c['place'] for c in contexts.values()})}")
    print(f"  types          : {dict(sorted(by_type.items()))}")
    print(f"  pool sizes     : {dict(sorted((t, len(v)) for t, v in pools.items()))}")


if __name__ == "__main__":
    main()

"""Build a multi-event news feed from REAL DocEE articles.

Why a second feed. The synthetic feed (`make_demo_feed.py`) is built from templated
disaster snippets whose only "place" is *the region* -- so type+location association has
nothing to work with, and two interfering earthquakes are genuinely indistinguishable
from text. That is a property of the feed, not of the method, and it makes the synthetic
feed the wrong instrument for testing association.

DocEE articles are real news: they name actual places, span 20 disaster types, and carry
gold `Casualties and Losses` spans. Interleaving several incidents gives a feed where
association is *observable* -- which is the regime the pipeline's `--associate` keying
targets, as distinct from the genuinely ambiguous regime that needs MHT.

No casualty trajectory exists for these articles, so this feed carries **no truth
series**: it tests ASSOCIATION (do observations land in the right event stream?), not
tracking accuracy. Each line keeps its source event type under ``_event`` so association
can be scored.

Time is synthesised. DocEE has no timestamps, so articles are laid on a pseudo-clock at a
fixed spacing, interleaved across events so several incidents are live at once -- the
condition that breaks a single-stream tracker.

    uv run python tools/ekf_showcase/make_docee_feed.py \
        --events 3 --per-event 12 --out datasets/ekf_showcase/feed_docee.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path

CASUALTY_TYPES = [
    "Earthquakes", "Floods", "Air Crash", "Fire", "Road Crash",
    "Gas Explosion", "Mine Collapses", "Riot", "Armed Conflict", "Shipwreck",
]
NUM = re.compile(r"\d")


def load_docee(paths):
    """(event_type, article) pairs that carry a casualty span containing a number."""
    by_type = defaultdict(list)
    for p in paths:
        if not p.is_file():
            continue
        for line in p.open(encoding="utf-8"):
            rec = json.loads(line)
            out = rec.get("output") or {}
            label = next((tl[0] for c in (out.get("classifications") or [])
                          if (tl := c.get("true_label"))), None)
            if label not in CASUALTY_TYPES:
                continue
            cas = (out.get("entities") or {}).get("Casualties and Losses") or []
            if not any(NUM.search(str(s)) for s in cas):
                continue
            text = rec.get("input") or ""
            if len(text) < 200:
                continue
            by_type[label].append({"text": text, "casualties": cas,
                                   "locations": (out.get("entities") or {}).get("Location") or []})
    return by_type


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="data")
    ap.add_argument("--events", type=int, default=3, help="distinct incidents in the feed")
    ap.add_argument("--per-event", type=int, default=12, help="articles per incident")
    ap.add_argument("--spacing", type=float, default=6.0, help="hours between articles")
    ap.add_argument("--out", default="datasets/ekf_showcase/feed_docee.jsonl")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    root = Path(args.data)
    by_type = load_docee([root / f"docee.{s}.jsonl" for s in ("test", "val", "train")])
    usable = sorted((t for t, v in by_type.items() if len(v) >= args.per_event),
                    key=lambda t: -len(by_type[t]))
    if len(usable) < args.events:
        raise SystemExit(f"only {len(usable)} types have >= {args.per_event} usable articles")

    rng = random.Random(args.seed)
    chosen = usable[: args.events]
    lines = []
    for slot in range(args.per_event):
        # Interleave: one article per event per round, so every incident stays live.
        for ev_i, etype in enumerate(chosen):
            art = by_type[etype][slot]
            t = (slot * len(chosen) + ev_i) * args.spacing
            lines.append({
                "t_hours": round(t, 2),
                "text": art["text"],
                # Association ground truth; a production feed would not carry it.
                "_event": etype,
                "_casualties": art["casualties"][:4],
            })
    rng.shuffle(lines)
    lines.sort(key=lambda r: r["t_hours"])

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for r in lines:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"wrote {out}")
    print(f"  incidents : {', '.join(chosen)}")
    print(f"  articles  : {len(lines)} ({args.per_event} per incident)")
    print(f"  time span : 0h .. {lines[-1]['t_hours']}h")
    print("  NOTE: no truth trajectory -- this feed tests ASSOCIATION, not tracking error.")


if __name__ == "__main__":
    main()

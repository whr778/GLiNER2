"""Rebuild casualty_docee with the current multi-event builder (the `_standalone` fix).

casualty_docee was built at 9b2764e (2026-08-09 19:37), 18 minutes before c5641a4 made
`_locate_in_slice` match whole numbers only. The pre-fix substring rule let small tolls
collide with dates ("2" inside "2026") and dropped them -- 3,756 train paragraphs stated a
unique toll with no gold -- and also read values out of larger numbers ("7" inside "517").

The original arguments were not recorded; they were RECOVERED by replaying the pre-fix
builder until it reproduced the published v1 byte-for-byte (texts and gold, all splits):
the 157 landed streams of `disaster_streams_docee250/train`, sorted, sliced 125/16/16,
`max_interference=3`, `seed=42`, no contexts (v1 has no `location` field). The rebuild uses
the same arguments, so documents are unchanged except where v1 had dropped or mis-read gold.
v1 stays on the Hub at revision 0ac8531.

    uv run python tools/data/rebuild_casualty_docee.py --out-dir data
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "datasets/disaster_streams"))
sys.path.insert(0, str(ROOT / "tools/data"))
import build_multievent_corpus as B  # noqa: E402
from _split import dumps_record  # noqa: E402

SNIPPETS = ROOT / "datasets/disaster_streams_docee250/train"
SLICES = {"train": (0, 125), "val": (125, 141), "test": (141, 157)}


def landed_streams(v1_dir: Path):
    """The 157 streams v1 was built from, recovered from v1's own paragraphs."""
    stream_of = {}
    for line in open(SNIPPETS / "observations.jsonl", encoding="utf-8"):
        o = json.loads(line)
        if o.get("text"):
            stream_of[o["text"]] = o["stream_id"]
    streams = set()
    for split in SLICES:
        for line in open(v1_dir / f"casualty_docee.{split}.jsonl", encoding="utf-8"):
            streams.update(stream_of[p] for p in json.loads(line)["input"].split("\n\n"))
    return sorted(streams)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--v1-dir", type=Path, required=True,
                    help="directory holding the v1 casualty_docee.{train,val,test}.jsonl")
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    landed = landed_streams(args.v1_dir)
    assert len(landed) == 157, len(landed)
    snippets = B.load_snippets(SNIPPETS)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for split, (a, b) in SLICES.items():
        keep = set(landed[a:b])
        examples, stats = B.build([s for s in snippets if s["stream"] in keep],
                                  max_interference=3, seed=42)
        with open(args.out_dir / f"casualty_docee.{split}.jsonl", "w", encoding="utf-8") as fh:
            for e in examples:
                fh.write(dumps_record(e.to_dict()) + "\n")
        print(f"{split}: {stats['docs']:,} docs, {stats['instances']:,} records, "
              f"{stats['located']:,} fields located, {stats['dropped_collision']:,} dropped as collisions")


if __name__ == "__main__":
    main()

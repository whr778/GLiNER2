"""Repair the record anchor in the purchased casualty corpora.

`annotate_casualty.py` stamped `{"mode": "natural", "anchor": "dead"}` on every row it
wrote, regardless of what the row reported. In natural mode the anchor field's mentions
delimit record instances, so a row that reports only injured, only missing, or only a
place declares an anchor it does not carry -- and the record head raises

    record 'casualty_report' declares anchor 'dead' but no matching field query
    was found in the layout

on the first batch that contains one. That killed the balanced multilingual run at step 0
after 26% of its training rows turned out to be affected.

The writer is fixed at the origin (`pick_anchor`); this repairs what already shipped.
Anchor only: the text, the spans and the field values are byte-identical afterwards, so
splits, hashes and every leakage check stay valid.

    uv run python tools/data/repair_casualty_anchors.py data/cas_ann_en.jsonl ...
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _split import dumps_record  # noqa: E402

ANCHOR_ORDER = ("dead", "injured", "missing", "location")

DEFAULT_FILES = (
    "data/cas_ann_en.jsonl",
    "data/cas_ann_ccnews.jsonl",
    "data/turkish_gate/cas_ann_tr.jsonl",
    "data/chinese_gate/cas_ann_zh.jsonl",
    "data/chinese_gate/cas_ann_zh2.jsonl",
)


def repair(path: Path) -> tuple[int, int, int]:
    """Rewrite ``path`` in place; return (rows, repaired, unanchorable)."""
    rows = repaired = unanchorable = 0
    out = []
    for line in path.open(encoding="utf-8"):
        rec = json.loads(line)
        body = rec.get("output") if isinstance(rec.get("output"), dict) else rec
        meta = (body.get("record_metadata") or {}).get("casualty_report")
        structs = body.get("json_structures") or []
        rows += 1
        if meta:
            present = {f for s in structs for f in (s.get("casualty_report") or {})}
            if meta.get("anchor") not in present:
                anchor = next((f for f in ANCHOR_ORDER if f in present), None)
                if anchor is None:
                    unanchorable += 1
                else:
                    meta["anchor"] = anchor
                    repaired += 1
        out.append(rec)
    with path.open("w", encoding="utf-8") as f:
        for rec in out:
            f.write(dumps_record(rec) + "\n")
    return rows, repaired, unanchorable


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("files", nargs="*", default=list(DEFAULT_FILES))
    a = ap.parse_args()

    for name in a.files:
        path = Path(name)
        if not path.is_file():
            print(f"[anchor] {name}: absent, skipped")
            continue
        rows, repaired, unanchorable = repair(path)
        print(f"[anchor] {name}: {rows:,} rows, re-anchored {repaired:,}, "
              f"unanchorable {unanchorable:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

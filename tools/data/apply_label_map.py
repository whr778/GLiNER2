"""Rewrite a corpus's labels through the trainer's OWN transform.

Corpora derived from the 137k pool -- the replay mixes, the Turkish dose arms, the
scaling-curve slices -- froze the pool's Chinese labels at build time and never saw the
translation their parents got. They present a Chinese label MENU while the base they
warm-start from presents an English one.

Label positions live at five different nestings (entity keys, relation names, event types
AND argument roles, classification menus and their answers, structure names AND field
names, plus record_metadata anchors). Rather than reimplement that a third time, this
applies `train.transform_record` -- the exact function the trainer uses -- so the data on
disk and the data the trainer would produce cannot disagree.

Spans are never touched: the map is keyed on labels, and the script verifies the input
text and every gold surface are byte-identical before writing.

    uv run python tools/data/apply_label_map.py --map tools/data/label_map_zh_all.json FILES...
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "train"))
from _split import dumps_record  # noqa: E402
import train as trainer  # noqa: E402

SPAN_KEYS = ("entity", "head", "tail", "triggers", "input", "text")


def surfaces(node, parent=None, out=None):
    """Collect every gold surface, so a rewrite that touches one is caught."""
    out = [] if out is None else out
    if isinstance(node, dict):
        for key, value in node.items():
            surfaces(value, key, out)
    elif isinstance(node, list):
        for item in node:
            surfaces(item, parent, out)
    elif isinstance(node, str) and parent in SPAN_KEYS:
        out.append(node)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("paths", nargs="+")
    parser.add_argument("--map", required=True, help="JSON file of {label: replacement}")
    args = parser.parse_args()

    mapping = json.loads(Path(args.map).read_text(encoding="utf-8"))
    # One map for every category: the keys are labels, and a label means the same thing
    # whichever position it occupies.
    fns = trainer._category_fns({c: {"rollup": False, "separator": ".", "map": mapping}
                                 for c in trainer.LABEL_CATEGORIES})

    for path in args.paths:
        src = Path(path)
        rows, changed = [], 0
        for line in src.open(encoding="utf-8"):
            record = json.loads(line)
            new = trainer.transform_record(record, fns)
            assert surfaces(record) == surfaces(new), f"{src}: a gold surface changed"
            assert record.get("input") == new.get("input"), f"{src}: input text changed"
            changed += record != new
            rows.append(new)
        with src.open("w", encoding="utf-8") as out:
            for record in rows:
                out.write(dumps_record(record) + "\n")
        print(f"{src}: {len(rows)} records, {changed} rewritten")


if __name__ == "__main__":
    main()

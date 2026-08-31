"""Rewrite Chinese CLASSIFICATION label values to the English names the rest of the mix uses.

`translate_labels.py` rewrites label KEYS. It does not touch classification label VALUES --
the `classifications[].labels` menu and its `true_label` -- so DocFEE kept a Chinese menu
while its own entity keys and the sibling ChFinAnn corpus had already moved to English. The
model was being shown a Chinese menu and scored against an English label space.

DocFEE and ChFinAnn are the same taxonomy: four of DocFEE's nine types ARE ChFinAnn types,
so they take ChFinAnn's canonical names rather than new ones. That unifies the two corpora
instead of giving one taxonomy two vocabularies.

    uv run python tools/data/unify_classification_labels.py data/docfee.train.jsonl ...
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _split import dumps_record  # noqa: E402

CJK = re.compile(r"[一-鿿]")

# Shared with ChFinAnn (whose menu already reads EquityFreeze/Overweight/Pledge/Underweight).
MAP = {
    "股权冻结": "EquityFreeze",
    "股东增持": "EquityOverweight",
    "股权质押": "EquityPledge",
    "股东减持": "EquityUnderweight",
    # DocFEE-only types, named in ChFinAnn's style.
    "破产清算": "BankruptcyLiquidation",
    "重大安全事故": "MajorSafetyAccident",
    "重大对外赔付": "MajorExternalPayout",
    "重大资产损失": "MajorAssetLoss",
    "高层死亡": "ExecutiveDeath",
    # text2json: a Traditional-Chinese entity type left behind by the key pass.
    "維基物種": "Wikispecies",
}


STRUCTURE_PARENTS = ("json_structures", "structures")


def convert(node, mapping, parent=None):
    """Rewrite label-valued strings in place, leaving every span untouched.

    Structure NAMES and field names are dict keys, not values, so they need rekeying
    rather than substitution -- the docfee pass did not have to do this because docfee
    carries no structures.
    """
    if isinstance(node, dict):
        for key in list(node):
            value = node[key]
            if parent in STRUCTURE_PARENTS and key in mapping:
                node[mapping[key]] = node.pop(key)
                key = mapping[key]
                value = node[key]
            if key in ("labels", "true_label", "type", "event_type", "task"):
                node[key] = ([mapping.get(v, v) for v in value]
                             if isinstance(value, list) else mapping.get(value, value))
            else:
                convert(value, mapping, key)
    elif isinstance(node, list):
        for item in node:
            convert(item, mapping, parent)


def main(paths, mapping=MAP):
    for path in paths:
        src = Path(path)
        rows, changed = [], 0
        for line in src.open(encoding="utf-8"):
            record = json.loads(line)
            before = json.dumps(record, ensure_ascii=False, sort_keys=True)
            convert(record, mapping)
            after = json.dumps(record, ensure_ascii=False, sort_keys=True)
            changed += before != after
            rows.append(record)
        with src.open("w", encoding="utf-8") as out:
            for record in rows:
                out.write(dumps_record(record) + "\n")
        print(f"{src}: {len(rows)} records, {changed} rewritten")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("paths", nargs="+")
    parser.add_argument("--map", help="JSON file of {label: replacement}; default is the "
                                      "built-in DocFEE/ChFinAnn map")
    args = parser.parse_args()
    table = json.loads(Path(args.map).read_text(encoding="utf-8")) if args.map else MAP
    main(args.paths, table)

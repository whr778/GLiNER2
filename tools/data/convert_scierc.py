"""Convert SciERC (scientific NER + relations) to GLiNER2 JSONL from the AI2 release.

Canonical source: http://nlp.cs.washington.edu/sciIE/data/sciERC_processed.tar.gz
(~695 MB -- it bundles ELMo embeddings we don't need). Pass ``--json`` to point at
an already-extracted ``processed_data/json/<split>.json`` and skip the download.

Each doc has token ``sentences`` (flattened to one token list), ``ner`` entries
``[start, end, type]`` (end-inclusive, document-global token indices), and
``relations`` ``[h_start, h_end, t_start, t_end, type]``. Maps to ``{type: [surface]}``
entities + ``{type: {head, tail}}`` relations. License: research use (AI2 / SciERC).

Usage::

    uv run python tools/data/convert_scierc.py --out data/scierc.jsonl
    uv run python tools/data/convert_scierc.py \\
        --json processed_data/json/train.json --out data/scierc.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _split import SplitWriter, add_split_args

URL = "http://nlp.cs.washington.edu/sciIE/data/sciERC_processed.tar.gz"


def _span(tokens: list, s: int, e: int) -> str:
    """Space-join the inclusive token range [s, e]."""
    return " ".join(tokens[s:e + 1]).strip()


def convert_doc(d: dict) -> dict | None:
    tokens = [t for sent in d.get("sentences", []) for t in sent]
    if not tokens:
        return None
    text = " ".join(tokens)

    entities: dict[str, list[str]] = {}
    for sent in d.get("ner", []):
        for span in sent:
            s, e, typ = span[0], span[1], span[2]
            surface = _span(tokens, s, e)
            if surface and isinstance(typ, str):
                bucket = entities.setdefault(typ, [])
                if surface not in bucket:
                    bucket.append(surface)

    relations = []
    for sent in d.get("relations", []):
        for r in sent:
            head, tail, typ = _span(tokens, r[0], r[1]), _span(tokens, r[2], r[3]), r[4]
            if head and tail and isinstance(typ, str):
                relations.append({typ: {"head": head, "tail": tail}})

    if not entities and not relations:
        return None
    output: dict = {"entities": entities} if entities else {}
    if relations:
        output["relations"] = relations
    return {"input": text, "output": output}


def _load_jsonl_text(args) -> str:
    if args.json:
        return Path(args.json).read_text(encoding="utf-8")
    member = f"processed_data/json/{args.split}.json"
    with tempfile.NamedTemporaryFile(suffix=".tar.gz") as tf:
        print(f"Downloading {URL} (~695 MB)...")
        urllib.request.urlretrieve(URL, tf.name)
        with tarfile.open(tf.name) as tar:
            f = tar.extractfile(member)
            return f.read().decode("utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True, type=Path,
                        help="Output JSONL base path (writes <base>.{train,val,test}.jsonl).")
    parser.add_argument("--json", default=None,
                        help="Local processed_data/json/<split>.json (skips the 695 MB download).")
    parser.add_argument("--split", default="train", help="Split to read when downloading.")
    parser.add_argument("--max-records", type=int, default=-1, help="Max docs to emit.")
    add_split_args(parser)
    args = parser.parse_args()

    raw = _load_jsonl_text(args)
    emitted = total_entities = total_relations = 0
    ent_types: set[str] = set()
    rel_types: set[str] = set()
    with SplitWriter(args.out, ratios=args.split_ratios, seed=args.split_seed) as writer:
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            record = convert_doc(json.loads(line))
            if record is None:
                continue
            writer.write(record)
            emitted += 1
            ents = record["output"].get("entities") or {}
            rels = record["output"].get("relations") or []
            total_entities += sum(len(v) for v in ents.values())
            total_relations += len(rels)
            ent_types.update(ents.keys())
            rel_types.update(next(iter(r)) for r in rels)
            if 0 <= args.max_records <= emitted:
                break

    print(f"Done. emitted={emitted} total_entities={total_entities} total_relations={total_relations} "
          f"entity_types={sorted(ent_types)} relation_types={len(rel_types)} {writer.summary()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

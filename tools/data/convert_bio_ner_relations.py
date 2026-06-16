"""Convert knowledgator/bio-NER-relations to GLiNER2 JSONL (entities + relations).

Source rows follow a BioC-style layout::

    {"id": ..., "document_id": ...,
     "passages": [{"text": ["<passage 1 text>"], ...}, ...],
     "entities": [{"id": "...T1", "type": "DNA", "text": ["DNA"],
                   "offsets": [[61, 64]], "normalized": [...]}, ...],
     "relations": [{"id": "...", "type": "bind",
                    "arg1_id": "...T1", "arg2_id": "...T3",
                    "normalized": []}, ...]}

The dataset has both entity annotations (50 types, dominated by ``umlsterm``)
and relation annotations (48 types: ``associated_with``, ``location_of``,
``bind``, etc.), with ~94% of rows carrying at least one relation.

This converter emits both: entities grouped by type into ``output.entities``,
and relations resolved via ``arg{1,2}_id`` lookup into ``output.relations``
using GLiNER2's ``{relation_name: {head, tail}}`` shape. Where a relation
type appears more than once, multiple records share the same relation name.

``--skip-types`` lets you drop noisy entity buckets (default
``umlsterm`` — auto-extracted UMLS concept matches that account for ~85% of
all entity assignments). Pass ``--skip-types ''`` to keep them.

Usage::

    uv run python tools/data/convert_bio_ner_relations.py \\
        --out data/bio_ner_relations.jsonl

    # Keep umlsterm entities
    uv run python tools/data/convert_bio_ner_relations.py \\
        --out data/bio_ner_relations.jsonl --skip-types ''
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _split import SplitWriter, add_split_args


def join_passages(passages) -> str:
    """Concatenate all passage text into a single document string."""
    if not isinstance(passages, list):
        return ""
    parts = []
    for psg in passages:
        if not isinstance(psg, dict):
            continue
        txt = psg.get("text")
        if isinstance(txt, list):
            parts.extend(t for t in txt if isinstance(t, str))
        elif isinstance(txt, str):
            parts.append(txt)
    return "\n".join(parts)


def first_surface(entity) -> str | None:
    """Return the first text element as the entity surface."""
    if not isinstance(entity, dict):
        return None
    txt = entity.get("text")
    if isinstance(txt, list) and txt:
        s = txt[0]
        if isinstance(s, str):
            return s.strip() or None
    elif isinstance(txt, str):
        return txt.strip() or None
    return None


def convert_row(row: dict, skip_types: set[str]) -> dict | None:
    """Convert one bio-NER-relations row; None if no usable entities."""
    text = join_passages(row.get("passages"))
    if not text:
        return None

    entities_by_id: dict[str, tuple[str, str]] = {}
    entities: dict[str, list[str]] = {}
    for ent in row.get("entities") or []:
        eid = ent.get("id")
        etype = ent.get("type")
        surface = first_surface(ent)
        if not isinstance(eid, str) or not isinstance(etype, str) or not surface:
            continue
        etype = etype.strip()
        if not etype or etype in skip_types:
            continue
        entities_by_id[eid] = (etype, surface)
        bucket = entities.setdefault(etype, [])
        if surface not in bucket:
            bucket.append(surface)

    if not entities:
        return None

    relations = []
    for rel in row.get("relations") or []:
        if not isinstance(rel, dict):
            continue
        rname = rel.get("type")
        a1 = rel.get("arg1_id")
        a2 = rel.get("arg2_id")
        if not isinstance(rname, str) or not isinstance(a1, str) or not isinstance(a2, str):
            continue
        rname = rname.strip()
        if not rname:
            continue
        head_info = entities_by_id.get(a1)
        tail_info = entities_by_id.get(a2)
        if not head_info or not tail_info:
            continue
        head = head_info[1]
        tail = tail_info[1]
        if head == tail:
            continue
        relations.append({rname: {"head": head, "tail": tail}})

    output: dict = {"entities": entities}
    if relations:
        output["relations"] = relations
    return {"input": text, "output": output}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True, type=Path,
                        help="Output JSONL base path (writes <base>.train.jsonl, "
                             ".val.jsonl, .test.jsonl).")
    parser.add_argument("--max-records", type=int, default=-1,
                        help="Maximum input records to emit (-1 = all).")
    parser.add_argument("--repo", default="knowledgator/bio-NER-relations",
                        help="HuggingFace dataset repo.")
    parser.add_argument("--split", default="train",
                        help="Dataset split to read.")
    parser.add_argument("--skip-types", default="umlsterm",
                        help="Comma-separated entity types to drop (default: "
                             "'umlsterm' — auto-matched UMLS concepts that "
                             "account for ~85%% of entity assignments). Pass "
                             "'' to keep everything.")
    add_split_args(parser)
    args = parser.parse_args()

    from datasets import load_dataset

    skip_types = {t.strip() for t in args.skip_types.split(",") if t.strip()}
    print(f"Loading {args.repo} split={args.split}...")
    ds = load_dataset(args.repo, split=args.split)

    emitted = 0
    skipped_empty = 0
    total_entities = 0
    total_relations = 0
    rows_with_rel = 0
    all_ent_types: set[str] = set()
    all_rel_types: set[str] = set()

    with SplitWriter(args.out, ratios=args.split_ratios, seed=args.split_seed) as writer:
        for row in ds:
            record = convert_row(row, skip_types)
            if record is None:
                skipped_empty += 1
                continue
            writer.write(record)
            emitted += 1
            ents = record["output"]["entities"]
            total_entities += sum(len(v) for v in ents.values())
            all_ent_types.update(ents.keys())
            rels = record["output"].get("relations") or []
            if rels:
                rows_with_rel += 1
                total_relations += len(rels)
                all_rel_types.update(next(iter(r.keys())) for r in rels)
            if 0 <= args.max_records <= emitted:
                break
            if emitted % 1000 == 0:
                print(f"  emitted={emitted}  rows_with_rel={rows_with_rel}")

    print(f"Done. emitted={emitted} skipped_empty={skipped_empty} "
          f"total_entities={total_entities} total_relations={total_relations} "
          f"ent_types={len(all_ent_types)} rel_types={len(all_rel_types)} "
          f"{writer.summary()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

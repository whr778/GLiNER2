"""Convert thunlp/docred (DocRED) to GLiNER2 JSONL (entities + relations).

DocRED is document-level relation extraction. Each row is::

    {"title": ...,
     "sents": [["Token", "list", "per", "sentence"], ...],
     "vertexSet": [[{"name": ..., "sent_id": 0, "pos": [start, end], "type": "ORG"},
                    ...more coref mentions...], ...one list per entity...],
     "labels": {"head": [e_i, ...], "tail": [e_j, ...],
                "relation_id": ["P17", ...], "relation_text": ["country", ...],
                "evidence": [[...], ...]}}

Each ``vertexSet`` entry is one entity (a coreference cluster of mentions); the
``labels`` columns are parallel arrays describing relations between entity
indices. This converter:

* joins every sentence's tokens with spaces into one document string;
* reconstructs each mention's surface from its sentence tokens at ``pos`` so the
  surface is guaranteed to appear verbatim in the text;
* groups mention surfaces by entity ``type`` into ``output.entities``;
* emits relations under their human-readable ``relation_text`` using GLiNER2's
  ``{relation_name: {head, tail}}`` shape, mapping head/tail entity indices to
  each entity's first mention surface.

The HF dataset ships a dataset script that newer ``datasets`` no longer runs, so
this reads the auto-converted parquet revision (``refs/convert/parquet``), whose
splits are ``train`` (annotated, labelled), ``validation`` (labelled), and
``test`` (no relation labels). Default reads ``train`` and partitions it 80/10/10.

Usage::

    uv run python tools/data/convert_docred.py --out data/docred.jsonl
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _split import SplitWriter, add_split_args


def join_text(sents) -> str:
    """Flatten all sentence token lists into one space-joined document string."""
    parts = []
    for sent in sents or []:
        if isinstance(sent, list):
            parts.extend(t for t in sent if isinstance(t, str))
    return " ".join(parts)


def mention_surface(sents, mention) -> str | None:
    """Reconstruct a mention's surface from its sentence tokens at ``pos``."""
    if not isinstance(mention, dict):
        return None
    sid = mention.get("sent_id")
    pos = mention.get("pos")
    if not isinstance(sid, int) or not isinstance(pos, (list, tuple)) or len(pos) != 2:
        return None
    if not (0 <= sid < len(sents)):
        return None
    start, end = pos
    toks = sents[sid][start:end]
    surface = " ".join(t for t in toks if isinstance(t, str)).strip()
    return surface or None


def convert_row(row: dict, skip_types: set[str]) -> dict | None:
    """Convert one DocRED row; None if it has no usable entities."""
    sents = row.get("sents")
    text = join_text(sents)
    if not text:
        return None

    entities: dict[str, list[str]] = {}
    rep_surface: dict[int, str] = {}  # entity index -> first mention surface (for relations)
    for idx, mentions in enumerate(row.get("vertexSet") or []):
        if not isinstance(mentions, list) or not mentions:
            continue
        for m in mentions:
            etype = m.get("type") if isinstance(m, dict) else None
            surface = mention_surface(sents, m)
            if not isinstance(etype, str) or not surface:
                continue
            etype = etype.strip()
            if not etype or etype in skip_types:
                continue
            bucket = entities.setdefault(etype, [])
            if surface not in bucket:
                bucket.append(surface)
            rep_surface.setdefault(idx, surface)

    if not entities:
        return None

    relations = []
    labels = row.get("labels") or {}
    heads = labels.get("head") or []
    tails = labels.get("tail") or []
    names = labels.get("relation_text") or []
    for h, t, rname in zip(heads, tails, names):
        if not isinstance(rname, str) or not rname.strip():
            continue
        head = rep_surface.get(h)
        tail = rep_surface.get(t)
        if not head or not tail or head == tail:
            continue
        relations.append({rname.strip(): {"head": head, "tail": tail}})

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
    parser.add_argument("--repo", default="thunlp/docred", help="HuggingFace dataset repo.")
    parser.add_argument("--revision", default="refs/convert/parquet",
                        help="Dataset revision (default: the auto-converted parquet "
                             "branch, since the original ships a dataset script).")
    parser.add_argument("--split", default="train",
                        help="Parquet split to read: train (annotated), validation, "
                             "or test (test has no relation labels). Default: train.")
    parser.add_argument("--skip-types", default="",
                        help="Comma-separated entity types to drop (DocRED types: "
                             "PER, ORG, LOC, TIME, NUM, MISC). Default: keep all.")
    add_split_args(parser)
    args = parser.parse_args()

    from datasets import load_dataset

    skip_types = {t.strip() for t in args.skip_types.split(",") if t.strip()}
    print(f"Loading {args.repo} revision={args.revision} split={args.split}...")
    ds = load_dataset(args.repo, revision=args.revision, split=args.split)

    emitted = skipped_empty = total_entities = total_relations = rows_with_rel = 0
    ent_types: set[str] = set()
    rel_types: set[str] = set()

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
            ent_types.update(ents.keys())
            rels = record["output"].get("relations") or []
            if rels:
                rows_with_rel += 1
                total_relations += len(rels)
                rel_types.update(next(iter(r.keys())) for r in rels)
            if 0 <= args.max_records <= emitted:
                break

    print(f"Done. emitted={emitted} skipped_empty={skipped_empty} "
          f"total_entities={total_entities} total_relations={total_relations} "
          f"rows_with_rel={rows_with_rel} ent_types={len(ent_types)} "
          f"rel_types={len(rel_types)} {writer.summary()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

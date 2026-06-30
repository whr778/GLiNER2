"""Convert tonytan48/Re-DocRED to GLiNER2 JSONL (entities + relations).

Re-DocRED is a revised, higher-quality version of DocRED with corrected
entity and relation annotations (Tan et al., 2022). The document schema
is similar to DocRED::

    {"title": ...,
     "sents": [["Token", "list", "per", "sentence"], ...],
     "vertexSet": [[{"name": ..., "sent_id": 0, "pos": [start, end],
                     "type": "ORG", "global_pos": [...], "index": "..."}, ...], ...],
     "labels": [{"h": 0, "t": 1, "r": "P17", "evidence": [0]}, ...]}

Key differences from DocRED:
* ``labels`` is a **list of dicts** ``{h, t, r, evidence}`` (not parallel arrays).
* ``r`` is a Wikidata property ID (e.g. ``"P17"``); this converter maps it to the
  same human-readable string that DocRED's ``relation_text`` field uses (e.g.
  ``"country"``), ensuring co-trained models see one label per relation.
* The HF dataset has canonical train/validation/test splits (3053/500/500 docs).
  Call this converter once per split — no random partitioning needed.

Usage::

    uv run python tools/data/convert_redocred.py --split train      --out data/redocred.train.jsonl
    uv run python tools/data/convert_redocred.py --split validation  --out data/redocred.val.jsonl
    uv run python tools/data/convert_redocred.py --split test        --out data/redocred.test.jsonl
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _split import dumps_record

# Relation-text strings match the DocRED relation_text field verbatim (derived
# from thunlp/docred) so co-trained models share one label per Wikidata P-ID.
_RELATION_TEXT: dict[str, str] = {
    "P6":    "head of government",
    "P17":   "country",
    "P19":   "place of birth",
    "P20":   "place of death",
    "P22":   "father",
    "P25":   "mother",
    "P26":   "spouse",
    "P27":   "country of citizenship",
    "P30":   "continent",
    "P31":   "instance of",
    "P35":   "head of state",
    "P36":   "capital",
    "P37":   "official language",
    "P39":   "position held",
    "P40":   "child",
    "P50":   "author",
    "P54":   "member of sports team",
    "P57":   "director",
    "P58":   "screenwriter",
    "P69":   "educated at",
    "P86":   "composer",
    "P102":  "member of political party",
    "P108":  "employer",
    "P112":  "founded by",
    "P118":  "league",
    "P123":  "publisher",
    "P127":  "owned by",
    "P131":  "located in the administrative territorial entity",
    "P136":  "genre",
    "P137":  "operator",
    "P140":  "religion",
    "P150":  "contains administrative territorial entity",
    "P155":  "follows",
    "P156":  "followed by",
    "P159":  "headquarters location",
    "P161":  "cast member",
    "P162":  "producer",
    "P166":  "award received",
    "P170":  "creator",
    "P171":  "parent taxon",
    "P172":  "ethnic group",
    "P175":  "performer",
    "P176":  "manufacturer",
    "P178":  "developer",
    "P179":  "series",
    "P190":  "sister city",
    "P194":  "legislative body",
    "P205":  "basin country",
    "P206":  "located in or next to body of water",
    "P241":  "military branch",
    "P264":  "record label",
    "P272":  "production company",
    "P276":  "location",
    "P279":  "subclass of",
    "P355":  "subsidiary",
    "P361":  "part of",
    "P364":  "original language of work",
    "P400":  "platform",
    "P403":  "mouth of the watercourse",
    "P449":  "original network",
    "P463":  "member of",
    "P488":  "chairperson",
    "P495":  "country of origin",
    "P527":  "has part",
    "P551":  "residence",
    "P569":  "date of birth",
    "P570":  "date of death",
    "P571":  "inception",
    "P576":  "dissolved, abolished or demolished",
    "P577":  "publication date",
    "P580":  "start time",
    "P582":  "end time",
    "P585":  "point in time",
    "P607":  "conflict",
    "P674":  "characters",
    "P676":  "lyrics by",
    "P706":  "located on terrain feature",
    "P710":  "participant",
    "P737":  "influenced by",
    "P740":  "location of formation",
    "P749":  "parent organization",
    "P800":  "notable work",
    "P807":  "separated from",
    "P840":  "narrative location",
    "P937":  "work location",
    "P1001": "applies to jurisdiction",
    "P1056": "product or material produced",
    "P1198": "unemployment rate",
    "P1336": "territory claimed by",
    "P1344": "participant of",
    "P1365": "replaces",
    "P1366": "replaced by",
    "P1376": "capital of",
    "P1412": "languages spoken, written or signed",
    "P1441": "present in work",
    "P3373": "sibling",
}


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
    """Convert one Re-DocRED row; None if it has no usable entities."""
    sents = row.get("sents")
    text = join_text(sents)
    if not text:
        return None

    entities: dict[str, list[str]] = {}
    rep_surface: dict[int, str] = {}
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
    for label in row.get("labels") or []:
        if not isinstance(label, dict):
            continue
        pid = label.get("r")
        if not pid:
            continue
        rname = _RELATION_TEXT.get(pid, pid)
        head = rep_surface.get(label.get("h"))
        tail = rep_surface.get(label.get("t"))
        if not head or not tail or head == tail:
            continue
        relations.append({rname: {"head": head, "tail": tail}})

    output: dict = {"entities": entities}
    if relations:
        output["relations"] = relations
    return {"input": text, "output": output}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--split", required=True,
                        choices=["train", "validation", "test"],
                        help="Which canonical split to convert.")
    parser.add_argument("--out", required=True, type=Path,
                        help="Output JSONL path.")
    parser.add_argument("--max-records", type=int, default=-1,
                        help="Maximum input records to emit (-1 = all).")
    parser.add_argument("--repo", default="tonytan48/Re-DocRED",
                        help="HuggingFace dataset repo.")
    parser.add_argument("--skip-types", default="",
                        help="Comma-separated entity types to drop. Default: keep all.")
    args = parser.parse_args()

    from datasets import load_dataset

    skip_types = {t.strip() for t in args.skip_types.split(",") if t.strip()}
    print(f"Loading {args.repo} split={args.split}...")
    ds = load_dataset(args.repo, split=args.split)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    emitted = skipped_empty = total_entities = total_relations = rows_with_rel = 0
    ent_types: set[str] = set()
    rel_types: set[str] = set()

    with args.out.open("w", encoding="utf-8") as fh:
        for row in ds:
            record = convert_row(row, skip_types)
            if record is None:
                skipped_empty += 1
                continue
            fh.write(dumps_record(record) + "\n")
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

    print(f"Done. split={args.split} emitted={emitted} skipped_empty={skipped_empty} "
          f"total_entities={total_entities} total_relations={total_relations} "
          f"rows_with_rel={rows_with_rel} ent_types={len(ent_types)} "
          f"rel_types={len(rel_types)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

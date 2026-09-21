"""Attach purchased entity TYPES to an event corpus, as a typing signal only.

Option 2's typed margin needs each candidate span's entity type. cmnee and duee carry event
gold and NO entity gold in the same record -- their entity gold was bought separately and
lives in `cmnee_ner` / `duee_ner`, keyed to the same documents. Measured: only casie has both
in one record, 798 of ~175,000 training records, so a typed margin trained on today's data
would touch under 0.5% of the mix.

This joins them on document text and writes `entity_types: {surface: [type, ...]}` onto each
record. It deliberately does NOT add an `entities` block: that would make the purchased gold
ENTITY SUPERVISION, and it is not exhaustive -- the entity head would learn that every
unannotated entity is absent. The types are evidence for typing a span, not a target.

MATCHING IS CONSERVATIVE, AND MEASURED. A wrong type is worse than no type: it puts the
margin on the wrong candidate and teaches something false. Sampled on real records:

    rule                       typed    sampled matches
    exact                      58.3%    --
    + arg in ent               62.2%    all correct ('23' in '23人', '第7舰队' in '美国海军第7舰队')
    + ent in arg, ratio 0.85   62.7%    all correct ('B-2轰炸机队' has 'B-2轰炸机')
    + ent in arg, ratio 0.70   66.1%    ~2 of 5 WRONG ('FELIN...的部队', a unit, typed Product)
    + ent in arg, ratio 0.50   74.3%    clear errors ('40多架民用飞机' typed Quantity)

So containment is accepted only when the entity covers >= `--min-cover` of the argument
(default 0.85), plus sub-span matches, plus exact. The looser rules are reachable by flag for
anyone who wants to measure the trade again rather than trust this table.

    uv run python tools/data/merge_entity_types.py \
        --events data/cmnee --entities data/cmnee_ner --out data/cmnee_typed
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _split import dumps_record  # noqa: E402  normalized UTF-8 write path


def surface_types(output: dict) -> dict:
    """surface -> {entity types} from one record's entity gold."""
    out: dict = {}
    for etype, surfaces in (output.get("entities") or {}).items():
        for s in surfaces or []:
            if isinstance(s, str) and s.strip():
                out.setdefault(s.strip(), set()).add(str(etype))
    return out


def load_entity_index(prefix: Path, split: str) -> dict:
    """document text -> {surface: {types}}, from the corpus carrying entity gold."""
    path = prefix.with_name(f"{prefix.name}.{split}.jsonl")
    index: dict = {}
    if not path.is_file():
        return index
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            rec = json.loads(line)
            key = str(rec.get("input") or rec.get("text") or "").strip()
            if key:
                index[key] = surface_types(rec.get("output") or rec.get("schema") or {})
    return index


def lookup(surface: str, surfaces: dict, min_cover: float, min_len: int = 2):
    """Types for one surface, or None. Exact first, then the conservative containments."""
    if surface in surfaces:
        return surfaces[surface]
    if len(surface) < min_len:
        return None
    best, best_len = None, 0
    for s, types in surfaces.items():
        if len(s) < min_len:
            continue
        # the surface is a SUB-SPAN of an annotated entity: '23' inside '23人'
        if surface in s and len(surface) >= 0.5 * len(s) and len(surface) > best_len:
            best, best_len = types, len(surface)
        # an annotated entity covers MOST of the surface: 'B-2轰炸机' in 'B-2轰炸机队'.
        # The ratio is what keeps '印度' out of '印度海军', where the contained entity is a
        # modifier and the head is something else entirely.
        if s in surface and len(s) >= min_cover * len(surface) and len(s) > best_len:
            best, best_len = types, len(s)
    return best


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--events", required=True, help="corpus prefix carrying event gold")
    ap.add_argument("--entities", required=True, help="corpus prefix carrying entity gold")
    ap.add_argument("--out", required=True, help="output corpus prefix")
    ap.add_argument("--min-cover", type=float, default=0.85)
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    ap.add_argument("--min-join-rate", type=float, default=0.90)
    args = ap.parse_args()

    ev, en, out = Path(args.events), Path(args.entities), Path(args.out)
    grand = Counter()
    for split in args.splits:
        src = ev.with_name(f"{ev.name}.{split}.jsonl")
        if not src.is_file():
            print(f"[merge] {ev.name} has no {split} split; skipping")
            continue
        index = load_entity_index(en, split)
        if not index:
            print(f"[merge] {en.name} has no {split} split; every record would be unjoined")
        dst = out.with_name(f"{out.name}.{split}.jsonl")
        n = joined = args_total = args_typed = 0
        with src.open(encoding="utf-8") as fh, dst.open("w", encoding="utf-8") as w:
            for line in fh:
                if not line.strip():
                    continue
                rec = json.loads(line)
                n += 1
                key = str(rec.get("input") or rec.get("text") or "").strip()
                surfaces = index.get(key)
                o = rec.get("output") or rec.get("schema") or {}
                if surfaces:
                    joined += 1
                    # INSIDE THE SCHEMA, not at the record's top level. Traced: the dataset's
                    # `__getitem__` returns only `(text, schema)`, so a top-level key is
                    # dropped at the first hop and never reaches collate. Inside the schema it
                    # survives to `batch.original_schemas[i]`, and the query builder ignores it
                    # -- verified: task_types stayed ['events'], no query was created for it.
                    o["entity_types"] = {s: sorted(t) for s, t in surfaces.items()}
                for e in (o.get("events") or []):
                    for a in (e.get("arguments") or []):
                        surface = str(a.get("entity") or "").strip()
                        if not surface:
                            continue
                        args_total += 1
                        if surfaces and lookup(surface, surfaces, args.min_cover):
                            args_typed += 1
                w.write(dumps_record(rec) + "\n")
        rate = joined / max(n, 1)
        cover = args_typed / max(args_total, 1)
        print(f"[merge] {split:5s} records={n:6d} joined={joined:6d} ({rate:5.1%})  "
              f"arguments={args_total:7d} typed={args_typed:7d} ({cover:5.1%})  -> {dst}")
        grand["records"] += n; grand["joined"] += joined
        grand["args"] += args_total; grand["typed"] += args_typed
        # A silent join failure would build a corpus of whatever happened to match.
        if index and rate < args.min_join_rate:
            raise SystemExit(f"[merge] REFUSING: {split} joined only {rate:.1%} "
                             f"(floor {args.min_join_rate:.0%})")
    print(f"[merge] TOTAL records={grand['records']:,} joined={grand['joined']:,} "
          f"({grand['joined']/max(grand['records'],1):.1%})  "
          f"arguments={grand['args']:,} typed={grand['typed']:,} "
          f"({grand['typed']/max(grand['args'],1):.1%})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

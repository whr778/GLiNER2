"""Convert ACE 2005 English annotations to GLiNER2 JSONL.

Co-trains entities, relations, and events from a single ACE document.
ACE 2005 (LDC2006T06) is the canonical event-extraction benchmark with
33 event subtypes, 7 entity types, and 6 relation types — and crucially
the events' argument fillers are themselves entities, so all three tasks
share the same surface vocabulary and benefit from joint training.

For each ``.apf.xml`` / ``.sgm`` pair the converter emits one record::

    {"input": "<doc body>",
     "output": {
         "entities":  {"PER.Individual": ["John Smith"], "ORG.Government": ["UN"]},
         "relations": [{"ORG-AFF.Employment": {"head": "John Smith", "tail": "UN"}}],
         "events":    [{"event_type": "Conflict.Attack",
                        "triggers": ["attacked"],
                        "arguments": [{"role": "Attacker", "entity": "John Smith"}]}]
     }}

Relations are resolved through their ``<relation_mention_argument>``
REFIDs: the converter walks the entity-mention table once, then looks up
each REFID to recover the argument's surface text. ROLE ``Arg-1`` maps to
``head`` and ``Arg-2`` to ``tail``. Relations whose either argument
cannot be resolved (REFID missing, surface text missing from the body)
are dropped.

Entity surfaces use each mention's **head** span by default (dropping
determiners and modifiers: "the vice president" -> "president"), which keeps
the surface vocabulary tight; pass ``--extent-offsets`` to use the full extent
instead. Relation and event entity-arguments inherit the same choice so every
surface in a record is consistent.

By default the converter stratifies the resulting records into
80/10/10 train/test/val splits using a greedy multi-label algorithm:

1. Build the per-record category set — every entity type, relation type,
   and event type the record contains.
2. Compute per-type targets: for *N* samples of a given type, targets are
   ``(1, 0, 0)`` if N=1, ``(1, 1, 0)`` if N=2, ``(1, 1, 1)`` if N=3, and
   the rounded 80/10/10 split otherwise.
3. Iteratively pick the rarest type with unplaced samples (lowest
   remaining count, ties broken by lowest total frequency, then by
   name). Place its next unplaced sample into whichever split has the
   biggest gap to its target. Bookkeeping updates for *all* types the
   placed sample touches.

Pass ``--no-stratify`` to write a single file at ``--out`` instead.

Output filenames follow the convention used by the other split-aware
converters: ``<base>.train.jsonl`` / ``<base>.test.jsonl`` /
``<base>.val.jsonl``. The ``.jsonl`` suffix on ``--out`` is stripped if
present, so ``--out data/ace2005.jsonl`` and ``--out data/ace2005`` are
equivalent.

Source layout — typical LDC delivery::

    ace_2005_td_v7/
      data/
        English/                 (or any locale code)
          bc/ bn/ nw/ ... /      (genre dirs)
            adj/ fp1/ fp2/ timex2norm/  (annotation-pass dirs; all 4 re-annotate
                                          the SAME documents)
              CNN_CF_*.sgm       (text)
              CNN_CF_*.apf.xml   (annotations)

Only the ``adj`` (adjudicated gold) folder is converted -- the other
annotation-pass folders duplicate the same documents and are skipped, so
converting the whole tree unfiltered would emit duplicate records.

Usage::

    uv run python tools/data/convert_ace2005.py \\
        --input /path/to/ace_2005_td_v7/data/English \\
        --out data/ace2005.jsonl

    # Top-level event/entity/relation types only (drop SUBTYPE)
    uv run python tools/data/convert_ace2005.py \\
        --input /path/to/ace_2005_td_v7/data/English \\
        --out data/ace2005_toplevel.jsonl --no-subtypes

    # Single-file output (no stratification)
    uv run python tools/data/convert_ace2005.py \\
        --input /path/to/ace_2005_td_v7/data/English \\
        --out data/ace2005.jsonl --no-stratify
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _mention_filter import MentionFilter, load_mention_filter  # noqa: E402
from _split import dumps_record  # noqa: E402
from _stratify import (  # noqa: E402
    coverage_summary,
    derive_split_paths,
    parse_ratios,
    record_categories,
    stratified_split,
)


TAG_RE = re.compile(r"<[^>]+>")
WS_RE = re.compile(r"\s+")


# ---------------------------------------------------------------------------
# SGM / APF parsing
# ---------------------------------------------------------------------------

def _strip_sgml(sgm_text: str) -> str:
    m = re.search(r"<TEXT>(.*?)</TEXT>", sgm_text, re.DOTALL | re.IGNORECASE)
    body = m.group(1) if m else sgm_text
    stripped = TAG_RE.sub(" ", body)
    return WS_RE.sub(" ", stripped).strip()


def _first_charseq_text(parent: ET.Element, sub_tag: str) -> Optional[str]:
    sub = parent.find(sub_tag)
    if sub is None:
        return None
    cs = sub.find("charseq")
    if cs is None or cs.text is None:
        return None
    return WS_RE.sub(" ", cs.text).strip() or None


def _mention_surface(emention: ET.Element, use_head: bool) -> Optional[str]:
    """Surface text for an entity mention: its ``head`` or full ``extent``.

    The head is the minimal syntactic span (e.g. "president"); the extent adds
    modifiers and determiners (e.g. "the vice president"). Head is the default so
    articles and modifiers don't inflate the surface vocabulary. Falls back to the
    extent when no head charseq is present (value/time mentions have no head).
    """
    if use_head:
        head = _first_charseq_text(emention, "head")
        if head:
            return head
    return _first_charseq_text(emention, "extent")


def _pair_sgm(apf_path: Path) -> Optional[Path]:
    stem = apf_path.name
    if stem.endswith(".apf.xml"):
        sgm = apf_path.with_name(stem[:-len(".apf.xml")] + ".sgm")
    else:
        sgm = apf_path.with_suffix(".sgm")
    return sgm if sgm.is_file() else None


def parse_apf(
    apf_path: Path,
    keep_subtypes: bool,
    mention_filter: Optional[MentionFilter] = None,
    stats: Optional[Counter] = None,
    use_head: bool = True,
) -> Optional[Dict[str, Any]]:
    """Parse one .apf.xml + .sgm pair into a single GLiNER2 record.

    Pulls entities, relations, and events together so co-training has
    aligned surface forms for argument fillers.

    ``mention_filter`` keeps only the allowed entity mention types (NAM/NOM/PRO);
    a dropped mention cascades to its relations and event arguments. ``stats``
    accumulates the per-category drop counts for the caller's summary.

    ``use_head`` (default) takes each entity mention's head span, dropping
    determiners/modifiers ("the vice president" -> "president"); set it False to
    keep the full extent. Relation and event entity-arguments inherit the same
    choice via their mention lookup, so a record's surfaces stay consistent.
    """
    if mention_filter is None:
        mention_filter = MentionFilter(None)
    if stats is None:
        stats = Counter()
    sgm_path = _pair_sgm(apf_path)
    if sgm_path is None:
        return None
    try:
        tree = ET.parse(apf_path)
    except ET.ParseError:
        return None
    root = tree.getroot()

    try:
        sgm_text = sgm_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    text = _strip_sgml(sgm_text)
    if not text:
        return None

    # ----- entities -----
    # entity_mention_type records the mention type (NAM/NOM/PRO) for EVERY entity
    # mention, unconditionally, so the event-argument cascade can tell an entity
    # mention from a value/time mention regardless of the extent-in-text check.
    # entity_mention_text holds only the kept (allowed + in-text) mentions and
    # drives entities + relations.
    entity_mention_text: Dict[str, str] = {}   # kept mention_id -> surface
    entity_mention_type: Dict[str, str] = {}   # every entity mention_id -> m_type
    filtered_mids: Set[str] = set()            # mentions removed by the filter
    entities_by_type: Dict[str, List[str]] = {}
    for entity in root.iter("entity"):
        etype = (entity.get("TYPE") or "").strip()
        esub = (entity.get("SUBTYPE") or "").strip()
        if not etype:
            continue
        full_type = f"{etype}.{esub}" if keep_subtypes and esub else etype
        for emention in entity.iter("entity_mention"):
            mid = emention.get("ID")
            if not mid:
                continue
            m_type = (emention.get("TYPE") or "").strip()
            entity_mention_type[mid] = m_type
            span_text = _mention_surface(emention, use_head)
            if not span_text or span_text not in text:
                continue
            if not mention_filter.allows(m_type):
                filtered_mids.add(mid)
                stats["filtered_mentions"] += 1
                continue
            entity_mention_text[mid] = span_text
            bucket = entities_by_type.setdefault(full_type, [])
            if span_text not in bucket:
                bucket.append(span_text)

    # ----- relations -----
    relations_out: List[Dict[str, Dict[str, str]]] = []
    seen_rel: Set[Tuple[str, str, str]] = set()
    for rel in root.iter("relation"):
        rtype = (rel.get("TYPE") or "").strip()
        rsub = (rel.get("SUBTYPE") or "").strip()
        if not rtype:
            continue
        rel_type = f"{rtype}.{rsub}" if keep_subtypes and rsub else rtype
        for rmention in rel.iter("relation_mention"):
            head: Optional[str] = None
            tail: Optional[str] = None
            dropped_by_filter = False
            for ramg in rmention.iter("relation_mention_argument"):
                refid = ramg.get("REFID")
                role = (ramg.get("ROLE") or "").strip()
                if not refid:
                    continue
                if refid not in entity_mention_text:
                    if refid in filtered_mids:
                        dropped_by_filter = True
                    continue
                surface = entity_mention_text[refid]
                if role == "Arg-1":
                    head = surface
                elif role == "Arg-2":
                    tail = surface
            if head and tail and head != tail:
                key = (rel_type, head, tail)
                if key in seen_rel:
                    continue
                seen_rel.add(key)
                relations_out.append({rel_type: {"head": head, "tail": tail}})
            elif dropped_by_filter:
                stats["filtered_relations"] += 1

    # ----- events -----
    events_out: List[Dict[str, Any]] = []
    for evt in root.iter("event"):
        etype = (evt.get("TYPE") or "").strip()
        esub = (evt.get("SUBTYPE") or "").strip()
        if not etype:
            continue
        event_type = f"{etype}.{esub}" if keep_subtypes and esub else etype
        for emention in evt.iter("event_mention"):
            anchor_text = _first_charseq_text(emention, "anchor")
            if not anchor_text or anchor_text not in text:
                continue
            arguments: List[Dict[str, Any]] = []
            seen_args: Set[Tuple[str, str]] = set()
            for arg in emention.iter("event_mention_argument"):
                role = (arg.get("ROLE") or "").strip()
                if not role:
                    continue
                # Cascade: drop an argument whose REFID is a filtered entity
                # mention. event_mention_argument REFID is mention-level
                # (e.g. "DOC-E1-3"), like relation_mention_argument. Value and
                # time arguments are not entity mentions, so they are kept.
                refid = arg.get("REFID")
                if refid in entity_mention_type and not mention_filter.allows(
                    entity_mention_type[refid]
                ):
                    stats["filtered_event_args"] += 1
                    continue
                # In head mode reuse the entity mention's head surface so a filler
                # matches its entity form; value/time args (no entity mention) and
                # extent mode fall back to the argument's own extent charseq.
                arg_text = entity_mention_text.get(refid) if use_head else None
                if not arg_text:
                    arg_text = _first_charseq_text(arg, "extent")
                if not arg_text or arg_text not in text:
                    continue
                key = (role, arg_text)
                if key in seen_args:
                    continue
                seen_args.add(key)
                arguments.append({"role": role, "entity": arg_text})
            events_out.append({
                "event_type": event_type,
                "triggers": [anchor_text],
                "arguments": arguments,
            })

    output: Dict[str, Any] = {}
    if entities_by_type:
        output["entities"] = entities_by_type
    if relations_out:
        output["relations"] = relations_out
    if events_out:
        output["events"] = events_out

    if not output:
        return None
    return {"input": text, "output": output}


def iter_apf_files(root: Path):
    """Yield .apf.xml files from the adjudicated ("adj") annotation folders only.

    The raw ACE 2005 LDC delivery has several annotation-pass folders per
    genre/language (``adj`` plus others) that re-annotate the SAME underlying
    documents. Walking the whole tree would convert each document once per
    folder, duplicating records. ``adj`` holds the adjudicated gold
    annotation and is the only folder that should be converted.
    """
    for path in root.rglob("*.apf.xml"):
        if "adj" in path.relative_to(root).parts:
            yield path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _stats(records: List[Dict[str, Any]]) -> str:
    n_ent = sum(len(r["output"].get("entities") or {}) for r in records)
    n_rel = sum(len(r["output"].get("relations") or []) for r in records)
    n_evt = sum(len(r["output"].get("events") or []) for r in records)
    return f"docs={len(records)} entity_types_sum={n_ent} relations={n_rel} events={n_evt}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", required=True, type=Path,
                        help="Root directory of the ACE 2005 corpus (typically "
                             "ace_2005_td_v7/data/English).")
    parser.add_argument("--out", required=True, type=Path,
                        help="Output JSONL base path. Stratified mode writes "
                             "<base>.train.jsonl / .test.jsonl / .val.jsonl; "
                             "with --no-stratify, --out is the single output file.")
    parser.add_argument("--max-records", type=int, default=-1,
                        help="Maximum documents to emit (-1 = all).")
    parser.add_argument("--no-subtypes", action="store_true",
                        help="Use only top-level event/entity/relation types "
                             "(drop SUBTYPE everywhere).")
    parser.add_argument("--extent-offsets", action="store_true",
                        help="Use each mention's full extent span (with "
                             "determiners/modifiers, e.g. 'the vice president') "
                             "instead of the default head span ('president'). "
                             "Head is the default because it drops articles and "
                             "modifiers that inflate the surface vocabulary.")
    parser.add_argument("--no-stratify", action="store_true",
                        help="Disable stratified split; write a single file at --out.")
    parser.add_argument("--split-ratios", type=parse_ratios, default=(0.8, 0.1, 0.1),
                        help="Comma-separated train,test,val ratios "
                             "(default: 0.8,0.1,0.1).")
    parser.add_argument("--split-seed", type=int, default=42,
                        help="Seed for the deterministic stratified placement.")
    parser.add_argument("--filter-config", type=Path, default=None,
                        help="YAML mention-type filter (see "
                             "tools/data/config/mention_filter.yaml). Keeps only "
                             "the allowed entity mention types (NAM/NOM/PRO); "
                             "dropped mentions cascade to their relations and "
                             "event arguments. Default: keep all.")
    args = parser.parse_args()

    if not args.input.is_dir():
        raise SystemExit(f"input directory not found: {args.input}")

    keep_subtypes = not args.no_subtypes
    use_head = not args.extent_offsets
    mention_filter = load_mention_filter(args.filter_config, "ace2005")
    stats: Counter = Counter()

    # ----- collect all records -----
    records: List[Dict[str, Any]] = []
    skipped = 0
    for apf_path in iter_apf_files(args.input):
        if 0 <= args.max_records <= len(records):
            break
        rec = parse_apf(apf_path, keep_subtypes=keep_subtypes,
                        mention_filter=mention_filter, stats=stats,
                        use_head=use_head)
        if rec is None:
            skipped += 1
            continue
        records.append(rec)

    print(f"Parsed: {_stats(records)} skipped_no_content={skipped}")
    if mention_filter.active:
        print(f"Mention filter ({mention_filter.describe()}): "
              f"dropped_mentions={stats['filtered_mentions']} "
              f"dropped_relations={stats['filtered_relations']} "
              f"dropped_event_args={stats['filtered_event_args']}")

    # ----- single-file mode -----
    if args.no_stratify:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", encoding="utf-8") as f:
            for rec in records:
                f.write(dumps_record(rec) + "\n")
        print(f"Wrote {len(records)} records -> {args.out}")
        return 0

    # ----- stratified mode -----
    train, test, val = stratified_split(
        records, ratios=args.split_ratios, seed=args.split_seed,
    )
    paths = derive_split_paths(args.out)
    paths["train"].parent.mkdir(parents=True, exist_ok=True)

    for split_name, slice_records in (("train", train), ("test", test), ("val", val)):
        with paths[split_name].open("w", encoding="utf-8") as f:
            for rec in slice_records:
                f.write(dumps_record(rec) + "\n")

    print(
        f"Stratified split (ratios={args.split_ratios}): "
        f"train={len(train)} test={len(test)} val={len(val)}\n"
        f"  {coverage_summary(records, train, test, val)}\n"
        f"  -> {paths['train']}, {paths['test']}, {paths['val']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

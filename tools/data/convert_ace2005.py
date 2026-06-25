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
                        "trigger": "attacked",
                        "arguments": [{"role": "Attacker", "entity": "John Smith"}]}]
     }}

Relations are resolved through their ``<relation_mention_argument>``
REFIDs: the converter walks the entity-mention table once, then looks up
each REFID to recover the argument's surface text. ROLE ``Arg-1`` maps to
``head`` and ``Arg-2`` to ``tail``. Relations whose either argument
cannot be resolved (REFID missing, extent text missing from the body)
are dropped.

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
            adj/ timex2norm/     (annotation dirs)
              CNN_CF_*.sgm       (text)
              CNN_CF_*.apf.xml   (annotations)

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
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
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


def _pair_sgm(apf_path: Path) -> Optional[Path]:
    stem = apf_path.name
    if stem.endswith(".apf.xml"):
        sgm = apf_path.with_name(stem[:-len(".apf.xml")] + ".sgm")
    else:
        sgm = apf_path.with_suffix(".sgm")
    return sgm if sgm.is_file() else None


def parse_apf(apf_path: Path, keep_subtypes: bool) -> Optional[Dict[str, Any]]:
    """Parse one .apf.xml + .sgm pair into a single GLiNER2 record.

    Pulls entities, relations, and events together so co-training has
    aligned surface forms for argument fillers.
    """
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
    entity_mention_text: Dict[str, str] = {}   # mention_id -> surface
    entities_by_type: Dict[str, List[str]] = {}
    for entity in root.iter("entity"):
        etype = (entity.get("TYPE") or "").strip()
        esub = (entity.get("SUBTYPE") or "").strip()
        if not etype:
            continue
        full_type = f"{etype}.{esub}" if keep_subtypes and esub else etype
        for emention in entity.iter("entity_mention"):
            mid = emention.get("ID")
            extent_text = _first_charseq_text(emention, "extent")
            if not mid or not extent_text or extent_text not in text:
                continue
            entity_mention_text[mid] = extent_text
            bucket = entities_by_type.setdefault(full_type, [])
            if extent_text not in bucket:
                bucket.append(extent_text)

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
            for ramg in rmention.iter("relation_mention_argument"):
                refid = ramg.get("REFID")
                role = (ramg.get("ROLE") or "").strip()
                if not refid or refid not in entity_mention_text:
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
                "trigger": anchor_text,
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
    yield from root.rglob("*.apf.xml")


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
    parser.add_argument("--no-stratify", action="store_true",
                        help="Disable stratified split; write a single file at --out.")
    parser.add_argument("--split-ratios", type=parse_ratios, default=(0.8, 0.1, 0.1),
                        help="Comma-separated train,test,val ratios "
                             "(default: 0.8,0.1,0.1).")
    parser.add_argument("--split-seed", type=int, default=42,
                        help="Seed for the deterministic stratified placement.")
    args = parser.parse_args()

    if not args.input.is_dir():
        raise SystemExit(f"input directory not found: {args.input}")

    keep_subtypes = not args.no_subtypes

    # ----- collect all records -----
    records: List[Dict[str, Any]] = []
    skipped = 0
    for apf_path in iter_apf_files(args.input):
        if 0 <= args.max_records <= len(records):
            break
        rec = parse_apf(apf_path, keep_subtypes=keep_subtypes)
        if rec is None:
            skipped += 1
            continue
        records.append(rec)

    print(f"Parsed: {_stats(records)} skipped_no_content={skipped}")

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

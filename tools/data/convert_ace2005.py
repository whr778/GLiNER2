"""Convert ACE 2005 English annotations to GLiNER2 JSONL.

Co-trains entities, relations, events, and entity coreference from a single ACE
document. ACE 2005 (LDC2006T06) is the canonical joint information-extraction
benchmark: 7 entity types (45 subtypes), 6 relation types (18 subtypes),
33 ``Type.Subtype`` event subtypes, and document-level entity & event coreference
clusters.

For each ``.apf.xml`` / ``.sgm`` pair in an ``adj/`` folder the converter emits
one record::

    {"input": "<doc headline + body>",
     "output": {
         "entities":  {"PER": ["John Smith", "He"], "ORG": ["AT&T Corp"]},
         "relations": [{"ORG-AFF": {"head": "John Smith", "tail": "AT&T Corp"}}],
         "events":    [{"event_type": "Conflict.Attack",
                        "triggers": ["attacked"],
                        "arguments": [{"role": "Attacker", "entity": "John Smith"}]}],
         "mention_types": {"John Smith": "NAM", "He": "PRO"}
     },
     "coref_clusters": [
         [{"text": "John Smith", "start": 0, "end": 10},
          {"text": "He", "start": 42, "end": 44}],
         [{"text": "AT&T Corp", "start": 21, "end": 30}]
     ]}

Taxonomy defaults match the standard ACE05 / OneIE benchmark and ``eb23a``'s
unified label space (``labels/unified-full-v4.yaml``):

* **Entities**: 7 top-level types (``PER``, ``ORG``, ``GPE``, ``LOC``, ``FAC``,
  ``VEH``, ``WEA`` -> mapped by ``unified-full-v4.yaml`` to ``Person``,
  ``Organization``, ``GPE``, ``Location``, ``Facility``, ``VEH``, ``WEA``). Pass
  ``--entity-subtypes`` for the 45 ``TYPE.SUBTYPE`` labels.
* **Relations**: 6 top-level types (``ART``, ``GEN-AFF``, ``ORG-AFF``,
  ``PART-WHOLE``, ``PER-SOC``, ``PHYS``). Pass ``--relation-subtypes`` for the
  18 ``TYPE.SUBTYPE`` labels.
* **Events**: 33 ``TYPE.SUBTYPE`` subtypes (``Conflict.Attack``, ``Life.Die``,
  etc.). Pass ``--no-event-subtypes`` for the 8 coarse types.

Pass ``--offsets`` to write event triggers and arguments as
``{"text": ..., "start": ..., "end": ...}`` character spans into the normalized
``input`` text (marking only the annotated occurrence during training while
remaining transparent to string-level evaluation). ``coref_clusters`` always
carries exact ``{"text", "start", "end"}`` mention spans per ``<entity>`` cluster.

Pass ``--merge-event-mentions`` to merge coreferent ``<event_mention>`` children
of the same ``<event>`` into a single multi-trigger event record (feeding
``record_coreferent_ownership`` and ``record_coref_link``). Default emits one
event dict per ``<event_mention>`` for OneIE / per-trigger comparability.

Entity surfaces use each mention's **head** span by default (dropping
determiners and modifiers: "the vice president" -> "president"); pass
``--extent-offsets`` to keep full extents across entities, relations, event
arguments, and ``coref_clusters``.

Usage::

    # Standard ACE05 + coref + offset-anchored event spans (recommended for eb23a):
    uv run python tools/data/convert_ace2005.py \\
        --input /path/to/ace_2005_td_v7/data/English \\
        --out data/ace2005.jsonl --offsets

    # Drop pronominal mentions (cascading to relations, event args, and clusters):
    uv run python tools/data/convert_ace2005.py \\
        --input /path/to/ace_2005_td_v7/data/English \\
        --out data/ace2005.jsonl --offsets \\
        --filter-config tools/data/config/mention_filter_nam_nom.yaml
"""

from __future__ import annotations

import argparse
import html
import re
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _mention_filter import MentionFilter, load_mention_filter  # noqa: E402
from _split import clean_text, dumps_record  # noqa: E402
from _stratify import (  # noqa: E402
    coverage_summary,
    derive_split_paths,
    parse_ratios,
    stratified_split,
)


SGM_TOKEN_RE = re.compile(r"<[^>]+>|&(?:#\d+|#x[0-9a-fA-F]+|\w+);|.", re.DOTALL)
BODY_RE = re.compile(r"<BODY\b[^>]*>(.*?)</BODY>", re.DOTALL | re.IGNORECASE)
TEXT_RE = re.compile(r"<TEXT\b[^>]*>(.*?)</TEXT>", re.DOTALL | re.IGNORECASE)
WS_RE = re.compile(r"\s+")
_NO_SPACE = object()


# ---------------------------------------------------------------------------
# SGM / APF parsing and exact offset projection
# ---------------------------------------------------------------------------

def _extract_sgml_with_map(sgm_text: str) -> Tuple[str, Dict[int, Tuple[int, int]]]:
    """Extract normalized document text and map LDC tagless char indices into it.

    LDC ACE 2005 ``<charseq START="s" END="e">`` counts 0-indexed inclusive
    characters across the tag-stripped, entity-unescaped ``.sgm`` document.
    ``<BODY>`` (preferring ``<HEADLINE>`` + ``<TEXT>`` when ``<BODY>`` is
    present, falling back to ``<TEXT>``) is extracted, HTML entities are
    unescaped, whitespace is collapsed, and ``clean_text`` normalization is
    applied while recording each LDC character's ``(start, end)`` in the final
    string.
    """
    m = BODY_RE.search(sgm_text) or TEXT_RE.search(sgm_text)
    c_start, c_end = (m.start(1), m.end(1)) if m else (0, len(sgm_text))

    stream: List[Tuple[str, Optional[int]]] = []
    ldc_idx = 0
    for tok_match in SGM_TOKEN_RE.finditer(sgm_text):
        tok = tok_match.group(0)
        in_body = c_start <= tok_match.start() < c_end
        if tok.startswith("<"):
            if in_body:
                stream.append((" ", None))
            continue
        decoded = html.unescape(tok)
        for ch in decoded:
            if in_body:
                stream.append((ch, ldc_idx))
            ldc_idx += 1

    out_parts: List[str] = []
    out_len = 0
    ldc_map: Dict[int, Tuple[int, int]] = {}
    pending_space: Any = _NO_SPACE

    for ch, pos in stream:
        norm = clean_text(ch)
        if not norm:
            continue
        if norm.isspace():
            if out_len > 0:
                if pending_space is _NO_SPACE or pos is not None:
                    pending_space = pos
            continue
        if pending_space is not _NO_SPACE:
            out_parts.append(" ")
            if isinstance(pending_space, int):
                ldc_map[pending_space] = (out_len, out_len + 1)
            out_len += 1
            pending_space = _NO_SPACE
        out_parts.append(norm)
        if pos is not None:
            ldc_map[pos] = (out_len, out_len + len(norm))
        out_len += len(norm)

    return "".join(out_parts), ldc_map


def _strip_sgml(sgm_text: str) -> str:
    """Return normalized document text from an ACE ``.sgm`` file."""
    text, _ = _extract_sgml_with_map(sgm_text)
    return text


def _resolve_charseq(
    parent: ET.Element,
    sub_tag: str,
    text: str,
    ldc_map: Dict[int, Tuple[int, int]],
) -> Tuple[Optional[str], Optional[Tuple[int, int]]]:
    """Return ``(surface, (start, end))`` in ``text`` for ``parent/<sub_tag>/charseq``."""
    sub = parent.find(sub_tag)
    if sub is None:
        return None, None
    cs = sub.find("charseq")
    if cs is None or cs.text is None:
        return None, None
    surface = clean_text(WS_RE.sub(" ", cs.text).strip())
    if not surface:
        return None, None

    s_attr, e_attr = cs.get("START"), cs.get("END")
    if s_attr is not None and e_attr is not None:
        try:
            s_ldc, e_ldc = int(s_attr), int(e_attr)
        except ValueError:
            s_ldc, e_ldc = -1, -2
        mapped = [ldc_map[k] for k in range(s_ldc, e_ldc + 1) if k in ldc_map]
        if mapped:
            cs_idx, ce_idx = mapped[0][0], mapped[-1][1]
            while cs_idx < ce_idx and text[cs_idx].isspace():
                cs_idx += 1
            while ce_idx > cs_idx and text[ce_idx - 1].isspace():
                ce_idx -= 1
            if text[cs_idx:ce_idx] == surface:
                return surface, (cs_idx, ce_idx)
            lo = max(0, cs_idx - 8)
            hi = min(len(text), cs_idx + len(surface) + 8)
            near = text.find(surface, lo, hi)
            if near != -1:
                return surface, (near, near + len(surface))

    if surface not in text:
        return None, None
    if s_attr is None and text.count(surface) == 1:
        idx = text.index(surface)
        return surface, (idx, idx + len(surface))
    return surface, None


def _mention_span(
    emention: ET.Element,
    use_head: bool,
    text: str,
    ldc_map: Dict[int, Tuple[int, int]],
) -> Tuple[Optional[str], Optional[Tuple[int, int]]]:
    """Surface text and ``(start, end)`` for an entity mention (head or extent)."""
    if use_head:
        head_text, head_span = _resolve_charseq(emention, "head", text, ldc_map)
        if head_text:
            return head_text, head_span
    return _resolve_charseq(emention, "extent", text, ldc_map)


def _as_gold_item(
    surface: str,
    span: Optional[Tuple[int, int]],
    offsets: bool,
    stats: Counter,
) -> Any:
    """Return ``{"text", "start", "end"}`` when ``offsets`` and ``span`` exist, else ``surface``."""
    if offsets:
        if span is not None:
            stats["offsets_placed"] += 1
            return {"text": surface, "start": span[0], "end": span[1]}
        stats["offsets_kept_as_string"] += 1
    return surface


def _pair_sgm(apf_path: Path) -> Optional[Path]:
    stem = apf_path.name
    if stem.endswith(".apf.xml"):
        sgm = apf_path.with_name(stem[:-len(".apf.xml")] + ".sgm")
    else:
        sgm = apf_path.with_suffix(".sgm")
    return sgm if sgm.is_file() else None


def parse_apf(
    apf_path: Path,
    keep_subtypes: Optional[bool] = None,
    mention_filter: Optional[MentionFilter] = None,
    stats: Optional[Counter] = None,
    use_head: bool = True,
    *,
    entity_subtypes: bool = False,
    relation_subtypes: bool = False,
    event_subtypes: bool = True,
    offsets: bool = False,
    merge_event_mentions: bool = False,
) -> Optional[Dict[str, Any]]:
    """Parse one ``.apf.xml`` + ``.sgm`` pair into a single GLiNER2 record.

    When ``keep_subtypes`` is explicitly passed (True or False), it sets all
    three ``entity_subtypes`` / ``relation_subtypes`` / ``event_subtypes`` flags
    for backwards compatibility. Otherwise the canonical ACE05 defaults apply:
    top-level entities (7), top-level relations (6), subtyped events (33).
    """
    if keep_subtypes is not None:
        entity_subtypes = keep_subtypes
        relation_subtypes = keep_subtypes
        event_subtypes = keep_subtypes
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
    text, ldc_map = _extract_sgml_with_map(sgm_text)
    if not text:
        return None

    # ----- entities & entity coreference clusters -----
    entity_mention_text: Dict[str, str] = {}
    entity_mention_span: Dict[str, Optional[Tuple[int, int]]] = {}
    mention_types_out: Dict[str, str] = {}
    entity_mention_type: Dict[str, str] = {}
    filtered_mids: Set[str] = set()
    entities_by_type: Dict[str, List[str]] = {}
    clusters_out: List[List[Dict[str, Any]]] = []

    for entity in root.iter("entity"):
        etype = (entity.get("TYPE") or "").strip()
        esub = (entity.get("SUBTYPE") or "").strip()
        if not etype:
            continue
        full_type = f"{etype}.{esub}" if entity_subtypes and esub else etype
        cluster_mentions: List[Dict[str, Any]] = []
        seen_cluster_spans: Set[Tuple[int, int]] = set()

        for emention in entity.iter("entity_mention"):
            mid = emention.get("ID")
            if not mid:
                continue
            m_type = (emention.get("TYPE") or "").strip()
            entity_mention_type[mid] = m_type
            span_text, span_offs = _mention_span(emention, use_head, text, ldc_map)
            if not span_text:
                continue
            if not mention_filter.allows(m_type):
                filtered_mids.add(mid)
                stats["filtered_mentions"] += 1
                continue
            entity_mention_text[mid] = span_text
            entity_mention_span[mid] = span_offs
            mention_types_out.setdefault(span_text, m_type)
            bucket = entities_by_type.setdefault(full_type, [])
            if span_text not in bucket:
                bucket.append(span_text)
            if span_offs is not None and span_offs not in seen_cluster_spans:
                seen_cluster_spans.add(span_offs)
                cluster_mentions.append({
                    "text": span_text,
                    "start": span_offs[0],
                    "end": span_offs[1],
                })
                stats["coref_mentions_placed"] += 1
            elif span_offs is None:
                stats["coref_mentions_unplaced"] += 1

        if cluster_mentions:
            cluster_mentions.sort(key=lambda m: (m["start"], m["end"]))
            clusters_out.append(cluster_mentions)

    clusters_out.sort(key=lambda c: (c[0]["start"], c[0]["end"]))

    # ----- relations -----
    relations_out: List[Dict[str, Dict[str, str]]] = []
    seen_rel: Set[Tuple[str, str, str]] = set()
    for rel in root.iter("relation"):
        rtype = (rel.get("TYPE") or "").strip()
        rsub = (rel.get("SUBTYPE") or "").strip()
        if not rtype:
            continue
        rel_type = f"{rtype}.{rsub}" if relation_subtypes and rsub else rtype
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

    # ----- non-entity value / timex2 lookup (for event arguments) -----
    non_entity_spans: Dict[str, Tuple[str, Optional[Tuple[int, int]]]] = {}
    for tag, mtag in (("timex2", "timex2_mention"), ("value", "value_mention")):
        for item in root.iter(tag):
            for m_elem in item.iter(mtag):
                mid = m_elem.get("ID")
                if not mid:
                    continue
                s_txt, s_off = _resolve_charseq(m_elem, "extent", text, ldc_map)
                if s_txt:
                    non_entity_spans[mid] = (s_txt, s_off)

    # ----- events -----
    events_out: List[Dict[str, Any]] = []
    for evt in root.iter("event"):
        etype = (evt.get("TYPE") or "").strip()
        esub = (evt.get("SUBTYPE") or "").strip()
        if not etype:
            continue
        event_type = f"{etype}.{esub}" if event_subtypes and esub else etype
        merged_triggers: List[Any] = []
        seen_trigs: Set[Any] = set()
        merged_args: List[Dict[str, Any]] = []
        seen_merged_args: Set[Any] = set()

        for emention in evt.iter("event_mention"):
            anchor_text, anchor_span = _resolve_charseq(emention, "anchor", text, ldc_map)
            if not anchor_text:
                continue
            trig_item = _as_gold_item(anchor_text, anchor_span, offsets, stats)
            trig_key = (anchor_text, anchor_span) if offsets and anchor_span is not None else anchor_text

            arguments: List[Dict[str, Any]] = []
            seen_args: Set[Any] = set()
            for arg in emention.iter("event_mention_argument"):
                role = (arg.get("ROLE") or "").strip()
                if not role:
                    continue
                refid = arg.get("REFID")
                if refid in entity_mention_type and not mention_filter.allows(
                    entity_mention_type[refid]
                ):
                    stats["filtered_event_args"] += 1
                    continue

                arg_text: Optional[str] = None
                arg_span: Optional[Tuple[int, int]] = None
                if use_head and refid in entity_mention_text:
                    arg_text = entity_mention_text[refid]
                    arg_span = entity_mention_span.get(refid)
                elif refid in non_entity_spans:
                    arg_text, arg_span = non_entity_spans[refid]
                if not arg_text:
                    arg_text, arg_span = _resolve_charseq(arg, "extent", text, ldc_map)
                if not arg_text:
                    continue

                dedup_key = (role, arg_text, arg_span) if offsets and arg_span is not None else (role, arg_text)
                if dedup_key in seen_args:
                    continue
                seen_args.add(dedup_key)
                arg_item = _as_gold_item(arg_text, arg_span, offsets, stats)
                arguments.append({"role": role, "entity": arg_item})

            if merge_event_mentions:
                if trig_key not in seen_trigs:
                    seen_trigs.add(trig_key)
                    merged_triggers.append(trig_item)
                for a, dk in zip(arguments, [
                    (a["role"], a["entity"]["text"], (a["entity"]["start"], a["entity"]["end"]))
                    if isinstance(a["entity"], dict) else (a["role"], a["entity"])
                    for a in arguments
                ]):
                    if dk not in seen_merged_args:
                        seen_merged_args.add(dk)
                        merged_args.append(a)
            else:
                events_out.append({
                    "event_type": event_type,
                    "triggers": [trig_item],
                    "arguments": arguments,
                })

        if merge_event_mentions and merged_triggers:
            events_out.append({
                "event_type": event_type,
                "triggers": merged_triggers,
                "arguments": merged_args,
            })

    output: Dict[str, Any] = {}
    if entities_by_type:
        output["entities"] = entities_by_type
    if relations_out:
        output["relations"] = relations_out
    if events_out:
        output["events"] = events_out
    if mention_types_out:
        output["mention_types"] = mention_types_out

    if not output:
        return None
    record: Dict[str, Any] = {"input": text, "output": output}
    if clusters_out:
        record["coref_clusters"] = clusters_out
    return record


def iter_apf_files(root: Path):
    """Yield ``.apf.xml`` files from the adjudicated (``adj``) folders only."""
    for path in sorted(root.rglob("*.apf.xml")):
        if "adj" in path.relative_to(root).parts:
            yield path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _stats(records: List[Dict[str, Any]]) -> str:
    n_ent = sum(len(r["output"].get("entities") or {}) for r in records)
    n_rel = sum(len(r["output"].get("relations") or []) for r in records)
    n_evt = sum(len(r["output"].get("events") or []) for r in records)
    n_clu = sum(len(r.get("coref_clusters") or []) for r in records)
    return (
        f"docs={len(records)} entity_types_sum={n_ent} "
        f"relations={n_rel} events={n_evt} coref_clusters={n_clu}"
    )


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
    parser.add_argument("--entity-subtypes", action="store_true",
                        help="Emit 45 entity TYPE.SUBTYPE labels instead of the "
                             "canonical 7 top-level types (PER/ORG/GPE/LOC/FAC/VEH/WEA).")
    parser.add_argument("--relation-subtypes", action="store_true",
                        help="Emit 18 relation TYPE.SUBTYPE labels instead of the "
                             "canonical 6 top-level types.")
    parser.add_argument("--no-event-subtypes", action="store_true",
                        help="Drop SUBTYPE on events (8 coarse types instead of 33).")
    parser.add_argument("--all-subtypes", action="store_true",
                        help="Emit TYPE.SUBTYPE on entities, relations, and events.")
    parser.add_argument("--no-subtypes", action="store_true",
                        help="Drop SUBTYPE everywhere (entities, relations, and events).")
    parser.add_argument("--offsets", action="store_true",
                        help="Write event triggers and arguments as "
                             "{text, start, end} spans into the normalized input.")
    parser.add_argument("--merge-event-mentions", action="store_true",
                        help="Merge coreferent <event_mention> children of one "
                             "<event> into a single multi-trigger event record.")
    parser.add_argument("--extent-offsets", action="store_true",
                        help="Use each mention's full extent span instead of the "
                             "default syntactic head span.")
    parser.add_argument("--no-stratify", action="store_true",
                        help="Disable stratified split; write a single file at --out.")
    parser.add_argument("--split-ratios", type=parse_ratios, default=(0.8, 0.1, 0.1),
                        help="Comma-separated train,test,val ratios "
                             "(default: 0.8,0.1,0.1).")
    parser.add_argument("--split-seed", type=int, default=42,
                        help="Seed for the deterministic stratified placement.")
    parser.add_argument("--filter-config", type=Path, default=None,
                        help="YAML mention-type filter (see "
                             "tools/data/config/mention_filter*.yaml). Default: keep all.")
    args = parser.parse_args()

    if not args.input.is_dir():
        raise SystemExit(f"input directory not found: {args.input}")
    if args.all_subtypes and args.no_subtypes:
        raise SystemExit("cannot combine --all-subtypes and --no-subtypes")

    entity_subtypes = (args.entity_subtypes or args.all_subtypes) and not args.no_subtypes
    relation_subtypes = (args.relation_subtypes or args.all_subtypes) and not args.no_subtypes
    event_subtypes = (not args.no_event_subtypes or args.all_subtypes) and not args.no_subtypes
    use_head = not args.extent_offsets
    mention_filter = load_mention_filter(args.filter_config, "ace2005")
    stats: Counter = Counter()

    records: List[Dict[str, Any]] = []
    skipped = 0
    for apf_path in iter_apf_files(args.input):
        if 0 <= args.max_records <= len(records):
            break
        rec = parse_apf(
            apf_path,
            mention_filter=mention_filter,
            stats=stats,
            use_head=use_head,
            entity_subtypes=entity_subtypes,
            relation_subtypes=relation_subtypes,
            event_subtypes=event_subtypes,
            offsets=args.offsets,
            merge_event_mentions=args.merge_event_mentions,
        )
        if rec is None:
            skipped += 1
            continue
        records.append(rec)

    print(f"Parsed: {_stats(records)} skipped_no_content={skipped}")
    print(
        f"Coref mentions: placed={stats['coref_mentions_placed']} "
        f"unplaced={stats['coref_mentions_unplaced']}"
    )
    if args.offsets:
        print(
            f"[offsets] placed={stats['offsets_placed']} "
            f"kept_as_string={stats['offsets_kept_as_string']}"
        )
    if mention_filter.active:
        print(
            f"Mention filter ({mention_filter.describe()}): "
            f"dropped_mentions={stats['filtered_mentions']} "
            f"dropped_relations={stats['filtered_relations']} "
            f"dropped_event_args={stats['filtered_event_args']}"
        )

    if args.no_stratify:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", encoding="utf-8") as f:
            for rec in records:
                f.write(dumps_record(rec) + "\n")
        print(f"Wrote {len(records)} records -> {args.out}")
        return 0

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

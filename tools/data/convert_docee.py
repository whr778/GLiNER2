"""Convert DocEE (Tong et al., NAACL 2022) to GLiNER2 JSONL.

DocEE is the largest publicly-available document-level event extraction
corpus: 27,485 documents, 59 event types, 356 argument-role types, and
180,528 argument instances. It follows a strict one-event-per-document
paradigm — the event type is a document-level label and arguments are
typed spans scattered through the text. The dataset does **not**
annotate triggers.

Because DocEE has no explicit triggers, by default this converter maps
each document into the most faithful GLiNER2 shape we can give it:

* ``output.entities``  — arguments grouped by their role ``type``
  (``Husband``, ``Court``, ``Cause``, ``Date``, …). 356 role-typed
  argument-NER buckets across the corpus.
* ``output.classifications``  — one record per doc with the document's
  ``event_type`` as ``true_label`` and the full 59-type vocabulary as
  ``labels``.

Pass ``--emit-events`` to additionally emit an ``output.events`` record
with a **synthetic** trigger: the event-type literal in square brackets
prepended to the document text (so the trigger always appears in the
input). This gives the events head some supervision at the cost of a
two-step inference protocol (predict event_type first, then re-feed
with the prefix); off by default.

DocEE data is distributed behind a Google Drive folder linked from
https://github.com/tongmeihan1995/docee — there is no public direct URL,
so you must download it manually. Pass the resulting JSON file via
``--input``. Expected layouts:

* ``DocEE-en.json``                  (all data)
* ``normal_setting/{train,dev,test}.json``  (canonical splits)
* ``cross_domain_setting/...``       (out-of-domain splits)

The converter reads the file (top-level list of records, or a dict
wrapping a list under ``data``/``records``/``examples``/``items``),
extracts text/event_type/annotations from a flexible set of likely
field names, and emits either:

* a stratified 80/10/10 train/test/val split (default), useful when
  you pass the all-data file; or
* a single output JSONL file when you pass ``--no-stratify``, useful
  when you're converting one canonical-split file at a time.

Usage::

    # Auto-stratify the all-data JSON into 80/10/10 splits:
    uv run python tools/data/convert_docee.py \\
        --input data/docee/DocEE-en.json \\
        --out data/docee.jsonl

    # Or use the canonical normal_setting splits (one converter call each):
    uv run python tools/data/convert_docee.py --no-stratify \\
        --input data/docee/normal_setting/train.json --out data/docee.train.jsonl
    uv run python tools/data/convert_docee.py --no-stratify \\
        --input data/docee/normal_setting/dev.json   --out data/docee.val.jsonl
    uv run python tools/data/convert_docee.py --no-stratify \\
        --input data/docee/normal_setting/test.json  --out data/docee.test.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _split import dumps_record  # noqa: E402
from _stratify import (  # noqa: E402
    coverage_summary,
    derive_split_paths,
    parse_ratios,
    stratified_split,
)


# Candidate field-name lists — DocEE has been distributed in slightly
# different shapes over time; try the most likely names in order.
TEXT_KEYS = ("text", "content", "body", "passage")
EVENT_TYPE_KEYS = ("event_type", "type", "label", "event")
ANNOTATIONS_KEYS = (
    "annotations", "args", "arguments", "labels",
    "meta", "metadata", "event_arguments",
)


def _normalise_record(raw: Any) -> Optional[Dict[str, Any]]:
    """Coerce a raw DocEE record into a uniform dict.

    The published JSON / pickle layout uses 4-element lists::

        [title, text, event_type, annotations]

    where ``annotations`` is either a list of ``{start, end, type, text}``
    dicts or a JSON-encoded string of the same. Older releases shipped
    everything as dicts. We accept either.
    """
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, (list, tuple)) and len(raw) >= 4:
        title, text, event_type, annotations = raw[0], raw[1], raw[2], raw[3]
        if isinstance(annotations, str):
            try:
                annotations = json.loads(annotations)
            except json.JSONDecodeError:
                annotations = []
        return {
            "title": title if isinstance(title, str) else None,
            "text": text if isinstance(text, str) else None,
            "event_type": event_type if isinstance(event_type, str) else None,
            "annotations": annotations if isinstance(annotations, list) else [],
        }
    return None


def _load_records(input_path: Path) -> List[Dict[str, Any]]:
    """Load DocEE records from a JSON file (top-level list, or dict-wrapped)."""
    with input_path.open(encoding="utf-8") as fh:
        data = json.load(fh)
    items: List[Any]
    if isinstance(data, list):
        items = data
    elif isinstance(data, dict):
        items = []
        for key in ("data", "records", "examples", "items"):
            if key in data and isinstance(data[key], list):
                items = data[key]
                break
    else:
        items = []
    out: List[Dict[str, Any]] = []
    for raw in items:
        rec = _normalise_record(raw)
        if rec is not None:
            out.append(rec)
    if not out:
        raise SystemExit(f"could not find a list of records in {input_path}")
    return out


def _first_value(rec: Dict[str, Any], keys) -> Any:
    """Return the first non-empty value among the candidate keys."""
    for k in keys:
        v = rec.get(k)
        if v not in (None, "", [], {}):
            return v
    return None


def _get_text(rec: Dict[str, Any]) -> Optional[str]:
    v = _first_value(rec, TEXT_KEYS)
    return v if isinstance(v, str) else None


def _get_event_type(rec: Dict[str, Any]) -> Optional[str]:
    v = _first_value(rec, EVENT_TYPE_KEYS)
    return v.strip() if isinstance(v, str) else None


def _get_annotations(rec: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract the list of ``{start, end, type, text}`` argument annotations.

    The upstream training pipeline stores the annotations as a JSON
    *string* inside the pickle records; the all-data JSON file is more
    likely to have them as a real list, but we accept both.
    """
    raw = _first_value(rec, ANNOTATIONS_KEYS)
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            return []
    if isinstance(raw, list):
        return [a for a in raw if isinstance(a, dict)]
    return []


def collect_event_types(records: List[Dict[str, Any]]) -> List[str]:
    """First pass — discover the full event_type vocabulary."""
    counts: Counter = Counter()
    for rec in records:
        et = _get_event_type(rec)
        if et:
            counts[et] += 1
    return sorted(counts.keys())


def convert_row(
    rec: Dict[str, Any],
    classification_labels: List[str],
    emit_events: bool,
) -> Optional[Dict[str, Any]]:
    text = _get_text(rec)
    event_type = _get_event_type(rec)
    if not isinstance(text, str) or not text.strip():
        return None
    if not event_type:
        return None

    # Bucket arguments by their role type; same surfaces feed both the
    # entities dict and the events arguments list (when --emit-events).
    entities: Dict[str, List[str]] = {}
    args_for_event: List[Dict[str, str]] = []
    for ann in _get_annotations(rec):
        role = ann.get("type")
        surface = ann.get("text")
        if not isinstance(role, str) or not isinstance(surface, str):
            continue
        role = role.strip()
        surface = surface.strip()
        if not role or not surface or surface not in text:
            continue
        bucket = entities.setdefault(role, [])
        if surface not in bucket:
            bucket.append(surface)
        args_for_event.append({"role": role, "entity": surface})

    output: Dict[str, Any] = {}
    if entities:
        output["entities"] = entities
    output["classifications"] = [{
        "task": "docee_event",
        "labels": list(classification_labels),
        "true_label": [event_type],
    }]

    if emit_events and args_for_event:
        # Synthesize a trigger by prepending the event-type literal to
        # the text. The original entity surfaces stay verbatim since
        # we only PREPEND, never modify the body.
        trigger = f"[{event_type}]"
        text = f"{trigger} {text}"
        output["events"] = [{
            "event_type": event_type,
            "trigger": trigger,
            "arguments": args_for_event,
        }]

    if not (output.get("entities") or output.get("events")):
        # Pure-classification records are still useful but we want at
        # least one extraction signal — drop docs with no usable args.
        return None
    return {"input": text, "output": output}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", required=True, type=Path,
                        help="Path to a DocEE JSON file (DocEE-en.json or "
                             "one of the canonical split files).")
    parser.add_argument("--out", required=True, type=Path,
                        help="Output JSONL base path. Default mode writes "
                             "<base>.train.jsonl / .test.jsonl / .val.jsonl "
                             "(stratified 80/10/10). With --no-stratify, "
                             "--out is the single output file.")
    parser.add_argument("--max-records", type=int, default=-1,
                        help="Maximum documents to emit (-1 = all).")
    parser.add_argument("--no-stratify", action="store_true",
                        help="Write a single output file at --out without "
                             "splitting (use when --input is already a "
                             "canonical train/dev/test file).")
    parser.add_argument("--emit-events", action="store_true",
                        help="Additionally emit an events block per record "
                             "with a synthetic trigger (the event-type "
                             "literal prepended to the doc text). Default "
                             "off — only entities + classification are "
                             "emitted.")
    parser.add_argument("--split-ratios", type=parse_ratios, default=(0.8, 0.1, 0.1),
                        help="Comma-separated train,test,val ratios "
                             "(default: 0.8,0.1,0.1).")
    parser.add_argument("--split-seed", type=int, default=42,
                        help="Seed for the deterministic stratified placement.")
    args = parser.parse_args()

    if not args.input.is_file():
        raise SystemExit(f"input not found: {args.input}")

    print(f"Loading {args.input}...")
    raw_records = _load_records(args.input)
    print(f"  loaded {len(raw_records):,} raw records")

    event_types = collect_event_types(raw_records)
    if not event_types:
        raise SystemExit(
            "could not discover any event_type values — check the "
            "input file's schema against the docstring's expected fields."
        )
    print(f"  discovered {len(event_types)} distinct event types "
          f"(classification vocab)")

    records: List[Dict[str, Any]] = []
    skipped = 0
    for rec in raw_records:
        if 0 <= args.max_records <= len(records):
            break
        out = convert_row(rec, event_types, emit_events=args.emit_events)
        if out is None:
            skipped += 1
            continue
        records.append(out)
    print(f"  converted {len(records):,} (skipped {skipped})")

    if not records:
        raise SystemExit("no usable records — check field names and offsets.")

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

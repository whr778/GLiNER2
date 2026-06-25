"""Convert WikiEvents (Li et al., NAACL 2021) to GLiNER2 JSONL.

WikiEvents pairs document-level event extraction (event triggers +
typed arguments, KAIROS ontology with ~50 event types and ~60 roles)
with entity-mention NER, so we emit both into the same record for
co-training. The dataset's ``relation_mentions`` field is always
empty across train/dev/test, so this converter does not produce
relations.

Source layout — one JSONL per split, hosted on a public S3 bucket from
https://github.com/raspberryice/gen-arg::

    https://gen-arg-data.s3.us-east-2.amazonaws.com/wikievents/data/train.jsonl
    https://gen-arg-data.s3.us-east-2.amazonaws.com/wikievents/data/dev.jsonl
    https://gen-arg-data.s3.us-east-2.amazonaws.com/wikievents/data/test.jsonl

Per record::

    {"doc_id": ..., "text": "<full doc>", "tokens": [...],
     "entity_mentions": [{"entity_type": "PER", "text": "Prayuth Chan-ocha",
                          "start": 11, "end": 15, ...}, ...],
     "event_mentions": [{"event_type": "Life.Injure.Unspecified",
                         "trigger": {"text": "injured", "start": 62, "end": 63},
                         "arguments": [{"role": "Victim",
                                        "text": "Terry Duffield",
                                        "entity_id": "..."}, ...]}, ...]}

Output (one record per source document)::

    {"input": "<text>",
     "output": {
         "entities": {"PER": ["Prayuth Chan-ocha", ...], "GPE": ["Thailand"], ...},
         "events":   [{"event_type": "Life.Injure.Unspecified",
                       "trigger": "injured",
                       "arguments": [{"role": "Victim", "entity": "Terry Duffield"}]}]
     }}

WikiEvents ships canonical train/dev/test splits, so the converter
emits a single JSONL file per call (mirroring MAVEN / RAMS / ACE 2005
conventions) — run it three times to populate all splits. ``--input``
accepts either a local path or an ``http(s)://`` URL; the URL is
streamed without buffering into memory.

Usage::

    # All three splits (S3 URLs are the defaults for --input):
    uv run python tools/data/convert_wikievents.py \\
        --input https://gen-arg-data.s3.us-east-2.amazonaws.com/wikievents/data/train.jsonl \\
        --out data/wikievents.train.jsonl

    uv run python tools/data/convert_wikievents.py \\
        --input https://gen-arg-data.s3.us-east-2.amazonaws.com/wikievents/data/dev.jsonl \\
        --out data/wikievents.dev.jsonl

    uv run python tools/data/convert_wikievents.py \\
        --input https://gen-arg-data.s3.us-east-2.amazonaws.com/wikievents/data/test.jsonl \\
        --out data/wikievents.test.jsonl

    # Or with the --split shorthand (downloads from the canonical URL):
    uv run python tools/data/convert_wikievents.py --split train --out data/wikievents.train.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path
from typing import Any, Dict, Iterable, List

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _split import dumps_record  # noqa: E402


S3_BASE = "https://gen-arg-data.s3.us-east-2.amazonaws.com/wikievents/data"


def _iter_jsonl_from_url(url: str) -> Iterable[Dict[str, Any]]:
    req = urllib.request.Request(url, headers={"User-Agent": "curl/8"})
    with urllib.request.urlopen(req, timeout=120) as r:
        for line in r:
            line = line.decode("utf-8").strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _iter_jsonl_from_file(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _iter_records(source: str) -> Iterable[Dict[str, Any]]:
    if source.startswith("http://") or source.startswith("https://"):
        yield from _iter_jsonl_from_url(source)
    else:
        p = Path(source)
        if not p.is_file():
            raise SystemExit(f"input not found: {source}")
        yield from _iter_jsonl_from_file(p)


def convert_row(row: Dict[str, Any]) -> Dict[str, Any] | None:
    """Convert one WikiEvents document to a GLiNER2 record; None if unusable."""
    text = row.get("text")
    if not isinstance(text, str) or not text.strip():
        return None

    # ---- entities ----
    entities_by_type: Dict[str, List[str]] = {}
    for mention in row.get("entity_mentions") or []:
        if not isinstance(mention, dict):
            continue
        etype = mention.get("entity_type")
        surface = mention.get("text")
        if not isinstance(etype, str) or not isinstance(surface, str):
            continue
        etype = etype.strip()
        surface = surface.strip()
        if not etype or not surface or surface not in text:
            continue
        bucket = entities_by_type.setdefault(etype, [])
        if surface not in bucket:
            bucket.append(surface)

    # ---- events ----
    events_out: List[Dict[str, Any]] = []
    for ev in row.get("event_mentions") or []:
        if not isinstance(ev, dict):
            continue
        etype = ev.get("event_type")
        trigger = ev.get("trigger") or {}
        trigger_text = trigger.get("text") if isinstance(trigger, dict) else None
        if not isinstance(etype, str) or not isinstance(trigger_text, str):
            continue
        etype = etype.strip()
        trigger_text = trigger_text.strip()
        if not etype or not trigger_text or trigger_text not in text:
            continue
        arguments: List[Dict[str, str]] = []
        seen_args: set = set()
        for arg in ev.get("arguments") or []:
            if not isinstance(arg, dict):
                continue
            role = arg.get("role")
            arg_text = arg.get("text")
            if not isinstance(role, str) or not isinstance(arg_text, str):
                continue
            role = role.strip()
            arg_text = arg_text.strip()
            if not role or not arg_text or arg_text not in text:
                continue
            key = (role, arg_text)
            if key in seen_args:
                continue
            seen_args.add(key)
            arguments.append({"role": role, "entity": arg_text})
        events_out.append({
            "event_type": etype,
            "trigger": trigger_text,
            "arguments": arguments,
        })

    output: Dict[str, Any] = {}
    if entities_by_type:
        output["entities"] = entities_by_type
    if events_out:
        output["events"] = events_out
    if not output:
        return None
    return {"input": text, "output": output}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    src_group = parser.add_mutually_exclusive_group(required=True)
    src_group.add_argument(
        "--input",
        help="Local path or URL of a WikiEvents jsonl file "
             "(train.jsonl / dev.jsonl / test.jsonl).",
    )
    src_group.add_argument(
        "--split", choices=("train", "dev", "test"),
        help="Convenience: download the named split from the canonical "
             f"S3 bucket ({S3_BASE}).",
    )
    parser.add_argument("--out", required=True, type=Path,
                        help="Output GLiNER2 JSONL path.")
    parser.add_argument("--max-records", type=int, default=-1,
                        help="Maximum documents to emit (-1 = all).")
    args = parser.parse_args()

    source = args.input if args.input else f"{S3_BASE}/{args.split}.jsonl"
    args.out.parent.mkdir(parents=True, exist_ok=True)

    emitted = 0
    skipped_empty = 0
    total_entities = 0
    total_events = 0
    total_args = 0
    all_ent_types: set = set()
    all_ev_types: set = set()
    all_roles: set = set()

    print(f"Reading {source}")
    with args.out.open("w", encoding="utf-8") as f:
        for idx, row in enumerate(_iter_records(source)):
            if 0 <= args.max_records <= idx:
                break
            record = convert_row(row)
            if record is None:
                skipped_empty += 1
                continue
            f.write(dumps_record(record) + "\n")
            emitted += 1
            ents = record["output"].get("entities") or {}
            for t, surfaces in ents.items():
                total_entities += len(surfaces)
                all_ent_types.add(t)
            for ev in record["output"].get("events") or []:
                total_events += 1
                all_ev_types.add(ev["event_type"])
                for arg in ev["arguments"]:
                    total_args += 1
                    all_roles.add(arg["role"])

    print(
        f"Done. emitted={emitted} skipped_empty={skipped_empty} "
        f"total_entities={total_entities} entity_types={len(all_ent_types)} "
        f"total_events={total_events} event_types={len(all_ev_types)} "
        f"total_arguments={total_args} roles={len(all_roles)} -> {args.out}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

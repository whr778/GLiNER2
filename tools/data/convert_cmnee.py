"""Convert CMNEE (Zhu et al., LREC-COLING 2024) to GLiNER2 JSONL.

CMNEE — *Chinese Military News Event Extraction* — is a 17,000-document
event-extraction corpus with both triggers and typed arguments, all
manually annotated against an 8 event-type / 11 argument-role schema.
Document-level, multi-event-per-document, character-offset annotations.
Released under LREC-COLING 2024 by Zhu et al. (NUDT).

This is the first Chinese corpus in the training mix, complementing
``gliner_multilingual`` (Chinese NER) and the English event corpora
(ACE / MAVEN / RAMS / WikiEvents / CASIE / DocEE).

Each input record::

    {"id": "...",
     "text": "<Chinese document>",
     "event_list": [
         {"event_type": "Manoeuvre",
          "trigger":   {"text": "执行", "offset": [18, 20]},
          "arguments": [{"role": "Subject", "text": "美国海军", "offset": [6, 10]},
                        {"role": "Date",    "text": "6月13日", "offset": [0, 5]}, ...]},
         ...
     ],
     "coref_arguments": [[...], [...]]}

…maps directly to GLiNER2's event shape:

    {"input": "<doc text>",
     "output": {"events": [{"event_type": "Manoeuvre",
                            "triggers": ["执行"],
                            "arguments": [{"role": "Subject", "entity": "美国海军"}, ...]}]}}

The ``coref_arguments`` clusters are kept as a top-level ``coref_clusters`` (mentions with
code-point offsets into the CLEANED text ``dumps_record`` writes; see add_coref_clusters.py).
The files in data/ are the DEDUPED splits; a fresh run must pass run_all_converters.sh's leakage gate.
Surfaces are verbatim-filtered against ``text``; the source's char
offsets are authoritative so this is essentially a defensive check.

CMNEE ships canonical train/valid/test splits (12k / 2k / 3k), so we
write one output file per call — mirroring WikiEvents / MAVEN / RAMS.
Run the converter three times, once per split.

Download: the data lives behind a Google Drive folder linked from the
upstream README; ``gdown`` wraps it::

    mkdir -p data/cmnee
    uv run --with gdown gdown --folder \\
        'https://drive.google.com/drive/folders/1nfKiSsu88oBeykUSYm7NGn4Q50_2GPS1' \\
        -O data/cmnee/

Usage::

    uv run python tools/data/convert_cmnee.py \\
        --input data/cmnee/CMNEE/train.json --out data/cmnee.train.jsonl
    uv run python tools/data/convert_cmnee.py \\
        --input data/cmnee/CMNEE/valid.json --out data/cmnee.val.jsonl
    uv run python tools/data/convert_cmnee.py \\
        --input data/cmnee/CMNEE/test.json  --out data/cmnee.test.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _split import clean_text, dumps_record  # noqa: E402
from add_coref_clusters import place  # noqa: E402


OFFSET_COUNTS: Dict[str, int] = {"placed": 0, "shifted": 0, "kept_as_string": 0}


def _offset_gold(text: str, cleaned: str, surface: str, offset: Any) -> Any:
    """``{text, start, end}`` in the CLEANED text ``dumps_record`` writes, from CMNEE's ``[start, end]`` character
    offsets; the surface string when no nearby shift reads exactly the surface (TODO #26)."""
    if isinstance(offset, (list, tuple)) and len(offset) == 2 and all(isinstance(x, int) for x in offset):
        for shift in (0, -1, 1, -2, 2):
            s, e = offset[0] + shift, offset[1] + shift
            if 0 <= s < e <= len(text) and text[s:e].strip() == surface:
                placed = place(cleaned, [[(surface, s, e)]], clean_text, text, {"kept": 0, "dropped": 0})
                if placed:
                    OFFSET_COUNTS["placed"] += 1
                    OFFSET_COUNTS["shifted"] += shift != 0
                    return placed[0][0]
    OFFSET_COUNTS["kept_as_string"] += 1
    return surface


def convert_row(row: Dict[str, Any], offsets: bool = False) -> Optional[Dict[str, Any]]:
    """Convert one CMNEE document into a GLiNER2 record; None if unusable. ``offsets`` writes triggers and
    arguments as ``{text, start, end}`` (TODO #26) so training marks only the annotated occurrence."""
    text = row.get("text")
    if not isinstance(text, str) or not text.strip():
        return None
    cleaned = clean_text(text) if offsets else None

    events_out: List[Dict[str, Any]] = []
    for ev in row.get("event_list") or []:
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
            entity: Any = _offset_gold(text, cleaned, arg_text, arg.get("offset")) if offsets else arg_text
            arguments.append({"role": role, "entity": entity})

        trig: Any = _offset_gold(text, cleaned, trigger_text, trigger.get("offset")) if offsets else trigger_text
        events_out.append({
            "event_type": etype,
            "triggers": [trig],
            "arguments": arguments,
        })

    if not events_out:
        return None
    clusters = [[(m["text"], m["offset"][0], m["offset"][1]) for m in c] for c in row.get("coref_arguments") or []]
    coref = place(clean_text(text), clusters, clean_text, text, {"kept": 0, "dropped": 0})
    return {"input": text, "output": {"events": events_out}, "coref_clusters": coref}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", required=True, type=Path,
                        help="Local path to a CMNEE split file "
                             "(train.json / valid.json / test.json).")
    parser.add_argument("--out", required=True, type=Path,
                        help="Output GLiNER2 events JSONL path.")
    parser.add_argument("--offsets", action="store_true",
                        help="write triggers and arguments as {text, start, end} into the cleaned text (TODO #26)")
    parser.add_argument("--max-records", type=int, default=-1,
                        help="Maximum documents to emit (-1 = all).")
    args = parser.parse_args()

    if not args.input.is_file():
        raise SystemExit(f"input not found: {args.input}")

    with args.input.open(encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, list):
        raise SystemExit(f"expected a JSON array of records, got {type(data).__name__}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    emitted = 0
    skipped_empty = 0
    total_events = 0
    total_args = 0
    all_event_types: set = set()
    all_roles: set = set()

    with args.out.open("w", encoding="utf-8") as f:
        for idx, row in enumerate(data):
            if 0 <= args.max_records <= idx:
                break
            record = convert_row(row, offsets=args.offsets)
            if record is None:
                skipped_empty += 1
                continue
            f.write(dumps_record(record) + "\n")
            emitted += 1
            for ev in record["output"]["events"]:
                total_events += 1
                all_event_types.add(ev["event_type"])
                for arg in ev["arguments"]:
                    total_args += 1
                    all_roles.add(arg["role"])

    if args.offsets:
        print(f"[offsets] placed {OFFSET_COUNTS['placed']} (of which shifted {OFFSET_COUNTS['shifted']}) | kept as string {OFFSET_COUNTS['kept_as_string']}")
    print(
        f"Done. emitted={emitted} skipped_empty={skipped_empty} "
        f"total_events={total_events} event_types={len(all_event_types)} "
        f"total_arguments={total_args} roles={len(all_roles)} -> {args.out}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

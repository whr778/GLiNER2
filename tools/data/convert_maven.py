"""Convert MAVEN trigger-only event detection JSONL to GLiNER2 events JSONL.

MAVEN (Wang et al., EMNLP 2020) is trigger-and-type detection only — no
argument annotations. Each input document carries a list of events with
typed mentions. This converter emits one GLiNER2 record per document
with the event-extraction shape::

    {"input": "<doc text>",
     "output": {"events": [
         {"event_type": "<type>", "trigger": "<surface>", "arguments": []},
         ...
     ]}}

Each MAVEN mention becomes one event with an empty ``arguments`` list, so
the trigger-detection part of the joint loss is trained while the
argument head sees no signal from MAVEN (use RAMS or ACE for that).

MAVEN is distributed by Tsinghua and Google Drive, not via HuggingFace —
the user must download ``train.jsonl`` / ``valid.jsonl`` / ``test.jsonl``
from https://github.com/THU-KEG/MAVEN-dataset and point ``--input`` at one
of them. ``test.jsonl`` has no gold labels (only candidates) and is
skipped automatically.

Usage::

    uv run python tools/data/convert_maven.py \\
        --input /path/to/maven/train.jsonl \\
        --out data/maven.train.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List


def _flatten_tokens(sentences: List[Dict[str, Any]]) -> List[str]:
    """Concatenate per-sentence token lists into one flat token list."""
    tokens: List[str] = []
    for sent in sentences:
        toks = sent.get("tokens") if isinstance(sent, dict) else None
        if isinstance(toks, list):
            tokens.extend(t for t in toks if isinstance(t, str))
    return tokens


def _sentence_token_offsets(sentences: List[Dict[str, Any]]) -> List[int]:
    """Cumulative start index of each sentence in the flat token list."""
    offsets: List[int] = []
    acc = 0
    for sent in sentences:
        offsets.append(acc)
        toks = sent.get("tokens") if isinstance(sent, dict) else None
        if isinstance(toks, list):
            acc += sum(1 for t in toks if isinstance(t, str))
    return offsets


def convert_row(row: Dict[str, Any]) -> Dict[str, Any] | None:
    """Convert one MAVEN document into a GLiNER2 record; None if unusable."""
    sentences = row.get("content")
    events_in = row.get("events")
    if not isinstance(sentences, list) or not sentences:
        return None
    tokens = _flatten_tokens(sentences)
    if not tokens:
        return None
    text = " ".join(tokens)
    sent_starts = _sentence_token_offsets(sentences)

    events_out: List[Dict[str, Any]] = []
    if isinstance(events_in, list):
        for evt in events_in:
            if not isinstance(evt, dict):
                continue
            etype = evt.get("type")
            mentions = evt.get("mention")
            if not isinstance(etype, str) or not isinstance(mentions, list):
                continue
            etype = etype.strip()
            if not etype:
                continue
            for m in mentions:
                if not isinstance(m, dict):
                    continue
                sid = m.get("sent_id")
                offset = m.get("offset")
                if not isinstance(sid, int) or not isinstance(offset, list) or len(offset) != 2:
                    continue
                if not (0 <= sid < len(sent_starts)):
                    continue
                s = sent_starts[sid] + int(offset[0])
                e = sent_starts[sid] + int(offset[1])
                if not (0 <= s < e <= len(tokens)):
                    continue
                trigger = " ".join(tokens[s:e]).strip()
                if not trigger or trigger not in text:
                    continue
                events_out.append({
                    "event_type": etype,
                    "trigger": trigger,
                    "arguments": [],
                })

    if not events_out:
        return None
    return {"input": text, "output": {"events": events_out}}


def _iter_jsonl(path: Path):
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", required=True, type=Path,
                        help="Local path to a MAVEN .jsonl file "
                             "(train.jsonl or valid.jsonl).")
    parser.add_argument("--out", required=True, type=Path,
                        help="Output GLiNER2 events JSONL path.")
    parser.add_argument("--max-records", type=int, default=-1,
                        help="Maximum input records to read (-1 = all).")
    args = parser.parse_args()

    if not args.input.is_file():
        raise SystemExit(f"input not found: {args.input}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    emitted = 0
    skipped_empty = 0
    total_events = 0
    all_types: set[str] = set()

    with args.out.open("w") as f:
        for idx, row in enumerate(_iter_jsonl(args.input)):
            if 0 <= args.max_records <= idx:
                break
            record = convert_row(row)
            if record is None:
                skipped_empty += 1
                continue
            f.write(json.dumps(record) + "\n")
            emitted += 1
            events = record["output"]["events"]
            total_events += len(events)
            all_types.update(e["event_type"] for e in events)
            if emitted % 200 == 0:
                print(f"  emitted={emitted}  skipped_empty={skipped_empty}  "
                      f"event_types={len(all_types)}")

    print(f"Done. emitted={emitted} skipped_empty={skipped_empty} "
          f"total_events={total_events} distinct_event_types={len(all_types)} "
          f"-> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

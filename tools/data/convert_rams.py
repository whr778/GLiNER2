"""Convert RAMS multi-sentence event linking jsonlines to GLiNER2 events JSONL.

RAMS (Ebner et al., ACL 2020) annotates events with trigger spans and
multi-sentence argument links. Each input document carries one event
(trigger + roles); arguments may live in neighbouring sentences.

Input shape (per line)::

    {
      "doc_key": "nw_...",
      "sentences": [["Three", "specific", ...], ...],
      "evt_triggers": [[trig_start, trig_end, [["life.die.n/a", 1.0]]]],
      "ent_spans": [[a_start, a_end, [["evt089arg02place", 1.0]]], ...],
      "gold_evt_links": [
          [[trig_start, trig_end], [a_start, a_end], "evt089arg02place"],
          ...
      ],
      ...
    }

We flatten ``sentences`` into one document, join tokens with spaces, and
slice surfaces by the (start, end) indices (both inclusive in RAMS). Each
gold link becomes an argument; the role name is the trailing word of the
``evtNargM<role>`` identifier.

Distribution: https://nlp.jhu.edu/rams/ (RAMS_1.0c.tar.gz). After
extracting, point ``--input`` at ``data/train.jsonlines`` /
``dev.jsonlines`` / ``test.jsonlines``.

Usage::

    uv run python tools/data/convert_rams.py \\
        --input /path/to/RAMS_1.0c/data/train.jsonlines \\
        --out data/rams.train.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _split import dumps_record  # noqa: E402


ROLE_RE = re.compile(r"^evt\d+arg\d+([A-Za-z]+)$")


def _flatten_tokens(sentences: List[List[str]]) -> List[str]:
    tokens: List[str] = []
    for sent in sentences:
        if isinstance(sent, list):
            tokens.extend(t for t in sent if isinstance(t, str))
    return tokens


def _slice(tokens: List[str], start: int, end: int) -> str | None:
    """Return ``tokens[start..end]`` (inclusive) joined with spaces."""
    if not (0 <= start <= end < len(tokens)):
        return None
    surface = " ".join(tokens[start:end + 1]).strip()
    return surface or None


def _parse_role(raw: str) -> str | None:
    """Extract the role name from a RAMS link id like ``evt089arg02victim``."""
    if not isinstance(raw, str):
        return None
    m = ROLE_RE.match(raw.strip())
    return m.group(1) if m else None


def _event_type_of(evt_trigger: List[Any]) -> str | None:
    if not isinstance(evt_trigger, list) or len(evt_trigger) < 3:
        return None
    type_list = evt_trigger[2]
    if not isinstance(type_list, list) or not type_list:
        return None
    first = type_list[0]
    if not isinstance(first, list) or not first:
        return None
    etype = first[0]
    return etype.strip() if isinstance(etype, str) and etype.strip() else None


def convert_row(row: Dict[str, Any]) -> Dict[str, Any] | None:
    sentences = row.get("sentences")
    triggers = row.get("evt_triggers")
    gold_links = row.get("gold_evt_links")
    if not isinstance(sentences, list) or not sentences:
        return None
    tokens = _flatten_tokens(sentences)
    if not tokens:
        return None
    text = " ".join(tokens)

    # Map (trig_start, trig_end) -> (event_type, surface)
    trigger_index: Dict[tuple, Dict[str, Any]] = {}
    if isinstance(triggers, list):
        for trig in triggers:
            if not isinstance(trig, list) or len(trig) < 2:
                continue
            try:
                s, e = int(trig[0]), int(trig[1])
            except (TypeError, ValueError):
                continue
            etype = _event_type_of(trig)
            surface = _slice(tokens, s, e)
            if etype is None or surface is None:
                continue
            trigger_index[(s, e)] = {
                "event_type": etype,
                "trigger": surface,
                "arguments": [],
            }

    # Apply gold links: for each link, find the trigger and append the arg.
    if isinstance(gold_links, list):
        for link in gold_links:
            if not isinstance(link, list) or len(link) != 3:
                continue
            trig_span, arg_span, raw_role = link
            if not (isinstance(trig_span, list) and len(trig_span) == 2):
                continue
            if not (isinstance(arg_span, list) and len(arg_span) == 2):
                continue
            try:
                ts, te = int(trig_span[0]), int(trig_span[1])
                as_, ae = int(arg_span[0]), int(arg_span[1])
            except (TypeError, ValueError):
                continue
            role = _parse_role(raw_role)
            if role is None:
                continue
            arg_surface = _slice(tokens, as_, ae)
            if arg_surface is None or arg_surface not in text:
                continue

            evt = trigger_index.get((ts, te))
            if evt is None:
                continue
            evt["arguments"].append({"role": role, "entity": arg_surface})

    events = [e for e in trigger_index.values() if e["trigger"] in text]
    if not events:
        return None
    return {"input": text, "output": {"events": events}}


def _iter_jsonl(path: Path):
    with path.open(encoding="utf-8") as fh:
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
                        help="Local path to a RAMS .jsonlines file "
                             "(train/dev/test).")
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
    total_args = 0
    all_types: set[str] = set()
    all_roles: set[str] = set()

    with args.out.open("w", encoding="utf-8") as f:
        for idx, row in enumerate(_iter_jsonl(args.input)):
            if 0 <= args.max_records <= idx:
                break
            record = convert_row(row)
            if record is None:
                skipped_empty += 1
                continue
            f.write(dumps_record(record) + "\n")
            emitted += 1
            for ev in record["output"]["events"]:
                total_events += 1
                total_args += len(ev["arguments"])
                all_types.add(ev["event_type"])
                for arg in ev["arguments"]:
                    all_roles.add(arg["role"])
            if emitted % 500 == 0:
                print(f"  emitted={emitted}  events={total_events}  "
                      f"event_types={len(all_types)}  roles={len(all_roles)}")

    print(f"Done. emitted={emitted} skipped_empty={skipped_empty} "
          f"total_events={total_events} total_arguments={total_args} "
          f"event_types={len(all_types)} roles={len(all_roles)} -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

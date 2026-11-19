"""Convert Wojood Hadath (Aljabari et al., ArabicNLP 2024) to GLiNER2 JSONL.

The release is sentence-level JSON, one file per official split. Each sentence
has event mentions and argument links. The readme names the fields ``event``
and ``argument``; the files use ``text`` and ``arg_text``. There is one event
category (a named occurrence: a battle, election, disaster, holiday), so every
trigger is typed ``Event``. Roles stay ``hasAgent``, ``hasLocation``, and
``hasDate``.

The JSON has no character offsets. A span is written as ``{text, start, end}``
only when it has one landing place: the trigger must occur once in the
sentence, and an argument must occur once inside that trigger or, failing
that, once in the sentence. Anything else is dropped rather than attached to
the wrong copy. Arguments are often inside the trigger mention.

Official splits leak: the same sentence appears in more than one of train, val,
and test. Held-out copy wins, test over val over train, matching
``dedupe_splits.py``.

License: CC-BY-4.0, plus the authors' limit to non-malicious, peaceful, and
non-military use. Cite Aljabari, Duaibes, Jarrar, and Khalilia (ArabicNLP 2024).
The converted splits are mirrored privately at ``whr778/wojood_hadath`` and
restored by ``tools/data/restore_from_hf.py``.

Usage::

    uv run python tools/data/convert_wojood_hadath.py \\
        --input /path/to/WojoodHadath --out data/wojood_hadath.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _split import clean_text, dumps_record, normalize_group_key  # noqa: E402

EVENT_TYPE = "Event"
SPLIT_FILES = {"train": "train/train.json", "val": "val/val.json", "test": "test/test.json"}
# test wins over val over train, the same precedence as dedupe_splits.py.
HELD_OUT_FIRST = ("test", "val", "train")


def _find_spans(text: str, surface: str) -> List[Tuple[int, int]]:
    """Non-overlapping occurrences of ``surface`` in ``text``."""
    if not surface:
        return []
    spans = []
    start = 0
    while True:
        i = text.find(surface, start)
        if i < 0:
            return spans
        spans.append((i, i + len(surface)))
        start = i + len(surface)


def _place_argument(text: str, surface: str, trigger: Tuple[int, int]) -> Optional[Tuple[int, int]]:
    """One span for ``surface``: inside the trigger if that is unique, else the sentence."""
    hits = _find_spans(text, surface)
    if not hits:
        return None
    ts, te = trigger
    inside = [h for h in hits if ts <= h[0] and h[1] <= te]
    if len(inside) == 1:
        return inside[0]
    if len(hits) == 1:
        return hits[0]
    return None


def convert_row(row: Dict[str, Any], stats: Counter) -> Optional[Dict[str, Any]]:
    """One Hadath sentence to a GLiNER2 record, or None when no trigger can be placed."""
    sentence = row.get("sentence")
    if not isinstance(sentence, str) or not sentence.strip():
        stats["row_empty"] += 1
        return None
    text = clean_text(sentence)
    events_out: List[Dict[str, Any]] = []
    for ev in row.get("events") or []:
        if not isinstance(ev, dict):
            continue
        surface = ev.get("text")
        if not isinstance(surface, str):
            stats["event_absent"] += 1
            continue
        surface = clean_text(surface).strip()
        hits = _find_spans(text, surface)
        if len(hits) != 1:
            stats["event_absent" if not hits else "event_ambiguous"] += 1
            continue
        trigger = hits[0]
        arguments: List[Dict[str, Any]] = []
        seen = set()
        for arg in ev.get("Arguments") or []:
            if not isinstance(arg, dict):
                continue
            role = arg.get("relation")
            arg_text = arg.get("arg_text")
            if not isinstance(role, str) or not isinstance(arg_text, str):
                stats["arg_absent"] += 1
                continue
            role = role.strip()
            arg_text = clean_text(arg_text).strip()
            if not role or not arg_text:
                stats["arg_absent"] += 1
                continue
            placed = _place_argument(text, arg_text, trigger)
            if placed is None:
                stats["arg_absent" if not _find_spans(text, arg_text) else "arg_ambiguous"] += 1
                continue
            key = (role, placed)
            if key in seen:
                stats["arg_duplicate"] += 1
                continue
            seen.add(key)
            a0, a1 = placed
            stats["arg_kept"] += 1
            arguments.append({
                "role": role,
                "entity": {"text": text[a0:a1], "start": a0, "end": a1},
            })
        stats["event_kept"] += 1
        events_out.append({
            "event_type": EVENT_TYPE,
            "triggers": [{"text": text[trigger[0]:trigger[1]], "start": trigger[0], "end": trigger[1]}],
            "arguments": arguments,
        })
    if not events_out:
        stats["row_dropped"] += 1
        return None
    stats["row_kept"] += 1
    return {"input": text, "output": {"events": events_out}}


def assign_splits(
    by_split: Dict[str, List[Dict[str, Any]]],
) -> Tuple[Dict[str, List[Dict[str, Any]]], Counter]:
    """Drop a sentence from train, then val, when the same text is held out.

    Within one split, the first copy of a text is the one that stays.
    """
    owner: Dict[str, str] = {}
    for split in HELD_OUT_FIRST:
        for rec in by_split.get(split) or []:
            owner.setdefault(normalize_group_key(rec["input"]), split)
    kept = {split: [] for split in HELD_OUT_FIRST}
    dropped: Counter = Counter()
    for split in ("train", "val", "test"):
        seen = set()
        for rec in by_split.get(split) or []:
            key = normalize_group_key(rec["input"])
            if owner[key] != split:
                dropped[f"{split} held in {owner[key]}"] += 1
                continue
            if key in seen:
                dropped[f"{split} duplicate"] += 1
                continue
            seen.add(key)
            kept[split].append(rec)
    return kept, dropped


def _word_edge_alignment(records: List[Dict[str, Any]]) -> Tuple[int, int]:
    """How many written spans sit on whitespace-token edges."""
    from gliner2.processing.word_splitter import WhitespaceTokenSplitter

    splitter = WhitespaceTokenSplitter()
    ok = total = 0
    for rec in records:
        text = rec["input"]
        edges_s = set()
        edges_e = set()
        for _, s, e in splitter(text, lower=True):
            edges_s.add(s)
            edges_e.add(e)
        for ev in rec["output"]["events"]:
            spans = list(ev["triggers"])
            spans.extend(arg["entity"] for arg in ev["arguments"])
            for span in spans:
                total += 1
                ok += span["start"] in edges_s and span["end"] in edges_e
    return ok, total


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", required=True, type=Path,
                        help="Directory that contains train/train.json, val/val.json, and test/test.json.")
    parser.add_argument("--out", required=True, type=Path,
                        help="Output base path; writes <base>.train.jsonl, <base>.val.jsonl, <base>.test.jsonl.")
    args = parser.parse_args()

    stats: Counter = Counter()
    converted: Dict[str, List[Dict[str, Any]]] = {}
    for split, rel in SPLIT_FILES.items():
        path = args.input / rel
        if not path.is_file():
            raise SystemExit(f"input not found: {path}")
        rows = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(rows, list):
            raise SystemExit(f"expected a JSON array in {path}")
        stats[f"source_{split}"] = len(rows)
        converted[split] = [rec for row in rows if (rec := convert_row(row, stats))]

    kept, dropped = assign_splits(converted)
    stem = args.out.with_suffix("") if args.out.suffix == ".jsonl" else args.out
    stem.parent.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        out_path = Path(f"{stem}.{split}.jsonl")
        with out_path.open("w", encoding="utf-8") as fh:
            for rec in kept[split]:
                fh.write(dumps_record(rec) + "\n")
        ok, total = _word_edge_alignment(kept[split])
        rate = 100.0 * ok / total if total else 0.0
        print(f"  {split}: {len(kept[split])} records, word-edge {ok}/{total} ({rate:.2f}%) -> {out_path}")

    print(
        "before split drop: events kept {event_kept} absent {event_absent} "
        "ambiguous {event_ambiguous}; args kept {arg_kept} absent {arg_absent} "
        "ambiguous {arg_ambiguous} duplicate {arg_duplicate}; rows dropped {row_dropped}".format_map(stats)
    )
    if dropped:
        print("split leakage dropped: " + ", ".join(f"{k}={v}" for k, v in sorted(dropped.items())))
    return 0


if __name__ == "__main__":
    sys.exit(main())

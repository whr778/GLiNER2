"""Merge RAMS's repeated passages into one record per passage.

RAMS annotates ONE event per example, so a passage with several events is stored as several
records of the same text, each carrying a single event. Trained as-is, every copy teaches the
model that the passage's OTHER events are absent -- false-negative supervision. Measured on
rams.train (2026-09-30): 840 of 6,322 passages repeat (1,847 records), and every repeat carries
different events: 124 split distinct triggers, 668 give one trigger conflicting types, 42 both.

Per passage, the events of all copies are unioned, then per trigger:

* identical events collapse to one;
* a specific subtype and its own catch-all (`x.y.hide` vs `x.y.n/a`) collapse to the SPECIFIC
  event, taking the union of both copies' arguments -- the n/a copy is a coarser label for the
  same mention (398 such triggers);
* genuinely different types for one trigger (different parents, or two specific subtypes)
  are kept as separate events: RAMS annotates the frame each example evokes, and a record per
  event type represents both.

The source corpus is untouched; this writes a derived corpus through `dumps_record`.

    uv run python tools/data/merge_rams_passages.py data/rams.train.jsonl data/rams_merged.train.jsonl
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _split import dumps_record, normalize_group_key  # noqa: E402


def _parent(event_type):
    return event_type.rsplit(".", 1)[0]


def merge_events(events):
    """Union a passage's events, collapsing duplicates and n/a catch-alls per trigger."""
    by_key = {}
    for ev in events:
        key = (ev["event_type"], tuple(ev.get("triggers") or []))
        slot = by_key.setdefault(key, {"event_type": ev["event_type"],
                                       "triggers": list(ev.get("triggers") or []),
                                       "arguments": []})
        for arg in ev.get("arguments") or []:
            if arg not in slot["arguments"]:
                slot["arguments"].append(arg)
    for (etype, trig), slot in list(by_key.items()):
        if not etype.endswith(".n/a"):
            continue
        specific = [k for k in by_key if k[1] == trig and k[0] != etype
                    and _parent(k[0]) == _parent(etype)]
        if specific:
            target = by_key[specific[0]]
            for arg in slot["arguments"]:
                if arg not in target["arguments"]:
                    target["arguments"].append(arg)
            del by_key[(etype, trig)]
    return list(by_key.values())


def merge_file(src, dst):
    """Write one record per passage; return (records in, records out, events in, events out)."""
    groups = defaultdict(list)
    order = []
    with open(src, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            rec = json.loads(line)
            key = normalize_group_key(rec["input"])
            if key not in groups:
                order.append(key)
            groups[key].append(rec)
    n_in = sum(map(len, groups.values()))
    ev_in = ev_out = 0
    with open(dst, "w", encoding="utf-8") as out:
        for key in order:
            copies = groups[key]
            events = [e for c in copies for e in (c["output"].get("events") or [])]
            merged = dict(copies[0])
            merged["output"] = {**copies[0]["output"], "events": merge_events(events)}
            ev_in += len(events)
            ev_out += len(merged["output"]["events"])
            out.write(dumps_record(merged) + "\n")
    return n_in, len(order), ev_in, ev_out


def main():
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    n_in, n_out, ev_in, ev_out = merge_file(src, dst)
    print(f"{src} -> {dst}: {n_in:,} records -> {n_out:,} passages; "
          f"{ev_in:,} events -> {ev_out:,} after collapsing duplicates and n/a catch-alls")


if __name__ == "__main__":
    main()

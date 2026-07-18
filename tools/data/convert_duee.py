"""Convert DuEE 1.0 (Li et al., NLPCC 2020) to GLiNER2 JSONL.

DuEE 1.0 is a large sentence-level Chinese event-extraction dataset (65 event
types, 121 argument roles) over real-world news. Each event carries a trigger
and typed argument roles, so it maps directly to GLiNER2 events::

    output.events = [{"event_type": <label>, "triggers": [<trigger>],
                      "arguments": [{"role": <role>, "entity": <surface>}, ...]}]

The canonical raw dataset is gated behind a Baidu LUGE account. This converter
uses the no-login HuggingFace mirror ``nlhappy/DuEE`` (already reformatted to
``{text, events:[{label, trigger:{text}, args:[{label, text}]}]}``). CAVEAT:
that mirror provides **train + validation only — there is no test split**, and
its licensing is the uploader's MIT label, not Baidu's original terms. So this
emits ``duee.train.jsonl`` + ``duee.val.jsonl`` (no ``duee.test.jsonl``).

Trigger/argument surfaces are matched verbatim against the sentence text (the
mirror also carries offsets, but GLiNER2 recovers spans by string match, as in
the other event converters).

Usage::

    uv run python tools/data/convert_duee.py --out data/duee.jsonl
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _split import dumps_record  # noqa: E402

HF_ID = "nlhappy/DuEE"
# HF split name -> our output split suffix. The mirror has no test split.
SPLIT_MAP = {"train": "train", "validation": "val"}


def convert_record(r: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    text = r.get("text")
    if not isinstance(text, str) or not text.strip():
        return None

    events_out: List[Dict[str, Any]] = []
    for ev in r.get("events") or []:
        if not isinstance(ev, dict):
            continue
        etype = ev.get("label")
        trigger_obj = ev.get("trigger") or {}
        trigger = trigger_obj.get("text") if isinstance(trigger_obj, dict) else None
        if not isinstance(etype, str) or not isinstance(trigger, str):
            continue
        etype, trigger = etype.strip(), trigger.strip()
        if not etype or not trigger or trigger not in text:
            continue

        arguments: List[Dict[str, str]] = []
        seen: set = set()
        for a in ev.get("args") or []:
            if not isinstance(a, dict):
                continue
            role, surface = a.get("label"), a.get("text")
            if not isinstance(role, str) or not isinstance(surface, str):
                continue
            role, surface = role.strip(), surface.strip()
            if not role or not surface or surface not in text:
                continue
            key = (role, surface)
            if key in seen:
                continue
            seen.add(key)
            arguments.append({"role": role, "entity": surface})

        events_out.append({
            "event_type": etype,
            "triggers": [trigger],
            "arguments": arguments,
        })

    if not events_out:
        return None
    return {"input": text, "output": {"events": events_out}}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True, type=Path,
                        help="Output base path; writes <base>.train.jsonl and "
                             "<base>.val.jsonl (no test split in the HF mirror).")
    parser.add_argument("--hf-id", default=HF_ID,
                        help=f"HuggingFace dataset id (default: {HF_ID}).")
    parser.add_argument("--max-records", type=int, default=-1,
                        help="Max records per split to emit (-1 = all).")
    args = parser.parse_args()

    from datasets import load_dataset
    print(f"Loading {args.hf_id} ...")
    ds = load_dataset(args.hf_id)

    stem = args.out.with_suffix("") if args.out.suffix == ".jsonl" else args.out
    for hf_split, out_split in SPLIT_MAP.items():
        if hf_split not in ds:
            print(f"  (skip: no '{hf_split}' split in {args.hf_id})")
            continue
        out_path = Path(f"{stem}.{out_split}.jsonl")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        n = skipped = 0
        with out_path.open("w", encoding="utf-8") as f:
            for r in ds[hf_split]:
                if 0 <= args.max_records <= n:
                    break
                rec = convert_record(r)
                if rec is None:
                    skipped += 1
                    continue
                f.write(dumps_record(rec) + "\n")
                n += 1
        print(f"  {out_split}: wrote {n} (skipped {skipped}) -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

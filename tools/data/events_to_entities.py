"""Reframe a trigger-only event dataset as an entity (typed-span) dataset.

MAVEN / mendeley_ed are event *detection* corpora: a typed trigger with
NO arguments. GLiNER2's event path needs >=1 role, so these can't train as
events. Event detection is typed-span classification, so map each event to an
entity: ``entity label = event_type``, ``mention = trigger surface``. Trigger
surfaces already appear verbatim in the text, so they validate as entities.

Splits: a real ``--val`` / ``--test`` is transformed 1:1; any split not supplied
is carved from train by ratio (deterministic, seeded), so every referenced
split file exists (the corpora loader opens each directly).

    uv run python tools/data/events_to_entities.py \
        --train data/maven.train.jsonl --out-base data/maven_ner \
        --carve-val 0.05 --carve-test 0.05
"""

import argparse
import json
import random
from pathlib import Path

from _split import dumps_record


def to_entity_record(rec: dict) -> dict:
    """Event record -> entity record: {event_type: [unique trigger surfaces]}."""
    labels: dict[str, list[str]] = {}
    for ev in rec.get("output", {}).get("events") or []:
        etype = ev.get("event_type")
        if not isinstance(etype, str) or not etype:
            continue
        for trig in ev.get("triggers") or []:
            if isinstance(trig, str) and trig and trig not in labels.setdefault(etype, []):
                labels[etype].append(trig)
    return {"input": rec["input"], "output": {"entities": labels}}


def _read(path: str) -> list[dict]:
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(to_entity_record(json.loads(line)))
    return out


def _write(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(dumps_record(r) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train", required=True, help="Event-format train JSONL.")
    ap.add_argument("--val", help="Event-format val JSONL (else carved from train).")
    ap.add_argument("--test", help="Event-format test JSONL (else carved from train).")
    ap.add_argument("--out-base", required=True, help="Output base, e.g. data/maven_ner.")
    ap.add_argument("--carve-val", type=float, default=0.0, help="Val fraction to carve from train.")
    ap.add_argument("--carve-test", type=float, default=0.0, help="Test fraction to carve from train.")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    train = _read(args.train)
    val = _read(args.val) if args.val else []
    test = _read(args.test) if args.test else []

    carve_val = args.carve_val if not args.val else 0.0
    carve_test = args.carve_test if not args.test else 0.0
    if carve_val + carve_test > 0:
        rng = random.Random(args.seed)
        rng.shuffle(train)
        n = len(train)
        n_val = int(n * carve_val)
        n_test = int(n * carve_test)
        val = val or train[:n_val]
        test = test or train[n_val:n_val + n_test]
        train = train[n_val + n_test:]

    base = Path(args.out_base)
    stem = base.with_suffix("") if base.suffix == ".jsonl" else base
    _write(Path(f"{stem}.train.jsonl"), train)
    _write(Path(f"{stem}.val.jsonl"), val)
    _write(Path(f"{stem}.test.jsonl"), test)
    print(f"{stem.name}: train={len(train)} val={len(val)} test={len(test)}")


if __name__ == "__main__":
    main()

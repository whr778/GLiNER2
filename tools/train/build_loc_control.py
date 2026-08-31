"""The location-supervision control: identical documents, different annotation generation.

The tr-dose curve trained its English half on `casualty_docee`, which carries ZERO
location supervision -- 25,154 records, not one `location` field. Every arm including the
zero-Turkish control then failed to bind English locations on Helene (1 of 96 dead
observations against the shipped extractor's 24 of 69). That was read first as Turkish
diluting English, then as an mmBERT/boundary regression. It was neither: the models were
never taught English locations.

This isolates that single variable. `casualty_natural` contains 97.9% of `casualty_docee`'s
documents at a later annotation generation, so the arm below is the SAME 13,080 documents
with the SAME replay -- only the labels change, from 0% location coverage to 93.0%.

If location binding on Helene returns, the diagnosis is confirmed and the full retrain is
built on a correct premise. If it does not, something else is wrong and the $17 retrain
would have been spent on the wrong fix.

    uv run python tools/train/build_loc_control.py
"""
from __future__ import annotations

import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))
from _split import dumps_record, normalize_group_key  # noqa: E402

OUT = "data/loc_control"
VAL_SHARE = 0.05


def load(path: str) -> list[dict]:
    return [json.loads(l) for l in Path(path).open(encoding="utf-8")]


def main() -> int:
    rng = random.Random(42)
    docee_keys = {normalize_group_key(r["input"])[:300]
                  for r in load("data/casualty_docee.train.jsonl")}
    natural = {normalize_group_key(r["input"])[:300]: r
               for r in load("data/casualty_natural.train.jsonl")}
    rows = [natural[k] for k in docee_keys if k in natural]
    print(f"{len(rows):,} documents: casualty_docee's own text, casualty_natural's labels")

    # Anchor must be a field the document actually has, or the record head raises.
    order = ("dead", "injured", "missing", "location")
    for r in rows:
        structs = r["output"].get("json_structures") or []
        present = {f for s in structs for f in (s.get("casualty_report") or {})}
        anchor = next((f for f in order if f in present), None)
        if anchor:
            r["output"]["record_metadata"] = {
                "casualty_report": {"mode": "natural", "anchor": anchor}}

    replay = load("data/replay_137k30.train.jsonl")
    rng.shuffle(rows)
    rng.shuffle(replay)

    n_val = int(len(rows) * VAL_SHARE)
    val_rows, train_rows = rows[:n_val], rows[n_val:]
    n_replay_val = min(400, len(replay) // 4)
    replay_val, replay_pool = replay[-n_replay_val:], replay[:-n_replay_val]

    val_all = val_rows + replay_val
    rng.shuffle(val_all)
    half = len(val_all) // 2
    val, test = val_all[:half], val_all[half:]

    n_replay = int(len(train_rows) / 0.7 * 0.3)
    train = train_rows + replay_pool[:n_replay]
    held = {normalize_group_key(r["input"])[:300] for r in val_all}
    train = [r for r in train if normalize_group_key(r["input"])[:300] not in held]
    rng.shuffle(train)

    for name, data in (("train", train), ("val", val), ("test", test)):
        Path(f"{OUT}.{name}.jsonl").write_text(
            "".join(dumps_record(r) + "\n" for r in data), encoding="utf-8")
        tasks = Counter(k for r in data for k in r["output"])
        print(f"  {OUT}.{name}.jsonl: {len(data):,} rows | {dict(tasks)}")

    tk = {normalize_group_key(r["input"])[:300] for r in train}
    vk = {normalize_group_key(r["input"])[:300] for r in val}
    sk = {normalize_group_key(r["input"])[:300] for r in test}
    print(f"[split hygiene] train n val {len(tk & vk)} | train n test {len(tk & sk)} "
          f"| val n test {len(vk & sk)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

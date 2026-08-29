"""Dose-curve mixes for teaching the casualty extractor Turkish.

The base is `whr778/gliner2-joint-boundary-mmbert-137k`. mmBERT, NOT the fastino
DeBERTa-v3 base the shipped casualty extractor used: that vocabulary is English-only
(128,011) and cannot represent Turkish at the tokenizer at all, so teaching it Turkish is
not a data problem. mmBERT tokenizes Turkish cleanly -- `kisi`, `kisinin` and `bin` are
single tokens -- which is what makes supervision the only variable here.

The failure being fixed: asked for a `location` the shipped extractor returns a digit,
78.2% of the time on Turkish against 5.8% on English, and that is the known signature of
a narrow fine-tune with NO replay.

Each arm therefore holds two things constant and varies one:
  FIXED   13,358 English casualty rows, plus 30% broad replay from mix_natural
  VARIED  Turkish casualty rows: 0 (control), 5K, 15K, all 31,263

**The control arm is not optional.** Every dose arm adds replay AND Turkish at once, so
without a 0-Turkish arm carrying the same replay, an improvement over the shipped model
cannot be attributed to either. The shipped model is not a matched baseline; it has no
replay.

Replay is EXACT, not a proxy: `build_137k_replay.py` samples the literal training pool of
the base, proportionally by corpus, train splits only. We own that base and its data, so
"did the original capability survive" is directly measurable against the eval the base was
scored on. It is also BROAD rather than more casualty rows -- replaying the narrow task
would guard the task being trained and leave the collapsed capabilities collapsed.

    uv run python tools/train/build_turkish_dose_mix.py
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))
from _split import dumps_record, normalize_group_key  # noqa: E402

DOSES = (0, 5000, 15000, 31263)
REPLAY_SHARE = 0.30
VAL_SHARE = 0.05


def load(path: str) -> list[dict]:
    return [json.loads(l) for l in Path(path).open(encoding="utf-8")]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--english", default="data/casualty_docee.train.jsonl")
    ap.add_argument("--turkish", default="data/turkish_gate/cas_ann_tr.jsonl")
    ap.add_argument("--replay", default="data/replay_137k30.train.jsonl",
                    help="EXACT replay from the base's own pool; rebuild per arm with "
                         "build_137k_replay.py --new-records N")
    ap.add_argument("--out-prefix", default="data/tr_dose")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    rng = random.Random(a.seed)
    en, tr, replay_pool = load(a.english), load(a.turkish), load(a.replay)
    for row in tr:
        row.pop("sources", None)   # audit-only field; not part of the training row
    rng.shuffle(tr)
    rng.shuffle(replay_pool)
    print(f"english {len(en):,} | turkish {len(tr):,} | replay pool {len(replay_pool):,}")

    # One val split, shared by every arm, so the arms are comparable. Turkish val comes
    # from the largest dose so the smaller arms are not evaluated on their own training rows.
    n_val_tr = int(len(tr) * VAL_SHARE)
    tr_val, tr_train = tr[:n_val_tr], tr[n_val_tr:]
    n_val_en = int(len(en) * VAL_SHARE)
    en_val, en_train = en[:n_val_en], en[n_val_en:]
    val = en_val + tr_val
    rng.shuffle(val)

    val_keys = {normalize_group_key(r["input"])[:300] for r in val}
    Path(f"{a.out_prefix}.val.jsonl").write_text(
        "".join(dumps_record(r) + "\n" for r in val), encoding="utf-8")
    print(f"val: {len(val):,} rows ({len(en_val):,} en + {len(tr_val):,} tr)")

    for dose in DOSES:
        take = tr_train[:dose]
        new = en_train + take
        n_replay = int(len(new) / (1 - REPLAY_SHARE) * REPLAY_SHARE)
        rows = new + replay_pool[:n_replay]
        # Drop val overlap AND internal duplicates. A duplicate is not contamination the
        # way a val leak is, but it silently oversamples whatever it duplicates, and the
        # arms differ in size so it would not even do so evenly across them.
        rows, seen = [r for r in rows], set()
        deduped = []
        for r in rows:
            key = normalize_group_key(r["input"])[:300]
            if key in val_keys or key in seen:
                continue
            seen.add(key)
            deduped.append(r)
        rows = deduped
        rng.shuffle(rows)                      # replay works only if every BATCH is mixed
        name = f"{a.out_prefix}{dose}"
        Path(f"{name}.train.jsonl").write_text(
            "".join(dumps_record(r) + "\n" for r in rows), encoding="utf-8")
        keys = [normalize_group_key(r["input"])[:300] for r in rows]
        dup = len(keys) - len(set(keys))
        leak = sum(1 for k in keys if k in val_keys)
        print(f"  {name}.train.jsonl: {len(rows):,} rows "
              f"(en {len(en_train):,} + tr {len(take):,} + replay {n_replay:,}) "
              f"dups={dup} val-leak={leak}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

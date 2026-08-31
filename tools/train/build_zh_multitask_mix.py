"""Splits for the Chinese multi-task corpus, warm-started from the 137k boundary base.

5,872 Simplified Chinese news articles annotated for all five GLiNER2 tasks in one pass
(entities, relations, events, classifications, structures) at 97.8% verbatim.

30% EXACT replay from the base's own 137k pool, not a proxy. Training a warm start on the
new task alone destroys what it is not training on -- measured twice here: the casualty
fine-tune returned a digit when asked for a `location`, and the base-v1 arms lost 23-39%
of general-domain entity F1.

The val/test splits carry replay too. Built from the new corpus alone they would report
only the tasks it contains, and whether the replay preserved anything would go unmeasured
-- the gap that made the Turkish dose curve unable to observe forgetting.

    uv run python tools/train/build_zh_multitask_mix.py
"""
from __future__ import annotations

import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))
from _split import dumps_record, normalize_group_key  # noqa: E402

OUT = "data/zh_multitask"
VAL_SHARE = 0.05
REPLAY_SHARE = 0.30


def main() -> int:
    rng = random.Random(42)
    rows = [json.loads(l) for l in
            Path("data/chinese_gate/zh_multitask_6k_en.jsonl").open(encoding="utf-8")]
    replay = [json.loads(l) for l in
              Path("data/replay_137k30.train.jsonl").open(encoding="utf-8")]
    rng.shuffle(rows)
    rng.shuffle(replay)
    print(f"chinese multi-task {len(rows):,} | replay pool {len(replay):,}")

    n_val = int(len(rows) * VAL_SHARE)
    zh_hold, zh_train = rows[: n_val * 2], rows[n_val * 2:]
    n_rep_hold = min(600, len(replay) // 4)
    rep_hold, rep_pool = replay[-n_rep_hold:], replay[: len(replay) - n_rep_hold]

    held = zh_hold + rep_hold
    rng.shuffle(held)
    half = len(held) // 2
    val, test = held[:half], held[half:]

    n_replay = int(len(zh_train) / (1 - REPLAY_SHARE) * REPLAY_SHARE)
    train = zh_train + rep_pool[:n_replay]
    held_keys = {normalize_group_key(r["input"])[:300] for r in held}
    train = [r for r in train if normalize_group_key(r["input"])[:300] not in held_keys]
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

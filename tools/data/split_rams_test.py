"""Split rams.test into a val and a blind test half.

RAMS shipped train + test only, so the EKF front-end config had to give it train alone
and it contributed no validation signal. This carves the existing held-out test into two
disjoint halves.

Two guards, both because RAMS specifically needs them:

* **Deduplicate first.** RAMS carries 13.7% duplicate documents inside train alone
  (TODO), so the same hazard is assumed for test. Splitting before deduplicating would
  put copies of one document on both sides of the new boundary -- val/test contamination
  created by the very script meant to produce a clean split.
* **Group by input text, not by row.** Two rows can annotate different events in the SAME
  document. Those must move together, or the model validates on a document it is tested
  on.

Deterministic: groups are ordered by hash, so the split is reproducible without a seed.

    uv run python tools/data/split_rams_test.py
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _split import dumps_record  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "data" / "rams.test.jsonl"
VAL = REPO / "data" / "rams.val.jsonl"
TEST = REPO / "data" / "rams.test.jsonl"
VAL_SHARE = 0.5


def main() -> None:
    rows = [json.loads(l) for l in SRC.open(encoding="utf-8") if l.strip()]
    print(f"read {len(rows)} rows from {SRC.name}")

    groups: dict[str, list] = {}
    for r in rows:
        key = hashlib.sha1(r["input"].encode("utf-8")).hexdigest()
        groups.setdefault(key, []).append(r)
    dupes = len(rows) - len(groups)
    print(f"{len(groups)} distinct documents ({dupes} duplicate rows folded in)")

    order = sorted(groups)                      # hash order: deterministic, no seed
    cut = int(len(order) * VAL_SHARE)
    val_keys, test_keys = set(order[:cut]), set(order[cut:])
    assert not (val_keys & test_keys)

    val_rows = [r for k in order if k in val_keys for r in groups[k]]
    test_rows = [r for k in order if k in test_keys for r in groups[k]]

    VAL.write_text("".join(dumps_record(r) + "\n" for r in val_rows), encoding="utf-8")
    TEST.write_text("".join(dumps_record(r) + "\n" for r in test_rows), encoding="utf-8")

    v = {hashlib.sha1(r["input"].encode()).hexdigest() for r in val_rows}
    t = {hashlib.sha1(r["input"].encode()).hexdigest() for r in test_rows}
    print(f"val  {len(val_rows):>4} rows / {len(v)} documents -> {VAL.name}")
    print(f"test {len(test_rows):>4} rows / {len(t)} documents -> {TEST.name}")
    print(f"document overlap val/test: {len(v & t)}")
    assert not (v & t), "val and test share a document"
    print("OK -- disjoint by document")


if __name__ == "__main__":
    main()

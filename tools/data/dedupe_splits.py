"""Make a corpus's train/val/test mutually disjoint, in place.

`check_leakage.py` detects contamination; this repairs it. The two share one
document key (`_split.normalize_group_key`), so a corpus repaired here reads
clean there.

**Precedence is test > val > train**, from the standing rule: test is a blind
test and must keep every document it has, so overlaps are resolved by dropping
the *training-side* copy. Concretely: dedupe within each split (keep first),
drop from val anything in test, drop from train anything in val or test. Test
is never reduced except by its own internal duplicates.

Some corpora are contaminated at the source rather than by our splitter. DocEE
ships `normal_setting/{train,dev,test}.json` already overlapping -- 56 train/val,
12 train/test, 26 val/test, plus 84 internal duplicates -- and `convert_docee.py`
honours those published splits, so it reproduces the defect faithfully every run.
For corpora like that this step is part of the build, not a one-off cleanup;
re-running the converter without it silently restores the leak.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Set

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _split import derive_split_paths, dumps_record, normalize_group_key  # noqa: E402

# test last: each split drops what the later, higher-precedence ones already hold.
SPLIT_ORDER = ("train", "val", "test")


def _load(path: Path) -> List[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _write(path: Path, records: List[dict]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(dumps_record(rec) + "\n")


def dedupe(paths: Dict[str, Path], dry_run: bool = False) -> int:
    """Drop duplicate and cross-split documents. Returns the number removed."""
    records = {s: _load(p) for s, p in paths.items()}
    keys = {
        s: [normalize_group_key(r.get("input", "")) for r in recs]
        for s, recs in records.items()
    }
    removed_total = 0

    for i, split in enumerate(SPLIT_ORDER):
        if split not in records:
            continue
        # Everything of higher precedence wins the document outright.
        blocked: Set[str] = set()
        for higher in SPLIT_ORDER[i + 1:]:
            blocked |= set(keys.get(higher, []))

        kept, seen, dropped_cross, dropped_dup = [], set(), 0, 0
        for rec, key in zip(records[split], keys[split]):
            if key in blocked:
                dropped_cross += 1
            elif key in seen:
                dropped_dup += 1
            else:
                seen.add(key)
                kept.append(rec)

        removed = len(records[split]) - len(kept)
        removed_total += removed
        print(
            f"  {split:5s} {len(records[split]):6,} -> {len(kept):6,}"
            f"   (-{dropped_dup} internal dupe, -{dropped_cross} held by a later split)"
        )
        if removed and not dry_run:
            _write(paths[split], kept)
        # Later splits compare against what actually survives here.
        records[split], keys[split] = kept, [normalize_group_key(r.get("input", "")) for r in kept]

    return removed_total


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("base", type=Path,
                    help="split base, e.g. data/docee -> data/docee.{train,val,test}.jsonl")
    ap.add_argument("--dry-run", action="store_true",
                    help="report what would be dropped without rewriting")
    args = ap.parse_args()

    paths = derive_split_paths(args.base)
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        raise SystemExit(f"missing split file(s): {', '.join(missing)}")

    print(f"{'DRY RUN: ' if args.dry_run else ''}{args.base} (precedence test > val > train)")
    total = dedupe(paths, dry_run=args.dry_run)
    print(f"  removed {total:,} record(s)" + (" -- nothing written" if args.dry_run else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

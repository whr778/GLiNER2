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

``--near-dup T`` also drops a lower-precedence document that CONTAINS a share >= T of a
higher-precedence document's 8-word shingles (``near_dup_eval.shingles``, the eval near-dup
gate's measure). Exact keys miss the same wire story re-published with a different header
or ending: WikiEvents ships test doc 10 92% inside train doc 113.
Read the pairs before trusting it: a shared footer trips it too (DocEE: ABC's copyright footer
fills >= 20% of short test docs, 761 false holds), so it is run per corpus, never across the board.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Set

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _split import derive_split_paths, dumps_record, normalize_group_key  # noqa: E402
from near_dup_eval import shingles  # noqa: E402

# test last: each split drops what the later, higher-precedence ones already hold.
SPLIT_ORDER = ("train", "val", "test")


def _load(path: Path) -> List[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _write(path: Path, records: List[dict]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(dumps_record(rec) + "\n")


def near_dup_holders(higher: List[str], texts: List[str], threshold: float) -> Set[int]:
    """Indices of ``texts`` that contain >= ``threshold`` of some ``higher`` doc's shingles."""
    hs = [shingles(t) for t in higher]
    index: Dict[int, List[int]] = {}
    for j, sh in enumerate(hs):
        for h in sh:
            index.setdefault(h, []).append(j)
    held = set()
    for i, t in enumerate(texts):
        count: Dict[int, int] = {}
        for h in shingles(t):
            for j in index.get(h, ()):
                count[j] = count.get(j, 0) + 1
        if any(c / len(hs[j]) >= threshold for j, c in count.items()):
            held.add(i)
    return held


def dedupe(paths: Dict[str, Path], dry_run: bool = False, near_dup: float | None = None) -> int:
    """Drop duplicate and cross-split documents (and, with ``near_dup``, near-duplicates of a
    later split's documents). Returns the number removed."""
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
        near: Set[int] = set()
        if near_dup is not None:
            later = [r.get("input", "") for h in SPLIT_ORDER[i + 1:] for r in records.get(h, [])]
            near = near_dup_holders(later, [r.get("input", "") for r in records[split]], near_dup) if later else set()

        kept, seen, dropped_cross, dropped_dup = [], set(), 0, 0
        for n, (rec, key) in enumerate(zip(records[split], keys[split])):
            if key in blocked or n in near:
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
    ap.add_argument("--near-dup", type=float, metavar="T",
                    help="also drop a document containing >= T of a later split's document's 8-word shingles (0.2)")
    args = ap.parse_args()

    paths = derive_split_paths(args.base)
    # Some converters emit `.dev.jsonl` instead of `.val.jsonl` (wikievents, rams).
    # Fall back to it rather than refusing to repair the corpus: the validation split
    # is the middle precedence tier, and skipping it would leave train/val overlap
    # unrepaired while reporting success on the pair it did check.
    if not paths["val"].exists():
        dev = Path(str(paths["val"]).replace(".val.jsonl", ".dev.jsonl"))
        if dev.exists():
            print(f"  using {dev.name} as the validation split")
            paths["val"] = dev
    # Operate on whichever splits exist. Refusing outright left train+val-only corpora
    # (warmstart_mix has no test) permanently unrepairable, which is worse than
    # repairing the pair that IS present -- precedence still holds among the survivors.
    present = {s: q for s, q in paths.items() if q.exists()}
    if len(present) < 2:
        absent = ", ".join(str(q) for s, q in paths.items() if s not in present)
        raise SystemExit(f"need at least two split files, found {len(present)}; "
                         f"absent: {absent}")
    if len(present) < len(paths):
        print(f"  no {'/'.join(s for s in paths if s not in present)} split; "
              f"repairing {'/'.join(present)} only")
    paths = present

    print(f"{'DRY RUN: ' if args.dry_run else ''}{args.base} (precedence test > val > train)")
    total = dedupe(paths, dry_run=args.dry_run, near_dup=args.near_dup)
    print(f"  removed {total:,} record(s)" + (" -- nothing written" if args.dry_run else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

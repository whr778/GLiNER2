"""Sample a proportional replay slice from the 137k joint_ie training pool.

Unlike `data/replay_pile30` -- which is a *proxy* for base-v1's unknown training data --
this is the **literal** training pool of `joint-boundary-mmbert-137k`. We own that base
and its data, so replaying it is exact: "did the original capability survive" becomes
directly measurable against the same eval the base was scored on.

**Proportional by corpus.** A flat random draw over the concatenated pool would sample
each corpus in proportion to its size anyway, but doing it per corpus makes the
composition exact rather than approximate at small slice sizes, and makes a short
corpus's contribution visible in the printout instead of silently rounding to zero.

Reads only TRAIN splits -- the base never saw val or test, so replaying those would
leak. Deterministic (seed 42).
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))
from _split import dumps_record, normalize_group_key  # noqa: E402

import yaml  # noqa: E402


def pool_files(config: Path) -> dict[str, Path]:
    """Every TRAIN file the given config trains on, keyed by corpus name."""
    cfg = yaml.safe_load(config.read_text(encoding="utf-8"))
    out: dict[str, Path] = {}
    for co in cfg["data"].get("corpora") or []:
        out[Path(co).name] = Path(f"{co}.train.jsonl")
    for name, v in (cfg["data"].get("event_files") or {}).items():
        out[name] = Path(v["train"])
    return {k: v for k, v in out.items() if v.exists()}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path,
                    default=Path("tools/train/config/joint-boundary-mmbert-137k.yaml"))
    ap.add_argument("--new-records", type=int, required=True,
                    help="size of the NEW data the replay accompanies")
    ap.add_argument("--replay-frac", type=float, default=0.30,
                    help="replay as a fraction of the TOTAL mixture (default 0.30)")
    ap.add_argument("--out", type=Path, default=Path("data/replay_137k30"))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    files = pool_files(args.config)
    sizes = {k: sum(1 for _ in p.open(encoding="utf-8")) for k, p in files.items()}
    pool_total = sum(sizes.values())
    # replay/(new+replay) = frac  ->  replay = new * frac/(1-frac)
    want = round(args.new_records * args.replay_frac / (1 - args.replay_frac))
    if want > pool_total:
        raise SystemExit(f"need {want:,} but pool holds only {pool_total:,}")

    rng = random.Random(args.seed)
    picked, rows = {}, []
    for name, path in sorted(files.items()):
        share = round(want * sizes[name] / pool_total)
        lines = [l for l in path.open(encoding="utf-8") if l.strip()]
        rng.shuffle(lines)
        take = lines[:share]
        picked[name] = len(take)
        rows.extend(take)
    rng.shuffle(rows)

    out_train = Path(f"{args.out}.train.jsonl")
    seen = set()
    with out_train.open("w", encoding="utf-8") as fh:
        for line in rows:
            rec = json.loads(line)
            seen.add(normalize_group_key(rec.get("input", "")))
            fh.write(dumps_record(rec) + "\n")
    # the trainer resolves a corpus as <base>.{train,val,test}; give it readable paths
    for split in ("val", "test"):
        Path(f"{args.out}.{split}.jsonl").write_text("", encoding="utf-8")

    print(f"pool {pool_total:,} records across {len(files)} corpora")
    print(f"new data {args.new_records:,}  ->  replay {len(rows):,} "
          f"({len(rows)/(args.new_records+len(rows))*100:.1f}% of the mixture)")
    print(f"{'corpus':22s} {'pool':>9s} {'sampled':>9s} {'share':>7s}")
    for name in sorted(picked):
        print(f"{name:22s} {sizes[name]:9,d} {picked[name]:9,d} "
              f"{picked[name]/max(len(rows),1)*100:6.1f}%")
    print(f"\nwrote {out_train}  ({len(seen):,} unique documents)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

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
    """Every TRAIN file the given config trains on, keyed by corpus name and deduplicated by path."""
    cfg = yaml.safe_load(config.read_text(encoding="utf-8"))
    raw: list[tuple[str, Path]] = []
    for co in cfg["data"].get("corpora") or []:
        raw.append((Path(co).name, Path(f"{co}.train.jsonl")))
    for name, v in (cfg["data"].get("event_files") or {}).items():
        if "train" in v:   # val/test-only entries (eb21: duee, cmnee, mendeley_ed) are never trained on
            raw.append((name, Path(v["train"])))
    out: dict[str, Path] = {}
    seen_paths: set[Path] = set()
    for name, p in raw:
        if not p.exists():
            continue
        rp = p.resolve()
        if rp in seen_paths:
            continue
        seen_paths.add(rp)
        out[name] = p
    return out


def base_eval_files(config: Path) -> list[Path]:
    """Every VAL and TEST file the given config evaluates on, deduplicated by path."""
    cfg = yaml.safe_load(config.read_text(encoding="utf-8"))
    data = cfg.get("data") or {}
    train_only = {Path(str(x)).name for x in (data.get("train_only") or [])}
    candidates: list[Path] = []
    for co in data.get("corpora") or []:
        if Path(co).name in train_only:
            continue
        for sp in ("val", "test"):
            candidates.append(Path(f"{co}.{sp}.jsonl"))
    for v in (data.get("event_files") or {}).values():
        for sp in ("val", "test"):
            if sp in v:
                candidates.append(Path(v[sp]))
    out: list[Path] = []
    seen: set[Path] = set()
    for p in candidates:
        if not p.exists():
            continue
        rp = p.resolve()
        if rp not in seen:
            seen.add(rp)
            out.append(p)
    return out


def _collect_doc_keys(paths: list[Path]) -> set[str]:
    keys: set[str] = set()
    for p in paths:
        if not p.exists():
            continue
        with p.open(encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    keys.add(normalize_group_key(json.loads(line).get("input", "")))
    return keys


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path,
                    default=Path("tools/train/config/base/joint-boundary-mmbert-137k.yaml"))
    ap.add_argument("--new-records", type=int, required=True,
                    help="size of the NEW data the replay accompanies")
    ap.add_argument("--replay-frac", type=float, default=0.30,
                    help="replay as a fraction of the TOTAL mixture (default 0.30)")
    ap.add_argument("--exclude-corpora", nargs="*", default=[],
                    help="corpus prefixes whose .val.jsonl and .test.jsonl documents must not be sampled")
    ap.add_argument("--out", type=Path, default=Path("data/replay_137k30"))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    eval_paths = base_eval_files(args.config)
    for prefix in args.exclude_corpora:
        for sp in ("val", "test"):
            eval_paths.append(Path(f"{prefix}.{sp}.jsonl"))
    forbidden = _collect_doc_keys(eval_paths)

    files = pool_files(args.config)
    filtered_lines: dict[str, list[str]] = {}
    excluded_count = 0
    for name, path in sorted(files.items()):
        clean = []
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                if normalize_group_key(json.loads(line).get("input", "")) in forbidden:
                    excluded_count += 1
                    continue
                clean.append(line)
        filtered_lines[name] = clean

    sizes = {k: len(v) for k, v in filtered_lines.items()}
    pool_total = sum(sizes.values())
    # replay/(new+replay) = frac  ->  replay = new * frac/(1-frac)
    want = round(args.new_records * args.replay_frac / (1 - args.replay_frac))
    if want > pool_total:
        raise SystemExit(f"need {want:,} but pool holds only {pool_total:,}")

    rng = random.Random(args.seed)
    picked, rows = {}, []
    for name in sorted(filtered_lines):
        share = round(want * sizes[name] / pool_total)
        lines = list(filtered_lines[name])
        rng.shuffle(lines)
        take = lines[:share]
        picked[name] = len(take)
        rows.extend((name, line) for line in take)
    rng.shuffle(rows)

    out_train = Path(f"{args.out}.train.jsonl")
    seen = set()
    with out_train.open("w", encoding="utf-8") as fh:
        for name, line in rows:
            # Keep the SOURCE corpus: train.py only setdefaults `_corpus` from the file name, so without
            # this every replay record is tagged with the replay file and the negatives injector and
            # `partial_annotation` look up the wrong pool.
            rec = dict(json.loads(line), _corpus=name)
            key = normalize_group_key(rec.get("input", ""))
            assert key not in forbidden, "forbidden eval document leaked into replay"
            seen.add(key)
            fh.write(dumps_record(rec) + "\n")
    # the trainer resolves a corpus as <base>.{train,val,test}; give it readable paths
    for split in ("val", "test"):
        Path(f"{args.out}.{split}.jsonl").write_text("", encoding="utf-8")

    print(f"pool {pool_total:,} clean records across {len(files)} corpora "
          f"(excluded {excluded_count:,} eval-overlapping rows)")
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

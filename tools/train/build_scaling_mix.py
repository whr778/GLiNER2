"""Build nested, proportional Stage-A subsamples for the mmBERT head-init scaling curve.

Option B of tools/events_working_papers/SCALING_CURVE_EXPERIMENT.md: assemble the
structure/argument-dense event corpora already on disk into 10K and 40K nested
subsets (the ~100K point uses the full train files directly). Deterministic
(seed 42); each corpus's 10K slice is a prefix of its 40K slice, so the mixes nest.
Each corpus contributes in proportion to its share of the ~100K pool. A small shared
val slice per corpus keeps epoch eval cheap (eval_loss is only used within a run to
pick that run's best checkpoint). RAMS and WikiEvents are excluded -- they are the
downstream targets, so including them here would leak.

Run from the repo root:  .venv/bin/python tools/train/build_scaling_mix.py
Outputs to data/scaling/ (gitignored, like the rest of data/).
"""
import random
from pathlib import Path

SEED = 42
TARGETS = {"s10k": 10_000, "s40k": 40_000}
VAL_CAP = 150  # per corpus, shared across all three sizes
CORPORA = [
    "chfinann", "docee", "docfee", "duee", "cmnee",
    "text2json", "maven", "events_biotech", "mendeley_ed", "casie",
]
DATA = Path("data")
OUT = DATA / "scaling"


def read_lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines()


def write_lines(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    trains = {c: read_lines(DATA / f"{c}.train.jsonl") for c in CORPORA}
    total = sum(len(v) for v in trains.values())
    rng = random.Random(SEED)
    print(f"pool: {total} train records across {len(CORPORA)} corpora")
    sums = {tag: 0 for tag in TARGETS}
    for c in CORPORA:
        lines = trains[c][:]
        rng.shuffle(lines)
        row = f"{c:16s} n={len(lines):6d}"
        for tag, target in TARGETS.items():
            k = round(len(lines) * target / total)
            write_lines(OUT / f"{c}.{tag}.train.jsonl", lines[:k])
            sums[tag] += k
            row += f"  {tag}={k:5d}"
        vp = DATA / f"{c}.val.jsonl"
        if vp.exists():
            vlines = read_lines(vp)
            rng.shuffle(vlines)
            write_lines(OUT / f"{c}.val.jsonl", vlines[:VAL_CAP])
            row += f"  val={min(len(vlines), VAL_CAP)}"
        else:
            row += "  val=NONE"
        print(row)
    print(f"totals: s10k={sums['s10k']}  s40k={sums['s40k']}  s100k={total} (full)")


if __name__ == "__main__":
    main()

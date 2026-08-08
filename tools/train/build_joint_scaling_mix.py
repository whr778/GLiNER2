"""Build nested Stage-A subsamples for the joint_ie BOUNDARY scaling curve.

Unlike :mod:`build_scaling_mix` (which slices the event pool only, for the span
curve), every point here carries **both** event and relation data at a fixed
composition, so the only factor varying across the curve is total volume --
not whether the relation head was warmed at all.

Sizes are **total mix records**: {10K, 40K, 100K, 137K}. The composition is the
full pool's own ratio, events 73.02% / relations 26.98%, so::

    total  10,000 -> events   7,302  relations  2,698
    total  40,000 -> events  29,209  relations 10,791
    total 100,000 -> events  73,023  relations 26,977
    total 137,052 -> events 100,080  relations 36,972   (the whole pool)

The 137K point is the full pool, so it reads the original ``data/*.train.jsonl``
directly and needs no slice. Within each pool a corpus contributes in proportion
to its share of that pool. Deterministic (seed 42) and **nested**: each corpus's
10K slice is a prefix of its 40K slice, which is a prefix of its 100K slice.

Writes to ``data/scaling_joint/`` with a ``j`` prefix, leaving the span curve's
``data/scaling/`` slices untouched -- the two experiments must not collide.

Run from the repo root::

    uv run python tools/train/build_joint_scaling_mix.py
"""
import random
from pathlib import Path

SEED = 42
TARGETS = {"j10k": 10_000, "j40k": 40_000, "j100k": 100_000}
VAL_CAP = 150  # per corpus, shared across sizes; epoch eval only picks a checkpoint

EVENT_CORPORA = [
    "chfinann", "docee", "docfee", "duee", "cmnee",
    "text2json", "maven", "events_biotech", "mendeley_ed", "casie",
]
# Non-leaking relation corpora. DocRED is EXCLUDED: Re-DocRED re-annotates the
# same documents, so including it would leak the downstream.
RELATION_CORPORA = ["sentence_rex", "bio_ner_relations", "biored", "scierc"]

DATA = Path("data")
OUT = DATA / "scaling_joint"


def read_lines(path: Path) -> list[str]:
    return [l for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def write_lines(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n" if lines else "", encoding="utf-8")


def shuffled(corpora: list[str]) -> dict[str, list[str]]:
    """Read each corpus and shuffle deterministically, so prefixes nest."""
    out = {}
    for name in corpora:
        lines = read_lines(DATA / f"{name}.train.jsonl")
        random.Random(SEED).shuffle(lines)
        out[name] = lines
    return out


def emit(pool: dict[str, list[str]], pool_target: int, tag: str) -> int:
    """Write one slice per corpus, each proportional to its share of the pool.

    Relation corpora are consumed through ``data.corpora``, which auto-derives
    ``<base>.{train,val,test}.jsonl`` **without checking existence**, so a sliced
    base must ship all three or training dies on a missing path. Val/test are
    capped copies of the originals (identical across sizes -- only train varies).
    """
    total = sum(len(v) for v in pool.values())
    written = 0
    for name, lines in pool.items():
        take = min(len(lines), round(pool_target * len(lines) / total))
        write_lines(OUT / f"{name}.{tag}.train.jsonl", lines[:take])
        for split in ("val", "test"):
            src = DATA / f"{name}.{split}.jsonl"
            rows = read_lines(src)[:VAL_CAP] if src.is_file() else []
            write_lines(OUT / f"{name}.{tag}.{split}.jsonl", rows)
        written += take
    return written


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    events = shuffled(EVENT_CORPORA)
    relations = shuffled(RELATION_CORPORA)
    n_events = sum(len(v) for v in events.values())
    n_relations = sum(len(v) for v in relations.values())
    grand = n_events + n_relations
    print(f"pool: events {n_events:,} + relations {n_relations:,} = {grand:,} "
          f"(events {n_events / grand:.2%})")

    for tag, target in TARGETS.items():
        ev = emit(events, round(target * n_events / grand), tag)
        rel = emit(relations, round(target * n_relations / grand), tag)
        print(f"  {tag:6} -> events {ev:7,}  relations {rel:6,}  total {ev + rel:7,}")

    # One shared val slice per corpus (all sizes reuse it).
    for name in EVENT_CORPORA + RELATION_CORPORA:
        src = DATA / f"{name}.val.jsonl"
        if src.is_file():
            write_lines(OUT / f"{name}.val.jsonl", read_lines(src)[:VAL_CAP])
    print(f"val slices (cap {VAL_CAP}/corpus) written to {OUT}/")
    print("137K point = the full pool; configs read data/*.train.jsonl directly.")


if __name__ == "__main__":
    main()

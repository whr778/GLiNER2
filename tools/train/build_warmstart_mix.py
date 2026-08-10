"""Build the warm-start mixture: 70% new tasks, 30% replay, shuffled together.

Why this shape. `mmbert-137k` cannot do `[C]` record extraction at all -- verified
directly: None at every threshold down to 0.01, with `enable_records: true` and a real
`record_decoder` module. The cause is the training mix. It contains a corpus NAMED
`text2json`, but every one of its 7,817 records supervises `entities`, not structures, so
the record head was never taught the task. The same audit explains weak NER: those 7,817
records were the ONLY entity supervision, 5.7% of a 137k mix.

So the new data is structure + NER, and replay guards what already works.

**Replay, not fine-tuning.** Training on the new tasks alone would forget the old ones,
which is not hypothetical here: the casualty fine-tune was trained on a narrow homogeneous
schema with zero replay and destroyed capabilities it was not training on (asked for a
`location` it returns a digit). 30% replay at the pool's own 73/27 event/relation ratio --
sampling only events would preserve one head and starve the other.

**Shuffled at the example level.** Replay only works if every BATCH is a mixture;
concatenated task blocks are close to the worst case. The trainer shuffles on load
(`ExtractorDataset(shuffle=is_train)`), and this writes a pre-shuffled file as well so the
mixture is right even if something downstream streams it in order. No length bucketing
exists in the pipeline, so tasks cannot be silently re-segregated by batching.

    uv run python tools/train/build_warmstart_mix.py
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

DATA = Path("data")
SEED = 42

# New capability: structure (the point) and NER (weakest existing head).
STRUCTURE = ["casualty_multi_loc", "paraloq_json"]
NER = ["pile_ner_def", "nuner_full"]
# Replay, at the 137k pool's own ratio.
EVENT_CORPORA = ["chfinann", "docee", "docfee", "duee", "cmnee",
                 "text2json", "maven", "events_biotech", "mendeley_ed", "casie"]
RELATION_CORPORA = ["sentence_rex", "bio_ner_relations", "biored"]


def read(name: str, max_chars: int = 0) -> list[str]:
    """Corpus lines, optionally dropping documents longer than `max_chars`.

    Profiled on the warm-start mixture (2026-08-10), and the cap is worth far more than
    the 2.4% of records it removes. Step cost grows superlinearly with sequence length --
    measured 374 tok -> 1.1s, 743 -> 2.4s, 1,913 -> 7.9s on the same model -- and the NER
    corpora carry a long tail the structure corpus does not have at all (4.5% of NER
    records over 4,000 chars, max 52,881; structure max 2,007).

    Two things follow. Those 4.5% carry ~half of all NER cost and ~34% of the WHOLE run's
    cost. And 15% of them are then discarded anyway: they blow `max_gold_per_query=32`
    (one query had 770 gold spans) and `on_capacity_exceeded=skip_sample` drops the
    sample. Measured: 6/40 long records skipped, 0/40 short. So the tail is compute spent
    to produce no gradient.
    """
    for cand in (DATA / f"{name}.train.jsonl", DATA / f"{name}.jsonl"):
        if cand.is_file():
            lines = [l for l in cand.read_text(encoding="utf-8").splitlines() if l.strip()]
            if not max_chars:
                return lines
            kept = [l for l in lines if len(json.loads(l).get("input") or "") <= max_chars]
            print(f"    [cap] {name}: dropped {len(lines) - len(kept):,} of {len(lines):,} "
                  f"over {max_chars:,} chars")
            return kept
    return []


def take(names: list[str], total: int, rng: random.Random, label: str,
         even: bool = False, max_chars: int = 0) -> list[str]:
    """Sample `total` lines across `names`.

    Replay is proportional to pool size, so the mixture keeps the ratio the model was
    originally trained on. NER is `even` instead: pool size there reflects how big a
    dataset someone published, not how useful it is, and proportional sampling buries
    `pile_ner_def` (typed labels WITH definitions -- the shape GLiNER2 actually queries
    with) under `nuner_full`, which is 20x larger and has no definitions.
    """
    pools = {n: read(n, max_chars) for n in names}
    pools = {n: v for n, v in pools.items() if v}
    have = sum(len(v) for v in pools.values())
    if not have:
        print(f"  [{label}] NOTHING FOUND for {names}")
        return []
    out: list[str] = []
    share = {n: (total / len(pools)) if even else (total * len(v) / have)
             for n, v in pools.items()}
    deficit = sum(max(0.0, share[n] - len(pools[n])) for n in pools)   # small pools cap out
    live = [n for n in pools if len(pools[n]) > share[n]]
    for n, v in pools.items():
        want = round(share[n] + (deficit / len(live) if n in live and live else 0))
        want = min(len(v), want)
        rng.shuffle(v)
        out.extend(v[:want])
        print(f"  [{label}] {n:22} {want:>7,} of {len(v):>8,}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--total", type=int, default=86_000)
    ap.add_argument("--new-frac", type=float, default=0.70)
    ap.add_argument("--out", default="data/warmstart_mix.train.jsonl")
    ap.add_argument("--val-frac", type=float, default=0.02)
    ap.add_argument("--max-chars", type=int, default=2000,
                    help="drop docs longer than this. 2000 is the knee: 1.87x faster with the "
                         "structure corpus intact at 34.1%%; 1000 is 5.4x but halves it")
    args = ap.parse_args()
    rng = random.Random(SEED)

    n_new = int(args.total * args.new_frac)
    n_old = args.total - n_new
    print(f"target {args.total:,}  new {n_new:,} ({args.new_frac:.0%})  replay {n_old:,}")

    print("NEW - structure (all of it; this is the capability being added)")
    structure = take(STRUCTURE, 10**9, rng, "struct", max_chars=args.max_chars)
    print("NEW - NER")
    ner = take(NER, max(0, n_new - len(structure)), rng, "ner", even=True,
               max_chars=args.max_chars)

    print("REPLAY - events (73%) and relations (27%), the pool's own ratio")
    # The cap applies to REPLAY too: text2json is 30.6% over 4,000 chars (max 102,068)
    # and lives in the event list, so a NER-only cap misses the actual cost driver.
    events = take(EVENT_CORPORA, round(n_old * 0.73), rng, "event", max_chars=args.max_chars)
    relations = take(RELATION_CORPORA, round(n_old * 0.27), rng, "rel", max_chars=args.max_chars)

    mix = structure + ner + events + relations
    rng.shuffle(mix)                      # every batch must be a mixture, not a block

    # A val slice is carved from the SAME shuffled mixture, so checkpoint selection sees
    # the new tasks. The blind test stays the curve's own event/relation files, untouched,
    # which is what keeps per-capability numbers comparable to the 137k baseline.
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    n_val = max(200, int(len(mix) * args.val_frac))
    val, train = mix[:n_val], mix[n_val:]
    out.write_text("\n".join(train) + "\n", encoding="utf-8")
    val_path = out.with_name(out.name.replace(".train.", ".val."))
    val_path.write_text("\n".join(val) + "\n", encoding="utf-8")
    print(f"val slice: {len(val):,} -> {val_path}")

    tot = len(mix)                      # proportions are of the WHOLE mixture, pre-split
    print(f"\nwrote {out}  {len(train):,} train + {len(val):,} val = {tot:,} records")
    for label, n in (("structure", len(structure)), ("ner", len(ner)),
                     ("events(replay)", len(events)), ("relations(replay)", len(relations))):
        print(f"   {label:20} {n:>7,}  {n / tot:6.1%}")
    print(f"   {'NEW total':20} {len(structure) + len(ner):>7,} "
          f"{(len(structure) + len(ner)) / tot:6.1%}")
    print(f"   {'REPLAY total':20} {len(events) + len(relations):>7,} "
          f"{(len(events) + len(relations)) / tot:6.1%}")


if __name__ == "__main__":
    main()

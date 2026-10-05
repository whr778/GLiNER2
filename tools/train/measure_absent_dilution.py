"""How much do ABSENT queries dilute the boundary loss? Measured on real training batches, no model.

The start/end boundary loss uses `global` reduction: sum over kept (query, position) cells
divided by their count. Queries with no gold (injected negatives, or naturally empty) add
cells to that denominator, so the gold-bearing share

    pi_P = N_P / (N_P + N_A)

is the factor every present-query gradient is scaled by. This runs the trainer's own
chunking, label transform, negatives injector and collator, counts N_P and N_A per chunk
from `boundary_keep`'s ingredients (query mask x word positions + 1), then simulates
micro-batches drawn from the natural corpus mix. Negatives ON vs OFF separates injected
dilution from natural.

    uv run python tools/train/measure_absent_dilution.py --config tools/train/config/base/eb18-balanced.yaml
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/train"))
import train as T  # noqa: E402
import yaml  # noqa: E402


def load_records(cfg, config, per_corpus, seed, split="train"):
    """({corpus: sampled records}, {corpus: total records}) from one split; val excludes train_only."""
    data = cfg["data"]
    files = T._dedupe_paths(T._split_files(data.get("corpora") or [], split, set(data.get("train_only") or ()))
                            + T._event_split(data.get("event_files") or {}, split), split)
    fns = T._category_fns(T.load_labels_cfg(cfg, config_path=config))
    sampled, totals = {}, {}
    for f in files:
        p = Path(f)
        if not p.is_file():
            continue
        lines = [l for l in p.open(encoding="utf-8") if l.strip()]
        name = str(p.relative_to("data")).split(".")[0] if p.is_relative_to("data") else p.name.split(".")[0]   # by PATH: casie and scaling_joint/casie are two files
        totals[name] = len(lines)
        recs = [T.transform_record(json.loads(l), fns) for l in random.Random(seed).sample(lines, min(per_corpus, len(lines)))]
        for r in recs:
            r["_corpus"] = name
        sampled[name] = recs
    return sampled, totals


def counts(batch):
    """(N_P, N_A) kept boundary cells for a one-chunk batch."""
    qmask = batch.query_marker_mask[0].bool()
    positions = int(batch.text_word_counts[0]) + 1
    has_gold = batch.targets.mention_mask[0].any(-1)[: qmask.shape[0]]
    n_p = int((qmask & has_gold).sum()) * positions
    n_a = int((qmask & ~has_gold).sum()) * positions
    return n_p, n_a


def measure(cfg, config, sampled, negatives, proc):
    from gliner2.training.chunking import chunk_records
    from gliner2.training.trainer import ExtractorCollator, ExtractorDataset
    tr, head = cfg["training"], cfg["model"]["boundary_head"]
    coll = ExtractorCollator(proc, is_training=True, max_len=tr.get("max_len"), architecture="boundary",
                             max_gold_per_query=int(head.get("max_gold_per_query", 32)),
                             on_capacity_exceeded=tr.get("on_capacity_exceeded", "raise"), error_policy="skip",
                             event_records=bool(head.get("event_records", False)))
    out = defaultdict(list)
    for corpus, recs in sampled.items():
        chunks = chunk_records(recs, tokenizer=proc.tokenizer, window_size=int(tr["max_len"]),
                               stride=int(tr["window_stride"]), show_progress=False)
        ds = ExtractorDataset(data=chunks, max_samples=-1, shuffle=False, seed=42, validate=False, negatives=negatives)
        for i in range(len(ds)):
            batch = coll([ds[i]])
            if batch is not None and batch.query_marker_mask.numel():
                out[corpus].append(counts(batch))
    return out


def simulate(cells, totals, batch_size, n_batches, seed):
    """pi_P per simulated micro-batch, corpora drawn in proportion to their train records."""
    names = [c for c in cells if cells[c]]
    weights = [totals[c] for c in names]
    rng = random.Random(seed)
    pis = []
    for _ in range(n_batches):
        picks = [rng.choice(cells[c]) for c in rng.choices(names, weights, k=batch_size)]
        n_p, n_a = sum(p for p, _ in picks), sum(a for _, a in picks)
        if n_p + n_a:
            pis.append(n_p / (n_p + n_a))
    return sorted(pis)


def summary(pis):
    q = lambda f: pis[int(f * (len(pis) - 1))]
    return f"mean {sum(pis) / len(pis):.3f} | p10 {q(0.1):.3f} p50 {q(0.5):.3f} p90 {q(0.9):.3f}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default="tools/train/config/base/eb18-balanced.yaml")
    ap.add_argument("--checkpoint", default="whr778/gliner2-eb18-balanced", help="only its processor is used")
    ap.add_argument("--per-corpus", type=int, default=300)
    ap.add_argument("--batches", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    from gliner2 import AutoExtractor
    from gliner2.training.negatives import NegativeLabels
    proc = AutoExtractor.from_pretrained(args.checkpoint, map_location="cpu").processor
    spec = importlib.util.spec_from_file_location("bnp", ROOT / "tools/data/build_negative_pools.py")
    bnp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bnp)
    tr = cfg["training"]
    neg = NegativeLabels(bnp.build_pools(cfg, Path(args.config), log=lambda *a: None), tr["negative_labels_per_dim"],
                         seed=int(tr.get("negative_label_seed", 42)), partial=cfg["data"].get("partial_annotation"))
    sampled, totals = load_records(cfg, args.config, args.per_corpus, args.seed)
    print(f"[dilution] {len(sampled)} corpora, {sum(map(len, sampled.values())):,} sampled records, batch {tr['batch_size']}")
    arms = {"negatives OFF": measure(cfg, args.config, sampled, None, proc),
            "negatives ON": measure(cfg, args.config, sampled, neg, proc)}
    print(f"[dilution] {'corpus':26s} {'chunks':>6s}  pi_P OFF  pi_P ON")
    for c in sorted(sampled):
        row = []
        for cells in arms.values():
            n_p, n_a = sum(p for p, _ in cells[c]), sum(a for _, a in cells[c])
            row.append(f"{n_p / (n_p + n_a):8.3f}" if n_p + n_a else "     n/a")
        print(f"[dilution] {c:26s} {len(arms['negatives ON'][c]):6d}  {'  '.join(row)}")
    for arm, cells in arms.items():
        print(f"[dilution] mix-weighted micro-batch pi_P, {arm:13s}: "
              f"{summary(simulate(cells, totals, int(tr['batch_size']), args.batches, args.seed))}")
    print(f"[dilution] {neg.composition_line()}")


if __name__ == "__main__":
    main()

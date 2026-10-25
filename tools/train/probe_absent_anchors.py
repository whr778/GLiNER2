"""Code review Issue 5b: do ABSENT event types reach the trigger gate as negatives in real training batches?

In natural mode the record head's object loss is zero (records.compute_group_loss) and the anchor gate decodes the
trigger candidates' PAIR logits, so the gate learns "this type is not here" only where the span losses see an event
query with no gold. This counts, per corpus, on the TRAINER'S OWN dataloader (negative_pools, negative_labels_per_dim
and every injection gradient_conflict.build reproduces -- `probe_absent_queries.py` calls the collator directly and
cannot see them): event ANCHOR (trigger) queries with and without gold, event ROLE queries, and entity queries.

    uv run tools/train/probe_absent_anchors.py --config tools/train/config/base/eb22d.yaml \\
        --checkpoint /Volumes/Development/tmp/eb22d_epochs/checkpoint-epoch-2 --records 60 --out absent.json
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import gradient_conflict as G  # noqa: E402

KINDS = ("anchor", "role", "entity")


def count_batch(batch) -> dict:
    """{kind: [present, absent]} over every sample in the batch."""
    out = {k: [0, 0] for k in KINDS}
    mask = batch.targets.mention_mask.any(-1)                       # [B, Q]: query carries gold
    for i, layout in enumerate(batch.query_layouts):
        specs = (batch.record_specs[i] if i < len(batch.record_specs) else None) or {}
        anchors = {s.anchor_query_id for s in specs.values() if s.mode == "natural"}
        roles = {f.query_id for s in specs.values() if s.mode == "natural" for f in s.fields} - anchors
        for q in layout.queries:
            kind = ("anchor" if q.query_id in anchors else "role" if q.query_id in roles
                    else "entity" if q.task_type == "entities" else None)
            if kind is not None and q.query_id < mask.shape[1]:
                out[kind][0 if bool(mask[i, q.query_id]) else 1] += 1
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--corpora", nargs="+", default=None, help="default: every train corpus with event gold")
    ap.add_argument("--records", type=int, default=60, help="per corpus")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    pool = [r for r in G.training_records(cfg, args.config, 10 ** 9, args.seed) if (r.get("output") or {}).get("events")]
    by = defaultdict(list)
    for r in pool:
        by[r["_corpus"]].append(r)
    names = args.corpora or sorted(by)
    print(f"[absent] event-carrying train corpora: {', '.join(f'{c} {len(by[c])}' for c in sorted(by))}", flush=True)
    model, trainer, _ = G.build(cfg, args.config, args.checkpoint, [dict(by[names[0]][0])], 1)
    result = {}
    for c in names:
        recs = random.Random(args.seed).sample(by[c], min(args.records, len(by[c])))
        loader = trainer._create_dataloader(trainer._prepare_data([dict(r) for r in recs], is_train=True), 1,
                                            shuffle=False, is_training=True)
        tot, docs = {k: [0, 0] for k in KINDS}, 0
        for batch in loader:
            for k, (p, a) in count_batch(batch).items():
                tot[k][0] += p
                tot[k][1] += a
            docs += 1
        result[c] = {"docs": docs, **{k: {"present": p, "absent": a} for k, (p, a) in tot.items()}}
        cell = lambda k: f"{tot[k][0]:>6} / {tot[k][1]:<6} ({tot[k][1] / max(docs, 1):5.1f} absent/doc)"
        print(f"[absent] {c:28s} docs {docs:4d} | anchor present/absent {cell('anchor')} | role {cell('role')} "
              f"| entity {cell('entity')}", flush=True)
    Path(args.out).write_text(json.dumps({"args": vars(args), "corpora": result}, indent=1, ensure_ascii=False),
                              encoding="utf-8")
    print(f"[absent] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()

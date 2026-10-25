"""Code review Issue 5a: how much event ARGUMENT supervision is dropped because the trigger was not proposed?

`compute_group_loss` (natural mode) resolves each gold record's trigger against the model's own candidates and
skips the record on a miss: no field loss for any of its arguments. Gold injection (`gold_injection_prob`, a
per-mention Bernoulli in the proposer) puts gold triggers into the candidates; the schedule holds 1.0 for the first
`gold_injection_hold_frac` of the planned steps, then anneals linearly to `gold_injection_end` (0.25). So the
dropped share is about (1 - p) x the model's own trigger miss rate, and it grows late in training.

`probe_anchor_gate.py` cannot measure this: it builds its own collator and never sets the schedule, so the head runs
at its constructor default p = 1.0 and the gate almost cannot fire. This runs gradient_conflict's training path
(trainer data preparation, negatives, train mode, dropout off) per corpus at each injection level -- `sched` (the
schedule at --step), and fixed values -- and reports, per corpus, the share of gold event RECORDS and of their gold
ARGUMENTS that reached no field loss. Records are read from the code's own counter (`records.anchor_gate_stats`);
arguments from a replica of the gate's resolution, which must count the same records or the tool refuses to report.

    uv run tools/train/probe_supervision_gate.py --config tools/train/config/base/eb22d.yaml \\
        --checkpoint /Volumes/Development/tmp/eb22d_epochs/checkpoint-epoch-2 --step 33172 \\
        --injection sched 0 0.25 --records 60 --out gate.json
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import gradient_conflict as G  # noqa: E402
from gliner2.models.boundary import records as R  # noqa: E402

TALLY = {"records": 0, "dropped": 0, "args": 0, "args_dropped": 0}


def replica_gate(group, records) -> None:
    """Count event records and their gold arguments the way compute_group_loss's natural-mode gate resolves them."""
    if group.spec.mode != "natural" or group.spec.task_type != "events":
        return
    qid = group.spec.anchor_query_id
    f_idx = group.field_query_ids.index(qid)
    seeds = {s[1] for s in group.instance_seed if s is not None and s[0] == f_idx}
    anchor_cols = R._span_index(group.field_spans[f_idx])
    for record in records:
        aft = record.field_for_query(qid)
        if aft is None or not aft.values:
            continue
        cols = R._resolve_value_cols(aft.values[0], anchor_cols)
        dropped = not cols or not any((c - 1) in seeds for c in cols)
        n_args = sum(len(f.values) for f in record.fields if f.query_id != qid)
        TALLY["records"] += 1
        TALLY["args"] += n_args
        if dropped:
            TALLY["dropped"] += 1
            TALLY["args_dropped"] += n_args


def spy(real):
    def compute_group_loss(group, records, *a, **k):
        replica_gate(group, records)
        return real(group, records, *a, **k)
    return compute_group_loss


def run(model, trainer, loader, p: float, batches: int) -> dict:
    for k in TALLY:
        TALLY[k] = 0
    R.reset_anchor_gate()
    model.boundary_head.set_gold_injection_prob(p)
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= batches:
                break
            torch.manual_seed(i)
            random.seed(i)
            model(batch)
    st = R.anchor_gate_stats().get("by_task") or {}
    ev = {o: st.get(f"events/{o}", 0) for o in ("trained", "anchor_not_proposed", "anchor_not_seeded")}
    official_dropped = ev["anchor_not_proposed"] + ev["anchor_not_seeded"]
    official_total = official_dropped + ev["trained"]
    assert (official_total, official_dropped) == (TALLY["records"], TALLY["dropped"]), \
        f"replica disagrees with the code's counter: official {official_total}/{official_dropped}, replica {TALLY}"
    return {"p": p, **ev, "records": TALLY["records"], "args": TALLY["args"],
            "record_dropped_share": TALLY["dropped"] / max(TALLY["records"], 1),
            "arg_dropped_share": TALLY["args_dropped"] / max(TALLY["args"], 1)}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--step", type=int, required=True)
    ap.add_argument("--planned-steps", type=int, default=82930)
    ap.add_argument("--injection", nargs="+", default=["sched", "0", "0.25"], help="sched and/or fixed probabilities")
    ap.add_argument("--corpora", nargs="+", default=None, help="default: every train corpus with event arguments")
    ap.add_argument("--records", type=int, default=60, help="per corpus")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    has_args = lambda r: any(e.get("arguments") for e in (r.get("output") or {}).get("events") or [] if isinstance(e, dict))
    by = defaultdict(list)
    for r in G.training_records(cfg, args.config, 10 ** 9, args.seed):
        if has_args(r):
            by[r["_corpus"]].append(r)
    names = args.corpora or sorted(by)
    print(f"[gate5a] train corpora with event arguments: {', '.join(f'{c} {len(by[c])}' for c in sorted(by))}", flush=True)
    model, trainer, _ = G.build(cfg, args.config, args.checkpoint, [dict(by[names[0]][0])], 1)
    model.train()
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.p = 0.0
    G.set_schedules(trainer, model, args.step, args.planned_steps)
    sched = float(model.boundary_head._gold_injection_prob)
    print(f"[gate5a] step {args.step} of {args.planned_steps}: scheduled gold injection p = {sched:.4f}", flush=True)
    R.compute_group_loss = spy(R.compute_group_loss)
    result = {"args": vars(args), "scheduled_p": sched, "corpora": {}}
    for c in names:
        recs = random.Random(args.seed).sample(by[c], min(args.records, len(by[c])))
        loader = trainer._create_dataloader(trainer._prepare_data([dict(r) for r in recs], is_train=True), 1,
                                            shuffle=False, is_training=True)
        rows = [run(model, trainer, loader, sched if p == "sched" else float(p), 10 ** 9) for p in args.injection]
        result["corpora"][c] = rows
        print(f"[gate5a] {c:28s} " + " | ".join(
            f"p={r['p']:.2f}: records dropped {r['record_dropped_share']:6.1%} of {r['records']}, "
            f"args dropped {r['arg_dropped_share']:6.1%} of {r['args']}" for r in rows), flush=True)
        Path(args.out).write_text(json.dumps(result, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"[gate5a] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()

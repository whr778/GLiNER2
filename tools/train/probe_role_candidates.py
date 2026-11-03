"""Where does a gold event role sit when the record loss looks at it?

On the training path (the trainer's collator, negatives, windows, dropout off),
each gold role filler of an event record is one of:

  candidate              the trigger was supervised and at least one gold span
                         is a candidate. The assignment probability is the max
                         sigmoid over the trained owners and the matched spans.
  not_in_pool            the trigger was supervised and no gold span is a candidate.
                         The field is withheld: it is not trained as empty.
  trigger_not_proposed   the gold trigger span is not a candidate, so the record
                         trains nothing.
  trigger_not_seeded     the trigger span is a candidate but no instance was seeded
                         from it. The record trains nothing.

An empty role is a real absence and is not a filler. Counts that do not match
the anchor gate, or a logit row that does not line up with the candidate spans,
are a refusal.

Injection is the schedule at the checkpoint's step, then zero. Zero is the
model's own proposals. The schedule is what the loss saw.

    uv run tools/train/probe_role_candidates.py \\
        --config tools/train/config/base/eb22e.yaml \\
        --checkpoint /Volumes/Development/tmp/autopsy_ckpts/whr778__gliner2-eb22e-epoch/checkpoint-epoch-1 \\
        --step 16586 --planned-steps 82930
"""
from __future__ import annotations

import argparse
import json
import random
import resource
import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import gradient_conflict as G  # noqa: E402
import train as T  # noqa: E402
from gliner2.models.boundary import records as R  # noqa: E402
from gliner2.training.trainer import GLiNER2Trainer  # noqa: E402

VAL = {
    "casie": "data/scaling_joint/casie.val.jsonl",
    "cmnee": "data/scaling_joint/cmnee.val.jsonl",
    "duee": "data/scaling_joint/duee.val.jsonl",
    "wikievents": "data/wikievents.dev.jsonl",
}
BUCKETS = ("candidate", "not_in_pool", "trigger_not_proposed", "trigger_not_seeded")
ROWS: list[dict] = []
RECORDS = {k: 0 for k in ("trained", "anchor_not_proposed", "anchor_not_seeded", "no_gold_anchor")}
CORPUS = ["?"]
COLLECT = [True]
SEEN: set[str] = set()


def load_val(cfg: dict, cfg_path: str, per_corpus: int, seed: int, max_chars: int) -> list[dict]:
    """Validation records that carry an event argument, label-mapped as train.py does."""
    fns = T._category_fns(T.load_labels_cfg(cfg, cfg_path))
    rng = random.Random(seed)
    kept, skipped = [], {}
    for name, path in VAL.items():
        rows = []
        for line in open(path, encoding="utf-8"):
            if not line.strip():
                continue
            rec = T.transform_record(json.loads(line), fns)
            text = rec.get("input") or ""
            events = (rec.get("output") or {}).get("events") or []
            if not any(isinstance(e, dict) and e.get("arguments") for e in events):
                continue
            if len(text) > max_chars:
                skipped[name] = skipped.get(name, 0) + 1
                continue
            rec["_corpus"] = name
            rows.append(rec)
        kept.extend(rng.sample(rows, min(per_corpus, len(rows))))
        print(f"[roles] {name}: {len(rows)} docs within {max_chars} chars, "
              f"skipped {skipped.get(name, 0)} longer, taking {min(per_corpus, len(rows))}", flush=True)
    return kept


def _matched_cols(alternatives, span_to_idx) -> list[int]:
    cols = []
    for span in alternatives:
        idx = span_to_idx.get((int(span[0]), int(span[1])))
        if idx is not None:
            cols.append(idx + 1)
    return cols


def classify(group, records, coreferent_ownership: bool) -> list[dict]:
    """One row per gold role filler. Event records in natural mode only."""
    if group.spec.mode != "natural" or group.spec.task_type != "events":
        return []
    anchor_qid = group.spec.anchor_query_id
    anchor_f = group.field_query_ids.index(anchor_qid)
    span_indices = [R._span_index(spans) for spans in group.field_spans]
    for f_idx, spans in enumerate(group.field_spans):
        row = group.assign_logits[f_idx]
        if row.shape[-1] != spans.shape[0] + 1:
            raise RuntimeError(
                f"assign columns {row.shape[-1]} != candidates {spans.shape[0]} + ABSENT"
            )
    seed_to_inst = {
        seed[1]: i for i, seed in enumerate(group.instance_seed)
        if seed is not None and seed[0] == anchor_f
    }
    out = []
    for record in records:
        aft = record.field_for_query(anchor_qid)
        if aft is None or not aft.values:
            RECORDS["no_gold_anchor"] += 1
            continue
        cols = R._resolve_value_cols(aft.values[0], span_indices[anchor_f])
        if not cols:
            fate, gate_key = "trigger_not_proposed", "anchor_not_proposed"
            owners: list[int] = []
        else:
            owners = (sorted({seed_to_inst[c - 1] for c in cols if (c - 1) in seed_to_inst})
                      if coreferent_ownership
                      else [i for i in [seed_to_inst.get(cols[0] - 1)] if i is not None])
            fate, gate_key = (None, "trained") if owners else ("trigger_not_seeded", "anchor_not_seeded")
        RECORDS[gate_key] += 1
        for f_idx, fspec in enumerate(group.field_specs):
            if fspec.query_id == anchor_qid or fspec.cardinality.is_scalar:
                continue
            target = record.field_for_query(fspec.query_id)
            if target is None or not target.values:
                continue
            for value in target.values:
                hit = _matched_cols(value, span_indices[f_idx])
                if fate is not None:
                    row = {"bucket": fate, "p": None, "corpus": CORPUS[0]}
                elif not hit:
                    row = {"bucket": "not_in_pool", "p": None, "corpus": CORPUS[0]}
                else:
                    probs = [
                        float(torch.sigmoid(group.assign_logits[f_idx][inst][col].detach().float()))
                        for inst in owners for col in hit
                    ]
                    row = {"bucket": "candidate", "p": max(probs), "corpus": CORPUS[0]}
                if row["bucket"] not in SEEN:
                    SEEN.add(row["bucket"])
                    print(f"[roles] first {row['bucket']}: corpus {CORPUS[0]} role {fspec.name} "
                          f"alts {len(value)} matched {len(hit)} owners {len(owners)} "
                          f"p {None if row['p'] is None else round(row['p'], 6)}", flush=True)
                out.append(row)
    return out


def _wrap(coreferent_ownership_default: bool):
    orig = R.compute_group_loss

    def wrapped(group, records, **opts):
        if COLLECT[0]:
            owns = opts.get("coreferent_ownership", coreferent_ownership_default)
            ROWS.extend(classify(group, records, bool(owns)))
        return orig(group, records, **opts)

    return orig, wrapped


def summarize(rows: list[dict]) -> dict:
    counts = {k: 0 for k in BUCKETS}
    by_corpus: dict[str, dict[str, int]] = {}
    probs, by_prob = [], {}
    for row in rows:
        counts[row["bucket"]] += 1
        name = row["corpus"] or "?"
        bucket = by_corpus.setdefault(name, {k: 0 for k in BUCKETS})
        bucket[row["bucket"]] += 1
        if row["p"] is not None:
            probs.append(row["p"])
            by_prob.setdefault(name, []).append(row["p"])
    n = sum(counts.values())
    probs.sort()
    def share(k):
        return counts[k] / n if n else 0.0
    above = lambda t: sum(p >= t for p in probs) / len(probs) if probs else 0.0
    return {
        "fillers": n,
        "counts": counts,
        "shares": {k: share(k) for k in BUCKETS},
        "by_corpus": by_corpus,
        "by_corpus_probability": {
            name: {"n": len(ps), "median": sorted(ps)[len(ps) // 2],
                   "share_at_least_0.1": sum(p >= 0.1 for p in ps) / len(ps)}
            for name, ps in by_prob.items()
        },
        "assignment_probability": {
            "n": len(probs),
            "median": probs[len(probs) // 2] if probs else None,
            "mean": sum(probs) / len(probs) if probs else None,
            "share_at_least_0.1": above(0.1),
            "share_at_least_0.5": above(0.5),
        },
    }


def check_gate() -> None:
    """The wrapper's record fates must equal the loss's own event anchor gate."""
    gate = R.anchor_gate_stats()["by_task"]
    for key, got in RECORDS.items():
        expected = gate.get(f"events/{key}", 0)
        if got != expected:
            raise RuntimeError(f"events/{key}: wrapper {got} != anchor gate {expected}")
    print("[roles] anchor gate matches: " + ", ".join(f"{k} {v}" for k, v in RECORDS.items()), flush=True)


def run_pass(model, loader, injection: float, label: str, corpus: str) -> int:
    """One corpus at this injection. Returns the number of batches. Rows accumulate."""
    R.reset_anchor_gate()
    for key in RECORDS:
        RECORDS[key] = 0
    before = len(ROWS)
    CORPUS[0] = corpus
    model.boundary_head.set_gold_injection_prob(injection)
    used = 0
    for batch in loader:
        with torch.no_grad():
            out = model(batch)
        loss = float(out.loss.detach())
        if loss != loss:
            raise RuntimeError("non-finite loss")
        used += 1
        print(f"[roles] {label} {corpus} batch {used}: tokens {tuple(batch.input_ids.shape)} "
              f"loss {loss:.4f} fillers {len(ROWS) - before} "
              f"rss {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30:.1f} GB",
              flush=True)
    check_gate()
    return used


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default="tools/train/config/base/eb22e.yaml")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--step", type=int, default=16586)
    ap.add_argument("--planned-steps", type=int, default=82930)
    ap.add_argument("--per-corpus", type=int, default=20)
    ap.add_argument("--max-chars", type=int, default=12000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--out", default="/Volumes/Development/tmp/role_candidates/eb22e_epoch1.json")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    records = load_val(cfg, args.config, args.per_corpus, args.seed, args.max_chars)
    schedule = GLiNER2Trainer._gold_injection_probability(
        args.step / args.planned_steps, 1.0, 0.25, 0.15)
    print(f"[roles] {len(records)} docs, step {args.step}/{args.planned_steps}, "
          f"scheduled injection {schedule:.4f}, device {args.device}", flush=True)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    model, trainer, loader = G.build(cfg, args.config, args.checkpoint, [dict(r) for r in records], 1, args.device)
    model.train()
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0
    G.set_schedules(trainer, model, args.step, args.planned_steps)
    owns = bool(model.boundary_settings.record_coreferent_ownership)
    by_corpus: dict[str, list[dict]] = {}
    for rec in records:
        by_corpus.setdefault(rec["_corpus"], []).append(rec)
    loaders = {}
    for name, subset in by_corpus.items():
        dataset = trainer._prepare_data([dict(r) for r in subset], is_train=True)
        loaders[name] = trainer._create_dataloader(dataset, 1, shuffle=False, is_training=True)
        print(f"[roles] {name}: {len(dataset)} training windows", flush=True)
    _orig, wrapped = _wrap(owns)
    R.compute_group_loss = wrapped
    try:
        for injection, label in ((schedule, "schedule"), (0.0, "injection-0")):
            ROWS.clear()
            SEEN.clear()
            for name, corpus_loader in loaders.items():
                run_pass(model, corpus_loader, injection, label, name)
            block = summarize(ROWS)
            block["injection"] = injection
            if label == "schedule":
                at_schedule = block
            else:
                at_zero = block
    finally:
        R.compute_group_loss = _orig
    result = {"args": vars(args), "scheduled_injection": schedule,
              "schedule": at_schedule, "injection_0": at_zero}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=1, ensure_ascii=False), encoding="utf-8")
    for name, block in (("schedule", at_schedule), ("injection 0", at_zero)):
        p = block["assignment_probability"]
        s = block["shares"]
        print(f"[roles] {name}: fillers {block['fillers']}  "
              f"candidate {s['candidate']:.1%}  not-in-pool {s['not_in_pool']:.1%}  "
              f"trigger-not-proposed {s['trigger_not_proposed']:.1%}  "
              f"trigger-not-seeded {s['trigger_not_seeded']:.1%}", flush=True)
        med = p["median"]
        print(f"[roles]   assignment probability on candidates: n {p['n']}  "
              f"median {None if med is None else round(med, 6)}  "
              f"mean {None if p['mean'] is None else round(p['mean'], 6)}  "
              f"share >= 0.1 {p['share_at_least_0.1']:.1%}  "
              f"share >= 0.5 {p['share_at_least_0.5']:.1%}", flush=True)
        for corpus, counts in block["by_corpus"].items():
            n = sum(counts.values())
            cp = block["by_corpus_probability"].get(corpus)
            med_c = None if cp is None else round(cp["median"], 6)
            ge = None if cp is None else f"{cp['share_at_least_0.1']:.1%}"
            print(f"[roles]   {corpus:12s} fillers {n:5d}  " + "  ".join(
                f"{k} {counts[k] / n:.1%}" for k in BUCKETS
            ) + f"  candidate P median {med_c}  >=0.1 {ge}", flush=True)
    print(f"[roles] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()

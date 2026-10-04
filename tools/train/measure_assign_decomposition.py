"""Is the trigger -> argument assignment a JUNCTION (pair identity) or mostly ROLE FIT? Measured.

`RecordDecoder._assign_logits` scores instance i, role field f, candidate j as

    (inst_proj(t_i) + field_proj(r_f)) . cand_proj(a_j)  =  P[i, j]  +  R[f, j]
    P = inst_proj(t_i) . cand_proj(a_j)    pairwise term: depends on the trigger AND the argument
    R = field_proj(r_f) . cand_proj(a_j)   role term: the SAME for every trigger

If R dominates, a candidate that fits a role scores high under EVERY trigger -- which would explain
why argument evidence cannot separate gold from false triggers (AUC ~0.5) and why training false
rows (negative instances) cost recall (pushing R down hurts true rows too). On real training-shaped
event batches (natural groups with gold), reports:

  spread      std of P vs std of R over role-field cells
  specificity per argument candidate, variance ACROSS TRIGGERS (P only) vs ACROSS CANDIDATES
  junction    AUC: gold trigger vs false trigger scoring the SAME gold argument (R cancels: pure P)
  row         AUC: gold argument vs other candidates in the gold trigger's row, by P, by R, by P+R

    uv run python tools/train/measure_assign_decomposition.py --checkpoint whr778/gliner2-eb18-balanced
"""
from __future__ import annotations

import argparse
import importlib.util
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/train"))
import measure_absent_dilution as D  # noqa: E402

import gliner2.models.boundary.records as R  # noqa: E402


def auc(pos, neg) -> float:
    if not pos or not neg:
        return float("nan")
    pos, neg = torch.tensor(pos), torch.tensor(neg)
    return float(((pos[:, None] > neg[None, :]).float() + 0.5 * (pos[:, None] == neg[None, :]).float()).mean())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default="tools/train/config/base/eb18-balanced.yaml")
    ap.add_argument("--checkpoint", default="whr778/gliner2-eb18-balanced")
    ap.add_argument("--batches", type=int, default=8)
    ap.add_argument("--false-triggers", type=int, default=8, help="top-scoring non-gold instances compared per group")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    tr, head = cfg["training"], cfg["model"]["boundary_head"]
    from gliner2 import AutoExtractor
    from gliner2.training.chunking import chunk_records
    from gliner2.training.trainer import ExtractorCollator, ExtractorDataset

    model = AutoExtractor.from_pretrained(args.checkpoint, map_location="cpu")
    dec = model.record_decoder
    proc = model.processor
    sampled, totals = D.load_records(cfg, args.config, 40, args.seed)
    event_corpora = [n for n in sorted(sampled)
                     if any(any(e.get("arguments") for e in r["output"].get("events") or []) for r in sampled[n])]
    coll = ExtractorCollator(proc, is_training=True, max_len=tr.get("max_len"), architecture="boundary",
                             max_gold_per_query=int(head.get("max_gold_per_query", 32)),
                             on_capacity_exceeded=tr.get("on_capacity_exceeded", "raise"), error_policy="skip",
                             event_records=True)

    pending = []
    real_assign = dec._assign_logits

    def assign_spy(inst_states, field_query_states, field_cand_states):
        with torch.no_grad():
            inst_q = dec.inst_proj(inst_states)
            field_q = dec.field_proj(field_query_states)
            parts = []
            for f, cand in enumerate(field_cand_states):
                if cand.shape[0] == 0:
                    parts.append(None)
                    continue
                cand_p = dec.cand_proj(cand)
                parts.append((inst_q @ cand_p.t(), field_q[f] @ cand_p.t()))   # P [Ni, Cf], R [Cf]
            out = real_assign(inst_states, field_query_states, field_cand_states)
            for f, part in enumerate(parts):
                if part is not None:
                    err = float((part[0] + part[1][None, :] - out[f][:, 1:]).abs().max())
                    stats["decomp_err"].append(err)
            pending.append(parts)
        return out

    dec._assign_logits = assign_spy
    stats = defaultdict(list)
    real_loss = R.compute_group_loss

    def loss_spy(group, records, *a, **k):
        parts = pending.pop(0) if pending else None
        if parts is not None and group.spec.mode == "natural" and records:
            collect(group, records, parts, args.false_triggers, stats)
        return real_loss(group, records, *a, **k)

    R.compute_group_loss = loss_spy
    rng = random.Random(args.seed)
    model.train()
    for b in range(args.batches):
        corpora = rng.choices(event_corpora, [totals[n] for n in event_corpora], k=int(tr["batch_size"]))
        recs = [rng.choice([r for r in sampled[c] if any(e.get("arguments") for e in r["output"].get("events") or [])])
                for c in corpora]
        chunks = chunk_records(recs, tokenizer=proc.tokenizer, window_size=int(tr["max_len"]),
                               stride=int(tr["window_stride"]), show_progress=False)
        ds = ExtractorDataset(data=chunks, max_samples=-1, shuffle=False, seed=42, validate=False, negatives=None)
        random.seed(args.seed + b)
        batch = coll([ds[i] for i in range(min(len(ds), int(tr["batch_size"])))])
        pending.clear()
        torch.manual_seed(args.seed + b)
        with torch.no_grad():
            model(batch)
    report(args.checkpoint, stats)


def collect(group, records, parts, k_false, stats) -> None:
    aq = group.spec.anchor_query_id
    af = group.field_query_ids.index(aq)
    spans = [R._span_index(s) for s in group.field_spans]
    seed = {s[1]: i for i, s in enumerate(group.instance_seed) if s is not None and s[0] == af}
    gold_insts = {}
    for rec in records:
        a = rec.field_for_query(aq)
        cols = R._resolve_value_cols(a.values[0], spans[af]) if a is not None and a.values else []
        if cols and (inst := seed.get(cols[0] - 1)) is not None:
            gold_insts[inst] = rec
    if not gold_insts:
        return
    false_insts = R._negative_instances(group, records, aq, set(gold_insts), k_false)
    for f, fs in enumerate(group.field_specs):
        if fs.cardinality.is_scalar or parts[f] is None:
            continue
        P, Rr = parts[f]
        stats["P_std"].append(float(P.std()))
        stats["R_std"].append(float(Rr.std()))
        if P.shape[0] > 1:
            stats["var_across_triggers"].append(float(P.var(dim=0).mean()))       # per candidate, across instances
            stats["var_across_candidates"].append(float((P.mean(0) + Rr).var()))  # column means across candidates
        for inst, rec in gold_insts.items():
            cols, _ = R._field_target_cols(fs, rec, spans[f])
            gold_j = sorted({c - 1 for c in cols if 0 <= c - 1 < P.shape[1]})
            if not gold_j:
                continue
            other = [j for j in range(P.shape[1]) if j not in gold_j]
            for j in gold_j:
                stats["junction_gold"].append(float(P[inst, j]))
                stats["junction_false"].extend(float(P[i, j]) for i in false_insts)
                stats["row_gold_P"].append(float(P[inst, j]))
                stats["row_gold_R"].append(float(Rr[j]))
                stats["row_gold_T"].append(float(P[inst, j] + Rr[j]))
            sample = other if len(other) <= 64 else random.Random(inst).sample(other, 64)
            stats["row_other_P"].extend(float(P[inst, j]) for j in sample)
            stats["row_other_R"].extend(float(Rr[j]) for j in sample)
            stats["row_other_T"].extend(float(P[inst, j] + Rr[j]) for j in sample)


def report(name: str, s) -> None:
    mean = lambda v: statistics.mean(v) if v else float("nan")
    print(f"[decomp] {name}: {len(s['P_std'])} role fields, {len(s['junction_gold'])} gold arguments, "
          f"{len(s['junction_false'])} false-trigger comparisons")
    print(f"[decomp] GATE         max |P + R - model assign logit| = {max(s['decomp_err']) if s['decomp_err'] else float('nan'):.2e}")
    print(f"[decomp] spread       std(P) {mean(s['P_std']):.3f} | std(R) {mean(s['R_std']):.3f}")
    print(f"[decomp] specificity  variance across TRIGGERS (P) {mean(s['var_across_triggers']):.3f} | "
          f"across CANDIDATES (column means) {mean(s['var_across_candidates']):.3f}")
    print(f"[decomp] junction     AUC gold-trigger vs false-trigger on the SAME gold argument (pure P): "
          f"{auc(s['junction_gold'], s['junction_false']):.3f}")
    print(f"[decomp] row          AUC gold argument vs other candidates in the gold row: by P "
          f"{auc(s['row_gold_P'], s['row_other_P']):.3f} | by R {auc(s['row_gold_R'], s['row_other_R']):.3f} | "
          f"by P+R {auc(s['row_gold_T'], s['row_other_T']):.3f}")


if __name__ == "__main__":
    main()

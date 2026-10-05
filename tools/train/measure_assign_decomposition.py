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
  junction    PER-COLUMN AUC: gold trigger vs each false trigger scoring the SAME gold argument (R cancels),
              by pure P and by the model's full score (incl. a junction link term)
  row         PER-ROW AUC: gold argument vs each other candidate in the gold trigger's row, by P, by R, by P+R
  (2026-10-05: the first version POOLED all gold vs all false scores across columns/rows; corrected.)

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
    # ALIGNMENT GATE: the captured P/R must be THIS group's -- matched by call order, so a skipped
    # loss call would shift every later group. Compare against the group's own logits.
    for f, part in enumerate(parts):
        if part is None or group.field_specs[f].cardinality.is_scalar:
            continue
        full = group.assign_logits[f].detach()[:, 1:]
        ok = part[0].shape == full.shape and float((part[0] + part[1][None, :] - full).abs().max()) < 1e-3
        stats["aligned"].append(ok)
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
            full = group.assign_logits[f].detach()
            other = [j for j in range(P.shape[1]) if j not in gold_j]
            for j in gold_j:
                # PER COLUMN: the gold trigger vs the false triggers scoring the SAME argument.
                for key, col in (("P", P[:, j]), ("full", full[:, 1 + j])):
                    g = float(col[inst])
                    for i in false_insts:
                        stats[f"col_{key}_wins"].append(1.0 if g > float(col[i]) else 0.5 if g == float(col[i]) else 0.0)
                # PER ROW: the gold argument vs the other candidates in the gold trigger's row.
                for key, row in (("P", P[inst]), ("R", Rr), ("T", P[inst] + Rr)):
                    g = float(row[j])
                    for o in other:
                        stats[f"row_{key}_wins"].append(1.0 if g > float(row[o]) else 0.5 if g == float(row[o]) else 0.0)


def report(name: str, s) -> None:
    mean = lambda v: statistics.mean(v) if v else float("nan")
    print(f"[decomp] {name}: {len(s['P_std'])} role fields")
    print(f"[decomp] GATE         max |P + R - model assign logit| = {max(s['decomp_err']) if s['decomp_err'] else float('nan'):.2e}")
    print(f"[decomp] spread       std(P) {mean(s['P_std']):.3f} | std(R) {mean(s['R_std']):.3f}")
    print(f"[decomp] specificity  variance across TRIGGERS (P) {mean(s['var_across_triggers']):.3f} | "
          f"across CANDIDATES (column means) {mean(s['var_across_candidates']):.3f}")
    m = lambda k: sum(s[k]) / len(s[k]) if s[k] else float("nan")
    al = s["aligned"]
    print(f"[decomp] ALIGNMENT    captured P+R equals the group's own logits in {sum(al)} of {len(al)} role fields "
          f"(must be ALL on an additive model; a junction model adds the link, so 0 there is expected)")
    print(f"[decomp] junction     PER-COLUMN AUC, gold vs false trigger on the SAME gold argument: pure P {m('col_P_wins'):.3f} | "
          f"full score (incl. link) {m('col_full_wins'):.3f}  [{len(s['col_P_wins'])} pairs]")
    print(f"[decomp] row          PER-ROW AUC, gold argument vs other candidates in the gold row: by P {m('row_P_wins'):.3f} | "
          f"by R {m('row_R_wins'):.3f} | by P+R {m('row_T_wins'):.3f}  [{len(s['row_P_wins'])} pairs]")


if __name__ == "__main__":
    main()

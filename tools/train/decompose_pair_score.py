"""Split each gold trigger's pair logit into the SparseBoundaryPairScorer's additive parts, per checkpoint.

The autopsy probe found the gold trigger top-ranked at every eb22c checkpoint while the LEVEL of all trigger scores
swings across the 0.3 anchor gate (CMNEE gold logit median -4.33 at 20k, +1.56 at 30k). The pair logit is a sum:
  compat   endpoint compatibility (gated start x end projection, plus the endpoint-difference term)
  start    start marginal at the candidate start;  end   end marginal at the candidate end
  prior    the proposer's compat_logits
  content  span content x query, plus the content bias
  inside   inside evidence;  length   length features x query
This recomputes each part from the scorer's own inputs, refuses to report when the parts do not sum to the real
logit, and prints, per checkpoint, the median of each part at the gold candidate and over all valid candidates.

    uv run tools/train/decompose_pair_score.py --config tools/train/config/base/eb22c-ddp-trace.yaml \\
        --checkpoints whr778/gliner2-eb22c-collapse-trace:checkpoint-4000 whr778/gliner2-eb22c-collapse-trace:checkpoint-5000 \\
        --corpora cmnee duee --docs 40 --out decompose.json
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import probe_event_collapse as P  # noqa: E402
import train as T  # noqa: E402
from gliner2.models.boundary.scoring import gather_boundary_states, interval_prefix_score, continuous_length_features  # noqa: E402

PARTS = ("compat", "start", "end", "prior", "content", "inside", "length")


def parts(sc, boundary_states, query_states, proposals, start_logits, end_logits, inside_prefix, text_lengths,
          text_states=None, text_mask=None, inside_prefix_mean=None) -> dict:
    """SparseBoundaryPairScorer.forward (eval mode), one [B,Q,C] tensor per additive part."""
    starts, ends, valid = proposals.indices[..., 0], proposals.indices[..., 1], proposals.valid_mask
    if proposals.score_start_states is not None and proposals.score_end_states is not None:
        s_proj, e_proj = proposals.score_start_states, proposals.score_end_states
    else:
        start_all, end_all = sc.project_endpoints(boundary_states)
        s_proj, e_proj = gather_boundary_states(start_all, starts), gather_boundary_states(end_all, ends)
    gate = torch.sigmoid(sc.query_gate(query_states))
    if sc.enable_rotary_endpoints:
        gate = gate.repeat_interleave(2, dim=-1)
    gate = gate.unsqueeze(2)
    compat = torch.zeros_like(starts, dtype=s_proj.dtype)
    if sc.reranker_endpoint_compat:
        per_head = (s_proj * gate * e_proj).reshape(*s_proj.shape[:-1], sc.multihead_pair_compat_heads, -1).sum(-1)
        compat = sc.compat_mix(per_head).squeeze(-1) / math.sqrt(sc.pair_dim)
    if sc.endpoint_difference_projection is not None:
        diff = s_proj - e_proj
        compat = compat + sc.endpoint_difference_projection(torch.cat((diff, diff.abs()), -1)).squeeze(-1)
    out = {"compat": compat,
           "start": torch.gather(start_logits, 2, starts.clamp(0, start_logits.shape[2] - 1)),
           "end": torch.gather(end_logits, 2, ends.clamp(0, end_logits.shape[2] - 1))}
    prior = proposals.compat_logits if proposals.compat_logits is not None else proposals.logits
    out["prior"] = torch.where(valid, prior, torch.zeros_like(prior))
    zero = torch.zeros_like(compat)
    out["content"] = zero
    if sc.content_pooler is not None:
        mean_prefix, lse_prefix = sc.content_pooler.build_prefix(text_states, text_mask)
        span = sc.content_pooler.pool(mean_prefix, lse_prefix, starts, ends, compat.dtype)
        coeff = sc.content_query_projection(query_states).unsqueeze(2)
        out["content"] = (span * coeff).sum(-1) / math.sqrt(span.shape[-1]) + sc.content_bias(span).squeeze(-1)
    out["inside"] = zero
    if sc.use_inside_evidence and inside_prefix is not None:
        interval = interval_prefix_score(inside_prefix, starts, ends, inside_prefix_mean).to(compat.dtype)
        w = (sc.inside_weight(query_states).squeeze(-1).unsqueeze(-1) if sc.query_conditioned_inside_weight
             else sc.inside_weight)
        out["inside"] = w * interval / torch.sqrt((ends - starts).clamp(min=1).to(compat.dtype))
    feats = continuous_length_features(starts, ends, text_lengths)
    out["length"] = (feats.to(compat.dtype) * sc.length_query_projection(query_states).unsqueeze(2)).sum(-1)
    return out


def decompose_doc(model, rec: dict, threshold: float, queries=("events",)) -> list[dict]:
    """probe_doc rows (events and/or entities), each with the gold candidate's parts and the per-part median over the query's valid candidates."""
    sc = model.boundary_head.pair_scorer
    orig, seen = sc.forward, []

    def hook(*a, **k):
        logits = orig(*a, **k)
        seen.append((parts(sc, *a, **k), logits, a[2].valid_mask))
        return logits

    sc.forward = hook
    try:
        rows = P.probe_doc(model, rec, threshold, queries)
    finally:
        sc.forward = orig
    if not rows:
        return []
    assert len(seen) == 1, f"pair scorer called {len(seen)} times for one single-chunk document"
    p, logits, valid = seen[0]
    for r in rows:
        if not r["reached"]:
            continue
        q, c = r["query_id"], r["gold_candidate"]
        total = sum(float(p[k][0, q, c]) for k in PARTS)
        assert abs(total - float(logits[0, q, c])) < 1e-3 and abs(total - r["gold_logit"]) < 1e-3, \
            f"parts sum {total:.4f} != scorer {float(logits[0, q, c]):.4f} / candidates {r['gold_logit']:.4f}"
        v = valid[0, q]
        r["gold_parts"] = {k: float(p[k][0, q, c]) for k in PARTS}
        r["level_parts"] = {k: float(p[k][0, q][v].median()) for k in PARTS}
    return rows


def summarize(name: str, rows: list[dict]) -> dict:
    reached = [r for r in rows if r.get("gold_parts")]
    med = lambda xs: round(statistics.median(xs), 3)
    s = {"checkpoint": name, "gold_events": len(rows), "decomposed": len(reached),
         "gold_logit": med([r["gold_logit"] for r in reached]),
         "gold": {k: med([r["gold_parts"][k] for r in reached]) for k in PARTS},
         "level": {k: med([r["level_parts"][k] for r in reached]) for k in PARTS}}
    print(f"[decompose] {name}: n {s['decomposed']}/{s['gold_events']} | gold logit {s['gold_logit']} | "
          f"level logit {med([r['level_logit'] for r in reached])} | gold rank {med([r['gold_rank'] for r in reached])} | "
          f"gold above gate {sum(r['gold_gate_prob'] >= P.THRESHOLD for r in reached) / len(rows):.3f}", flush=True)
    for kind in ("gold", "level"):
        print(f"[decompose]   {kind:5s} " + " ".join(f"{k} {s[kind][k]:+.3f}" for k in PARTS), flush=True)
    return s


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoints", nargs="+", required=True, help="local dirs or repo:subfolder")
    ap.add_argument("--corpora", nargs="+", default=["cmnee"], help="event_files names, else data/<name>.val.jsonl")
    ap.add_argument("--queries", nargs="+", default=["events"], choices=["events", "entities"])
    ap.add_argument("--docs", type=int, default=40)
    ap.add_argument("--threshold", type=float, default=0.3)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    P.THRESHOLD = args.threshold
    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    fns, passthrough = T._category_fns(T.load_labels_cfg(cfg, args.config)), T.labels_passthrough(cfg)
    docs = {}
    for c in args.corpora:
        path = cfg["data"]["event_files"][c]["val"] if c in cfg["data"]["event_files"] else f"data/{c}.val.jsonl"
        T._fetch_if_missing(path)
        docs[c] = [r for r in T.read_transformed([path], fns, passthrough)
                   if any(r["output"].get(q) for q in args.queries)][:args.docs]
    result = {"args": vars(args), "checkpoints": {}}
    for ref in args.checkpoints:
        model = P.load_checkpoint(ref)
        per = {}
        for c, recs in docs.items():
            rows = [row for r in recs for row in decompose_doc(model, r, args.threshold, args.queries)]
            for kind in sorted({r["kind"] for r in rows}):
                sub = [r for r in rows if r["kind"] == kind]
                per[f"{c}/{kind}"] = {"summary": summarize(f"{ref} [{c} {kind}]", sub), "rows": sub}
        result["checkpoints"][ref] = per
        Path(args.out).write_text(json.dumps(result, indent=1, ensure_ascii=False), encoding="utf-8")
        del model
    print(f"[decompose] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()

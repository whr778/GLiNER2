"""The life of one training sample: every stage on the REAL pipeline, printed, ending in its gradient.

Stages (each prints what it actually holds):
  S0 raw JSONL record             S4 collator: query layout, gold per query, record groups
  S1 label transform (labels_file) S5 forward: proposals, reach before injection, heads
  S2 sliding-window chunking       S6 every loss term, its weight, its contribution; the
  S3 schema sampling + negatives      weighted sum must reproduce the model's total (a gate)
                                   S7 per-term gradient norm on the shared encoder and on
                                      the whole model, and the clip factor at max_grad_norm

CPU, one record, no optimiser step. Used by tools/events_working_papers/LIFE_OF_A_SAMPLE.md.

    uv run python tools/train/trace_sample_life.py --corpus casie --config tools/train/config/base/eb18-balanced.yaml
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
from collections import Counter
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/train"))
import train as T  # noqa: E402


def stage(name: str) -> None:
    print(f"\n[life] ===== {name} =====")


def pick_record(path: str, max_chars: int, seed: int) -> dict:
    lines = [l for l in open(path, encoding="utf-8") if l.strip()]
    for l in random.Random(seed).sample(lines, len(lines)):
        rec = json.loads(l)
        if (rec.get("output") or {}).get("events") and len(rec["input"]) <= max_chars:
            return rec
    raise SystemExit(f"no event record under {max_chars} chars in {path}")


def summarize_output(out: dict) -> str:
    ents = out.get("entities") or {}
    evs = out.get("events") or []
    return (f"entities {[(k, len(v)) for k, v in ents.items()] if isinstance(ents, dict) else ents} | "
            f"events {[(e.get('event_type'), e.get('triggers'), len(e.get('arguments') or [])) for e in evs]} | "
            f"absent_events {list((out.get('absent_events') or {}).keys())}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default="tools/train/config/base/eb18-balanced.yaml")
    ap.add_argument("--checkpoint", default="whr778/gliner2-eb18-balanced")
    ap.add_argument("--corpus", default="casie")
    ap.add_argument("--max-chars", type=int, default=700)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--absent-reduction", default=None, help="override, e.g. separate (eb19 candidate)")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    tr, head_cfg = cfg["training"], cfg["model"]["boundary_head"]

    from gliner2 import AutoExtractor
    from gliner2.training.chunking import chunk_records
    from gliner2.training.negatives import NegativeLabels
    from gliner2.training.trainer import ExtractorCollator, ExtractorDataset

    spec = importlib.util.spec_from_file_location("bnp", ROOT / "tools/data/build_negative_pools.py")
    bnp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bnp)
    path = bnp._corpus_train_paths(cfg)[args.corpus]

    stage("S0 raw record")
    raw = pick_record(path, args.max_chars, args.seed)
    print(f"[life] file {path} | chars {len(raw['input'])}")
    print(f"[life] text: {raw['input'][:300]!r}")
    print(f"[life] gold: {summarize_output(raw['output'])}")

    stage("S1 label transform (labels_file)")
    fns = T._category_fns(T.load_labels_cfg(cfg, args.config))
    rec = T.transform_record(raw, fns)
    rec["_corpus"] = args.corpus
    print(f"[life] labels_file {cfg.get('labels_file')}")
    print(f"[life] gold: {summarize_output(rec['output'])}")

    stage("S2 sliding-window chunking")
    model = AutoExtractor.from_pretrained(args.checkpoint, map_location="cpu")
    proc = model.processor
    chunks = chunk_records([rec], tokenizer=proc.tokenizer, window_size=int(tr["max_len"]),
                           stride=int(tr["window_stride"]), show_progress=False)
    print(f"[life] window {tr['max_len']} stride {tr['window_stride']} -> {len(chunks)} chunk(s); "
          f"subword tokens {len(proc.tokenizer(rec['input'])['input_ids'])}")

    stage("S3 schema sampling + injected negatives (ExtractorDataset.__getitem__)")
    pools = bnp.build_pools(cfg, Path(args.config), log=lambda *a: None)
    neg = NegativeLabels(pools, tr["negative_labels_per_dim"], seed=int(tr.get("negative_label_seed", 42)),
                         partial=cfg["data"].get("partial_annotation"))
    ds = ExtractorDataset(data=chunks, max_samples=-1, shuffle=False, seed=42, validate=False, negatives=neg)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    text, schema = ds[0]
    print(f"[life] schema after sampling + injection: {summarize_output(schema)}")
    print(f"[life] injector: {neg.composition_line()}")

    stage("S4 collator: queries and targets")
    coll = ExtractorCollator(proc, is_training=True, max_len=tr.get("max_len"), architecture="boundary",
                             max_gold_per_query=int(head_cfg.get("max_gold_per_query", 32)),
                             on_capacity_exceeded="raise", error_policy="raise",
                             event_records=bool(head_cfg.get("event_records", False)))
    random.seed(args.seed)
    batch = coll([(text, schema)])
    layout = batch.query_layouts[0]
    gold = batch.targets.mention_mask[0].sum(-1)
    print(f"[life] words {int(batch.text_word_counts[0])} | subword ids {tuple(batch.input_ids.shape)} | queries {len(layout.queries)}")
    for q in layout.queries:
        print(f"[life]   q{q.query_id:<3d} {q.task_type:9s} {str(q.task_name)[:24]:24s} {str(q.role_name)[:20]:20s} gold {int(gold[q.query_id])}"
              + ("   <- ABSENT" if int(gold[q.query_id]) == 0 else ""))
    print(f"[life] record groups: {[(g.task_name if hasattr(g, 'task_name') else str(g)[:60]) for g in (batch.record_specs[0] or [])]}"[:400])

    stage("S5 forward (train mode)")
    if args.absent_reduction:
        T._apply_boundary_head_overrides(model, {"absent_reduction": args.absent_reduction})
    s = model.boundary_head.settings
    print(f"[life] settings: reduction {s.loss_reduction}, absent_reduction {s.absent_reduction}, start/end_top_k "
          f"{s.start_top_k}/{s.end_top_k}, training_candidate_budget {s.training_candidate_budget}, "
          f"negative_query_ratio {s.negative_query_ratio}, attn {model.config.attn_implementation}")
    model.train()
    torch.manual_seed(args.seed)
    out = model(batch)
    stats = getattr(model.boundary_head, "_last_proposal_stats", None)
    if stats is not None and getattr(stats, "gold_hit_without_injection", None) is not None:
        print(f"[life] proposal gold reach BEFORE injection: {stats.gold_hit_without_injection}")

    stage("S6 loss terms -> total")
    L = {k: v for k, v in out.losses.items() if torch.is_tensor(v) and v.dim() == 0}
    lw = model.boundary_head.loss_weights
    h = model.boundary_head
    weights = {
        "start_loss": lw.get("start", 1.0), "end_loss": lw.get("end", 1.0), "pair_loss": lw.get("pair", 1.0),
        "inside_loss": lw.get("inside", 0.5),
        "soft_iou_loss": s.soft_iou_aux_weight * h._soft_iou_scale,
        "rerank_listwise_loss": s.rerank_listwise_weight, "proposal_loss": s.proposal_loss_weight,
        "consistency_loss": s.consistency_loss_weight * h._consistency_scale,
        "abstention_loss": s.abstention_loss_weight, "count_loss": s.count_loss_weight,
    }
    span_sum = sum(weights[k] * L[k] for k in weights)
    print(f"[life] {'term':22s} {'value':>10s} {'weight':>8s} {'contrib':>10s}")
    for k, w in weights.items():
        print(f"[life] {k:22s} {float(L[k]):10.5f} {w:8.3f} {float(w * L[k]):10.5f}")
    print(f"[life] span total: recomputed {float(span_sum):.6f} vs head total_loss(before outer) -- see gate below")
    outer = {k: float(L[k]) for k in ("classification_loss", "record_object_loss", "record_field_loss", "relation_loss") if k in L}
    print(f"[life] outer terms: {outer}")
    rest = float(out.loss) - float(span_sum) - sum(outer.values())
    print(f"[life] GATE: model loss {float(out.loss):.6f} = span {float(span_sum):.6f} + outer {sum(outer.values()):.6f} "
          f"+ residual {rest:.2e}  (nonzero residual = an unaccounted term or weight)")

    stage("S7 gradient per term")
    enc = [p for n, p in model.named_parameters() if p.requires_grad and n.startswith("encoder")]
    allp = [p for p in model.parameters() if p.requires_grad]
    terms = {**{k: weights[k] * L[k] for k in weights}, **{k: L[k] for k in outer}}
    total_g = torch.autograd.grad(out.loss, allp, retain_graph=True, allow_unused=True)
    gn2 = float(sum((g.float() ** 2).sum() for g in total_g if g is not None))
    norm = lambda gs: float(torch.sqrt(sum((g.float() ** 2).sum() for g in gs if g is not None)) or 0.0)
    print(f"[life] {'term':22s} {'|grad| enc':>11s} {'|grad| all':>11s} {'share of total':>15s}")
    for k, t in terms.items():
        if not t.requires_grad:
            print(f"[life] {k:22s} no graph (constant)")
            continue
        g_enc = torch.autograd.grad(t, enc, retain_graph=True, allow_unused=True)
        g_all = torch.autograd.grad(t, allp, retain_graph=True, allow_unused=True)
        dot = float(sum((a.float() * b.float()).sum() for a, b in zip(g_all, total_g) if a is not None and b is not None))
        print(f"[life] {k:22s} {norm(g_enc):11.5f} {norm(g_all):11.5f} {dot / gn2:15.4f}")
    gn = gn2 ** 0.5
    mg = float(tr.get("max_grad_norm", 1.0))
    print(f"[life] TOTAL |grad| {gn:.5f}; max_grad_norm {mg} -> clip factor {min(1.0, mg / gn):.4f} "
          f"(before gradient_accumulation_steps={tr.get('gradient_accumulation_steps')} averaging)")


if __name__ == "__main__":
    main()

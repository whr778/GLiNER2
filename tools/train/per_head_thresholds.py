"""Per-head decision thresholds for a trained checkpoint, picked on VAL and scored on the blind test ONCE.

The end-of-run sweep picks ONE global threshold by support-weighted strict micro-F1 over every category;
``pick_per_head_thresholds`` reads the same sweep and takes each decode head's own best (entity, relation,
classification, structure, event -- the three event categories share one decode). This tool runs that on an
existing checkpoint without retraining:

    uv run tools/train/per_head_thresholds.py --config CFG --checkpoint DIR [--grid 0.1 0.3 0.5]
        [--per-corpus 8] [--test] [--device cpu] --out picks.json

``--per-corpus N`` samples N val (and test) records per corpus for a laptop read; omit it for the full splits.
``--picks entity=0.3 relation=0.5 ...`` overrides the val picks (the splice gate is only exercised when heads differ).
``--test`` scores the test split at the global pick and at the per-head picks, and prints the splice gate:
for every head, the per-head run's metrics for that head's categories must EQUAL the single-threshold run
at that head's value (a bled splice fails it). The picks go to ``--out``, never into the checkpoint.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train as T  # noqa: E402
from gliner2 import AutoExtractor  # noqa: E402
from gliner2.training.eval_metrics import (  # noqa: E402
    HEAD_CATEGORIES, _selection_score, compute_metrics, compute_metrics_per_head, per_head_report,
    pick_per_head_thresholds, sweep_thresholds,
)
from gliner2.training.trainer import ExtractorDataset  # noqa: E402

MISMATCH_TOL = 1e-9


def split_records(cfg: dict, split: str, per_corpus: int | None, seed: int = 0) -> list:
    """The config's val/test records (train_only excluded), optionally N per corpus."""
    d = cfg["data"]; train_only = set(d.get("train_only") or []); rng = random.Random(seed); out = []
    for c in d["corpora"]:
        p = Path(f"{c}.{split}.jsonl")
        if c in train_only or not p.exists():
            continue
        T._fetch_if_missing(str(p))
        rows = [json.loads(l) for l in open(p, encoding="utf-8")]
        if per_corpus:
            rows = rng.sample(rows, min(per_corpus, len(rows)))
        for r in rows:
            r["_corpus"] = c
        out.extend(rows)
    return out


def splice_gate(per_head: dict, picks: dict, by_threshold: dict) -> list:
    """Per head, the per-head metrics for its categories vs the single-threshold run at its pick; [] passes."""
    bad = []
    for head, t in picks.items():
        for cat in HEAD_CATEGORIES[head]:
            for metric in ("strict_micro_f1", "strict_support", "relaxed_micro_f1"):
                key = f"eval_{cat}_{metric}"
                a, b = per_head.get(key), by_threshold[t].get(key)
                if (a is None) != (b is None) or (a is not None and abs(a - b) > MISMATCH_TOL):
                    bad.append(f"{key}: per-head {a} vs single-run@{t} {b}")
    return bad


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--grid", type=float, nargs="+", default=[0.1, 0.3, 0.5, 0.7, 0.9])
    ap.add_argument("--per-corpus", type=int, default=None)
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--picks", nargs="*", default=[], help="head=threshold overrides applied after the val sweep")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    if args.device == "cpu":
        torch.backends.mps.is_available = lambda: False
    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    ev = T._parse_eval_settings(cfg, args.config, None)
    gd = dict(chunk_size=ev["chunk_size"], chunk_overlap=ev["chunk_overlap"], global_decode=ev["global_decode"],
              global_decode_config=ev["global_decode_config"])
    kw = dict(batch_size=ev["batch_size"], stopwords=ev["stopwords"], **gd)
    model = AutoExtractor.from_pretrained(args.checkpoint, map_location="cpu").to(args.device).eval()
    val = split_records(cfg, "val", args.per_corpus)
    print(f"[per-head] val: {len(val)} records from {len({r['_corpus'] for r in val})} corpora | grid {args.grid} | {gd}", flush=True)
    global_t, _, by_t = sweep_thresholds(model, ExtractorDataset(val, shuffle=False, validate=False), thresholds=args.grid, **kw)
    picks = pick_per_head_thresholds(by_t, global_t)
    for item in args.picks:
        head, _, t = item.partition("=")
        assert head in HEAD_CATEGORIES, head
        picks[head] = float(t)
    report = per_head_report(by_t, picks, global_t) if not args.picks else {}
    if args.picks:
        print(f"[per-head] picks OVERRIDDEN from the command line: {picks}")
    print(f"[per-head] global pick {global_t} (selection {_selection_score(by_t[global_t]):.4f})")
    for head, row in report.items():
        print(f"[per-head] {head:15s} {row['threshold']}  strict micro-F1 {row['f1_at_own']:.4f} vs {row['f1_at_global']:.4f} at global"
              f"{'  (flat)' if row['flat'] else ''}")
    result = {"checkpoint": args.checkpoint, "config": args.config, "grid": args.grid, "per_corpus": args.per_corpus,
              "val_records": len(val), "global_threshold": global_t, "per_head_thresholds": picks, "per_head_report": report,
              "val_by_threshold": {str(t): m for t, m in by_t.items()}}
    if args.test:
        test = split_records(cfg, "test", args.per_corpus)
        ds = ExtractorDataset(test, shuffle=False, validate=False)
        print(f"[per-head] test: {len(test)} records; scoring at global {global_t} and per-head {picks}", flush=True)
        single = {t: compute_metrics(model, ds, threshold=t, report=False, **kw) for t in sorted({global_t, *picks.values()})}
        merged = compute_metrics_per_head(model, ds, picks, global_t, report=False, **kw)
        bad = splice_gate(merged, picks, single)
        print(f"[per-head] splice gate: {'PASS' if not bad else 'FAIL'} ({len(bad)} mismatches)")
        for line in bad:
            print(f"[per-head]   {line}")
        print(f"{'category':16s} {'global@' + str(global_t):>14s} {'per-head':>10s}")
        for cat in [c for cats in HEAD_CATEGORIES.values() for c in cats]:
            g, m = single[global_t].get(f"eval_{cat}_strict_micro_f1"), merged.get(f"eval_{cat}_strict_micro_f1")
            if g is not None or m is not None:
                print(f"{cat:16s} {g if g is not None else float('nan'):14.4f} {m if m is not None else float('nan'):10.4f}")
        result.update({"test_records": len(test), "test_global": single[global_t], "test_per_head": merged,
                       "test_single_by_threshold": {str(t): m for t, m in single.items()}, "splice_gate_mismatches": bad})
    Path(args.out).write_text(json.dumps(result, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"[per-head] wrote {args.out}")


if __name__ == "__main__":
    main()

"""Pick a trained checkpoint's decision threshold under a MENU on val, then score the blind test ONCE.

    uv run tools/train/sweep_menu_threshold.py --config tools/train/config/ab/menudose-treatment.yaml \\
        --checkpoint whr778/gliner2-menudose-treatment --menu app:news55 --out treatment.sweep.json

MENU_SPEC.md: a threshold is an operating point, so it is chosen under the menu production sends, not
the gold menu. The choice is `sweep_thresholds`' own (support-weighted strict micro F1). The test split
is scored once, at the chosen threshold, under the menu AND under the gold menu (the recall cost).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train as T  # noqa: E402

GRID = (0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)   # 0.8+ for an over-firing model


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True, help="local dir or Hub repo id")
    ap.add_argument("--menu", required=True, help="widened:K | corpus_full | app:<name>")
    ap.add_argument("--grid", type=float, nargs="+", default=list(GRID))
    ap.add_argument("--limit", type=int, help="score only the first N records of each split (a trace, not a result)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    from gliner2 import AutoExtractor
    from gliner2.training.eval_metrics import _selection_score, compute_metrics, sweep_thresholds
    from gliner2.training.trainer import ExtractorDataset

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    ev = T._parse_eval_settings(cfg, args.config, corpus_data=[])
    gd = dict(chunk_size=ev["chunk_size"], chunk_overlap=ev["chunk_overlap"], global_decode=ev["global_decode"],
              global_decode_config=ev["global_decode_config"])
    model = AutoExtractor.from_pretrained(args.checkpoint)

    def split(mode, name):
        recs, menus, report = T.menu_split(cfg, args.config, mode, name)
        n = args.limit or len(recs)
        return recs[:n], menus[:n], report

    val, vmenus, vrep = split(args.menu, "val")
    print(f"[sweep] {args.checkpoint} on val under {args.menu}: {len(val)} records ({vrep}), grid {args.grid}")
    thr, _, by_t = sweep_thresholds(model, ExtractorDataset(val, shuffle=False, validate=False), thresholds=args.grid,
                                    batch_size=ev["batch_size"], menus=vmenus, **gd)
    print(f"[sweep] chose threshold {thr} (" + ", ".join(f"{t}: {_selection_score(m):.4f}" for t, m in by_t.items()) + ")")

    result = {"checkpoint": args.checkpoint, "menu": args.menu, "chosen_threshold": thr,
              "val_by_threshold": {str(t): m for t, m in by_t.items()}, "test": {}}
    for mode in (args.menu, "gold"):
        recs, menus, rep = split(mode, "test")
        print(f"[test] {mode} at threshold {thr}: {len(recs)} records")
        m = compute_metrics(model, ExtractorDataset(recs, shuffle=False, validate=False), batch_size=ev["batch_size"],
                            threshold=thr, menus=menus, **gd) or {}
        result["test"][mode] = dict(m, menu_report=rep)
    Path(args.out).write_text(json.dumps(result, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"[sweep] wrote {args.out}")


if __name__ == "__main__":
    main()

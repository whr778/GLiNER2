"""Score several checkpoints on ONE held-out set, each at its own best threshold.

The arms of the real-vs-synthetic comparison cannot be read against each other on
their own test splits: ``train.py`` builds train/val/test from the same ``corpora:``
list, so every arm reports on its own corpus. Real news is harder than generated
passages, and a cross-arm delta would measure that difficulty as much as data
quality. This scores every arm on a fixed set none of them trained on.

**Best-vs-best, not fixed-threshold.** A fine-tune shifts its own operating point --
the synthetic arm's sweep picked 0.7 on its own val while the base model's optimum
was 0.4 -- so comparing at a single threshold penalises whichever arm is furthest
from it. Each leg is swept and reported at its own optimum, which is the method the
existing base-v1 (0.5320) and synthetic (0.4136) numbers already use.

``--stride`` scores every Nth record, for sweeping the grid cheaply before
confirming the chosen thresholds on the full set.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import yaml  # noqa: E402
from gliner2 import AutoExtractor  # noqa: E402
from gliner2.training import sweep_thresholds  # noqa: E402
from gliner2.training.trainer import ExtractorDataset  # noqa: E402
from gliner2.training.metrics import _selection_score  # noqa: E402
from train import _read_records, _split_files  # noqa: E402

GRID = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)


def load_records(config_path: str, split: str, stride: int) -> list:
    """Read the config's corpora for one split, optionally strided."""
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    files = _split_files(cfg["data"].get("corpora") or [], split)
    records = _read_records(files)
    return records[::stride] if stride > 1 else records


def score_leg(name: str, checkpoint: str, records: list, batch_size: int,
              grid: tuple) -> dict:
    """Sweep one checkpoint over the grid and return its best operating point."""
    print(f"\n=== {name} :: {checkpoint} ===", flush=True)
    model = AutoExtractor.from_pretrained(checkpoint)
    dataset = ExtractorDataset(records, shuffle=False, validate=False)
    best_thr, best_metrics, all_metrics = sweep_thresholds(
        model, dataset, thresholds=grid, batch_size=batch_size,
    )
    del model

    by_threshold = {
        str(t): {
            "selection_score": _selection_score(m),
            "entity_strict_micro_f1": m.get("eval_entity_strict_micro_f1"),
            "entity_strict_precision": m.get("eval_entity_strict_precision"),
            "entity_strict_recall": m.get("eval_entity_strict_recall"),
            "entity_fair_micro_f1": m.get("eval_entity_fair_micro_f1"),
            "entity_strict_support": m.get("eval_entity_strict_support"),
        }
        for t, m in all_metrics.items()
    }
    leg = {
        "checkpoint": checkpoint,
        "chosen_threshold": best_thr,
        "selection_score": _selection_score(best_metrics),
        "entity_strict_micro_f1": best_metrics.get("eval_entity_strict_micro_f1"),
        "entity_strict_precision": best_metrics.get("eval_entity_strict_precision"),
        "entity_strict_recall": best_metrics.get("eval_entity_strict_recall"),
        "entity_fair_micro_f1": best_metrics.get("eval_entity_fair_micro_f1"),
        "entity_strict_support": best_metrics.get("eval_entity_strict_support"),
        "by_threshold": by_threshold,
    }
    print(f"[{name}] best threshold {best_thr}  "
          f"entity strict F1 {leg['entity_strict_micro_f1']:.4f}  "
          f"fair {leg['entity_fair_micro_f1']:.4f}", flush=True)
    return leg


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="Config naming the held-out corpora.")
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--leg", action="append", required=True, metavar="NAME=PATH",
                    help="Repeatable. Checkpoint to score, e.g. base=fastino/gliner2-base-v1")
    ap.add_argument("--out", required=True, type=Path, help="Where to write the JSON.")
    ap.add_argument("--batch-size", type=int, default=4,
                    help="4: batch 16 OOMs on pile_ner_def, whose records carry large label sets.")
    ap.add_argument("--stride", type=int, default=1,
                    help="Score every Nth record (cheap grid pass before confirming on all).")
    ap.add_argument("--grid", type=float, nargs="+", default=list(GRID))
    args = ap.parse_args()

    records = load_records(args.config, args.split, args.stride)
    print(f"[preservation] {len(records)} {args.split} records "
          f"(stride {args.stride}) over grid {args.grid}", flush=True)

    legs = {}
    for spec in args.leg:
        name, _, path = spec.partition("=")
        legs[name] = score_leg(name, path, records, args.batch_size, tuple(args.grid))

    payload = {
        "config": args.config,
        "split": args.split,
        "stride": args.stride,
        "n_records": len(records),
        "grid": args.grid,
        "legs": legs,
    }
    args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[preservation] wrote {args.out}")

    print(f"\n{'leg':28s} {'thr':>5s} {'strict F1':>10s} {'fair F1':>9s} {'P':>7s} {'R':>7s}")
    for name, leg in legs.items():
        print(f"{name:28s} {leg['chosen_threshold']:5.1f} "
              f"{leg['entity_strict_micro_f1']:10.4f} {leg['entity_fair_micro_f1']:9.4f} "
              f"{leg['entity_strict_precision']:7.4f} {leg['entity_strict_recall']:7.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

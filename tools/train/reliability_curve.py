"""Is the model UNDER-CONFIDENT, or is a low threshold genuinely trading precision away?

THE QUESTION THE SWEEP CANNOT ANSWER. A threshold sweep reports F1 at each operating point
and nothing about WHY the optimum sits where it does. Two very different situations produce
the same curve:

  UNDER-CONFIDENT   predictions scored 0.3 are right ~70% of the time. The probabilities are
                    compressed toward zero, the threshold is merely undoing that, and the
                    model is better than its own scores say.

  HONEST            predictions scored 0.3 are right ~30% of the time. The probabilities mean
                    what they say, and a low threshold is buying recall with precision --
                    a real trade, not a correction.

The fixes differ. The first is calibration (or a loss reweighting); the second is a genuine
quality ceiling, and lowering the threshold is spending precision you may not want to spend.

WHY WE EXPECT COMPRESSION HERE, which is exactly why it must be measured rather than
assumed: eb17 trains with `abstention_loss_weight` 0.2, `count_loss_weight` 0.2,
`negative_query_ratio` 0.5 and `struct_pos_weight` 4.0. Four terms that all push toward
"emit nothing". Under BCE, 0.5 is the right decode point only if the positive/negative mix
at inference matches training -- and by construction here it does not.

READ THE GAP, NOT THE LEVEL. `empirical - mean_conf` per bin is the calibration error.
Positive means under-confident (right more often than it claims).

    uv run python tools/train/reliability_curve.py \
        --config tools/train/config/base/eb17-best.yaml \
        --checkpoint whr778/gliner2-eb17-best --max-records 800
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

BINS = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0001)


def _bin(c):
    for i in range(len(BINS) - 1):
        if BINS[i] <= c < BINS[i + 1]:
            return i
    return len(BINS) - 2


def main() -> int:
    import yaml

    import train as T
    from gliner2 import AutoExtractor
    from gliner2.training.eval_metrics import _gold_entity_set, _schema_from_gold

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--split", default="val", choices=("val", "test"))
    ap.add_argument("--max-records", type=int, default=800)
    ap.add_argument("--threshold", type=float, default=0.01,
                    help="decode FLOOR -- must be low, or the bins below it are empty by "
                         "construction and the curve only describes what survived the gate")
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    data = cfg.get("data") or {}
    files = T._dedupe_paths(
        T._split_files(data.get("corpora") or [], args.split,
                       set(data.get("train_only") or ()))
        + T._event_split(data.get("event_files") or {}, args.split), args.split)
    fns = T._category_fns(T.load_labels_cfg(cfg, config_path=args.config))

    records = []
    for f in files:
        p = Path(f)
        if not p.is_file():
            continue
        for line in p.open(encoding="utf-8"):
            if line.strip():
                records.append(T.transform_record(json.loads(line), fns))
    import random
    random.Random(42).shuffle(records)
    records = records[: args.max_records]
    print(f"[reliability] {len(records):,} {args.split} records, decode floor "
          f"{args.threshold}", flush=True)

    model = AutoExtractor.from_pretrained(args.checkpoint, map_location=args.device)
    hits = defaultdict(int)
    tot = defaultdict(int)
    conf_sum = defaultdict(float)

    for n, rec in enumerate(records):
        out = rec.get("output") or {}
        text = rec.get("input") or rec.get("text") or ""
        if not text or not (out.get("entities")):
            continue
        gold = _gold_entity_set(out)
        schema = _schema_from_gold(out)
        try:
            pred = model.extract(text, schema, threshold=args.threshold,
                                 include_confidence=True)
        except Exception as exc:
            print(f"[reliability] record {n} raised {type(exc).__name__}: {exc}", flush=True)
            continue
        for label, spans in (pred.get("entities") or {}).items():
            if not isinstance(spans, list):
                continue
            for sp in spans:
                if not isinstance(sp, dict):
                    continue
                surf, c = sp.get("text"), sp.get("confidence")
                if not isinstance(surf, str) or not isinstance(c, (int, float)):
                    continue
                b = _bin(float(c))
                tot[b] += 1
                conf_sum[b] += float(c)
                if (label, surf) in gold:
                    hits[b] += 1
        if n and n % 200 == 0:
            print(f"[reliability] {n} records", flush=True)

    if not sum(tot.values()):
        print("\n[reliability] *** NO PREDICTIONS SCORED. The curve says NOTHING -- check "
              "the decode floor and that the split carries entity gold. ***")
        return 1

    print(f"\n[reliability] {sum(tot.values()):,} predicted entity spans\n")
    print(f"  {'bin':>12} {'n':>7} {'mean_conf':>10} {'empirical':>10} {'gap':>8}")
    rows = []
    for b in sorted(tot):
        n_b = tot[b]
        mc = conf_sum[b] / n_b
        emp = hits[b] / n_b
        rows.append({"bin_lo": BINS[b], "bin_hi": BINS[b + 1], "n": n_b,
                     "mean_conf": round(mc, 4), "empirical": round(emp, 4),
                     "gap": round(emp - mc, 4)})
        print(f"  [{BINS[b]:.1f},{BINS[b+1]:.1f}) {n_b:>7,} {mc:>10.4f} {emp:>10.4f} "
              f"{emp - mc:>+8.4f}")

    wtd = sum(r["n"] * r["gap"] for r in rows) / sum(r["n"] for r in rows)
    ece = sum(r["n"] * abs(r["gap"]) for r in rows) / sum(r["n"] for r in rows)
    print(f"\n[reliability] support-weighted gap {wtd:+.4f}   ECE {ece:.4f}")
    if wtd > 0.05:
        print("[reliability] UNDER-CONFIDENT: right more often than it claims, so a low "
              "threshold is CORRECTING the scores rather than buying recall with precision.")
    elif wtd < -0.05:
        print("[reliability] OVER-confident: it claims more than it delivers.")
    else:
        print("[reliability] WELL CALIBRATED: the probabilities mean what they say, so a low "
              "threshold IS a precision/recall trade, not a correction.")
    if args.out:
        Path(args.out).write_text(json.dumps(rows, indent=2), encoding="utf-8")
        print(f"[reliability] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

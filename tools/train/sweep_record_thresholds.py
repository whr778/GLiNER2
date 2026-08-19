"""Score the boundary record head across its own decode thresholds.

`structure` read exactly 0.0000 on every model we have ever measured. One cause was
decisive and one is a real but secondary cost:

1. **The cause.** `record_metadata` was dropped in `runtime.py` before it reached the
   processor, so no `RecordSpec` was compiled and the head decoded NOTHING at any
   threshold. Fixed. This alone accounts for every zero.
2. **A miscalibration on top of it.** `record_anchor_threshold` defaults to 0.5, which is
   well above where this head is confident. Do NOT state that 0.5 decodes nothing: with
   (1) fixed, 0.5 still scores 0.0052 / 0.0654 / 0.0760 strict F1 at 40k / 100k / 137k.
   Only the 10k model reaches 0.0 there. The default costs roughly a third of the
   attainable F1 (137k: 0.076 at 0.5 vs 0.112 at 0.10) -- worth sweeping, not the reason
   the metric read zero.

`threshold_sweep` in the trainer sweeps the general decision threshold; it does not touch
`record_anchor_threshold` / `record_field_threshold`, so nothing has ever calibrated this
head.

**Only structure-bearing records are scored, and that is exact rather than an
approximation.** `_schema_from_gold` omits the `structures` key for a record with no gold
structures, so the model is never asked for one there and cannot produce a structure
false positive. Filtering them out changes no count and turns a 15,456-record pass into
856.

**The decode is NONDETERMINISTIC by about +/-1 false positive.** Two identical
single-threshold runs over the same 148 records returned FP=8 and FP=7; TP and FN were
stable. So do not read a 1-FP difference between runs as a real change, and do not chase
it when comparing this tool against a slower implementation. Verified before trusting the
one-pass optimisation below -- the optimisation reproduces the slow path within exactly
that noise band.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parent))

import yaml  # noqa: E402
from gliner2 import AutoExtractor  # noqa: E402
from gliner2.training.eval_metrics import (  # noqa: E402
    _gold_structure_set,
    _pred_structure_set,
    _schema_from_gold,
)

GRID = (0.5, 0.3, 0.2, 0.15, 0.10, 0.07, 0.05, 0.03)


def load_structure_records(config: Path) -> List[dict]:
    """Every test record in the config that carries gold structures."""
    cfg = yaml.safe_load(config.read_text(encoding="utf-8"))
    paths = [f"{co}.test.jsonl" for co in (cfg["data"].get("corpora") or [])]
    paths += [v["test"] for v in (cfg["data"].get("event_files") or {}).values() if "test" in v]
    out = []
    for p in paths:
        f = Path(p)
        if not f.exists() or f.stat().st_size == 0:
            continue
        for line in f.open(encoding="utf-8"):
            rec = json.loads(line)
            if (rec.get("output") or {}).get("json_structures"):
                out.append(rec)
    return out


def score_all_thresholds(model, records: List[dict], batch_size: int,
                         grid: List[float]) -> dict:
    """Score EVERY threshold from a single encoder pass per record.

    The record group logits do not depend on the decode thresholds -- only
    `decode_group`'s filtering does. Re-running `batch_extract` per threshold therefore
    recomputed identical logits N times; measured at 0.66 records/sec that turned a
    45-minute job into a 6-hour one.

    So hook `_decode_records`, which receives everything the decode needs (groups,
    spans, offsets, text) and reads its cutoffs from `self.boundary_settings`. For each
    record it is invoked once per threshold, cheaply, off the forward pass that already
    ran. Only one record's tensors are live at a time -- caching all of them would hold
    the whole test set's candidate states in memory.
    """
    texts, golds, schemas = [], [], []
    for rec in records:
        schema = _schema_from_gold(rec["output"])
        if not schema.get("structures"):
            continue
        texts.append(rec.get("input") or rec.get("text") or "")
        golds.append(rec["output"])
        schemas.append(schema)
    if not texts:
        return {t: (0, 0, 0) for t in grid}

    engine = type(model)
    original = engine._decode_records
    base_settings = model.boundary_settings
    per_threshold: dict = {t: [] for t in grid}

    def multi_decode(self, *a, **kw):
        """Decode this sample at every threshold; return the base one to the caller."""
        result_at = {}
        for thr in grid:
            self.boundary_settings = dataclasses.replace(
                base_settings,
                record_anchor_threshold=thr,
                record_field_threshold=thr,
                record_anchor_proposal_threshold=min(
                    thr, base_settings.record_anchor_proposal_threshold),
            )
            result_at[thr] = original(self, *a, **kw)
        self.boundary_settings = base_settings
        for thr in grid:
            per_threshold[thr].append(result_at[thr])
        return result_at[grid[0]]

    engine._decode_records = multi_decode
    try:
        for i in range(0, len(texts), batch_size):
            model.batch_extract(texts[i:i + batch_size], schemas[i:i + batch_size],
                                batch_size=batch_size, threshold=0.5)
    finally:
        engine._decode_records = original
        model.boundary_settings = base_settings

    out = {}
    for thr in grid:
        preds = per_threshold[thr]
        if len(preds) != len(golds):
            raise RuntimeError(
                f"threshold {thr}: {len(preds)} decodes for {len(golds)} records -- "
                "the hook missed samples, do not trust these numbers")
        tp = fp = fn = 0
        for gold, pred in zip(golds, preds):
            g, p = _gold_structure_set(gold), _pred_structure_set(pred)
            tp += len(g & p); fp += len(p - g); fn += len(g - p)
        out[thr] = (tp, fp, fn)
    return out


def prf(tp: int, fp: int, fn: int) -> tuple:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--grid", type=float, nargs="+", default=list(GRID))
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="cap records, for a smoke run")
    ap.add_argument("--device", default=None, help="cpu | mps | cuda; default auto")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    records = load_structure_records(args.config)
    if args.limit:
        records = records[:args.limit]
    print(f"[record-sweep] {len(records)} structure-bearing test records", flush=True)

    model = AutoExtractor.from_pretrained(
        str(args.checkpoint),
        attn_implementation="eager",          # CPU/MPS have no FA2
        **({"map_location": args.device} if args.device else {}),
    )
    model.eval()
    base = model.boundary_settings

    # one encoder pass per record, every threshold decoded off it
    counts = score_all_thresholds(model, records, args.batch_size, list(args.grid))

    rows = []
    print(f"{'thr':>6s} {'P':>8s} {'R':>8s} {'F1':>8s} {'TP':>6s} {'FP':>6s} {'FN':>6s}")
    for thr in args.grid:
        tp, fp, fn = counts[thr]
        p, r, f = prf(tp, fp, fn)
        rows.append({"threshold": thr, "precision": p, "recall": r, "f1": f,
                     "tp": tp, "fp": fp, "fn": fn})
        print(f"{thr:6.2f} {p:8.4f} {r:8.4f} {f:8.4f} {tp:6d} {fp:6d} {fn:6d}", flush=True)

    best = max(rows, key=lambda x: x["f1"])
    print(f"\nBEST threshold {best['threshold']} -> structure strict F1 {best['f1']:.4f}")
    if args.out:
        args.out.write_text(json.dumps(
            {"config": str(args.config), "checkpoint": str(args.checkpoint),
             "n_records": len(records), "by_threshold": rows, "best": best},
            indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[record-sweep] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

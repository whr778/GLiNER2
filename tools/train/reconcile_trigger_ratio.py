"""Why does the instance probe say 0.96:1 and the blind test say 3.78:1?

THE GAP. `probe_object_score_separation.py` labels DECODED INSTANCES good/spurious by
anchor span and reports 0.96 spurious per good on val event corpora. The blind test's
Ortmann accounting reports 8,658 new FP per 2,293 recovered FN = 3.78:1 on the full test
split. Both describe "what lowering the record gate admits", and they disagree 4x.

THREE CANDIDATES, and this separates them by running the EVAL's own accounting on the
PROBE's own data:

  population  -- val event corpora here vs the full 26,724-record test split there
  granularity -- instances here vs trigger SPANS after formatting/dedup there
  strictness  -- anchor-span match here vs event TYPE + trigger TEXT there

If the eval run below lands near 0.96 the gap is POPULATION. If it lands near 3.78 the
gap is ACCOUNTING (granularity and/or strictness) and the probe's ratio is not comparable
to the blind test's at all.

    uv run python tools/train/reconcile_trigger_ratio.py \
        --config tools/train/config/base/eb17-best.yaml \
        --checkpoint whr778/gliner2-eb17-best --max-records 600
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

CATS = ("COR", "FN", "FP", "BES", "BEO", "BEL", "LBE", "LE")


def main() -> int:
    import yaml

    import train as T
    from gliner2.training.eval_metrics import evaluate_checkpoint

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--max-records", type=int, default=600)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--span-threshold", type=float, default=0.3)
    ap.add_argument("--anchor", type=float, default=0.1)
    ap.add_argument("--map-location", default="cpu",
                    help="cpu by default: MPS dies with 'Invalid buffer size: 151.56 GiB' "
                         "on the long val docs, and resolve_device picks MPS on this Mac.")
    ap.add_argument("--chunk-size", type=int, default=512,
                    help="whole-document decode blows up attention on the long val docs. "
                         "BOTH arms get the same chunking, so the RATIO -- which is what "
                         "this tool reports -- is unaffected by the choice.")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    data, tc = cfg.get("data") or {}, cfg.get("training") or {}
    files = T._dedupe_paths(T._event_split(data.get("event_files") or {}, args.split),
                            args.split)
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
    random.Random(int(tc.get("seed", 42))).shuffle(records)
    records = records[: args.max_records]
    print(f"[rec] {len(records):,} {args.split} event records from {len(files)} file(s) "
          f"-- THE PROBE'S OWN POPULATION", flush=True)

    def score(overrides, label):
        print(f"[rec] scoring {label} ...", flush=True)
        m = evaluate_checkpoint(args.checkpoint, records, batch_size=args.batch_size,
                                threshold=args.span_threshold,
                                map_location=args.map_location,
                                chunk_size=args.chunk_size,
                                boundary_overrides=overrides) or {}
        return {c: float(m.get(f"eval_event_trigger_error_{c}", 0.0)) for c in CATS}

    control = score({}, "control (record gate at the checkpoint default)")
    treat = score({"record_anchor_threshold": args.anchor,
                   "record_anchor_threshold_wins": True,
                   "record_anchor_proposal_threshold": args.anchor},
                  f"anchor {args.anchor}")

    print(f"\n  {'category':10} {'control':>10} {'anchor':>10} {'delta':>10}")
    d = {}
    for c in CATS:
        d[c] = treat[c] - control[c]
        print(f"  {c:10} {control[c]:>10.0f} {treat[c]:>10.0f} {d[c]:>+10.0f}")

    recovered = -d["FN"]
    partial = sum(d[c] for c in ("BES", "BEO", "BEL", "LBE", "LE"))
    print()
    if recovered <= 0 or d["FP"] <= 0:
        print("[rec] *** the gate did not move triggers on this sample -- "
              "nothing to reconcile, and this is UNMEASURED rather than agreement. ***")
        return 1
    ratio = d["FP"] / recovered
    print(f"  FN recovered                    : {recovered:>8.0f}")
    print(f"  of which exactly correct        : {d['COR']:>8.0f}")
    print(f"  of which boundary/label errors  : {partial:>8.0f}")
    print(f"  reconciles? {d['COR'] + partial:.0f} vs {recovered:.0f}")
    print(f"  new FP                          : {d['FP']:>8.0f}")
    print(f"\n  EVAL-ACCOUNTING ratio on the PROBE's data = {ratio:.2f} : 1")
    print(f"  probe instance ratio                      = 0.96 : 1")
    print(f"  blind-test eval ratio (full test split)   = 3.78 : 1")
    print()
    if abs(ratio - 0.96) < abs(ratio - 3.78):
        print("[rec] VERDICT: close to the PROBE -> the gap is POPULATION. The probe's AUC")
        print("[rec] is measured on a population that resembles this one, so it transfers.")
    else:
        print("[rec] VERDICT: close to the BLIND TEST -> the gap is ACCOUNTING, not")
        print("[rec] population. The probe counts instances, the eval counts trigger spans")
        print("[rec] under a stricter match, and the two ratios are NOT comparable.")
    if args.out:
        Path(args.out).write_text(json.dumps(
            {"control": control, "treatment": treat, "delta": d,
             "eval_ratio_on_probe_data": ratio}, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

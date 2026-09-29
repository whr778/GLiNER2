"""Sweep the RECORD DECODE knobs on VALIDATION. Records decode at their own gates.

WHY. Record decode selects instances with
``sigmoid(object_logits) >= record_anchor_threshold``, default **0.5**, and the ordinary
``threshold_sweep`` never touches it -- that one calibrates the SPAN decision threshold. So
every event number this programme has quoted was read at 0.5, while the 137k STRUCTURE
reference needed **0.1** to reach its best (0.1119). Events have never been swept on an
event-records base. This is TODO/O4 plus the rest of that tier.

IT IS THE LEVER S13 IMPLICATES. The anchor supervision gate measured 0 of 362 event records
dropped -- the gold trigger is essentially ALWAYS among the model's candidates -- so recall
is lost when instances are SELECTED, not when they are proposed. These are the selection
gates.

ONE AXIS AT A TIME. Each axis is swept with every other held at the checkpoint's own value.
A joint sweep would confound them and cannot say which knob moved anything.

TWO TRAPS, BOTH ALREADY PAID FOR HERE:

  `boundary_settings` IS A FROZEN DATACLASS, and `from_pretrained` builds it from the
  checkpoint's config inside __init__, so a plain setattr raises or lands too late and is
  silently dropped. `evaluate_checkpoint(boundary_overrides=...)` rebuilds the settings AND
  syncs the head's own reference -- two places, and updating one is a no-op. This script
  goes through that path rather than reinventing it.

  A FLAT SWEEP IS INDISTINGUISHABLE FROM "the threshold does not matter" and has been
  mistaken for it. If nothing moves, this refuses to report rather than printing a tidy
  table of identical rows.

PICK HERE, SCORE THE BLIND TEST ONCE, ELSEWHERE. This reads VAL by construction and has no
--split. Sweeping on test and quoting the best is fitting the test set; a "+0.049 win" on
this programme was already lost that way.

    uv run python tools/train/sweep_record_anchor_threshold.py \
        --config tools/train/config/base/eb17-best.yaml \
        --checkpoint out/eb17-best/best --out sweep.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

# Each axis: (name, grid). `record_anchor_proposal_threshold` must stay <= the anchor
# threshold -- configuration.py validates it and raises otherwise -- so its grid stops at
# the anchor default of 0.5.
AXES = {
    # Extended DOWNWARD 2026-09-29 on evidence: the first run scored 0.5/0.4/0.3/0.2
    # and every head was still RISING at 0.2 (event_argument strict 2.2x, event_type
    # +59% across that range). The optimum is below the old grid, so the passes belong
    # at the bottom end. 0.4 dropped -- it sat on a smooth stretch and bought nothing.
    "record_anchor_threshold": (0.5, 0.3, 0.2, 0.1, 0.05, 0.02, 0.01),
    "record_field_threshold": (0.5, 0.3, 0.2, 0.1, 0.05),
    "record_temperature": (1.0, 1.5, 2.0, 3.0),
}
# `record_anchor_proposal_threshold` is DELIBERATELY ABSENT. It is declared in
# configuration.py (default 0.2, "lower rescue threshold"), validated against
# record_anchor_threshold, and READ BY NOTHING -- no decoder, no loss. Contrast
# `relation_argument_proposal_threshold`, which is read at model.py:1434. Sweeping it would
# be guaranteed-flat for a reason that has nothing to do with the model.
WATCH = ("event_argument", "event_trigger", "event_type", "structure", "entity")


def _row(metrics, label, value):
    row = {"axis": label, "value": value}
    for head in WATCH:
        for regime in ("strict", "relaxed"):
            k = f"eval_{head}_{regime}_micro_f1"
            if k in metrics:
                row[f"{head}_{regime}"] = round(float(metrics[k]), 4)
    return row


def main() -> int:
    import yaml

    import train as T
    from gliner2.training.eval_metrics import evaluate_checkpoint

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--axes", default=",".join(AXES),
                    help="which axes to sweep, comma separated")
    ap.add_argument("--span-thresholds", default="",
                    help="also sweep the SPAN threshold over these, e.g. 0.3,0.1,0.05,0.02")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-records", type=int, default=0,
                    help="subsample val for SHAPE; the shipped pick must use the full split")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    data, ev = cfg.get("data") or {}, cfg.get("eval") or {}
    span_thr = float(ev.get("threshold", 0.5))

    files = T._dedupe_paths(
        T._split_files(data.get("corpora") or [], "val", set(data.get("train_only") or ()))
        + T._event_split(data.get("event_files") or {}, "val"), "val")
    fns = T._category_fns(T.load_labels_cfg(cfg, config_path=args.config))

    records = []
    for f in files:
        p = Path(f)
        if not p.is_file():
            continue
        for line in p.open(encoding="utf-8"):
            if line.strip():
                records.append(T.transform_record(json.loads(line), fns))
    if args.max_records and len(records) > args.max_records:
        import random
        random.Random(42).shuffle(records)
        records = records[: args.max_records]
        print(f"[sweep] SUBSAMPLED to {len(records):,} val records -- for SHAPE only. "
              f"Re-run on the full split before shipping a threshold.")
    print(f"[sweep] {len(records):,} VAL records from {len(files)} files\n", flush=True)

    print("[sweep] boundary_head override record_anchor_threshold_wins=True -- without it "
          "the record gate IS the span gate and the sweep is flat")

    rows = []
    for axis in [a.strip() for a in args.axes.split(",") if a.strip()]:
        if axis not in AXES:
            print(f"[sweep] unknown axis {axis!r}, skipping")
            continue
        print(f"[sweep] === {axis} (span threshold held at {span_thr}) ===", flush=True)
        for value in AXES[axis]:
            # `record_anchor_threshold_wins` is what makes a record-threshold sweep
            # mean anything: without it the record gate IS the span gate and every row
            # is identical. Set for EVERY axis, because the monkeypatch this replaced
            # forced `threshold=None` globally -- keeping it global keeps these rows
            # comparable with the 2026-09-29 run.
            overrides = {axis: value, "record_anchor_threshold_wins": True}
            # THE DEAD SETTING STILL GATES THE LIVE ONE. `record_anchor_proposal_threshold`
            # is read by NOTHING (no decoder, no loss -- verified 2026-09-29), which is why
            # it was dropped from AXES. But `validate_boundary_head` still enforces
            # proposal <= anchor, so sweeping the anchor below its 0.2 default raises
            #   ValueError: record_anchor_proposal_threshold (0.2) must be <= (0.1)
            # and kills the run mid-sweep. That is exactly what happened on the first A100
            # attempt: four rows scored, then dead at 0.1. The OLD tool clamped this; I
            # removed the axis and lost the clamp with it.
            #
            # Setting it EQUAL to the anchor is safe precisely because it is dead code --
            # it satisfies the validator and can change nothing else.
            if axis == "record_anchor_threshold":
                overrides["record_anchor_proposal_threshold"] = value
            m = evaluate_checkpoint(
                args.checkpoint, records, batch_size=args.batch_size, threshold=span_thr,
                boundary_overrides=overrides,
            ) or {}
            row = _row(m, axis, value)
            rows.append(row)
            print(f"[sweep]   {value:<6} " + "  ".join(
                f"{k}={v}" for k, v in row.items() if k not in ("axis", "value")), flush=True)

    for st in [s.strip() for s in args.span_thresholds.split(",") if s.strip()]:
        print(f"[sweep] === span threshold {st} (record knobs at checkpoint defaults) ===",
              flush=True)
        m = evaluate_checkpoint(args.checkpoint, records, batch_size=args.batch_size,
                                threshold=float(st)) or {}
        row = _row(m, "span_threshold", float(st))
        rows.append(row)
        print(f"[sweep]   {st:<6} " + "  ".join(
            f"{k}={v}" for k, v in row.items() if k not in ("axis", "value")), flush=True)

    if not rows:
        print("[sweep] nothing swept")
        return 1

    # THE FLAT GATE, per axis. Identical rows within an axis mean the override never
    # reached the decoder, which reads exactly like "this knob does not matter".
    bad = []
    for axis in sorted({r["axis"] for r in rows}):
        sub = [r for r in rows if r["axis"] == axis]
        if len(sub) < 2:
            continue
        moved = {k for k in sub[0] if k not in ("axis", "value")
                 and len({r.get(k) for r in sub}) > 1}
        print(f"\n[sweep] {axis}: {len(moved)} metric(s) moved" +
              (f" -> {sorted(moved)}" if moved else ""))
        if not moved:
            bad.append(axis)
    if bad:
        print(f"\n[sweep] *** FLAT on {bad}: every value produced identical metrics. Either "
              f"the override did not reach the decoder, or no record groups are present. "
              f"Those axes measured NOTHING -- do not read them as 'no effect'. ***")

    key = "event_argument_strict"
    scored = [r for r in rows if key in r]
    if scored:
        best = max(scored, key=lambda r: r[key])
        base = next((r[key] for r in scored if r["value"] in (0.5, 1.0)), None)
        print(f"\n[sweep] best {key} = {best[key]} at {best['axis']}={best['value']}"
              + (f"  (default gives {base})" if base is not None else ""))
        print("[sweep] PICKED ON VALIDATION. Score the blind test ONCE at this point.")
    if args.out:
        Path(args.out).write_text(json.dumps(rows, indent=2), encoding="utf-8")
        print(f"[sweep] wrote {args.out}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())

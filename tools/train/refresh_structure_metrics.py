"""Correct the `structure` metrics in a finished run and regenerate its model card.

Every model we trained reported `structure` as exactly 0.0000. That was a decode bug,
not the head: `record_metadata` was dropped in `runtime.py` before it reached the
processor, so no `RecordSpec` was compiled and the record head emitted nothing at any
threshold. Fixed, but the stored `test_metrics.json` and the pushed model cards still
carry the wrong zeros.

Nothing is retrained. The corrected number comes from
`sweep_record_thresholds.py`, which re-scores the SAME checkpoint across the record
head's own decode thresholds -- a sweep no existing tool performed, since
`threshold_sweep` moves the general decision threshold and never touches
`record_anchor_threshold` / `record_field_threshold`.

The card is regenerated through the real `build_model_card`, not patched textually, so a
refreshed card cannot drift from the shape training produces. The original zero is kept
in the JSON under `structure_*_superseded` -- the point is to correct the record, not to
erase that it was wrong.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import yaml  # noqa: E402
from model_card import build_model_card, canonical_dataset_key  # noqa: E402
from train import _dataset_counts as dataset_counts  # noqa: E402


def _load(path: Path):
    """The saved artifact, or None. Absence is normal for older runs."""
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def patch_metrics(metrics: dict, best: dict, at_default: dict | None = None) -> dict:
    """Replace the STRICT structure metrics with the swept values.

    Only strict. `sweep_record_thresholds` scores exact `(name, field, value)` triples
    via `_gold_structure_set` / `_pred_structure_set`; it computes no relaxed variant.
    Writing the strict number into the relaxed keys would be inventing a measurement,
    so the relaxed keys are left at their (unmeasured) values and flagged.

    Key names follow the file's existing convention -- `..._micro_precision`, not
    `..._precision`. Getting that wrong leaves the real keys at 0.0 and the model card
    silently drops the row, which is how this was caught.
    """
    out = dict(metrics)
    for name, val in (("micro_f1", best["f1"]),
                      ("micro_precision", best["precision"]),
                      ("micro_recall", best["recall"])):
        key = f"eval_structure_strict_{name}"
        sup = f"{key}_superseded"
        # IDEMPOTENT: never overwrite an existing _superseded. Re-running the tool on an
        # already-patched file would otherwise archive the patched value as if it were
        # the original, destroying the only record of the zero this exists to document.
        if key in out and sup not in out:
            out[sup] = out[key]
        out[key] = float(val)
    out["eval_structure_strict_support"] = int(best["tp"] + best["fn"])
    out["structure_record_threshold"] = float(best["threshold"])

    # What the 0.5 default actually costs THIS model. Do not generalise one model's
    # ratio onto the others: the penalty runs from 100% at 10k (0.0 there) to 32% at
    # 137k, so a single blanket sentence is wrong on most cards.
    if at_default is not None:
        out["structure_default_threshold"] = float(at_default["threshold"])
        out["structure_f1_at_default_threshold"] = float(at_default["f1"])

    # The sweep computes micro only. Strict MACRO is left over from the broken decode and
    # is still exactly 0.0; the card does not render it, but leaving an unflagged zero in
    # the JSON invites the same misreading the micro zeros caused.
    for name in ("macro_f1", "macro_precision", "macro_recall"):
        key = f"eval_structure_strict_{name}"
        if key in out:
            out[f"{key}_unmeasured"] = out.pop(key)

    # DELETE the relaxed structure keys rather than leave them at their stale zeros.
    # The sweep measures strict exact-match only, and the model card renders
    # "strict -> relaxed", so leaving them produced `0.039 -> 0.000` -- reading as if
    # relaxed were WORSE than strict, which is impossible by construction. The renderer
    # falls back to showing strict alone when the relaxed value is absent, which is the
    # honest presentation of what was actually measured.
    for name in ("micro_f1", "micro_precision", "micro_recall",
                 "macro_f1", "macro_precision", "macro_recall", "support"):
        key = f"eval_structure_relaxed_{name}"
        if key in out:
            out[f"{key}_unmeasured"] = out.pop(key)
    out["structure_relaxed_note"] = (
        "eval_structure_relaxed_* was NOT re-measured -- the record-threshold sweep "
        "scores strict exact-match only -- so those keys are moved to *_unmeasured "
        "rather than shown. Re-run the sweep with a relaxed matcher to restore them."
    )
    out["structure_correction_note"] = (
        "structure re-measured after the record_metadata decode fix (runtime.py dropped "
        "the key before the processor, so no RecordSpec compiled and the record head "
        "decoded nothing at ANY threshold), at the head's own swept threshold. The "
        "*_superseded values are the original zeros: they reflected a decode that never "
        "ran, not a head that could not perform."
    )
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--run-dir", type=Path, required=True, help="out/<run>, containing best/")
    ap.add_argument("--sweep", type=Path, required=True, help="sweep_record_thresholds JSON")
    ap.add_argument("--date", help=(
        "Date to stamp on the card. The card labels this 'Trained on', so defaulting to "
        "today silently backdates nothing but FORWARD-dates a model trained earlier -- "
        "the 10k/40k cards went out reading 2026-08-19 for an 2026-08-18 run. Pass the "
        "original date when refreshing an already-published card."))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    best_dir = args.run_dir / "best"
    sweep = json.loads(args.sweep.read_text(encoding="utf-8"))
    best = sweep["best"]

    tm_path = args.run_dir / "test_metrics.json"
    metrics = json.loads(tm_path.read_text(encoding="utf-8")) if tm_path.is_file() else {}
    old = metrics.get("eval_structure_strict_micro_f1")
    at_default = next((r for r in sweep["by_threshold"] if r["threshold"] == 0.5), None)
    patched = patch_metrics(metrics, best, at_default)

    print(f"{args.run_dir.name}:")
    print(f"  structure strict F1 {old} -> {best['f1']:.4f} at record threshold {best['threshold']}")

    corpora = cfg["data"].get("corpora") or []
    event_files = cfg["data"].get("event_files") or {}
    dataset_keys = [canonical_dataset_key(Path(c).name) for c in corpora] + [
        canonical_dataset_key(n) for n, by in event_files.items()
        if any(p and Path(p).is_file() for p in by.values())]

    # Feed the card EVERY artifact training fed it. Rebuilding from the YAML alone is
    # lossy in ways that are easy to miss and were caught only by diffing against the
    # pushed README: per-split dataset counts, duration and throughput all disappear,
    # `weight_decay`/`seed` read None (they live in training_config.json, not the YAML),
    # and the threshold reverts from the CALIBRATED 0.1 to the YAML's 0.5. A refreshed
    # card must differ from the pushed one in the structure row and nothing else.
    results = _load(args.run_dir / "train_results.json")
    training_cfg = _load(args.run_dir / "training_config.json") or {}
    eval_metrics = _load(best_dir / "eval_metrics.json")
    sweep_tbl = _load(best_dir / "threshold_sweep.json") or {}
    threshold = sweep_tbl.get("chosen_threshold", (cfg.get("eval") or {}).get("threshold"))

    class _Cfg:
        """Attribute view over the SAVED training config, falling back to the YAML."""
        def __init__(self):
            self.experiment_name = (training_cfg.get("experiment_name")
                                    or (cfg.get("training") or {}).get("experiment_name")
                                    or args.run_dir.name)
            self.output_dir = training_cfg.get("output_dir", str(args.run_dir))

        def __getattr__(self, k):
            if k in training_cfg:
                return training_cfg[k]
            return (cfg.get("training") or {}).get(k)

    model_cfg = cfg.get("model") or {}
    card = build_model_card(
        model_name=_Cfg().experiment_name,
        base_model=model_cfg.get("encoder") or model_cfg.get("pretrained"),
        cfg=cfg, config=_Cfg(), dataset_keys=dataset_keys, results=results,
        eval_metrics=eval_metrics, test_metrics=patched,
        generated_at=args.date or datetime.now().strftime("%Y-%m-%d"),
        dataset_counts=dataset_counts(corpora, event_files),
        threshold=threshold,
        threshold_calibrated="chosen_threshold" in sweep_tbl,
    )

    if args.dry_run:
        print("  [dry-run] card regenerated, nothing written")
        return 0

    tm_path.write_text(json.dumps(patched, indent=2, ensure_ascii=False), encoding="utf-8")
    (best_dir / "test_metrics.json").write_text(
        json.dumps(patched, indent=2, ensure_ascii=False), encoding="utf-8")
    (best_dir / "MODEL_CARD.md").write_text(card, encoding="utf-8")
    print(f"  wrote {tm_path}, {best_dir}/test_metrics.json, {best_dir}/MODEL_CARD.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

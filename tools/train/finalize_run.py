"""Re-run the post-training tail (blind test + model card) on a finished run.

Training can complete and save a perfectly good checkpoint and still leave the
run unfinished, because the blind test and the model card come AFTER the last
save. When that tail crashes you lose the card and the test metrics -- not the
model. Retraining to recover them wastes hours of GPU for work already done.

This replays only the tail against an existing ``best/`` directory, using
train.py's own helpers so the output is identical to what a clean run would
have produced.

    uv run python tools/train/finalize_run.py \
        --config tools/train/config/joint-boundary-mmbert-10k.yaml

Reads the calibrated threshold from ``best/threshold_sweep.json`` when present,
so the card reports the same cutoff the run chose rather than re-sweeping.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import yaml  # noqa: E402

from train import (  # noqa: E402
    TrainingConfig,
    _event_split,
    _read_records,
    _run_blind_test,
    _split_files,
    _write_model_card,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="The run's training YAML.")
    parser.add_argument("--skip-blind-test", action="store_true",
                        help="Write the card only (use when the test split is unavailable).")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    config = TrainingConfig(**cfg["training"])
    best = Path(config.output_dir) / "best"
    if not best.is_dir():
        raise SystemExit(f"no best/ checkpoint at {best}; nothing to finalize")

    data = cfg.get("data") or {}
    corpora = data.get("corpora") or []
    event_files = data.get("event_files") or {}
    ev = cfg.get("eval") or {}

    # Prefer the threshold the run actually calibrated to; fall back to the config.
    threshold = ev.get("threshold", 0.5)
    calibrated = False
    sweep = best / "threshold_sweep.json"
    if sweep.is_file():
        threshold = json.loads(sweep.read_text(encoding="utf-8"))["chosen_threshold"]
        calibrated = True
        print(f"[finalize] Using calibrated threshold={threshold} from {sweep}")

    results_path = Path(config.output_dir) / "train_results.json"
    results = json.loads(results_path.read_text(encoding="utf-8")) if results_path.is_file() else None

    # Reuse metrics already on disk when skipping the blind test, so regenerating a card
    # (e.g. after a card-generator fix) costs no GPU and loses no numbers.
    test_metrics = None
    saved = best / "test_metrics.json"
    if args.skip_blind_test and saved.is_file():
        test_metrics = json.loads(saved.read_text(encoding="utf-8"))
        print(f"[finalize] Reusing {saved} ({len(test_metrics)} keys); blind test skipped")
    if not args.skip_blind_test:
        test_files = _split_files(corpora, "test") + _event_split(event_files, "test")
        test_data = _read_records(test_files) if test_files else []
        if test_data:
            test_metrics = _run_blind_test(
                best, test_data, ev.get("batch_size", 8), threshold,
                ev.get("eval_by_language", False),
                dict(chunk_size=None, chunk_overlap=128,
                     global_decode=False, global_decode_config=None),
            )
            if test_metrics:
                for target in (Path(config.output_dir) / "test_metrics.json",
                               best / "test_metrics.json"):
                    target.write_text(json.dumps(test_metrics, indent=2, ensure_ascii=False),
                                      encoding="utf-8")
                print(f"[finalize] Wrote blind-test metrics ({len(test_metrics)} keys)")
        else:
            print("[finalize] No test split; skipping blind test.")

    _write_model_card(
        cfg, config, corpora, event_files, results, test_metrics, best,
        threshold=threshold, threshold_calibrated=calibrated,
    )
    card = best / "MODEL_CARD.md"
    print(f"[finalize] {'Wrote' if card.is_file() else 'FAILED to write'} {card}")


if __name__ == "__main__":
    main()

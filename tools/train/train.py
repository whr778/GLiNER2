"""Train a fresh GLiNER2 from a YAML config.

Run::

    uv run python tools/train/train.py tools/train/config/mmbert-small-focal.yaml

The config has four sections:

* ``model``    - encoder id + struct-loss settings, passed to
  ``GLiNER2.from_encoder``. Keys other than ``encoder`` are forwarded as-is,
  so a config may omit settings it doesn't use (e.g. focal params under ``bce``).
* ``training`` - fields forwarded verbatim to :class:`TrainingConfig`.
* ``eval``     - ``batch_size`` / ``threshold`` for the metrics hook and the
  blind test pass.
* ``data``     - ``corpora`` base paths (``<name>.{train,val,test}.jsonl``) and
  an ``event_files`` map of ``{name: {train,val,test}}``. Event splits are
  included only if the file exists on disk, so a config runs with any subset
  present. See ``tools/train/config/`` for examples.

Results land in ``<output_dir>/train_results.json`` and the blind-test metrics
in ``<output_dir>/test_metrics.json``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from pprint import pprint
from typing import Dict, List

import yaml

from gliner2 import GLiNER2
from gliner2.training import estimate_eta, evaluate_checkpoint, make_compute_metrics
from gliner2.training.trainer import GLiNER2Trainer, TrainingConfig


def _split_files(corpora: List[str], suffix: str) -> List[str]:
    return [f"{c}.{suffix}.jsonl" for c in corpora]


def _event_split(event_files: Dict[str, Dict[str, str]], suffix: str) -> List[str]:
    paths: List[str] = []
    for by_split in event_files.values():
        p = by_split.get(suffix)
        if p and Path(p).is_file():
            paths.append(p)
    return paths


def main(config_path: str) -> None:
    cfg = yaml.safe_load(Path(config_path).read_text())

    model_cfg = dict(cfg["model"])
    encoder = model_cfg.pop("encoder")
    model = GLiNER2.from_encoder(encoder, **model_cfg)

    config = TrainingConfig(**cfg["training"])

    data = cfg.get("data") or {}
    corpora = data.get("corpora") or []
    event_files = data.get("event_files") or {}
    train_data = _split_files(corpora, "train") + _event_split(event_files, "train")
    eval_data = _split_files(corpora, "val") + _event_split(event_files, "val")
    test_data = _split_files(corpora, "test") + _event_split(event_files, "test")

    eval_cfg = cfg.get("eval") or {}
    eval_bs = eval_cfg.get("batch_size", 8)
    eval_thr = eval_cfg.get("threshold", 0.5)

    trainer = GLiNER2Trainer(
        model, config,
        eval_data=eval_data,
        compute_metrics=make_compute_metrics(batch_size=eval_bs, threshold=eval_thr),
    )
    estimate_eta(model, train_data, config)
    results = trainer.train(train_data=train_data)
    pprint(results)

    results_path = Path(config.output_dir) / "train_results.json"
    results_path.write_text(
        json.dumps(results, indent=2, default=lambda o: o.to_dict() if hasattr(o, "to_dict") else str(o))
    )
    print(f"[train] Wrote results to {results_path}")

    best = Path(config.output_dir) / "best"
    if not best.is_dir():
        print(f"\n[blind test] No 'best' checkpoint at {best}; skipping.")
        return

    print(f"\n[blind test] Loading {best} and scoring against {len(test_data)} held-out splits...")
    test_metrics = evaluate_checkpoint(best, test_data, batch_size=eval_bs, threshold=eval_thr)
    if not test_metrics:
        print("[blind test] No metrics produced (empty test set?).")
        return

    metrics_path = Path(config.output_dir) / "test_metrics.json"
    metrics_path.write_text(json.dumps(test_metrics, indent=2))
    print(f"\n[blind test] Wrote metrics to {metrics_path}")

    print("\n===== Blind test metrics =====")
    for key in sorted(test_metrics):
        val = test_metrics[key]
        if isinstance(val, float):
            print(f"  {key}: {val:.4f}")
        elif isinstance(val, int):
            print(f"  {key}: {val}")

    for category in ("entity", "relation", "classification", "event_trigger", "event_argument"):
        report_key = f"eval_{category}_classification_report"
        if report_key in test_metrics:
            print(f"\n--- {category} classification report ---")
            print(test_metrics[report_key])


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: train.py <config.yaml>")
        raise SystemExit(1)
    main(sys.argv[1])

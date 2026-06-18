"""Train a fresh GLiNER2 from a YAML config.

Run::

    uv run python tools/train/train.py --config tools/train/config/mmbert-small-focal.yaml

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
* ``labels``   - optional label transforms applied identically to train, val,
  and test::

      labels:
        rollup: true        # ORG.Media -> ORG (keep the parent segment)
        separator: "."      # split character for roll-up (default ".")
        map:                # rename labels after roll-up
          ORG: ORGANIZATION

  Roll-up runs first, then ``map``, across entity labels, relation names,
  event types, event-argument roles, and classification labels (plus
  ``entity_descriptions`` keys). Labels that collide after transform are
  merged, not dropped. Omit the section for no transform.

Results land in ``<output_dir>/train_results.json`` and the blind-test metrics
in ``<output_dir>/test_metrics.json``.
"""

from __future__ import annotations

import argparse
import json
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


# ---------------------------------------------------------------------------
# Label transforms (optional ``labels`` config section)
# ---------------------------------------------------------------------------

def _label_fn(rollup: bool, separator: str, mapping: Dict[str, str]):
    """Roll a label up to its parent (first ``separator`` segment) then remap it."""
    def fn(label: str) -> str:
        if rollup and separator in label:
            label = label.split(separator, 1)[0]
        return mapping.get(label, label)
    return fn


def _dedup(seq: List) -> List:
    """Order-preserving dedup."""
    out, seen = [], set()
    for x in seq:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out


def _transform_entities(entities: Dict, fn) -> Dict:
    """Re-key entities by transformed label, MERGING collisions.

    Gold values are surface lists (merged + deduped); schema values are
    description strings (first non-empty wins). Collapsing e.g.
    ORG.Media + ORG.Government -> ORG keeps every surface.
    """
    out: Dict = {}
    for label, value in entities.items():
        new = fn(label)
        if isinstance(value, list):
            out.setdefault(new, []).extend(value)
        elif not out.get(new):
            out[new] = value
    return {k: (_dedup(v) if isinstance(v, list) else v) for k, v in out.items()}


def _transform_descriptions(desc: Dict, fn) -> Dict:
    """Re-key entity_descriptions, keeping the first non-empty on collision."""
    out: Dict = {}
    for label, text in desc.items():
        new = fn(label)
        if not out.get(new):
            out[new] = text
    return out


def _transform_relations(relations: List, fn) -> List:
    out = []
    for rel in relations:
        if isinstance(rel, dict):
            out.append({fn(name): fields for name, fields in rel.items()})
        else:
            out.append(rel)
    return out


def _transform_events(events: List, fn) -> List:
    out = []
    for ev in events:
        if not isinstance(ev, dict):
            out.append(ev)
            continue
        new = dict(ev)
        if isinstance(ev.get("event_type"), str):
            new["event_type"] = fn(ev["event_type"])
        if "arguments" in ev:
            args = []
            for arg in ev.get("arguments") or []:
                if isinstance(arg, dict) and isinstance(arg.get("role"), str):
                    a = dict(arg)
                    a["role"] = fn(arg["role"])
                    args.append(a)
                else:
                    args.append(arg)
            new["arguments"] = args
        out.append(new)
    return out


def _transform_classifications(cls_list: List, fn) -> List:
    out = []
    for c in cls_list:
        if not isinstance(c, dict):
            out.append(c)
            continue
        nc = dict(c)
        if isinstance(c.get("labels"), list):
            nc["labels"] = _dedup([fn(x) if isinstance(x, str) else x for x in c["labels"]])
        tl = c.get("true_label")
        if isinstance(tl, str):
            nc["true_label"] = fn(tl)
        elif isinstance(tl, list):
            nc["true_label"] = _dedup([fn(x) if isinstance(x, str) else x for x in tl])
        out.append(nc)
    return out


def _transform_container(container: Dict, fn) -> Dict:
    """Apply ``fn`` to every label-bearing field in a gold/schema dict."""
    out = dict(container)
    if isinstance(container.get("entities"), dict):
        out["entities"] = _transform_entities(container["entities"], fn)
    if isinstance(container.get("entity_descriptions"), dict):
        out["entity_descriptions"] = _transform_descriptions(container["entity_descriptions"], fn)
    if isinstance(container.get("relations"), list):
        out["relations"] = _transform_relations(container["relations"], fn)
    if isinstance(container.get("events"), list):
        out["events"] = _transform_events(container["events"], fn)
    if isinstance(container.get("classifications"), list):
        out["classifications"] = _transform_classifications(container["classifications"], fn)
    return out


def transform_record(record: Dict, fn) -> Dict:
    """Return a copy of ``record`` with labels transformed in its gold container.

    Handles both the training (``output``) and schema (``schema``) formats.
    """
    rec = dict(record)
    for key in ("output", "schema"):
        if isinstance(record.get(key), dict):
            rec[key] = _transform_container(record[key], fn)
    return rec


def _read_records(paths: List[str]) -> List[Dict]:
    records: List[Dict] = []
    for p in paths:
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    return records


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

    # Optional label transforms, applied identically to train/val/test.
    labels_cfg = cfg.get("labels") or {}
    rollup = bool(labels_cfg.get("rollup", False))
    separator = labels_cfg.get("separator", ".")
    mapping = labels_cfg.get("map") or {}
    if rollup or mapping:
        fn = _label_fn(rollup, separator, mapping)
        train_data = [transform_record(r, fn) for r in _read_records(train_data)]
        eval_data = [transform_record(r, fn) for r in _read_records(eval_data)]
        test_data = [transform_record(r, fn) for r in _read_records(test_data)]
        print(f"[labels] rollup={rollup} separator={separator!r} map={len(mapping)} entries; "
              f"transformed {len(train_data)}/{len(eval_data)}/{len(test_data)} train/val/test records")

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
    parser = argparse.ArgumentParser(description="Train GLiNER2 from a YAML config.")
    parser.add_argument("--config", required=True, help="Path to the YAML config file.")
    args = parser.parse_args()
    main(args.config)

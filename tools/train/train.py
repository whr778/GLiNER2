"""Train a fresh GLiNER2 from a YAML config.

Run::

    uv run python tools/train/train.py --config tools/train/config/mmbert-small-focal.yaml

The config has four sections:

* ``model``    - either ``encoder`` (a raw HF encoder bootstrapped with fresh
  heads via ``GLiNER2.from_encoder``; remaining keys like ``max_width`` /
  ``struct_loss`` are forwarded to it) or ``pretrained`` (a saved GLiNER2
  checkpoint continued via ``GLiNER2.from_pretrained``; remaining keys override
  the loaded ``model.config``). Exactly one of the two must be set.
* ``training`` - fields forwarded verbatim to :class:`TrainingConfig`. Multi-GPU:
  preferred is DistributedDataParallel via ``torchrun --nproc_per_node=N
  tools/train/train.py --config ...`` (auto-detected from ``LOCAL_RANK``; here
  ``batch_size`` is per-GPU). Alternatively set ``data_parallel: true`` for
  single-process ``nn.DataParallel`` (``batch_size`` is the total split across
  GPUs). See TRAINING.md section 4c.
* ``eval``     - ``batch_size`` / ``threshold`` for the metrics hook and the
  blind test pass.
* ``data``     - ``corpora`` base paths (``<name>.{train,val,test}.jsonl``) and
  an ``event_files`` map of ``{name: {train,val,test}}``. Event splits are
  included only if the file exists on disk, so a config runs with any subset
  present. See ``tools/train/config/`` for examples.
* ``labels``   - optional per-category label transforms, applied identically to
  train, val, and test. Each category (``entities``, ``relations``, ``events``,
  ``classifications``) has its own ``rollup`` / ``separator`` / ``map``::

      labels:
        entities:
          rollup: true        # ORG.Media -> ORG (keep the parent segment)
          separator: "."      # split character for roll-up (default ".")
          map:                # rename labels after roll-up
            ORG: ORGANIZATION
        events:
          rollup: true
          separator: "."
          map: {}

  Per category, roll-up runs first then ``map``. ``entities`` also covers
  ``entity_descriptions`` keys; ``events`` covers both event types and argument
  roles. Labels colliding after transform are merged, not dropped. Omit a
  category (or the whole section) to leave it untouched.

Results land in ``<output_dir>/train_results.json`` and the blind-test metrics
in ``<output_dir>/test_metrics.json``.
"""

from __future__ import annotations

import argparse
import json
import os
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


LABEL_CATEGORIES = ("entities", "relations", "events", "classifications")


def _category_fns(labels_cfg: Dict) -> Dict:
    """Build ``{category: label_fn}`` from a nested ``labels`` config section.

    Each category (``entities``, ``relations``, ``events``, ``classifications``)
    has its own ``rollup`` / ``separator`` / ``map``. A category with neither an
    active rollup nor a map is skipped. ``events`` covers both event types and
    argument roles; the ``entities`` fn also applies to ``entity_descriptions``.
    """
    if any(k in labels_cfg for k in ("rollup", "separator", "map")):
        raise ValueError(
            "labels: uses the removed flat form. Nest rollup/separator/map under a "
            f"category, one of {LABEL_CATEGORIES}."
        )
    fns: Dict = {}
    for cat in LABEL_CATEGORIES:
        block = labels_cfg.get(cat) or {}
        rollup = bool(block.get("rollup", False))
        separator = block.get("separator", ".")
        mapping = block.get("map") or {}
        if rollup or mapping:
            fns[cat] = _label_fn(rollup, separator, mapping)
    return fns


def _transform_container(container: Dict, fns: Dict) -> Dict:
    """Apply each category's fn to its label-bearing fields in a gold/schema dict."""
    out = dict(container)
    ent = fns.get("entities")
    if ent:
        if isinstance(container.get("entities"), dict):
            out["entities"] = _transform_entities(container["entities"], ent)
        if isinstance(container.get("entity_descriptions"), dict):
            out["entity_descriptions"] = _transform_descriptions(container["entity_descriptions"], ent)
    rel = fns.get("relations")
    if rel and isinstance(container.get("relations"), list):
        out["relations"] = _transform_relations(container["relations"], rel)
    ev = fns.get("events")
    if ev and isinstance(container.get("events"), list):
        out["events"] = _transform_events(container["events"], ev)
    cls = fns.get("classifications")
    if cls and isinstance(container.get("classifications"), list):
        out["classifications"] = _transform_classifications(container["classifications"], cls)
    return out


def transform_record(record: Dict, fns: Dict) -> Dict:
    """Return a copy of ``record`` with per-category label transforms applied.

    Handles both the training (``output``) and schema (``schema``) formats.
    """
    rec = dict(record)
    for key in ("output", "schema"):
        if isinstance(record.get(key), dict):
            rec[key] = _transform_container(record[key], fns)
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


def _build_model(model_cfg: Dict):
    """Build the model from the ``model`` config section.

    ``pretrained`` loads a saved GLiNER2 checkpoint (continue/fine-tune its
    trained heads via ``from_pretrained``); ``encoder`` bootstraps fresh heads
    on a raw HF encoder via ``from_encoder``. Exactly one must be set.

    On the ``pretrained`` path, ``map_location`` / ``quantize`` / ``compile`` go
    to ``from_pretrained``; any remaining keys (e.g. ``struct_loss``) override
    the loaded ``model.config`` -- use only loss-related overrides, not
    structural ones like ``max_width`` that are baked into the saved weights.
    """
    model_cfg = dict(model_cfg)
    pretrained = model_cfg.pop("pretrained", None)
    if pretrained is not None:
        load_kwargs = {k: model_cfg.pop(k) for k in ("map_location", "quantize", "compile") if k in model_cfg}
        model = GLiNER2.from_pretrained(pretrained, **load_kwargs)
        for key, value in model_cfg.items():
            setattr(model.config, key, value)
        return model
    encoder = model_cfg.pop("encoder")
    return GLiNER2.from_encoder(encoder, **model_cfg)


def main(config_path: str) -> None:
    # Under torchrun (DDP) only rank 0 estimates ETA and writes results/blind-test;
    # all ranks run trainer.train(). LOCAL_RANK is unset (-> -1) for single-process.
    is_main = int(os.environ.get("LOCAL_RANK", -1)) <= 0

    cfg = yaml.safe_load(Path(config_path).read_text())

    model = _build_model(cfg["model"])

    config = TrainingConfig(**cfg["training"])

    data = cfg.get("data") or {}
    corpora = data.get("corpora") or []
    event_files = data.get("event_files") or {}
    train_data = _split_files(corpora, "train") + _event_split(event_files, "train")
    eval_data = _split_files(corpora, "val") + _event_split(event_files, "val")
    test_data = _split_files(corpora, "test") + _event_split(event_files, "test")

    # Optional per-category label transforms, applied identically to train/val/test.
    fns = _category_fns(cfg.get("labels") or {})
    if fns:
        train_data = [transform_record(r, fns) for r in _read_records(train_data)]
        eval_data = [transform_record(r, fns) for r in _read_records(eval_data)]
        test_data = [transform_record(r, fns) for r in _read_records(test_data)]
        print(f"[labels] transforms: {', '.join(sorted(fns))}; "
              f"transformed {len(train_data)}/{len(eval_data)}/{len(test_data)} train/val/test records")

    eval_cfg = cfg.get("eval") or {}
    eval_bs = eval_cfg.get("batch_size", 8)
    eval_thr = eval_cfg.get("threshold", 0.5)

    trainer = GLiNER2Trainer(
        model, config,
        eval_data=eval_data,
        compute_metrics=make_compute_metrics(batch_size=eval_bs, threshold=eval_thr),
    )
    if is_main:
        estimate_eta(model, train_data, config)
    results = trainer.train(train_data=train_data)
    # pprint(results)

    # Only rank 0 writes results and runs the blind test; other ranks are done.
    if not is_main:
        return

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
        for regime in ("strict", "relaxed"):
            report_key = f"eval_{category}_{regime}_classification_report"
            if report_key in test_metrics:
                print(f"\n--- {category} {regime} classification report ---")
                print(test_metrics[report_key])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train GLiNER2 from a YAML config.")
    parser.add_argument("--config", required=True, help="Path to the YAML config file.")
    args = parser.parse_args()
    main(args.config)

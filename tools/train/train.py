"""Train a fresh GLiNER2 from a YAML config.

Run::

    uv run python tools/train/train.py --config tools/train/config/mmbert-small-focal.yaml

The config has four sections:

* ``model``    - either ``encoder`` (a raw HF encoder bootstrapped with fresh
  heads via ``AutoExtractor.from_encoder``; remaining keys like ``max_width`` /
  ``struct_loss`` are forwarded to it) or ``pretrained`` (a saved GLiNER2
  checkpoint continued via ``AutoExtractor.from_pretrained``; remaining keys override
  the loaded ``model.config``). Exactly one of the two must be set.
  Optional ``architecture`` picks the head stack: ``span`` (default, unchanged
  behaviour) or ``boundary``. Boundary configs take a ``boundary_head:`` block
  and must NOT set ``max_width`` -- that is a span-head field, and removing the
  span width cap is the point of the boundary architecture. On the ``pretrained``
  path a declared ``architecture`` is checked against the checkpoint, so warm
  starting the wrong architecture fails loudly instead of silently.
* ``training`` - fields forwarded verbatim to :class:`TrainingConfig`. Multi-GPU is
  DistributedDataParallel and there is NO config flag for it -- it is enabled by how
  you launch::

      uv run torchrun --standalone --nproc_per_node=N tools/train/train.py --config ...

  ``LOCAL_RANK`` is read from the environment into ``config.local_rank``, and the
  trainer's distributed path engages on ``local_rank >= 0``. ``batch_size`` is then
  PER GPU, so halve ``gradient_accumulation_steps`` to keep the effective batch --
  moving such a config back to one GPU without restoring accumulation silently halves
  it. ``data_parallel`` is a DEPRECATED NO-OP: the single-process ``nn.DataParallel``
  path is gone, and setting it true does nothing at all. See TRAINING.md section 4.
* ``eval``     - ``batch_size`` / ``threshold`` for the metrics hook and the
  blind test pass. Optional ``stopword_languages`` (list of ISO 639-2 codes)
  enables multilingual stopword filtering in relaxed metrics; defaults to the
  English-only built-in set. Optional ``stopword_yaml`` (relative path, default
  ``stopwords.yaml``) supplements stopwordsiso with a user-maintained YAML file.
  Optional ``threshold_sweep`` calibrates the decision threshold against the
  val set after training: ``true`` uses a default 5-point grid (0.1/0.3/0.5/
  0.7/0.9), or give an explicit list of candidates. The chosen threshold
  replaces ``threshold`` for the blind test and is recorded in
  ``best/threshold_sweep.json`` and the model card. Unset/false keeps
  ``threshold`` as-is (today's behavior). Each candidate is a full forward
  pass over the val set, so keep the grid small.
* ``data``     - ``corpora`` base paths (``<name>.{train,val,test}.jsonl``) and
  an ``event_files`` map of ``{name: {train,val,test}}``. Event splits are
  included only if the file exists on disk, so a config runs with any subset
  present. See ``tools/train/config/`` for examples.
* ``labels``   - optional per-category label transforms, applied identically to
  train, val, and test. Each category (``entities``, ``relations``, ``events``,
  ``classifications``, ``structures``) has its own ``rollup`` / ``separator`` /
  ``map``::

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
  ``entity_descriptions`` keys; ``structures`` covers ``json_structures`` names
  AND their field names, carrying ``record_metadata`` keys and ``anchor`` with
  the rename; ``events`` covers both event types and argument
  roles. Labels colliding after transform are merged, not dropped. Omit a
  category (or the whole section) to leave it untouched.

Results land in ``<output_dir>/train_results.json`` and the blind-test metrics
in ``<output_dir>/test_metrics.json``. The best checkpoint's eval metrics are
written to ``<output_dir>/eval_metrics.json`` and ``<output_dir>/best/``; the
blind-test metrics are also copied into ``<output_dir>/best/test_metrics.json``,
so each metrics file sits alongside the model it describes. A human-readable
``<output_dir>/best/MODEL_CARD.md`` is generated at the end of training,
covering the base model, training procedure and date, the datasets actually
used (with per-dataset licenses), the best metrics, and an effective-license
determination sourced from ``tools/train/dataset_registry.yaml``.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from pprint import pprint
from typing import Dict, List

import yaml

from gliner2 import AutoExtractor
from gliner2.inference.schema import derive_schema
from gliner2.training import estimate_eta, evaluate_checkpoint, make_compute_metrics, sweep_thresholds
from gliner2.training.metrics import DEFAULT_THRESHOLD_GRID, _selection_score, make_sweeping_compute_metrics
from gliner2.training.trainer import ExtractorDataset, GLiNER2Trainer, TrainingConfig


def _dataset_counts(corpora: List[str], event_files: Dict[str, Dict[str, str]]) -> Dict[str, Dict[str, int]]:
    """Count non-empty lines per split for each dataset."""
    def _count(path: str) -> int:
        try:
            with open(path, encoding="utf-8") as f:
                return sum(1 for line in f if line.strip())
        except OSError:
            return 0

    # Key by the SAME canonical key the card resolves datasets under, or the lookup
    # misses and the count columns render blank: `sentence_rex.j10k` is credited to
    # `sentence_rex`, and an event_files label like `synthetic_coerced` to its registry
    # name. Counts for slices/aliases of one dataset are summed rather than overwritten.
    from model_card import canonical_dataset_key

    counts: Dict[str, Dict[str, int]] = {}

    def _add(key: str, split: str, n: int) -> None:
        counts.setdefault(key, {})
        counts[key][split] = counts[key].get(split, 0) + n

    for c in (corpora or []):
        key = canonical_dataset_key(Path(c).name)
        counts.setdefault(key, {})
        for split in ("train", "val", "test"):
            p = f"{c}.{split}.jsonl"
            if Path(p).is_file():
                _add(key, split, _count(p))
    for name, by_split in (event_files or {}).items():
        key = canonical_dataset_key(name)
        counts.setdefault(key, {})
        for split, path in (by_split or {}).items():
            if path and Path(path).is_file():
                _add(key, split, _count(path))
    return counts


def _split_files(corpora: List[str], suffix: str) -> List[str]:
    paths = [f"{c}.{suffix}.jsonl" for c in corpora]
    for p in paths:
        _fetch_if_missing(p)
    return paths


def _fetch_if_missing(path: str) -> None:
    """Pull ``path`` from the Hub when it is absent locally and the registry says where.

    A corpus entry may carry ``hf_jsonl: <repo>``, a dataset repo holding the
    GLiNER2 JSONL splits under their exact local basenames. This is deliberately
    NOT the registry's ``hf_id``, which is upstream attribution for the model card
    and for most corpora points at raw parquet or BIO tags, not converted records.

    Private repos work as-is: huggingface_hub reads HF_TOKEN from the environment.
    """
    p = Path(path)
    if p.is_file():
        return
    from model_card import canonical_dataset_key, load_registry
    registry = load_registry()
    # A directory that holds its own JSONL wins over the per-corpus repo. Resolving by
    # basename alone is ambiguous across directories: scaling_joint/chfinann.val.jsonl is
    # a 150-record slice, and whr778/chfinann would answer with the 3,204-record val.
    repo = (registry.get("jsonl_dirs") or {}).get(p.parent.name)
    if not repo:
        if p.parent.name != "data":
            # A file in a SUBDIRECTORY is a slice, and its basename is shared with the
            # parent corpus: data/scaling/casie.val.jsonl and data/casie.val.jsonl are
            # different files. Falling back to the corpus repo returns the parent's full
            # split with no error. Refuse instead, and say what would fix it.
            print(f"[data] {path} is under data/{p.parent.name}/ with no 'jsonl_dirs' "
                  f"entry; NOT fetching by basename, which would return the parent "
                  f"corpus's file. Add {p.parent.name} to jsonl_dirs in "
                  f"dataset_registry.yaml.")
            return
        key = canonical_dataset_key(p.name.rsplit(".", 2)[0])
        repo = (registry.get("datasets", {}).get(key) or {}).get("hf_jsonl")
    if not repo:
        return
    from huggingface_hub import hf_hub_download
    print(f"[data] {path} missing; fetching {p.name} from {repo}")
    hf_hub_download(repo_id=repo, filename=p.name, repo_type="dataset",
                    local_dir=str(p.parent))


def _event_split(event_files: Dict[str, Dict[str, str]], suffix: str) -> List[str]:
    """Resolve the event-file paths for one split, fetching what is absent.

    This used to silently drop any path that was not already on disk, while the
    `corpora` path (`_split_files`) fetched from the Hub. On a fresh box that meant a
    blind test quietly scored ONLY the corpora -- 3 files instead of 11 -- and reported
    the subset as if it were the whole test set. A missing corpus is now fetched, and
    one that still cannot be resolved is named rather than dropped in silence.
    """
    paths: List[str] = []
    for name, by_split in event_files.items():
        # A malformed entry is not the same as an absent split, and YAML makes the two
        # look identical. `{train:data/x.jsonl}` -- one missing space -- parses as the
        # single key "train:data/x.jsonl" with value None, so .get("train") is None and
        # the corpus is skipped as though the config never asked for it. That is how a
        # run ends up reporting entity and relation metrics with no event metrics at all,
        # while the data sits on disk. Name the keys we actually saw.
        if not isinstance(by_split, dict) or not (
            {"train", "val", "test"} & set(by_split)
        ):
            print(f"[data] event_files.{name} declares no train/val/test key "
                  f"(got {sorted(by_split) if isinstance(by_split, dict) else type(by_split).__name__}); "
                  f"check for a missing space after a colon. NOT scored")
            continue
        p = by_split.get(suffix)
        if not p:
            continue
        _fetch_if_missing(p)
        if Path(p).is_file():
            paths.append(p)
        else:
            print(f"[data] {p} unavailable and not fetchable; NOT scored")
    return paths


def _build_streaming_data(hf_streaming: Dict, config: TrainingConfig):
    """Resolve a ``data.hf_streaming`` block into a streaming train dataset and
    bounded in-memory val/test lists. Nothing is written to disk.

    Returns ``(train_dataset, val_records, test_records)`` where the train set is
    a ``StreamingExtractorDataset`` (lazy, low-memory) and val/test are lists
    capped by label class via ``eval_min_per_class``.
    """
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))
    from hf_stream import SOURCES, cap_by_class
    from gliner2.training.trainer import StreamingExtractorDataset

    name = hf_streaming.get("source")
    if name not in SOURCES:
        raise ValueError(f"unknown hf_streaming source {name!r}; have {sorted(SOURCES)}")
    if config.max_steps <= 0:
        raise ValueError("data.hf_streaming requires training.max_steps > 0 "
                         "(a streaming dataset has no length to derive epochs from).")

    src = SOURCES[name]
    langs = hf_streaming.get("langs", "all")
    cap = int(hf_streaming.get("eval_min_per_class", 0) or 0)
    buf = int(hf_streaming.get("shuffle_buffer", 10000) or 10000)
    classes = set(src.classes)

    train_ds = StreamingExtractorDataset(
        make_iter=lambda: src.records(langs, "train"), shuffle_buffer=buf)
    val = list(cap_by_class(src.records(langs, "validation"), cap, src.classes_of, classes))
    test = list(cap_by_class(src.records(langs, "test"), cap, src.classes_of, classes))
    print(f"[hf_streaming] source={name} langs={langs} eval_min_per_class={cap} "
          f"-> val={len(val)} test={len(test)} records (train streams lazily; "
          f"nothing written to disk)")
    return train_ds, val, test


def _write_model_card(
    cfg, config, corpora, event_files, results, test_metrics, best: Path,
    threshold: float, threshold_calibrated: bool,
) -> None:
    """Generate MODEL_CARD.md in the best checkpoint folder. Never fatal -- a
    card bug must not lose a model that already trained for hours."""
    try:
        from datetime import datetime
        from model_card import build_model_card

        eval_metrics = None
        ev = best / "eval_metrics.json"
        if ev.is_file():
            eval_metrics = json.loads(ev.read_text())
        # Datasets actually used: every corpus, plus event sets whose files exist.
        # Scaling slices (sentence_rex.j10k) are credited to the parent dataset, so
        # the card reports the real corpus and its real license.
        from model_card import canonical_dataset_key
        dataset_keys = [canonical_dataset_key(Path(c).name) for c in corpora] + [
            canonical_dataset_key(name) for name, by_split in (event_files or {}).items()
            if any(p and Path(p).is_file() for p in by_split.values())
        ]
        # A streamed source (data.hf_streaming) is a real training dataset too;
        # add it and override its registry language with the run's actual langs.
        registry = None
        hf_streaming = (cfg.get("data") or {}).get("hf_streaming")
        if hf_streaming and hf_streaming.get("source"):
            from model_card import load_registry
            src_key = hf_streaming["source"]
            dataset_keys.append(src_key)
            registry = load_registry()
            entry = (registry.get("datasets") or {}).get(src_key)
            langs = hf_streaming.get("langs", "all")
            if entry is not None and isinstance(langs, list) and langs:
                registry["datasets"][src_key] = {**entry, "language": [str(x) for x in langs]}
        model_cfg = cfg.get("model") or {}
        card = build_model_card(
            model_name=getattr(config, "experiment_name", None) or Path(config.output_dir).name,
            base_model=model_cfg.get("encoder") or model_cfg.get("pretrained"),
            cfg=cfg, config=config, dataset_keys=dataset_keys, results=results,
            eval_metrics=eval_metrics, test_metrics=test_metrics or None,
            generated_at=datetime.now().strftime("%Y-%m-%d"), registry=registry,
            dataset_counts=_dataset_counts(corpora, event_files),
            threshold=threshold, threshold_calibrated=threshold_calibrated,
        )
        path = best / "MODEL_CARD.md"
        path.write_text(card, encoding="utf-8")
        print(f"[model card] Wrote {path}")
    except Exception as e:  # noqa: BLE001 - never let a card bug abort the run
        print(f"[model card] Skipped (generation failed): {e}")


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


def _transform_structures(structures: List, meta: Dict, fn):
    """Re-key json_structures names AND their field names, carrying record_metadata.

    ``record_metadata`` is keyed by structure name and its ``anchor`` names a field, so a
    rename that misses either leaves a structure the boundary head cannot decode -- it
    returns {} with no error (gliner2/processor.py). Both move with the rename.
    """
    out = []
    for entry in structures:
        if not isinstance(entry, dict):
            out.append(entry)
            continue
        renamed = {}
        for name, body in entry.items():
            new_name = fn(name)
            if isinstance(body, dict):
                fields = renamed.setdefault(new_name, {})
                for field, value in body.items():
                    new_field = fn(field)
                    if not fields.get(new_field):
                        fields[new_field] = value
            elif new_name not in renamed:
                renamed[new_name] = body
        out.append(renamed)

    new_meta = {}
    for name, block in (meta or {}).items():
        new_block = dict(block) if isinstance(block, dict) else block
        if isinstance(new_block, dict) and isinstance(new_block.get("anchor"), str):
            new_block["anchor"] = fn(new_block["anchor"])
        new_meta.setdefault(fn(name), new_block)
    return out, new_meta


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


LABEL_CATEGORIES = ("entities", "relations", "events", "classifications", "structures")


def load_labels_cfg(cfg: Dict, config_path: str = "") -> Dict:
    """Return the ``labels`` section, merging a shared ``labels_file`` underneath it.

    Every model warm-started from one base has to present that base's label space, so the
    unified maps live in ONE file that each config references rather than 25 copies that
    drift. A category given inline in the config wins over the shared file, which is how a
    run opts out of one category without forking the whole map.
    """
    shared: Dict = {}
    ref = cfg.get("labels_file")
    if ref:
        path = Path(ref)
        if not path.is_absolute() and config_path:
            beside = Path(config_path).resolve().parent / path
            # Resolve relative to the CONFIG, not the working directory. Falling back to
            # a cwd-relative path silently loads whatever `labels/unified.yaml` happens to
            # sit next to wherever the run was started -- or, more often, throws naming
            # only the relative path and not where it actually looked.
            if not beside.exists() and not path.exists():
                raise FileNotFoundError(
                    f"labels_file {ref!r} not found; looked in {beside} and {path.resolve()}")
            path = beside if beside.exists() else path
        shared = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        shared = shared.get("labels", shared)
    merged = dict(shared)
    merged.update(cfg.get("labels") or {})
    return merged


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
    st = fns.get("structures")
    if st and isinstance(container.get("json_structures"), list):
        out["json_structures"], meta = _transform_structures(
            container["json_structures"], container.get("record_metadata") or {}, st)
        if meta:
            out["record_metadata"] = meta
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


def _detect_lang(text: str) -> str:
    """Return 3-char ISO 639-2 code for text using lumi_language_id + langcodes."""
    from lumi_language_id import detect_language as _lid
    import langcodes
    lang2, _ = _lid(text)
    return langcodes.Language.get(lang2).to_alpha3()


def _annotate_languages(records: List[Dict]) -> List[Dict]:
    """Stamp each record dict with '_lang' (ISO 639-2) in-place; returns the list."""
    print("[lang] Detecting language for each sample (this may take a moment)...")
    for rec in records:
        text = rec.get("input") or rec.get("text") or ""
        rec["_lang"] = _detect_lang(text.strip()) if text.strip() else "und"
    return records


"""Boundary-head keys that are baked into the saved weights. Overriding one on
the ``pretrained`` path cannot work -- the modules are already built -- so we
refuse rather than apply half of it."""
_STRUCTURAL_BOUNDARY_KEYS = frozenset({
    "enable_records", "enable_relations", "boundary_dim", "pair_dim",
    "boundary_refinement_layers", "boundary_ffn_multiplier", "candidate_pool",
    "multihead_pair_compat_heads", "relation_heads_per_type",
    "relation_tails_per_type", "directional_relation_states",
    "relation_biaffine_content",
    # Measured 2026-09-05 by diffing state_dict keys across each flag: these eight
    # ADD OR REMOVE parameter tensors and were unguarded. Unguarded is worse than
    # refusing, because the override still lands on `model.config` and is SAVED --
    # so the run trains the checkpoint's architecture while writing a config that
    # describes a different one, and the next `from_pretrained` builds the modules
    # the config asks for and dies on a state-dict mismatch. It poisons the
    # checkpoint rather than the run. `from_pretrained` already names most of these
    # in its own load error (models/boundary/model.py); that list and this set had
    # drifted apart.
    "enable_abstention", "enable_count_head", "enable_span_content",
    "boundary_attention_layers", "candidate_attention_layers",
    "query_attention_layers", "endpoint_difference_features",
    "query_conditioned_inside_weight",
    # A key diff cannot see a flag that RESIZES an always-built module -- the
    # tensor names are identical and only the shapes move. Re-measured comparing
    # shapes: these three. `record_dim` and `record_instance_queries` are named
    # nowhere in the loader's error either, so they would have failed a warm start
    # with a bare shape-mismatch traceback.
    "enable_rotary_endpoints", "record_dim", "record_instance_queries",
    # The remainder of what `from_pretrained` itself declares incompatible. The
    # guard is now a superset of that list: whatever the loader refuses to LOAD,
    # the trainer refuses to SAVE.
    "candidate_attention_heads", "pool_boundary_top_k", "pool_size",
})


def _apply_boundary_head_overrides(model, overrides: Dict) -> None:
    """Apply a config's ``model.boundary_head`` block to an already-loaded model.

    ``from_pretrained`` builds ``boundary_settings`` from the CHECKPOINT's config
    inside ``__init__``, so the plain ``setattr(model.config, ...)`` the other
    overrides use lands too late and is silently dropped -- the model keeps the
    checkpoint's values. Measured: a config setting
    ``boundary_head.task_loss_weights`` produced ``task_loss_weights=None`` on the
    built model, i.e. a treatment arm that is a silent duplicate of its control.

    Loss-related keys are merged and ``boundary_settings`` is rebuilt. Structural
    keys raise, because the modules they size were already constructed.
    """
    from gliner2.configuration import BoundaryHeadSettings, validate_boundary_head

    current = dict(getattr(model.config, "boundary_head", None) or {})
    conflicting = sorted(
        k for k, v in overrides.items()
        if k in _STRUCTURAL_BOUNDARY_KEYS and current.get(k) != v
    )
    if conflicting:
        raise SystemExit(
            f"[config] boundary_head keys {conflicting} are structural and cannot be "
            f"overridden on the `pretrained` path -- the checkpoint's modules are "
            f"already built. Retrain from `encoder:`, or drop these keys."
        )
    current.update(overrides)
    model.config.boundary_head = current
    settings = BoundaryHeadSettings(**validate_boundary_head(current))
    model.boundary_settings = settings

    # The HEAD holds its own reference to a settings object, built in its __init__
    # from the checkpoint's config. Rebuilding only `model.boundary_settings` left
    # every knob the head reads through `self.settings` -- boundary_negative_weight,
    # the soft_iou/rerank/proposal/count weights, negative_query_ratio,
    # task_loss_weight_scope -- silently at its checkpoint value. Measured: a config
    # setting task_loss_weight_scope="all" produced scope="all" on the model and
    # "span" on the head, i.e. a treatment arm inert in exactly the way dfaaa2a was
    # meant to end.
    head = getattr(model, "boundary_head", None)
    if head is not None:
        head.settings = settings
        # Two values are COPIED at construction rather than read live, so assigning
        # settings does not move them.
        head.hard_negatives_per_positive = settings.hard_negatives_per_positive
        head.minimum_hard_negatives = settings.minimum_hard_negatives
        # A third, found 2026-09-05. `use_inside_evidence` adds no parameters, so it
        # is exactly the kind of behavioural knob this function exists to support --
        # but it is copied into TWO places at construction (the head, and the pair
        # scorer it is passed to as a constructor argument) and BOTH are what the
        # forward pass reads. Measured on fastino/gliner2.5-multi-v1: overriding it
        # to False left config and settings at False and both readers at True, i.e.
        # a treatment arm identical to its control, reported as applied.
        head.use_inside_evidence = settings.use_inside_evidence
        head.pair_scorer.use_inside_evidence = settings.use_inside_evidence


def _build_model(model_cfg: Dict):
    """Build the model from the ``model`` config section.

    ``pretrained`` loads a saved checkpoint (continue/fine-tune its trained heads
    via ``from_pretrained``); ``encoder`` bootstraps fresh heads on a raw HF
    encoder via ``from_encoder``. Exactly one must be set.

    ``architecture`` selects the head stack (``"span"`` default, ``"boundary"``
    for the boundary head). Both paths dispatch through ``AutoExtractor``: on the
    ``pretrained`` path a declared ``architecture`` is passed through, so a config
    asking for ``boundary`` against a span checkpoint fails loudly instead of
    silently training the wrong architecture.

    On the ``pretrained`` path, ``map_location`` / ``quantize`` / ``compile`` go
    to ``from_pretrained``; any remaining keys (e.g. ``struct_loss``) override
    the loaded ``model.config`` -- use only loss-related overrides, not
    structural ones like ``max_width`` that are baked into the saved weights.
    """
    model_cfg = dict(model_cfg)
    architecture = model_cfg.pop("architecture", None)
    pretrained = model_cfg.pop("pretrained", None)
    if pretrained is not None:
        load_kwargs = {k: model_cfg.pop(k) for k in ("map_location", "quantize", "compile") if k in model_cfg}
        if architecture is not None:
            load_kwargs["architecture"] = architecture
        model = AutoExtractor.from_pretrained(pretrained, **load_kwargs)
        for key, value in model_cfg.items():
            if key == "boundary_head":
                continue                      # needs more than setattr; see below
            setattr(model.config, key, value)
        if "boundary_head" in model_cfg:
            _apply_boundary_head_overrides(model, model_cfg["boundary_head"])
        return model
    encoder = model_cfg.pop("encoder")
    return AutoExtractor.from_encoder(
        encoder, architecture=architecture or "span", **model_cfg
    )


def _collect_lang_codes(data) -> set:
    """Collect unique ISO 639-2 _lang codes from training records.

    Accepts a list of file-path strings (JSONL) or already-materialised dicts.
    Skips undetermined ('und') and missing values.
    """
    codes: set = set()
    for item in data:
        if isinstance(item, str):
            try:
                with open(item, encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            lang = json.loads(line).get("_lang")
                            if lang and lang != "und":
                                codes.add(lang)
                        except (json.JSONDecodeError, AttributeError):
                            pass
            except OSError:
                pass
        elif isinstance(item, dict):
            lang = item.get("_lang")
            if lang and lang != "und":
                codes.add(lang)
    return codes


def _print_stopword_report(codes: list, yaml_path) -> None:
    """Print a per-language table: code, name, word count, source."""
    try:
        import langcodes
        import stopwordsiso
    except ImportError:
        print(f"[stopwords] {len(codes)} languages: {', '.join(codes)}")
        return

    yaml_data: dict = {}
    if yaml_path and yaml_path.exists():
        with open(yaml_path, encoding="utf-8") as fh:
            yaml_data = yaml.safe_load(fh) or {}

    print(f"\n[stopwords] Building stopword set from {len(codes)} languages:")
    print(f"  {'Code':<6}  {'Language':<30}  {'Words':>5}  Source")
    print(f"  {'-'*6}  {'-'*30}  {'-'*5}  {'-'*20}")

    total = 0
    for code in codes:
        try:
            lang = langcodes.Language.get(code)
            alpha2 = lang.language
        except Exception:
            print(f"  {code:<6}  {'(unknown)':<30}  {'0':>5}  none")
            continue

        try:
            name = lang.display_name("en") or code
        except Exception:
            name = code

        iso_words: set = set(stopwordsiso.stopwords(alpha2)) if stopwordsiso.has_lang(alpha2) else set()
        yaml_words: set = set(str(w) for w in (yaml_data.get(code) or []))
        n = len(iso_words | yaml_words)
        total += n

        if iso_words and yaml_words:
            source = "stopwordsiso + yaml"
        elif iso_words:
            source = "stopwordsiso"
        elif yaml_words:
            source = "yaml"
        else:
            source = "none"

        print(f"  {code:<6}  {name:<30}  {n:>5}  {source}")

    print(f"  {'':6}  {'':30}  {'-----':>5}")
    print(f"  {'total':<6}  {'':30}  {total:>5}\n")


def _build_eval_stopwords(eval_cfg: Dict, config_path: str, corpus_data=None):
    """Build the stopword set for eval, augmented from training corpus _lang codes.

    Unions ``eval.stopword_languages`` (config list) with language codes found
    in corpus_data records (via ``_lang`` field). Falls back to the English-only
    default when neither source produces any codes.
    """
    from gliner2.training.metrics import _DEFAULT_STOPWORDS

    config_codes: set = set(eval_cfg.get("stopword_languages") or [])

    detected_codes: set = set()
    if corpus_data:
        detected_codes = _collect_lang_codes(corpus_data)
        if detected_codes:
            print(f"[stopwords] Detected {len(detected_codes)} language codes in corpus")

    lang_codes = sorted(config_codes | detected_codes)

    if not lang_codes:
        return _DEFAULT_STOPWORDS

    from gliner2.training.stopwords import build_stopwords

    yaml_name = eval_cfg.get("stopword_yaml", "stopwords.yaml")
    yaml_path = Path(config_path).parent / yaml_name
    extra = yaml_path if yaml_path.exists() else None

    _print_stopword_report(lang_codes, extra)

    return build_stopwords(lang_codes, extra_yaml=extra)


def _print_blind_test(metrics: Dict) -> None:
    """Print the detailed blind-test report followed by a compact micro summary."""
    from gliner2.training.metrics import _print_micro_report

    if not metrics:
        return

    print("\n===== Blind test metrics =====")
    for key in sorted(metrics):
        val = metrics[key]
        if isinstance(val, float):
            print(f"  {key}: {val:.4f}")
        elif isinstance(val, int):
            print(f"  {key}: {val}")

    for category in ("entity", "relation", "classification", "event_type",
                     "event_trigger", "event_argument", "event"):
        for regime in ("strict", "relaxed"):
            report_key = f"eval_{category}_{regime}_classification_report"
            if report_key in metrics:
                print(f"\n--- {category} {regime} classification report ---")
                print(metrics[report_key])

    print("\n===== Blind test summary =====")
    _print_micro_report(metrics)


def _blind_test_by_language(
    best: Path,
    test_data,
    eval_bs: int,
    eval_thr: float,
    chunk_size: int = None,
    chunk_overlap: int = 128,
    global_decode: bool = False,
    global_decode_config=None,
) -> Dict:
    """Run the blind test per language then over all data; return aggregate metrics."""
    from collections import defaultdict
    from gliner2 import AutoExtractor
    from gliner2.training.metrics import compute_metrics, _print_micro_report
    from gliner2.training.trainer import ExtractorDataset

    # Materialise file paths to dicts so we can annotate and filter.
    if test_data and isinstance(test_data[0], str):
        test_data = _read_records(test_data)

    _annotate_languages(test_data)

    gd = dict(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap,
        global_decode=global_decode, global_decode_config=global_decode_config,
    )

    by_lang: Dict[str, List[Dict]] = defaultdict(list)
    for rec in test_data:
        by_lang[rec.get("_lang", "und")].append(rec)

    print(f"\n[blind test] Loading {best} for per-language evaluation...")
    model = AutoExtractor.from_pretrained(str(best))

    per_lang: Dict[str, Dict] = {}
    for lang in sorted(by_lang):
        subset = by_lang[lang]
        print(f"\n[blind test] Processing language: {lang}  ({len(subset)} samples)")
        ds = ExtractorDataset(subset, shuffle=False, validate=False)
        lang_metrics = compute_metrics(model, ds, batch_size=eval_bs, threshold=eval_thr, **gd) or {}
        per_lang[lang] = lang_metrics
        _print_blind_test(lang_metrics)

    print(f"\n[blind test] All languages combined ({len(test_data)} samples)")
    ds_all = ExtractorDataset(test_data, shuffle=False, validate=False)
    all_metrics = compute_metrics(model, ds_all, batch_size=eval_bs, threshold=eval_thr, **gd) or {}
    _print_blind_test(all_metrics)

    print("\n===== Blind test summary by language =====")
    for lang in sorted(per_lang):
        _print_micro_report(per_lang[lang], label=lang)
    _print_micro_report(all_metrics, label="all")

    return all_metrics


def _parse_eval_settings(cfg: Dict, config_path: str, corpus_data, overrides: Dict = None) -> Dict:
    """Resolve the ``eval:`` block into inference settings, applying any CLI
    ``overrides`` (threshold / chunk_size / chunk_overlap / global_decode).
    Shared by training's end-of-run blind test and the standalone eval CLI."""
    overrides = overrides or {}
    eval_cfg = cfg.get("eval") or {}
    chunk_explicit = "chunk_size" in overrides

    chunk_size = overrides.get("chunk_size", eval_cfg.get("chunk_size"))
    chunk_overlap = overrides.get("chunk_overlap", eval_cfg.get("chunk_overlap", 128))
    gd_raw = overrides.get("global_decode", eval_cfg.get("global_decode", False))
    global_decode = bool(gd_raw)
    global_decode_config = None
    if isinstance(gd_raw, dict):
        from gliner2.inference.global_decode import GlobalDecodeConfig
        gd_params = {**gd_raw}
        if "single_filler_roles" in gd_params:
            gd_params["single_filler_roles"] = frozenset(gd_params["single_filler_roles"])
        global_decode_config = GlobalDecodeConfig(**gd_params)
    # global_decode implies chunking; default the window to training max_len,
    # unless the caller explicitly set chunk_size (e.g. --chunk-size 0 = whole-doc).
    if global_decode and chunk_size is None and not chunk_explicit:
        chunk_size = (cfg.get("training") or {}).get("max_len", 384)

    return {
        "batch_size": eval_cfg.get("batch_size", 8),
        "threshold": overrides.get("threshold", eval_cfg.get("threshold", 0.5)),
        "by_language": eval_cfg.get("eval_by_language", False),
        "threshold_sweep": eval_cfg.get("threshold_sweep"),
        "metric_sweep": eval_cfg.get("metric_sweep", False),
        "stopwords": _build_eval_stopwords(eval_cfg, config_path, corpus_data=corpus_data),
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "global_decode": global_decode,
        "global_decode_config": global_decode_config,
    }


def _run_blind_test(best, split_data, batch_size, threshold, by_language, gd_kwargs) -> Dict:
    """Score ``split_data`` against the ``best`` checkpoint (per-language or
    combined) and return the metrics dict. Shared by train and eval."""
    if by_language:
        return _blind_test_by_language(best, split_data, batch_size, threshold, **gd_kwargs)
    # split_data is a list of RECORDS only when label transforms ran; otherwise it
    # is the list of split FILES, and calling those "samples" reads as a corpus of 1.
    unit = "records" if split_data and isinstance(split_data[0], dict) else "files"
    print(f"\n[blind test] Loading {best} and scoring against {len(split_data)} {unit}...")
    metrics = evaluate_checkpoint(best, split_data, batch_size=batch_size, threshold=threshold, **gd_kwargs) or {}
    _print_blind_test(metrics)
    return metrics


def evaluate_config(config_path: str, split: str = "test", checkpoint: str = None,
                    overrides: Dict = None) -> Dict:
    """Load a saved checkpoint for ``config_path`` and score its ``val`` or
    ``test`` split without retraining, writing ``<split>_metrics.json`` next to
    the checkpoint. Reuses the same blind-test path as training."""
    cfg = yaml.safe_load(Path(config_path).read_text())
    data = cfg.get("data") or {}
    suffix = "val" if split == "val" else "test"
    split_data = (_split_files(data.get("corpora") or [], suffix)
                  + _event_split(data.get("event_files") or {}, suffix))
    fns = _category_fns(load_labels_cfg(cfg, config_path))
    if fns:
        split_data = [transform_record(r, fns) for r in _read_records(split_data)]
    if not split_data:
        print(f"[eval] config has no {split} split; nothing to score.")
        return {}

    ev = _parse_eval_settings(cfg, config_path, corpus_data=split_data, overrides=overrides)
    best = Path(checkpoint) if checkpoint else Path(cfg["training"]["output_dir"]) / "best"
    if not best.is_dir():
        raise SystemExit(f"[eval] no checkpoint directory: {best}")

    gd_kwargs = dict(
        chunk_size=ev["chunk_size"], chunk_overlap=ev["chunk_overlap"],
        global_decode=ev["global_decode"], global_decode_config=ev["global_decode_config"],
    )
    # Log what is being scored, for the same reason training does: a <split>_metrics.json
    # records the numbers and nothing about the corpus behind them. Per-field coverage is
    # the part that matters -- a metric computed over a split with no `location` gold says
    # nothing about location, and reads identically to one that does.
    _log_composition(None, split_data if split == "val" else None,
                     split_data if split != "val" else None, streaming=False)
    metrics = _run_blind_test(best, split_data, ev["batch_size"], ev["threshold"], ev["by_language"], gd_kwargs)
    if metrics:
        fname = f"{split}_metrics.json"
        out_dir = Path(cfg["training"]["output_dir"])
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / fname).write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
        (best / fname).write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
        print(f"[eval] wrote {out_dir / fname} and {best / fname}")
    else:
        print("[eval] no metrics produced (empty split?).")
    return metrics


def _log_composition(train_data, eval_data, test_data, streaming: bool) -> None:
    """Print the FINAL composition of each split, just before train() is called.

    The log is the only durable record of what a run actually trained on: configs get
    edited, corpora get regenerated, and a model card weeks later shows metrics without
    the mixture behind them. Two failures this exists to make visible, both of which cost
    real GPU time here:

      * `casualty_docee` carries ZERO location supervision -- 25,154 records, not one
        `location` field. Four dose arms trained on it before anyone noticed, and the
        resulting inability to bind English locations was misread first as Turkish
        diluting English and then as an architecture regression.
      * A structure with no `record_metadata` cannot be decoded on the BOUNDARY path and
        fails SILENTLY -- extraction returns {} with no error.

    Per-FIELD coverage is what surfaces the first; the metadata line surfaces the second.
    Neither is visible in a record count.
    """
    import json as _json
    from collections import Counter, defaultdict

    if streaming:
        print("[composition] streaming train set -- no bounded record list to summarise")

    def summarise(name, data):
        if data and isinstance(data[0], str):
            records = _read_records(data)
        else:
            records = data or []
        if not records:
            print(f"[composition] {name}: empty")
            return
        tasks = Counter()
        struct_n = Counter()
        struct_f = defaultdict(Counter)
        anchors = defaultdict(Counter)
        for r in records:
            out = r.get("output") or {}
            for k in out:
                tasks[k] += 1
            for s in out.get("json_structures") or []:
                for rec_name, fields in (s or {}).items():
                    struct_n[rec_name] += 1
                    for f, v in (fields or {}).items():
                        if v not in (None, ""):
                            struct_f[rec_name][f] += 1
            for rec_name, meta in (out.get("record_metadata") or {}).items():
                anchors[rec_name][f"{meta.get('mode')}/{meta.get('anchor')}"] += 1
        print(f"[composition] {name}: {len(records):,} records | "
              + "  ".join(f"{k}={v:,}" for k, v in tasks.most_common()))
        for rec_name, n in sorted(struct_n.items()):
            fields = struct_f.get(rec_name, {})
            ranked = sorted(fields.items(), key=lambda kv: -kv[1])
            cov = "  ".join(f"{f}={c/n:.0%}" for f, c in ranked[:8])
            if len(ranked) > 8:
                cov += f"  (+{len(ranked) - 8} more)"
            print(f"[composition]   structure {rec_name}: {n:,} records  {cov}")
            zero = [f for f in ("location", "dead", "injured", "missing") if f not in fields]
            if zero and rec_name == "casualty_report":
                print(f"[composition]   WARNING {rec_name}: NO SUPERVISION for "
                      f"{', '.join(zero)} -- the model cannot learn "
                      f"{'them' if len(zero) > 1 else 'it'}")
            if not anchors.get(rec_name):
                print(f"[composition]   WARNING {rec_name}: no record_metadata -- a "
                      f"BOUNDARY model cannot decode this and returns {{}} silently")

    for name, data in (("train", train_data), ("val", eval_data), ("test", test_data)):
        if data is not None:
            summarise(name, data)


def _enforce_split_hygiene(train_data, eval_data, test_data, policy, is_main,
                           *, streaming: bool = False):
    """Drop cross-split contamination and exact duplicates before training.

    Splits arrive as file-path lists; they are materialised here so the check sees
    documents rather than filenames, and the cleaned lists are passed on directly
    (the trainer accepts records as readily as paths).
    """
    from gliner2.training.split_hygiene import check_and_clean

    if policy == "off":
        return train_data, eval_data, test_data

    def materialise(data):
        if not data:
            return data
        return _read_records(data) if isinstance(data[0], str) else list(data)

    # A streaming train source is a lazy iterator with no length; it cannot be
    # materialised, so gate val against test and leave train alone.
    train, val, test, report = check_and_clean(
        None if streaming else materialise(train_data),
        materialise(eval_data), materialise(test_data),
        policy=policy,
    )
    if streaming:
        train = train_data
    if is_main:
        print(report.format())
    return train, val, test


def main(config_path: str) -> None:
    # Under torchrun (DDP) only rank 0 estimates ETA and writes results/blind-test;
    # all ranks run trainer.train(). LOCAL_RANK is unset (-> -1) for single-process.
    is_main = int(os.environ.get("LOCAL_RANK", -1)) <= 0

    cfg = yaml.safe_load(Path(config_path).read_text())

    model = _build_model(cfg["model"])

    # Popped before TrainingConfig: it is a data-pipeline gate, not a trainer field.
    split_hygiene = str((cfg.get("training") or {}).pop("split_hygiene", "drop"))

    config = TrainingConfig(**cfg["training"])

    # DDP auto-detect: torchrun sets LOCAL_RANK per process; copy it into the
    # config so the trainer's distributed path (config.local_rank >= 0) engages.
    # Absent (single-process) -> config keeps its default -1 -> single-device.
    if "LOCAL_RANK" in os.environ:
        config.local_rank = int(os.environ["LOCAL_RANK"])

    data = cfg.get("data") or {}
    corpora = data.get("corpora") or []
    event_files = data.get("event_files") or {}
    train_data = _split_files(corpora, "train") + _event_split(event_files, "train")
    eval_data = _split_files(corpora, "val") + _event_split(event_files, "val")
    test_data = _split_files(corpora, "test") + _event_split(event_files, "test")

    # Optional per-category label transforms, applied identically to train/val/test.
    fns = _category_fns(load_labels_cfg(cfg, config_path))
    if fns:
        train_data = [transform_record(r, fns) for r in _read_records(train_data)]
        eval_data = [transform_record(r, fns) for r in _read_records(eval_data)]
        test_data = [transform_record(r, fns) for r in _read_records(test_data)]
        print(f"[labels] transforms: {', '.join(sorted(fns))}; "
              f"transformed {len(train_data)}/{len(eval_data)}/{len(test_data)} train/val/test records")

    # Streaming source (data.hf_streaming): the train set streams lazily from HF
    # (never written to disk, never fully resident), val/test are bounded in-memory
    # lists capped by label class. Replaces any disk train corpora for this run.
    streaming = bool(data.get("hf_streaming"))
    if streaming:
        if corpora or event_files:
            print("[hf_streaming] note: disk corpora/event_files are ignored for the "
                  "train stream (streaming replaces the train source).")
        train_data, eval_data, test_data = _build_streaming_data(data["hf_streaming"], config)

    # GATE, before a single step: train/val/test must be mutually disjoint.
    # Measured 2026-08-15 these configs were NOT -- joint-boundary-mmbert-137k shared
    # 1,080 documents between train and test (7.03% of the blind set), warmstart-natural
    # 299 (1.95%), every one a corpus leaking into its own splits through row-wise
    # SplitWriter routing. A blind test scored partly on trained documents is not a
    # measurement, so this runs by default; `training.split_hygiene: warn` reproduces a
    # pre-gate run unchanged, and `off` skips it. Placed after the streaming branch
    # because that REPLACES the lists; a streaming train set has no bounded document
    # set to compare, so only its val/test are gated.
    train_data, eval_data, test_data = _enforce_split_hygiene(
        train_data, eval_data, test_data, split_hygiene, is_main, streaming=streaming,
    )

    # Co-locate the training schema on the model so it ships in config.json (best/
    # final checkpoints + HF Hub) and every consumer -- the extractor, the viewer --
    # gets the ontology this model was trained on. Open-vocabulary default, not a
    # limit. Skipped for streaming (no bounded record set to union over).
    if is_main and not streaming and train_data:
        recs = train_data if isinstance(train_data[0], dict) else _read_records(train_data)
        model.config.default_schema = derive_schema(recs) or None
        sch = model.config.default_schema or {}
        ov = sch.get("open_vocab") or []
        print(f"[schema] co-located training schema on the model -> config.json: "
              f"{len(sch.get('entities') or [])} entity types, "
              f"{len(sch.get('events') or {})} event types, "
              f"{len(sch.get('relations') or [])} relations, "
              f"{len(sch.get('classifications') or [])} classification tasks"
              + (f"; open-vocab: {', '.join(ov)}" if ov else ""))

    # Resolve the eval: block (threshold_sweep, windowed chunk_size/chunk_overlap,
    # global_decode, stopwords) into inference settings. See the eval CLI in
    # tools/train/eval.py for scoring a checkpoint without retraining.
    corpus_data = (eval_data + test_data) if streaming else (train_data + eval_data + test_data)
    ev = _parse_eval_settings(cfg, config_path, corpus_data=corpus_data)
    eval_bs = ev["batch_size"]
    eval_thr = ev["threshold"]
    eval_by_language = ev["by_language"]
    threshold_sweep_cfg = ev["threshold_sweep"]
    chunk_size = ev["chunk_size"]
    chunk_overlap = ev["chunk_overlap"]
    global_decode = ev["global_decode"]
    global_decode_config = ev["global_decode_config"]
    eval_stopwords = ev["stopwords"]
    metric_sweep_cfg = ev["metric_sweep"]

    # metric_sweep: sweep the decision threshold each epoch and select the best
    # checkpoint at its own best threshold. Needed when the loss (bce_posweight)
    # shifts the score distribution so a fixed threshold no longer reflects the
    # model's operating point (metric_for_best at 0.5 becomes near-zero noise).
    if metric_sweep_cfg and eval_data:
        sweep_grid = DEFAULT_THRESHOLD_GRID if metric_sweep_cfg is True else list(metric_sweep_cfg)
        print(f"[metric sweep] per-epoch checkpoint selection sweeps threshold over "
              f"{list(sweep_grid)}, maximizing {config.metric_for_best}")
        compute_metrics_hook = make_sweeping_compute_metrics(
            metric_key=config.metric_for_best,
            thresholds=sweep_grid,
            greater_is_better=config.greater_is_better,
            batch_size=eval_bs, stopwords=eval_stopwords,
            chunk_size=chunk_size, chunk_overlap=chunk_overlap,
            global_decode=global_decode, global_decode_config=global_decode_config,
        )
    else:
        compute_metrics_hook = make_compute_metrics(
            batch_size=eval_bs, threshold=eval_thr, stopwords=eval_stopwords,
            chunk_size=chunk_size, chunk_overlap=chunk_overlap,
            global_decode=global_decode, global_decode_config=global_decode_config,
        )

    trainer = GLiNER2Trainer(
        model, config,
        eval_data=eval_data,
        compute_metrics=compute_metrics_hook,
    )
    if is_main and not streaming:
        # ETA counts records up front; a streaming source has no length.
        estimate_eta(model, train_data, config)
    if is_main:
        _log_composition(train_data, eval_data, test_data, streaming)

    results = trainer.train(train_data=train_data)
    # pprint(results)

    # Only rank 0 writes results and runs the blind test; other ranks are done.
    if not is_main:
        return

    results_path = Path(config.output_dir) / "train_results.json"
    results_path.write_text(
        json.dumps(results, indent=2, ensure_ascii=False,
                   default=lambda o: o.to_dict() if hasattr(o, "to_dict") else str(o))
    )
    print(f"[train] Wrote results to {results_path}")

    best = Path(config.output_dir) / "best"
    if not best.is_dir():
        print(f"\n[blind test] No 'best' checkpoint at {best}; skipping.")
        return

    if threshold_sweep_cfg and not eval_data:
        print("\n[threshold sweep] No val data; skipping, keeping threshold="
              f"{eval_thr}.")
    elif threshold_sweep_cfg:
        thresholds = DEFAULT_THRESHOLD_GRID if threshold_sweep_cfg is True else list(threshold_sweep_cfg)
        print(f"\n[threshold sweep] Loading {best} to calibrate against "
              f"{len(eval_data)} val samples over {thresholds}...")
        sweep_model = AutoExtractor.from_pretrained(str(best))
        eval_records = _read_records(eval_data) if eval_data and isinstance(eval_data[0], str) else eval_data
        sweep_ds = ExtractorDataset(eval_records, shuffle=False, validate=False)
        eval_thr, sweep_best_metrics, sweep_all = sweep_thresholds(
            sweep_model, sweep_ds, thresholds=thresholds, batch_size=eval_bs, stopwords=eval_stopwords,
            chunk_size=chunk_size, chunk_overlap=chunk_overlap,
            global_decode=global_decode, global_decode_config=global_decode_config,
        )
        print(f"[threshold sweep] Chose threshold={eval_thr} "
              f"(support-weighted strict micro-F1={_selection_score(sweep_best_metrics):.4f}); "
              f"this only recalibrates the decision cutoff, it does not retrain the model.")
        sweep_path = best / "threshold_sweep.json"
        sweep_path.write_text(json.dumps(
            {"chosen_threshold": eval_thr, "by_threshold": {str(t): m for t, m in sweep_all.items()}},
            indent=2, ensure_ascii=False,
        ))
        print(f"[threshold sweep] Wrote {sweep_path}")
        del sweep_model

    gd_kwargs = dict(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap,
        global_decode=global_decode, global_decode_config=global_decode_config,
    )
    test_metrics = _run_blind_test(best, test_data, eval_bs, eval_thr, eval_by_language, gd_kwargs)

    if test_metrics:
        metrics_path = Path(config.output_dir) / "test_metrics.json"
        metrics_path.write_text(json.dumps(test_metrics, indent=2, ensure_ascii=False))
        best_metrics_path = best / "test_metrics.json"
        best_metrics_path.write_text(json.dumps(test_metrics, indent=2, ensure_ascii=False))
        print(f"\n[blind test] Wrote metrics to {metrics_path} and {best_metrics_path}")
    else:
        print("[blind test] No metrics produced (empty test set?).")

    # ----- model card (best/ exists at this point) -----
    _write_model_card(
        cfg, config, corpora, event_files, results, test_metrics, best,
        threshold=eval_thr, threshold_calibrated=bool(threshold_sweep_cfg),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train GLiNER2 from a YAML config.")
    parser.add_argument("--config", required=True, help="Path to the YAML config file.")
    args = parser.parse_args()
    main(args.config)

"""Evaluation metrics for the GLiNER2 trainer.

Provides a ready-to-use ``compute_metrics`` callable for
:class:`gliner2.training.trainer.GLiNER2Trainer` that scores entities,
relations, and classifications.

The trainer calls ``compute_metrics(model, eval_dataset)`` once per evaluation
pass and merges the returned dict into its own metrics. This module
reconstructs a per-example schema from each gold record, runs
``model.batch_extract`` to get predictions, then tallies per-label TP/FP/FN
and reports:

* ``eval_<category>_micro_{precision,recall,f1}``
* ``eval_<category>_macro_{precision,recall,f1}``
* ``eval_<category>_support`` — total gold count
* ``eval_<category>_classification_report`` — multi-line text table

where ``<category>`` is one of ``entity``, ``relation``, ``classification``.
Categories that don't appear anywhere in the eval set are silently omitted.

Match semantics:

* **Entities** — exact ``(label, surface)`` set match (case-sensitive,
  whitespace-stripped, deduped within a record).
* **Relations** — exact ``(name, head, tail)`` set match.
* **Classifications** — exact ``(task, label)`` set match; multi-label
  predictions are unrolled and scored individually.

Example::

    from gliner2.training import GLiNER2Trainer, TrainingConfig, make_compute_metrics

    trainer = GLiNER2Trainer(
        model, TrainingConfig(...),
        compute_metrics=make_compute_metrics(batch_size=16, threshold=0.5),
    )
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Callable, Dict, Iterable, List, Set, Tuple


def make_compute_metrics(
    batch_size: int = 8,
    threshold: float = 0.5,
) -> Callable[[Any, Any], Dict[str, Any]]:
    """Return a ``compute_metrics`` callable bound to the given inference settings.

    The returned function has the signature ``(model, eval_dataset) -> dict``
    expected by :class:`GLiNER2Trainer`.
    """
    def _hook(model, eval_dataset) -> Dict[str, Any]:
        return compute_metrics(
            model, eval_dataset, batch_size=batch_size, threshold=threshold,
        )
    return _hook


def compute_metrics(
    model,
    eval_dataset,
    batch_size: int = 8,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """Score ``eval_dataset`` and return a flat metrics dict.

    Args:
        model: A loaded :class:`GLiNER2` model (or ``Extractor`` subclass).
        eval_dataset: An indexable dataset where ``eval_dataset[i]`` yields
            ``(text, gold_output_dict)``. The trainer's ``ExtractorDataset``
            satisfies this contract.
        batch_size: Forwarded to ``model.batch_extract``.
        threshold: Forwarded to ``model.batch_extract``.

    Returns:
        Flat ``{metric_name: value}`` dict. Empty if the eval set has no
        records with usable gold structure.
    """
    texts: List[str] = []
    golds: List[Dict] = []
    schemas: List[Dict] = []

    for i in range(len(eval_dataset)):
        text, output = eval_dataset[i]
        if not isinstance(text, str) or not isinstance(output, dict):
            continue
        schema = _schema_from_gold(output)
        if not schema:
            continue
        texts.append(text)
        golds.append(output)
        schemas.append(schema)

    if not texts:
        return {}

    preds = model.batch_extract(
        texts, schemas, batch_size=batch_size, threshold=threshold,
    )

    ent_tp, ent_fp, ent_fn = Counter(), Counter(), Counter()
    rel_tp, rel_fp, rel_fn = Counter(), Counter(), Counter()
    cls_tp, cls_fp, cls_fn = Counter(), Counter(), Counter()
    has_entities = has_relations = has_classifications = False

    for gold, pred in zip(golds, preds):
        g, p = _gold_entity_set(gold), _pred_entity_set(pred)
        if g or p:
            has_entities = True
            _tally(g, p, ent_tp, ent_fp, ent_fn, key=lambda x: x[0])

        g, p = _gold_relation_set(gold), _pred_relation_set(pred)
        if g or p:
            has_relations = True
            _tally(g, p, rel_tp, rel_fp, rel_fn, key=lambda x: x[0])

        g, p = _gold_classification_pairs(gold), _pred_classification_pairs(pred, gold)
        if g or p:
            has_classifications = True
            _tally(g, p, cls_tp, cls_fp, cls_fn, key=lambda x: x[0])

    metrics: Dict[str, Any] = {}
    if has_entities:
        metrics.update(_finalize("entity", ent_tp, ent_fp, ent_fn))
    if has_relations:
        metrics.update(_finalize("relation", rel_tp, rel_fp, rel_fn))
    if has_classifications:
        metrics.update(_finalize("classification", cls_tp, cls_fp, cls_fn))
    return metrics


# ---------------------------------------------------------------------------
# Schema reconstruction
# ---------------------------------------------------------------------------

def _schema_from_gold(output: Dict) -> Dict:
    """Build a schema dict the model can predict against, from a gold record."""
    schema: Dict[str, Any] = {}

    entities = output.get("entities")
    if isinstance(entities, dict) and entities:
        schema["entities"] = {label: "" for label in entities.keys()}

    relations = output.get("relations")
    if isinstance(relations, list) and relations:
        names: List[str] = []
        for rel in relations:
            if isinstance(rel, dict):
                for name in rel.keys():
                    if name not in names:
                        names.append(name)
        if names:
            schema["relations"] = [{name: {"head": "", "tail": ""}} for name in names]

    classifications = output.get("classifications")
    if isinstance(classifications, list) and classifications:
        cls_schema = []
        for c in classifications:
            if not isinstance(c, dict):
                continue
            task = c.get("task")
            labels = c.get("labels")
            if isinstance(task, str) and isinstance(labels, list) and labels:
                cls_schema.append({"task": task, "labels": list(labels), "true_label": ["N/A"]})
        if cls_schema:
            schema["classifications"] = cls_schema

    return schema


# ---------------------------------------------------------------------------
# Gold / prediction extraction
# ---------------------------------------------------------------------------

def _gold_entity_set(output: Dict) -> Set[Tuple[str, str]]:
    out: Set[Tuple[str, str]] = set()
    ents = output.get("entities") or {}
    if not isinstance(ents, dict):
        return out
    for label, surfaces in ents.items():
        if not isinstance(label, str) or not isinstance(surfaces, list):
            continue
        for s in surfaces:
            if isinstance(s, str) and s.strip():
                out.add((label, s.strip()))
    return out


def _pred_entity_set(pred: Dict) -> Set[Tuple[str, str]]:
    out: Set[Tuple[str, str]] = set()
    ents = pred.get("entities") or {}
    if not isinstance(ents, dict):
        return out
    for label, items in ents.items():
        if not isinstance(label, str) or not isinstance(items, list):
            continue
        for item in items:
            text = None
            if isinstance(item, str):
                text = item
            elif isinstance(item, dict):
                text = item.get("text")
            if isinstance(text, str) and text.strip():
                out.add((label, text.strip()))
    return out


def _gold_relation_set(output: Dict) -> Set[Tuple[str, str, str]]:
    out: Set[Tuple[str, str, str]] = set()
    for rel in output.get("relations") or []:
        if not isinstance(rel, dict):
            continue
        for name, fields in rel.items():
            if not isinstance(name, str) or not isinstance(fields, dict):
                continue
            head, tail = fields.get("head"), fields.get("tail")
            if isinstance(head, str) and isinstance(tail, str) and head.strip() and tail.strip():
                out.add((name, head.strip(), tail.strip()))
    return out


def _pred_relation_set(pred: Dict) -> Set[Tuple[str, str, str]]:
    out: Set[Tuple[str, str, str]] = set()
    block = pred.get("relation_extraction") or {}
    if not isinstance(block, dict):
        return out
    for name, items in block.items():
        if not isinstance(name, str) or not isinstance(items, list):
            continue
        for item in items:
            if isinstance(item, dict):
                head, tail = item.get("head"), item.get("tail")
                if isinstance(head, str) and isinstance(tail, str) and head.strip() and tail.strip():
                    out.add((name, head.strip(), tail.strip()))
    return out


def _gold_classification_pairs(output: Dict) -> Set[Tuple[str, str]]:
    out: Set[Tuple[str, str]] = set()
    for c in output.get("classifications") or []:
        if not isinstance(c, dict):
            continue
        task = c.get("task")
        true_label = c.get("true_label")
        if not isinstance(task, str):
            continue
        if isinstance(true_label, str):
            if true_label.strip():
                out.add((task, true_label.strip()))
        elif isinstance(true_label, list):
            for lbl in true_label:
                if isinstance(lbl, str) and lbl.strip():
                    out.add((task, lbl.strip()))
    return out


def _pred_classification_pairs(pred: Dict, gold: Dict) -> Set[Tuple[str, str]]:
    """For each gold task, read pred[task] (str or list) and emit (task, label)."""
    out: Set[Tuple[str, str]] = set()
    tasks: Set[str] = set()
    for c in gold.get("classifications") or []:
        if isinstance(c, dict) and isinstance(c.get("task"), str):
            tasks.add(c["task"])
    for task in tasks:
        v = pred.get(task)
        if v is None:
            continue
        if isinstance(v, str):
            if v.strip():
                out.add((task, v.strip()))
        elif isinstance(v, list):
            for item in v:
                if isinstance(item, str) and item.strip():
                    out.add((task, item.strip()))
                elif isinstance(item, dict) and isinstance(item.get("label"), str):
                    out.add((task, item["label"].strip()))
        elif isinstance(v, dict) and isinstance(v.get("label"), str):
            out.add((task, v["label"].strip()))
    return out


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _tally(
    gold: Iterable,
    pred: Iterable,
    tp: Counter,
    fp: Counter,
    fn: Counter,
    key: Callable[[Tuple], str],
) -> None:
    g, p = set(gold), set(pred)
    for item in g & p:
        tp[key(item)] += 1
    for item in p - g:
        fp[key(item)] += 1
    for item in g - p:
        fn[key(item)] += 1


def _pr_f1(tp_n: int, fp_n: int, fn_n: int) -> Tuple[float, float, float]:
    p = tp_n / (tp_n + fp_n) if (tp_n + fp_n) > 0 else 0.0
    r = tp_n / (tp_n + fn_n) if (tp_n + fn_n) > 0 else 0.0
    f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return p, r, f


def evaluate_checkpoint(
    checkpoint_dir,
    test_data,
    batch_size: int = 8,
    threshold: float = 0.5,
    map_location: str = None,
) -> Dict[str, Any]:
    """Load a saved GLiNER2 checkpoint and run :func:`compute_metrics` on a test set.

    Use this for a blind final-test pass after ``trainer.train()`` returns,
    typically pointing at ``out/<run>/best``.

    Args:
        checkpoint_dir: Local path (or HF repo id) of a saved GLiNER2 checkpoint.
        test_data: Anything :class:`ExtractorDataset` accepts (JSONL paths, list
            of records, list of ``InputExample``). Materialised here so callers
            don't have to build the dataset themselves.
        batch_size: Inference batch size.
        threshold: Confidence threshold for prediction.
        map_location: Forwarded to ``GLiNER2.from_pretrained``.

    Returns:
        Same metric dict as :func:`compute_metrics`, with keys namespaced
        under ``eval_<category>_…``. Empty if the test set has no records
        with usable gold structure.
    """
    # Imported here so this module stays importable without a model checkout.
    from gliner2 import GLiNER2
    from gliner2.training.trainer import ExtractorDataset

    model = GLiNER2.from_pretrained(str(checkpoint_dir), map_location=map_location)
    dataset = ExtractorDataset(test_data, shuffle=False, validate=False)
    return compute_metrics(
        model, dataset, batch_size=batch_size, threshold=threshold,
    )


def _finalize(prefix: str, tp: Counter, fp: Counter, fn: Counter) -> Dict[str, Any]:
    labels = sorted(set(tp) | set(fp) | set(fn))
    if not labels:
        return {}

    total_tp = sum(tp.values())
    total_fp = sum(fp.values())
    total_fn = sum(fn.values())
    micro_p, micro_r, micro_f = _pr_f1(total_tp, total_fp, total_fn)

    per_label = []
    macro_p_sum = macro_r_sum = macro_f_sum = 0.0
    for lbl in labels:
        p, r, f = _pr_f1(tp[lbl], fp[lbl], fn[lbl])
        support = tp[lbl] + fn[lbl]
        per_label.append((lbl, p, r, f, support))
        macro_p_sum += p
        macro_r_sum += r
        macro_f_sum += f
    n = len(labels)
    macro_p = macro_p_sum / n
    macro_r = macro_r_sum / n
    macro_f = macro_f_sum / n

    lines = []
    lines.append(f"{'label':<40} {'precision':>10} {'recall':>10} {'f1':>10} {'support':>10}")
    lines.append("-" * 82)
    for lbl, p, r, f, sup in per_label:
        display = lbl if len(lbl) <= 40 else lbl[:38] + ".."
        lines.append(f"{display:<40} {p:>10.4f} {r:>10.4f} {f:>10.4f} {sup:>10d}")
    lines.append("-" * 82)
    overall_support = total_tp + total_fn
    lines.append(f"{'micro avg':<40} {micro_p:>10.4f} {micro_r:>10.4f} {micro_f:>10.4f} {overall_support:>10d}")
    lines.append(f"{'macro avg':<40} {macro_p:>10.4f} {macro_r:>10.4f} {macro_f:>10.4f} {overall_support:>10d}")
    report = "\n".join(lines)

    return {
        f"eval_{prefix}_micro_precision": micro_p,
        f"eval_{prefix}_micro_recall": micro_r,
        f"eval_{prefix}_micro_f1": micro_f,
        f"eval_{prefix}_macro_precision": macro_p,
        f"eval_{prefix}_macro_recall": macro_r,
        f"eval_{prefix}_macro_f1": macro_f,
        f"eval_{prefix}_support": overall_support,
        f"eval_{prefix}_classification_report": report,
    }

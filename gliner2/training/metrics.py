"""Evaluation metrics for the GLiNER2 trainer.

Provides a ready-to-use ``compute_metrics`` callable for
:class:`gliner2.training.trainer.GLiNER2Trainer` that scores entities,
relations, and classifications.

The trainer calls ``compute_metrics(model, eval_dataset)`` once per evaluation
pass and merges the returned dict into its own metrics. This module
reconstructs a per-example schema from each gold record, runs
``model.batch_extract`` to get predictions, then tallies per-label TP/FP/FN
and reports, for both a ``strict`` and a ``relaxed`` regime:

* ``eval_<category>_<regime>_micro_{precision,recall,f1}``
* ``eval_<category>_<regime>_macro_{precision,recall,f1}``
* ``eval_<category>_<regime>_support`` — total gold count
* ``eval_<category>_<regime>_classification_report`` — multi-line text table

where ``<category>`` is one of ``entity``, ``relation``, ``classification``,
``event_type``, ``event_trigger``, ``event_argument``, ``event`` and
``<regime>`` is ``strict`` or ``relaxed``. Categories absent from the eval set
are silently omitted.

Match semantics — **strict** is exact, **relaxed** keeps the discrete
type/label parts exact but lets surfaces partially overlap (one-to-one matched,
so relaxed never scores below strict):

* **Entities** — strict ``(label, surface)``; relaxed = label exact + surface
  overlap. (case-sensitive/exact for strict; normalized + overlap for relaxed.)
* **Relations** — strict ``(name, head, tail)``; relaxed = name exact + head and
  tail surfaces overlap.
* **Classifications** — strict ``(task, label)``; relaxed = task exact + label
  overlap. Multi-label predictions are unrolled and scored individually.
* **Event types** — ``(event_type,)`` presence. There is no surface to relax,
  so strict == relaxed.
* **Event triggers** — strict ``(event_type, trigger)``; relaxed = event_type
  exact + trigger surface overlap (consistent with entity/relation relaxed).
* **Event arguments** — strict ``(event_type, role, entity, trigger)``; relaxed =
  ``(event_type, role)`` exact + entity overlap, dropping the trigger link.
* **Event (overall)** — one combined score over event types + triggers +
  arguments: their TP/FP/FN are summed (micro is the aggregate). Per-label rows
  are namespaced ``type:``/``trigger:``/``arg:`` so the report stays honest; the
  event type is counted in all three buckets at their respective granularities.

"Overlap" means substring containment or a shared non-stopword token after
lowercasing/whitespace-normalization.

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

    # strict (exact) and relaxed (partial-overlap) TP/FP/FN per category.
    ent_s, ent_r = _counters(), _counters()
    rel_s, rel_r = _counters(), _counters()
    cls_s, cls_r = _counters(), _counters()
    ety_s, ety_r = _counters(), _counters()
    et_s, et_r = _counters(), _counters()
    ea_s, ea_r = _counters(), _counters()
    has_entities = has_relations = has_classifications = False
    has_event_types = has_event_triggers = has_event_arguments = False

    for gold, pred in zip(golds, preds):
        g, p = _gold_entity_set(gold), _pred_entity_set(pred)
        if g or p:
            has_entities = True
            _tally(g, p, *ent_s, key=lambda x: x[0])
            _match_relaxed(_items_entity(g), _items_entity(p), *ent_r)

        g, p = _gold_relation_set(gold), _pred_relation_set(pred)
        if g or p:
            has_relations = True
            _tally(g, p, *rel_s, key=lambda x: x[0])
            _match_relaxed(_items_relation(g), _items_relation(p), *rel_r)

        g, p = _gold_classification_pairs(gold), _pred_classification_pairs(pred, gold)
        if g or p:
            has_classifications = True
            _tally(g, p, *cls_s, key=lambda x: x[0])
            _match_relaxed(_items_classification(g), _items_classification(p), *cls_r)

        # Events — score type detection, triggers, and arguments separately.
        # Relaxed drops the trigger link (trigger = event_type presence;
        # argument = type/role/entity).
        g_ety, p_ety = _gold_event_type_set(gold), _pred_event_type_set(pred)
        if g_ety or p_ety:
            has_event_types = True
            _tally(g_ety, p_ety, *ety_s, key=lambda x: x[0])
            _match_relaxed(_items_event_type(g_ety), _items_event_type(p_ety), *ety_r)

        g_trig, p_trig = _gold_event_trigger_set(gold), _pred_event_trigger_set(pred)
        if g_trig or p_trig:
            has_event_triggers = True
            _tally(g_trig, p_trig, *et_s, key=lambda x: x[0])
            _match_relaxed(_items_trigger(g_trig), _items_trigger(p_trig), *et_r)

        g_arg, p_arg = _gold_event_argument_set(gold), _pred_event_argument_set(pred)
        if g_arg or p_arg:
            has_event_arguments = True
            _tally(g_arg, p_arg, *ea_s, key=lambda x: x[1])
            _match_relaxed(_items_argument(g_arg), _items_argument(p_arg), *ea_r)

    # Overall event score: sum the type, trigger, and argument counters
    # (namespaced so the per-label report keeps distinct rows; micro is the
    # aggregate over all three).
    evt_s = _combine_counters({"type": ety_s, "trigger": et_s, "arg": ea_s})
    evt_r = _combine_counters({"type": ety_r, "trigger": et_r, "arg": ea_r})
    has_events = has_event_types or has_event_triggers or has_event_arguments

    metrics: Dict[str, Any] = {}
    for present, prefix, strict, relaxed in (
        (has_entities, "entity", ent_s, ent_r),
        (has_relations, "relation", rel_s, rel_r),
        (has_classifications, "classification", cls_s, cls_r),
        (has_event_types, "event_type", ety_s, ety_r),
        (has_event_triggers, "event_trigger", et_s, et_r),
        (has_event_arguments, "event_argument", ea_s, ea_r),
        (has_events, "event", evt_s, evt_r),
    ):
        if present:
            metrics.update(_finalize(prefix, "strict", *strict))
            metrics.update(_finalize(prefix, "relaxed", *relaxed))
    _print_micro_report(metrics)
    return metrics


def _print_micro_report(metrics: Dict[str, Any]) -> None:
    """Print a compact micro precision/recall/F1 line per category, strict -> relaxed."""
    categories = ("entity", "relation", "classification", "event_type",
                  "event_trigger", "event_argument", "event")
    present = [c for c in categories if f"eval_{c}_strict_micro_f1" in metrics]
    if not present:
        return

    def val(cat: str, regime: str, metric: str) -> float:
        return metrics.get(f"eval_{cat}_{regime}_micro_{metric}", 0.0)

    print("\n[eval] micro precision / recall / f1  (strict -> relaxed)")
    for c in present:
        print(f"  {c:<15} "
              f"P={val(c, 'strict', 'precision'):.4f}->{val(c, 'relaxed', 'precision'):.4f}  "
              f"R={val(c, 'strict', 'recall'):.4f}->{val(c, 'relaxed', 'recall'):.4f}  "
              f"F1={val(c, 'strict', 'f1'):.4f}->{val(c, 'relaxed', 'f1'):.4f}")


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

    # Events: derive {event_type: [role, ...]} from the union of gold mentions.
    events = output.get("events")
    if isinstance(events, list) and events:
        events_schema: Dict[str, List[str]] = {}
        for ev in events:
            if not isinstance(ev, dict):
                continue
            etype = ev.get("event_type")
            if not isinstance(etype, str) or not etype.strip():
                continue
            roles = events_schema.setdefault(etype, [])
            for arg in ev.get("arguments") or []:
                if not isinstance(arg, dict):
                    continue
                role = arg.get("role")
                if isinstance(role, str) and role.strip() and role not in roles:
                    roles.append(role)
        # Drop event types that ended up role-less; the schema needs ≥1 role.
        events_schema = {k: v for k, v in events_schema.items() if v}
        if events_schema:
            schema["events"] = events_schema

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


def _rel_endpoint(v) -> str:
    """Coerce a relation head/tail to its surface string.

    The inference engine emits relation instances in several shapes depending on
    include_spans/include_confidence: a plain surface string, or a ``{"text": ...}``
    dict (with spans/confidence). Returns the stripped surface, or None.
    """
    if isinstance(v, dict):
        v = v.get("text")
    return v.strip() if isinstance(v, str) and v.strip() else None


def _pred_relation_set(pred: Dict) -> Set[Tuple[str, str, str]]:
    out: Set[Tuple[str, str, str]] = set()
    block = pred.get("relation_extraction") or {}
    if not isinstance(block, dict):
        return out
    for name, items in block.items():
        if not isinstance(name, str) or not isinstance(items, list):
            continue
        for item in items:
            # The engine's default output is a (head, tail) tuple; with
            # include_spans/confidence it's a {"head": ..., "tail": ...} dict.
            if isinstance(item, dict):
                head, tail = _rel_endpoint(item.get("head")), _rel_endpoint(item.get("tail"))
            elif isinstance(item, (tuple, list)) and len(item) == 2:
                head, tail = _rel_endpoint(item[0]), _rel_endpoint(item[1])
            else:
                head = tail = None
            if head and tail:
                out.add((name, head, tail))
    return out


def _gold_event_trigger_set(output: Dict) -> Set[Tuple[str, str]]:
    """Set of ``(event_type, trigger_surface)`` over the gold events."""
    out: Set[Tuple[str, str]] = set()
    events = output.get("events") or []
    if not isinstance(events, list):
        return out
    for ev in events:
        if not isinstance(ev, dict):
            continue
        etype = ev.get("event_type")
        trigger = ev.get("trigger")
        if isinstance(etype, str) and isinstance(trigger, str):
            etype = etype.strip()
            trigger = trigger.strip()
            if etype and trigger:
                out.add((etype, trigger))
    return out


def _pred_event_trigger_set(pred: Dict) -> Set[Tuple[str, str]]:
    """Same shape as the gold trigger set, sourced from the ``event_extraction`` block."""
    out: Set[Tuple[str, str]] = set()
    block = pred.get("event_extraction") or {}
    if not isinstance(block, dict):
        return out
    for etype, mentions in block.items():
        if not isinstance(etype, str) or not isinstance(mentions, list):
            continue
        for ev in mentions:
            if not isinstance(ev, dict):
                continue
            trigger = ev.get("trigger")
            if isinstance(trigger, dict):
                trigger = trigger.get("text")
            if isinstance(trigger, str) and trigger.strip():
                out.add((etype, trigger.strip()))
    return out


def _gold_event_type_set(output: Dict) -> Set[Tuple[str]]:
    """Set of ``(event_type,)`` over gold events — event-type presence."""
    return {(et,) for et, _trig in _gold_event_trigger_set(output)}


def _pred_event_type_set(pred: Dict) -> Set[Tuple[str]]:
    """Set of ``(event_type,)`` over predicted events — event-type presence."""
    return {(et,) for et, _trig in _pred_event_trigger_set(pred)}


def _gold_event_argument_set(output: Dict) -> Set[Tuple[str, str, str, str]]:
    """Set of ``(event_type, role, entity, trigger)`` over gold arguments.

    Including the trigger lets the metric distinguish identical (type, role,
    entity) tuples that come from different event mentions in the same text.
    Aggregation uses the role as the per-label key.
    """
    out: Set[Tuple[str, str, str, str]] = set()
    events = output.get("events") or []
    if not isinstance(events, list):
        return out
    for ev in events:
        if not isinstance(ev, dict):
            continue
        etype = ev.get("event_type")
        trigger = ev.get("trigger")
        if not isinstance(etype, str) or not isinstance(trigger, str):
            continue
        etype, trigger = etype.strip(), trigger.strip()
        if not etype or not trigger:
            continue
        for arg in ev.get("arguments") or []:
            if not isinstance(arg, dict):
                continue
            role = arg.get("role")
            entity = arg.get("entity")
            if not isinstance(role, str) or not isinstance(entity, str):
                continue
            role, entity = role.strip(), entity.strip()
            if role and entity:
                out.add((etype, role, entity, trigger))
    return out


def _pred_event_argument_set(pred: Dict) -> Set[Tuple[str, str, str, str]]:
    """Same shape as the gold argument set, sourced from ``event_extraction``."""
    out: Set[Tuple[str, str, str, str]] = set()
    block = pred.get("event_extraction") or {}
    if not isinstance(block, dict):
        return out
    for etype, mentions in block.items():
        if not isinstance(etype, str) or not isinstance(mentions, list):
            continue
        for ev in mentions:
            if not isinstance(ev, dict):
                continue
            trigger = ev.get("trigger")
            if isinstance(trigger, dict):
                trigger = trigger.get("text")
            if not isinstance(trigger, str) or not trigger.strip():
                continue
            trigger = trigger.strip()
            for arg in ev.get("arguments") or []:
                if not isinstance(arg, dict):
                    continue
                role = arg.get("role")
                entity = arg.get("entity")
                if isinstance(entity, dict):
                    entity = entity.get("text")
                if not isinstance(role, str) or not isinstance(entity, str):
                    continue
                role, entity = role.strip(), entity.strip()
                if role and entity:
                    out.add((etype, role, entity, trigger))
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


# ---------------------------------------------------------------------------
# Relaxed (partial-overlap) matching
# ---------------------------------------------------------------------------

_STOPWORDS = {"the", "a", "an", "of", "in", "on", "at", "to", "for", "and",
              "or", "by", "with", "from", "as", "is", "are", "was", "were"}


def _counters() -> Tuple[Counter, Counter, Counter]:
    return Counter(), Counter(), Counter()


def _combine_counters(
    parts: Dict[str, Tuple[Counter, Counter, Counter]],
) -> Tuple[Counter, Counter, Counter]:
    """Sum several ``(tp, fp, fn)`` counter-triples into one.

    Each source's per-label keys are namespaced (``type:``/``trigger:``/``arg:``)
    so the merged report keeps distinct rows and never silently folds an
    event-type row into a same-named trigger row. Micro totals are unaffected.
    """
    tp, fp, fn = Counter(), Counter(), Counter()
    for ns, (t, f, n) in parts.items():
        for src, dst in ((t, tp), (f, fp), (n, fn)):
            for k, v in src.items():
                dst[f"{ns}:{k}"] += v
    return tp, fp, fn


def _normalize(s: str) -> str:
    """Lowercase and collapse whitespace."""
    return " ".join(s.lower().split())


def _overlap(a: str, b: str) -> bool:
    """Partial-surface match: substring containment, or a shared non-stopword
    token (length >= 2), so common words like 'the'/'of' don't create matches."""
    na, nb = _normalize(a), _normalize(b)
    if not na or not nb:
        return False
    if na == nb or na in nb or nb in na:
        return True
    ta = {t for t in na.split() if len(t) >= 2 and t not in _STOPWORDS}
    tb = {t for t in nb.split() if len(t) >= 2 and t not in _STOPWORDS}
    return bool(ta & tb)


def _match_relaxed(gold_items, pred_items, tp: Counter, fp: Counter, fn: Counter) -> None:
    """One-to-one relaxed match, accumulating per-key TP/FP/FN.

    Each item is ``(discrete, surfaces, key)``: ``discrete`` (type/label parts)
    must match exactly; each ``surfaces`` component need only overlap. Two passes
    guarantee relaxed dominates strict: pass 1 pairs normalized-equal surfaces (a
    superset of the strict exact matches), pass 2 pairs the leftovers by overlap.
    Items with empty ``surfaces`` (e.g. relaxed triggers) match on ``discrete``
    alone. Inputs are pre-sorted by the caller for deterministic greedy order.
    """
    g_used = [False] * len(gold_items)
    p_done = [False] * len(pred_items)

    def run(match) -> None:
        for pi, (pd, ps, pk) in enumerate(pred_items):
            if p_done[pi]:
                continue
            for gi, (gd, gs, _gk) in enumerate(gold_items):
                if g_used[gi] or pd != gd or len(ps) != len(gs):
                    continue
                if all(match(x, y) for x, y in zip(ps, gs)):
                    g_used[gi] = p_done[pi] = True
                    tp[pk] += 1
                    break

    run(lambda x, y: _normalize(x) == _normalize(y))  # exact (normalized) first
    run(_overlap)                                      # then partial overlap

    for pi, (_pd, _ps, pk) in enumerate(pred_items):
        if not p_done[pi]:
            fp[pk] += 1
    for gi, (_gd, _gs, gk) in enumerate(gold_items):
        if not g_used[gi]:
            fn[gk] += 1


def _items_entity(s):
    return sorted(((label,), (surface,), label) for label, surface in s)


def _items_relation(s):
    return sorted(((name,), (head, tail), name) for name, head, tail in s)


def _items_classification(s):
    return sorted(((task,), (label,), task) for task, label in s)


def _items_event_type(s):
    # event-type presence; no surface, so relaxed collapses to strict
    return sorted(((et,), (), et) for (et,) in s)


def _items_trigger(s):
    # relaxed trigger = event_type exact + trigger surface overlap
    return sorted(((et,), (trigger,), et) for et, trigger in s)


def _items_argument(s):
    # relaxed argument = (event_type, role, entity), dropping the trigger link
    triples = {(et, role, ent) for et, role, ent, _trig in s}
    return sorted(((et, role), (ent,), role) for et, role, ent in triples)


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


def _finalize(prefix: str, regime: str, tp: Counter, fp: Counter, fn: Counter) -> Dict[str, Any]:
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
        f"eval_{prefix}_{regime}_micro_precision": micro_p,
        f"eval_{prefix}_{regime}_micro_recall": micro_r,
        f"eval_{prefix}_{regime}_micro_f1": micro_f,
        f"eval_{prefix}_{regime}_macro_precision": macro_p,
        f"eval_{prefix}_{regime}_macro_recall": macro_r,
        f"eval_{prefix}_{regime}_macro_f1": macro_f,
        f"eval_{prefix}_{regime}_support": overall_support,
        f"eval_{prefix}_{regime}_classification_report": report,
    }

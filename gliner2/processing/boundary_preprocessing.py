"""Build boundary query layouts and targets from transformed processor records.

Queries are enumerated in *transformed group order* — every extractive schema
child (``[E]``/``[C]``/``[R]`` marker) becomes one boundary query, in the same
order the boundary model's ``encode()`` builds its query states. Classification
groups contribute no extractive queries (they are scored by the shared
classifier). This ordering is deliberate so ``batch.targets`` aligns 1:1 with
``query_states`` even when the processor shuffles task order during training.
"""

from __future__ import annotations

from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

import torch

from gliner2.models.base import QueryLayout, QuerySpec
from gliner2.processing.records import (
    RecordFieldSpec,
    RecordSpec,
    compile_record_specs,
)
from gliner2.processing.targets import (
    MentionTarget,
    RecordFieldTarget,
    RecordTarget,
    TargetGraph,
    inclusive_tokens_to_boundary_pair,
    pad_target_graphs,
)
from collections import Counter

from gliner2.processing.layouts import validate_target_graph

# Every marker the processor emits a QUERY MARKER for must appear here, or the layout
# under-counts queries while `query_states`/`query_mask` still carry one entry per
# marker -- and gold injection then compares a [B,Q_gold,G] mask against a [B,Q_marker]
# one and dies. "[V]" (event roles: trigger + arguments) was missing, so every event
# group produced ZERO layout queries; `inference/runtime.py` already listed all four.
_EXTRACTIVE_MARKERS = ("[E]", "[C]", "[R]", "[V]")

# Unalignable entity surfaces skipped under on_missing_surface="skip", by field name.
# Counted rather than silently dropped so the loss is reportable; read it with
# `missing_surface_counts()`.
_MISSING_SURFACES: "Counter[str]" = Counter()


def missing_surface_counts() -> "Counter[str]":
    """Entity fields whose surfaces could not be aligned, and how often."""
    return _MISSING_SURFACES


# Entity labels whose query was SUPPRESSED because every one of their gold surfaces
# failed to align, by label name. Counted so the suppression is reportable rather than
# another silent change of supervision.
_SUPPRESSED_QUERIES: "Counter[str]" = Counter()


def suppressed_query_counts() -> "Counter[str]":
    """Entity labels dropped from a document's menu, and how often."""
    return _SUPPRESSED_QUERIES


def _orphaned_entity_fields(fields, labels) -> set:
    """Field indices whose gold is non-empty but NONE of it aligns.

    Such a label is offered to the model with nothing to find, so it trains as ABSENT
    while the entity is in the text -- a false negative the corpus manufactured by
    accident (measured at 1.41% of label queries). An EMPTY gold list is a different
    thing entirely and is never orphaned: that is an INTENTIONALLY injected negative
    (``{label: []}``, the shape `NegativeLabels.inject` writes), which must survive.
    """
    if not labels or labels[0] == 0 or len(labels) < 2:
        return set()
    seen, aligned = set(), set()
    for instance in labels[1]:
        for field_index, positions in enumerate(instance):
            if field_index >= len(fields) or not positions:
                continue
            seen.add(field_index)
            for pos in positions:
                if tuple(pos) != (-1, -1):
                    aligned.add(field_index)
                    break
    return seen - aligned


def _dedupe_contained_spans(
    pairs: Sequence[Tuple[int, int]],
) -> List[Tuple[int, int]]:
    """Keep only the maximal spans among ``pairs``, dropping contained ones.

    ``pairs`` are half-open ``(start, end)`` token spans that all belong to one
    query in one sample. Independently re-searching each listed value's tokens
    (see ``Processor._build_outputs``) can match a short value's tokens as the
    leading/trailing tokens of an unrelated, longer value's own, separately
    verified match -- e.g. a bare listed value ``"Kiox"`` matching inside every
    ``"Kiox 300"``/``"Kiox 400C"`` occurrence of a longer, distinct listed value
    under the *same* query. Those short matches are sub-spans of a different,
    real mention, not standalone occurrences, so they are phantom gold.

    This applies the standard "longest match, non-overlapping" span-resolution
    policy to the *reconstructed* positions rather than assuming the raw
    per-value search results are already clean: drop ``(s1, e1)`` iff a
    *different* pair ``(s2, e2)`` in the same list satisfies
    ``s2 <= s1 and e1 <= e2``. Exact duplicates collapse to one instance
    first (containment requires a genuinely different, larger pair). Ordinary
    partial/crossing overlaps that are not containment (e.g. ``(0, 3)`` and
    ``(2, 5)``) are left untouched -- this is deliberately narrow and only
    resolves the strict sub-span case described above.
    """
    ordered = list(dict.fromkeys(pairs))
    kept = []
    for i, (s1, e1) in enumerate(ordered):
        is_contained = any(
            s2 <= s1 and e1 <= e2 and (s2, e2) != (s1, e1)
            for j, (s2, e2) in enumerate(ordered)
            if j != i
        )
        if not is_contained:
            kept.append((s1, e1))
    return kept


def _unalignable_entity_error(
    *,
    field_name: str,
    sample_idx: int,
    original_texts: Optional[Sequence[str]],
    original_schemas: Optional[Sequence[Optional[Mapping[str, Any]]]],
) -> ValueError:
    """Build an actionable ``ValueError`` for an entity that failed to token-align.

    ``sample_idx`` is the batch-relative index inside this collated batch, not
    a row/line number in the caller's source dataset -- this function has no
    visibility into dataset ordering, and the DataLoader may have shuffled.
    When the caller supplies ``original_texts``/``original_schemas`` (as
    ``SchemaTransformer._add_boundary_metadata`` does), the message includes
    the literal listed value(s) for the failing entity type and a text
    snippet, so a caller does not have to reverse-engineer the tokenizer from
    scratch to find the offending annotation.
    """
    listed_values: Any = None
    if original_schemas is not None and 0 <= sample_idx < len(original_schemas):
        schema = original_schemas[sample_idx]
        if isinstance(schema, Mapping):
            listed_values = schema.get("entities", {}).get(field_name)

    snippet = None
    if original_texts is not None and 0 <= sample_idx < len(original_texts):
        text = original_texts[sample_idx]
        snippet = text if len(text) <= 240 else f"{text[:240]}\u2026"

    detail = [
        f"entity {field_name!r} was not found in sample {sample_idx} "
        "(a batch-relative index into this collated batch, not a row/line "
        "number in your source dataset)."
    ]
    if listed_values is not None:
        detail.append(f"Listed value(s) for {field_name!r}: {listed_values!r}.")
    if snippet is not None:
        detail.append(f"Sample text: {snippet!r}.")
    detail.append(
        "A listed value can be an exact character substring of the text yet still "
        "fail this check: the tokenizer (WhitespaceTokenSplitter) merges a "
        "hyphen/underscore-joined run (e.g. 'ledremote-abs-bes3-my2023') into one "
        "token, so a standalone-token search for a sub-part of that run (e.g. 'abs') "
        "finds nothing even though it is a literal substring. Call "
        "SchemaTransformer.find_unalignable_entities(text, entities) on this "
        "(text, schema) example before training to check alignment ahead of time."
    )
    return ValueError(" ".join(detail))


def _extractive_fields(schema_tokens: Sequence[str]) -> list[str]:
    return [
        str(schema_tokens[i + 1])
        for i, token in enumerate(schema_tokens[:-1])
        if token in _EXTRACTIVE_MARKERS
    ]


def _group_name(schema_tokens: Sequence[str], task_type: str) -> str:
    if task_type == "entities":
        return "entities"
    if len(schema_tokens) > 2:
        return str(schema_tokens[2]).split(" [DESCRIPTION] ")[0]
    return task_type


def _iter_inclusive_spans(value: Any) -> Iterator[tuple[int, int]]:
    """Yield inclusive ``(start, end)`` spans from a structure-label field.

    Field values are lists of ``(start, end)`` tuples (possibly nested for choice
    fields); ``(-1, -1)`` marks "not found" and is skipped.
    """
    if value is None:
        return
    if (
        isinstance(value, tuple)
        and len(value) == 2
        and all(isinstance(x, int) for x in value)
    ):
        if value != (-1, -1):
            yield value  # type: ignore[misc]
        return
    if isinstance(value, (list, tuple)):
        for sub in value:
            yield from _iter_inclusive_spans(sub)


def _resolve_field_spans(
    raw_value: Any, text_length: int
) -> List[Tuple[int, int]]:
    """Half-open in-range spans from a structure-label field value."""
    spans: List[Tuple[int, int]] = []
    for (s, e_inc) in _iter_inclusive_spans(raw_value):
        if 0 <= s <= e_inc < text_length:
            spans.append(inclusive_tokens_to_boundary_pair(s, e_inc))
    return spans


def _apply_occurrence_policy(
    spans: List[Tuple[int, int]],
    policy: str,
    fspec: RecordFieldSpec,
) -> List[Tuple[int, int]]:
    if not spans:
        return spans
    if not fspec.cardinality.is_scalar:
        # List fields keep every detected value; policy governs scalar surfaces.
        return spans
    if policy == "first":
        return [spans[0]]
    if policy == "error_on_ambiguous" and len(spans) > 1:
        raise ValueError(
            f"ambiguous scalar field {fspec.name!r}: {len(spans)} occurrences; "
            "provide explicit offsets or use occurrence_policy='latent_all'/'first'."
        )
    return spans


def _build_record_field_target(
    fspec: RecordFieldSpec,
    instance: Sequence[Any],
    text_length: int,
    policy: str,
) -> RecordFieldTarget:
    raw = instance[fspec.role_index] if fspec.role_index < len(instance) else None
    spans = _apply_occurrence_policy(
        _resolve_field_spans(raw, text_length), policy, fspec
    )
    if fspec.cardinality.is_scalar:
        values: Tuple[Tuple[Tuple[int, int], ...], ...] = (
            (tuple(spans),) if spans else ()
        )
    else:
        values = tuple((sp,) for sp in spans)
    return RecordFieldTarget(query_id=fspec.query_id, values=values)


def _build_sample_records(
    specs: Mapping[int, RecordSpec],
    sample_labels: Sequence[Any],
    text_length: int,
) -> List[RecordTarget]:
    records: List[RecordTarget] = []
    for task_index, spec in specs.items():
        if task_index >= len(sample_labels):
            continue
        labels = sample_labels[task_index]
        if not labels or labels[0] == 0:
            continue
        _, instances = labels
        for inst_idx, instance in enumerate(instances):
            fields_t = [
                _build_record_field_target(
                    fspec, instance, text_length, spec.occurrence_policy
                )
                for fspec in spec.fields
            ]
            records.append(
                RecordTarget(
                    instance_id=f"{task_index}:{inst_idx}",
                    task_index=task_index,
                    fields=tuple(fields_t),
                    anchor_query_id=spec.anchor_query_id,
                )
            )
    return records


def _pack_record_targets(
    graphs: Sequence[TargetGraph],
    record_specs: Sequence[Mapping[int, RecordSpec]],
):
    """Pack record span supervision once in the collator."""
    max_groups = max((len(specs) for specs in record_specs), default=0)
    if max_groups == 0:
        return None
    grouped = [
        [
            [record for record in graph.records if record.task_index == task_index]
            for task_index in specs
        ]
        for graph, specs in zip(graphs, record_specs)
    ]
    max_records = max(
        (len(records) for sample in grouped for records in sample), default=0
    )
    max_fields = max(
        (len(spec.fields) for specs in record_specs for spec in specs.values()),
        default=0,
    )
    max_spans = max(
        (
            sum(len(alternatives) for alternatives in field.values)
            for graph in graphs
            for record in graph.records
            for field in record.fields
        ),
        default=0,
    )
    max_records = max(max_records, 1)
    max_fields = max(max_fields, 1)
    max_spans = max(max_spans, 1)
    spans = torch.zeros(
        len(graphs), max_groups, max_records, max_fields, max_spans, 2,
        dtype=torch.long,
    )
    span_mask = torch.zeros(spans.shape[:-1], dtype=torch.bool)
    record_mask = torch.zeros(
        len(graphs), max_groups, max_records, dtype=torch.bool
    )
    field_query_ids = torch.zeros(
        len(graphs), max_groups, max_fields, dtype=torch.long
    )
    field_mask = torch.zeros_like(field_query_ids, dtype=torch.bool)
    scalar_fields = torch.zeros_like(field_mask)
    modes = torch.full((len(graphs), max_groups), -1, dtype=torch.long)
    anchor_fields = torch.full_like(modes, -1)
    group_mask = torch.zeros_like(modes, dtype=torch.bool)
    for batch_index, (sample_groups, specs) in enumerate(
        zip(grouped, record_specs)
    ):
        for group_index, (records, spec) in enumerate(
            zip(sample_groups, specs.values())
        ):
            group_mask[batch_index, group_index] = True
            modes[batch_index, group_index] = {
                "natural": 0,
                "latent": 1,
                "anchorless": 2,
            }[spec.mode]
            for field_index, field_spec in enumerate(spec.fields):
                field_query_ids[
                    batch_index, group_index, field_index
                ] = field_spec.query_id
                field_mask[batch_index, group_index, field_index] = True
                scalar_fields[
                    batch_index, group_index, field_index
                ] = field_spec.cardinality.is_scalar
                if field_spec.query_id == spec.anchor_query_id:
                    anchor_fields[batch_index, group_index] = field_index
            for record_index, record in enumerate(records):
                record_mask[batch_index, group_index, record_index] = True
                for field_index, field_spec in enumerate(spec.fields):
                    target = record.field_for_query(field_spec.query_id)
                    if target is None:
                        continue
                    flat = [
                        span
                        for alternatives in target.values
                        for span in alternatives
                    ]
                    if flat:
                        count = len(flat)
                        spans[
                            batch_index,
                            group_index,
                            record_index,
                            field_index,
                            :count,
                        ] = torch.as_tensor(flat, dtype=torch.long)
                        span_mask[
                            batch_index,
                            group_index,
                            record_index,
                            field_index,
                            :count,
                        ] = True
    return (
        spans,
        span_mask,
        record_mask,
        field_query_ids,
        field_mask,
        scalar_fields,
        modes,
        anchor_fields,
        group_mask,
    )


def pack_record_routes(
    record_specs: Sequence[Mapping[int, RecordSpec]],
) -> Optional[Tuple[torch.Tensor, ...]]:
    """Pack record routing for batched inference."""
    max_groups = max((len(specs) for specs in record_specs), default=0)
    if max_groups == 0:
        return None
    max_fields = max(
        (len(spec.fields) for specs in record_specs for spec in specs.values()),
        default=1,
    )
    batch_size = len(record_specs)
    field_query_ids = torch.zeros(
        batch_size, max_groups, max_fields, dtype=torch.long
    )
    field_mask = torch.zeros_like(field_query_ids, dtype=torch.bool)
    scalar_fields = torch.zeros_like(field_mask)
    modes = torch.full((batch_size, max_groups), -1, dtype=torch.long)
    anchor_fields = torch.full_like(modes, -1)
    group_mask = torch.zeros_like(modes, dtype=torch.bool)
    mode_ids = {"natural": 0, "latent": 1, "anchorless": 2}

    for batch_index, specs in enumerate(record_specs):
        for group_index, spec in enumerate(specs.values()):
            group_mask[batch_index, group_index] = True
            modes[batch_index, group_index] = mode_ids[spec.mode]
            for field_index, field_spec in enumerate(spec.fields):
                field_query_ids[batch_index, group_index, field_index] = (
                    field_spec.query_id
                )
                field_mask[batch_index, group_index, field_index] = True
                scalar_fields[batch_index, group_index, field_index] = (
                    field_spec.cardinality.is_scalar
                )
                if field_spec.query_id == spec.anchor_query_id:
                    anchor_fields[batch_index, group_index] = field_index

    return (
        field_query_ids,
        field_mask,
        scalar_fields,
        modes,
        anchor_fields,
        group_mask,
    )


def _pair_gap(pair: Tuple[int, int, int, int]) -> Tuple[int, int, int]:
    """Token gap between a head and a tail span (0 if they touch or overlap); ties go to the earliest pair.

    A text-level relation lists every occurrence of its head and of its tail. Labelling every
    occurrence pair positive teaches a local pair scorer that mentions sentences apart are related
    (professorbob_re x20.6 pairs per edge); ``relation_gold_pairs: nearest`` keeps the closest one.
    """
    hs, he, ts, te = pair
    return (max(0, ts - he, hs - te), min(hs, ts), max(he, te))


def _pack_relation_routes(
    layouts: Sequence[QueryLayout],
    relation_gold: Sequence[Sequence[Sequence[Tuple[int, int, int, int]]]],
):
    """Pack relation type/query membership and gold edges on CPU."""
    max_relations = max((len(sample) for sample in relation_gold), default=0)
    if max_relations == 0:
        return None
    max_queries = max((len(layout.queries) for layout in layouts), default=0)
    max_pairs = max(
        (len(pairs) for sample in relation_gold for pairs in sample),
        default=0,
    )
    max_pairs = max(max_pairs, 1)
    gold_pairs = torch.zeros(
        len(layouts), max_relations, max_pairs, 4, dtype=torch.long
    )
    gold_mask = torch.zeros(
        len(layouts), max_relations, max_pairs, dtype=torch.bool
    )
    head_member = torch.zeros(
        len(layouts), max_relations, max_queries, dtype=torch.bool
    )
    tail_member = torch.zeros_like(head_member)
    relation_valid = torch.zeros(
        len(layouts), max_relations, dtype=torch.bool
    )
    allow_self = torch.zeros_like(relation_valid)
    for batch_index, (layout, sample_gold) in enumerate(zip(layouts, relation_gold)):
        grouped_queries: dict[int, list[int]] = {}
        for query in layout.queries:
            if query.task_type == "relations":
                grouped_queries.setdefault(query.task_index, []).append(query.query_id)
        for relation_index, query_ids in enumerate(grouped_queries.values()):
            relation_valid[batch_index, relation_index] = True
            if query_ids:
                head_member[batch_index, relation_index, query_ids[0]] = True
            if len(query_ids) > 1:
                tail_member[batch_index, relation_index, query_ids[1]] = True
            pairs = sample_gold[relation_index]
            if pairs:
                gold_pairs[
                    batch_index, relation_index, :len(pairs)
                ] = torch.as_tensor(pairs, dtype=torch.long)
                gold_mask[batch_index, relation_index, :len(pairs)] = True
    return (
        gold_pairs,
        gold_mask,
        head_member,
        tail_member,
        relation_valid,
        allow_self,
    )


def _label_shapes(sample_labels):
    """Per-task ``(count, n_instances)`` for a sample's gold -- shape only, never surfaces."""
    out = []
    for lab in sample_labels or ():
        if isinstance(lab, (list, tuple)) and len(lab) == 2:
            count, instances = lab
            out.append((count, len(instances) if hasattr(instances, "__len__") else None))
        else:
            out.append(None)
    return out


def build_boundary_batch_metadata(
    *,
    schema_tokens_list: Sequence[Sequence[Sequence[str]]],
    task_types: Sequence[Sequence[str]],
    structure_labels: Sequence[Sequence[Any]],
    text_lengths: Sequence[int],
    is_training: bool,
    max_gold_per_query: Optional[int],
    record_metadata_list: Optional[Sequence[Optional[Mapping[str, Any]]]] = None,
    field_dtypes_list: Optional[Sequence[Optional[Mapping[str, Any]]]] = None,
    build_targets: Optional[bool] = None,
    on_capacity_exceeded: str = "raise",
    on_missing_surface: str = "raise",
    event_records: bool = False,
    suppress_orphaned_queries: bool = False,
    ignore_missing_entities: bool = False,
    dedupe_contained_entity_spans: bool = True,
    relation_gold_pairs: str = "all",
    original_texts: Optional[Sequence[str]] = None,
    original_schemas: Optional[Sequence[Optional[Mapping[str, Any]]]] = None,
) -> tuple:
    """Build layouts, optional padded targets, and compiled record specs.

    Supports every extractive task type (entities, JSON structures, relations);
    classification groups are skipped (no extractive queries). Missing surfaces
    (``(-1, -1)``) are treated as absent and skipped rather than raising, so
    optional fields and unlabeled types are handled gracefully. When a group
    declares record metadata, its gold instance grouping is preserved as
    :class:`RecordTarget` objects (identity by coordinate, never surface text).

    ``build_targets`` decouples supervision construction from training mode:
    when ``None`` (default) it follows ``is_training``, but an evaluation
    collator can pass ``build_targets=True`` to obtain gold targets for a
    finite eval loss while still running the model in eval mode. Gold is used
    only for loss computation; proposal gold injection stays gated on
    ``model.training`` so eval remains unbiased.

    ``dedupe_contained_entity_spans`` (default ``True``) filters an entity
    query's raw per-value token matches down to the maximal, non-contained
    ones before they become gold mentions -- see :func:`_dedupe_contained_spans`
    for the exact rationale and rule. Scoped to ``task_type == "entities"``
    only: JSON-structure and relation fields feed a separate, per-occurrence
    record/edge-target consumer (:func:`_build_record_field_target`,
    the relations ``gold_pairs`` block below) that reads raw positions
    directly and is not touched by this filter, so filtering only their
    aggregate query-level mentions here would make the two consumers
    disagree about which spans are gold for the same field. Set to ``False``
    to restore the pre-fix behavior (every independently found position
    becomes a gold mention, including phantom sub-spans).

    ``original_texts``/``original_schemas`` are optional and used only to
    enrich the ``ValueError`` raised for an unalignable entity (literal listed
    value(s) and a text snippet); omit them to get the older, terser message.

    Returns ``(layouts, targets, record_specs)`` where ``record_specs`` is a
    per-sample tuple of ``{task_index: RecordSpec}`` mappings.
    """
    if ignore_missing_entities:  # upstream's name for on_missing_surface="skip"
        on_missing_surface = "skip"
    if build_targets is None:
        build_targets = is_training
    layouts = []
    graphs = []
    # Marker slots removed because the query was suppressed. The encoder counts every
    # marker; leaving a suppressed one in makes query_mask one wider than the gold.
    dropped_markers: List[List[int]] = []
    record_specs_out: List[Dict[int, RecordSpec]] = []
    relation_gold_out: List[List[List[Tuple[int, int, int, int]]]] = []

    batch_size = len(schema_tokens_list)
    if not (
        len(task_types) == len(structure_labels) == len(text_lengths) == batch_size
    ):
        raise ValueError("boundary metadata batch fields have inconsistent lengths")

    for sample_idx, (sample_schemas, sample_types, sample_labels, text_length) in enumerate(
        zip(schema_tokens_list, task_types, structure_labels, text_lengths)
    ):
        if not (len(sample_schemas) == len(sample_types) == len(sample_labels)):
            raise ValueError(
                f"boundary sample {sample_idx} schema/task/label counts do not match"
            )
        queries: list[QuerySpec] = []
        mentions: list[MentionTarget] = []
        sample_relation_gold: List[List[Tuple[int, int, int, int]]] = []
        query_id = 0
        marker_at = 0
        dropped: List[int] = []

        for task_index, (schema_tokens, task_type, labels) in enumerate(
            zip(sample_schemas, sample_types, sample_labels)
        ):
            if task_type == "classifications":
                continue  # not an extractive query; handled by the classifier

            fields = _extractive_fields(schema_tokens)
            if not fields:
                continue
            name = _group_name(schema_tokens, task_type)

            # A label whose gold is entirely unalignable is offered with nothing to find,
            # which trains it as ABSENT while the entity sits in the text. Suppressing the
            # query teaches nothing instead of teaching a falsehood. ENTITIES ONLY and
            # deliberately: dropping a structure field or event role would remove an
            # anchor's field query, which is exactly the
            # "declares anchor X but no matching field query" abort.
            orphaned = (
                _orphaned_entity_fields(fields, labels)
                if suppress_orphaned_queries and task_type == "entities" and build_targets
                else frozenset()
            )

            field_query_ids = []
            for role_index, field in enumerate(fields):
                if role_index in orphaned:
                    # `None` keeps this list positionally aligned with `fields` so the
                    # mention loop still indexes by field_index; `query_id` stays
                    # contiguous because it only advances for queries that survive.
                    # The marker slot is dropped with the query, or the encoder still
                    # emits a query state the gold tensor does not have a row for.
                    field_query_ids.append(None)
                    dropped.append(marker_at)
                    _SUPPRESSED_QUERIES[field] += 1
                else:
                    field_query_ids.append(query_id)
                    queries.append(
                        QuerySpec(
                            query_id=query_id,
                            task_index=task_index,
                            task_type=task_type,
                            task_name=name,
                            role_index=role_index,
                            role_name=field,
                            field_path=(name, field) if name != "entities" else (field,),
                            extractive=True,
                        )
                    )
                    query_id += 1
                marker_at += 1

            if task_type == "relations":
                gold_pairs: List[Tuple[int, int, int, int]] = []
                if build_targets and labels and labels[0] != 0 and len(fields) >= 2:
                    for instance in labels[1]:
                        if len(instance) < 2:
                            continue
                        pairs = [
                            (hs, he + 1, ts, te + 1)
                            for hs, he in _iter_inclusive_spans(instance[0])
                            for ts, te in _iter_inclusive_spans(instance[1])
                            if 0 <= hs <= he < text_length and 0 <= ts <= te < text_length
                        ]
                        if relation_gold_pairs == "nearest" and pairs:
                            pairs = [min(pairs, key=_pair_gap)]
                        gold_pairs.extend(pairs)
                sample_relation_gold.append(gold_pairs)

            if not build_targets or not labels or labels[0] == 0:
                continue

            _, instances = labels
            # Accumulate entity-query positions across every instance/listed
            # value before emitting mentions, so containment filtering (below)
            # sees the *whole* set of independently found positions for this
            # query in this sample -- a phantom short match can come from a
            # different instance than the longer match that actually explains
            # it, not only from a list-valued field within one instance.
            entity_pairs: Dict[int, List[Tuple[int, int]]] = {}
            for instance in instances:
                for field_index, positions in enumerate(instance):
                    if field_index >= len(field_query_ids):
                        break
                    if field_query_ids[field_index] is None:
                        continue  # query suppressed: no query, so no mention to attach
                    if task_type == "entities":
                        # A labeled entity surface that cannot be located is
                        # normally an annotation error and must raise rather than
                        # silently shrink supervision. But boundary word-alignment
                        # also fails on surfaces that ARE present verbatim -- most
                        # often in languages without whitespace delimiters -- and
                        # aborting a multi-hour run over those is worse than losing
                        # the single mention. ``on_missing_surface="skip"`` drops
                        # just that mention and keeps the record's other gold.
                        for start, end_inclusive in positions:
                            if (start, end_inclusive) == (-1, -1):
                                if on_missing_surface == "skip":
                                    _MISSING_SURFACES[fields[field_index]] += 1
                                    continue
                                raise _unalignable_entity_error(
                                    field_name=fields[field_index],
                                    sample_idx=sample_idx,
                                    original_texts=original_texts,
                                    original_schemas=original_schemas,
                                )
                            start, end = inclusive_tokens_to_boundary_pair(start, end_inclusive)
                            entity_pairs.setdefault(
                                field_query_ids[field_index], []
                            ).append((start, end))
                    else:
                        # JSON/relation fields may legitimately be absent within
                        # an instance; treat "not found" as absent and skip.
                        for start, end_inclusive in _iter_inclusive_spans(positions):
                            start, end = inclusive_tokens_to_boundary_pair(start, end_inclusive)
                            mentions.append(
                                MentionTarget(field_query_ids[field_index], start, end)
                            )
            # NOTE: intentionally not named ``query_id`` -- that name is the
            # running per-sample query-id counter mutated above and read by
            # every later task_index iteration; shadowing it here would
            # corrupt query-id assignment for every task processed afterward.
            for entity_query_id, pairs in entity_pairs.items():
                kept_pairs = (
                    _dedupe_contained_spans(pairs)
                    if dedupe_contained_entity_spans
                    else pairs
                )
                for start, end in kept_pairs:
                    mentions.append(MentionTarget(entity_query_id, start, end))

        layout = QueryLayout(queries=tuple(queries))
        layouts.append(layout)
        dropped_markers.append(dropped)

        sample_meta = (
            record_metadata_list[sample_idx]
            if record_metadata_list is not None and sample_idx < len(record_metadata_list)
            else None
        )
        sample_dtypes = (
            field_dtypes_list[sample_idx]
            if field_dtypes_list is not None and sample_idx < len(field_dtypes_list)
            else None
        )
        try:
            specs = compile_record_specs(
                query_layout=layout,
                record_metadata=sample_meta,
                field_dtypes=sample_dtypes,
                event_records=event_records,
            )
        except ValueError as exc:
            # SELF-DIAGNOSING RAISE. This failure names a SHAPE but not the SAMPLE, and the
            # shape alone cost a day: the on-disk data is clean (0 of 202,211 records have an
            # anchor missing from its group), schema sampling never drops an anchor (0 of
            # 155,080 draws), `_transform_record` never drops one (0 of 62,032), and a
            # single-process collate over 203,736 chunks does not reproduce it -- twice.
            #
            # It cannot be reproduced on the operator's machine AT ALL: trainer.py forces
            # `num_workers = 0` on darwin and pins Linux to `fork`, so local runs and GPU
            # runs execute structurally different data paths. The evidence therefore has to
            # travel off the box with the exception.
            #
            # Gold is printed as COUNTS, never surfaces: a surface can be a whole document
            # and can carry licensed corpus text that must not land in a run log.
            worker = None
            try:
                from torch.utils.data import get_worker_info
                info = get_worker_info()
                worker = None if info is None else info.id
            except Exception:
                pass
            raise ValueError(
                f"{exc}\n"
                f"  [diag] sample_idx={sample_idx} worker={worker} "
                f"batch_samples={len(schema_tokens_list)}\n"
                f"  [diag] record_metadata={sample_meta!r}\n"
                f"  [diag] task_types={list(sample_types)!r}\n"
                f"  [diag] gold_shape={_label_shapes(sample_labels)!r}\n"
                f"  [diag] schema_tokens={[list(t) for t in sample_schemas]!r}"
            ) from exc
        record_specs_out.append(specs)
        relation_gold_out.append(sample_relation_gold)

        if build_targets:
            records = _build_sample_records(specs, sample_labels, text_length)
            graph = TargetGraph(mentions=tuple(mentions), records=tuple(records))
            validate_target_graph(graph, layout, text_length)
            graphs.append(graph)

    targets = None
    if build_targets:
        targets = pad_target_graphs(
            graphs,
            [layout.extractive_count() for layout in layouts],
            text_lengths,
            max_gold_per_query=max_gold_per_query,
            on_capacity_exceeded=on_capacity_exceeded,
            build_dense=False,
        )
        targets.record_targets = _pack_record_targets(graphs, record_specs_out)
        targets.edge_targets = _pack_relation_routes(layouts, relation_gold_out)
    return tuple(layouts), targets, tuple(record_specs_out), dropped_markers


def compact_query_markers(indices, mask, groups, dropped: Sequence[Sequence[int]]):
    """Remove suppressed query markers. Kept order matches the layout's query ids."""
    if indices is None or not any(dropped):
        return indices, mask, groups
    kept_rows = []
    for i, drop in enumerate(dropped):
        gone = set(drop)
        width = int(mask[i].sum()) if mask is not None else indices.shape[1]
        kept_rows.append([j for j in range(width) if j not in gone])
    width = max((len(row) for row in kept_rows), default=0)
    new_indices = indices.new_zeros((indices.shape[0], width))
    new_mask = torch.zeros((indices.shape[0], width), dtype=torch.bool)
    new_groups = groups.new_zeros((indices.shape[0], width)) if groups is not None else None
    for i, keep in enumerate(kept_rows):
        if not keep:
            continue
        new_indices[i, : len(keep)] = indices[i, keep]
        new_mask[i, : len(keep)] = True
        if new_groups is not None:
            new_groups[i, : len(keep)] = groups[i, keep]
    return new_indices, new_mask, new_groups


__all__ = ["build_boundary_batch_metadata"]

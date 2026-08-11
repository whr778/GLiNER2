"""Public boundary extractor class = shared runtime + boundary model core."""

from __future__ import annotations

import bisect
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

import torch

from gliner2.inference.candidate_decoder import token_boundaries_to_character_offsets
from gliner2.inference.runtime import ExtractorRuntimeMixin
from gliner2.models.boundary.model import (
    BoundaryExtractorModel,
    _group_scored_candidates,
)
from gliner2.joint_ie.candidate_scores import _query_type_name
from gliner2.models.base import (
    QueryLayout,
    QuerySpec,
    display_query_type,
    qualified_query_type,
)
from gliner2.models.boundary.records import decode_group


def _layout_from_ext_specs(specs: List[Dict[str, Any]]) -> QueryLayout:
    """Build a `QueryLayout` from one sample's `ext_specs`.

    `ext_specs` enumerates exactly the extractive queries in candidate-slot order
    (classifications never take a slot), so `query_id == candidate query index`.
    Deriving from it — rather than `batch.query_layouts`, which only the fast
    routing path populates — types mentions and relation endpoints from a single
    source, so edge keys reference mention keys by construction.
    """
    return QueryLayout(queries=tuple(
        QuerySpec(
            query_id=qid,
            task_index=spec["group_index"],
            task_type=spec["task_type"],
            task_name=spec["task_name"],
            role_index=spec["field_index"],
            role_name=spec["field_name"],
            field_path=(spec["task_name"], spec["field_name"]),
            extractive=True,
        )
        for qid, spec in enumerate(specs)
    ))


def _relation_value(
    head: Tuple[str, int, int],
    tail: Tuple[str, int, int],
    score: float,
    include_confidence: bool,
    include_spans: bool,
):
    """Format one decoded relation pair; shared by the greedy and joint paths.

    ``head``/``tail`` are ``(surface, char_start, char_end)``.
    """
    if include_spans:
        value = {
            "head": {"text": head[0], "start": head[1], "end": head[2]},
            "tail": {"text": tail[0], "start": tail[1], "end": tail[2]},
        }
        if include_confidence:
            value["head"]["confidence"] = score
            value["tail"]["confidence"] = score
        return value
    if include_confidence:
        return {
            "head": {"text": head[0], "confidence": score},
            "tail": {"text": tail[0], "confidence": score},
        }
    return (head[0], tail[0])


def _resolve_flat_spans(
    scored: List[Tuple[float, int, int]]
) -> List[Tuple[float, int, int]]:
    """Maximum-total-score non-overlapping subset via interval scheduling."""
    if not scored:
        return []
    spans = sorted(scored, key=lambda item: (item[2], item[1], -item[0]))
    ends = [end for _, _, end in spans]
    predecessors = [
        bisect.bisect_right(ends, start, 0, index) - 1
        for index, (_, start, _) in enumerate(spans)
    ]
    best = [0.0] * (len(spans) + 1)
    choose = [False] * len(spans)
    for index, (score, _, _) in enumerate(spans):
        with_span = score + best[predecessors[index] + 1]
        without_span = best[index]
        if with_span > without_span:
            best[index + 1] = with_span
            choose[index] = True
        else:
            best[index + 1] = without_span
    kept: List[Tuple[float, int, int]] = []
    index = len(spans) - 1
    while index >= 0:
        if choose[index] and (
            spans[index][0] + best[predecessors[index] + 1] > best[index]
        ):
            kept.append(spans[index])
            index = predecessors[index]
        else:
            index -= 1
    return sorted(kept, key=lambda item: (-item[0], item[1], item[2]))


def _resolve_spans(
    scored: List[Tuple[float, int, int]], policy: str
) -> List[Tuple[float, int, int]]:
    if policy == "flat":
        return _resolve_flat_spans(scored)
    ranked = sorted(scored, key=lambda item: (-item[0], item[1], item[2]))
    if policy == "nested":
        return ranked
    if policy == "longest":
        kept = []
        for candidate in ranked:
            _, start, end = candidate
            if any(
                kept_start <= start
                and end <= kept_end
                and (kept_start < start or end < kept_end)
                for _, kept_start, kept_end in kept
            ):
                continue
            kept.append(candidate)
        return kept
    raise ValueError(f"unknown boundary overlap policy {policy!r}")


class BoundaryExtractor(ExtractorRuntimeMixin, BoundaryExtractorModel):
    """Boundary architecture with the shared public extraction runtime.

    Overrides ``_extract_from_batch`` with the sparse candidate path: encode →
    boundary head → threshold + flat-span resolution → exact half-open
    token→character conversion. Entities, classification, and enabled
    record/event schemas and enabled sparse relation decoding are supported.
    """

    architecture = "boundary"

    def _extract_from_batch(
        self,
        batch,
        threshold: float,
        metadata_list: List[Dict],
        include_confidence: bool,
        include_spans: bool,
    ) -> List[Dict[str, Any]]:
        core = self._encode_core(batch)
        has_queries = core["query_states"].shape[1] > 0
        candidates = None
        probs = None
        grouped_candidates = None
        null_probs = None
        if has_queries:
            out = self.boundary_head(
                core["text_states"], core["text_mask"],
                core["query_states"], core["query_mask"],
                return_candidates=True,
            )
            candidates = out.candidates
            probs = torch.sigmoid(
                candidates.pair_logits
                / self.boundary_settings.pair_temperature
            )
            grouped_candidates = _group_scored_candidates(
                candidates,
                threshold=threshold,
                probabilities=probs,
                count_log_rates=out.count_log_rates,
                adaptive_threshold=self.boundary_settings.adaptive_threshold,
            )
            if out.null_logits is not None:
                null_probs = torch.sigmoid(out.null_logits).float().cpu()

        results: List[Dict[str, Any]] = []
        for i in range(len(batch)):
            sample: Dict[str, Any] = {}
            overlap_policy = (
                metadata_list[i].get("_overlap_policy")
                or self.boundary_settings.overlap_policy
            )
            specs = core["ext_specs"][i] if has_queries else []
            layout = _layout_from_ext_specs(specs)
            offset = core["word_offsets"][i]
            start_map = batch.start_mappings[i]
            end_map = batch.end_mappings[i]
            text = batch.original_texts[i]
            text_len = len(start_map)

            joint = (
                self.boundary_settings.decode_mode == "joint"
                and candidates is not None
            )
            if not joint:
                # In joint mode records come out of the beam instead, via role
                # edges -- running this too would double-emit them.
                record_results = self._decode_records(
                    batch, i, core, candidates, offset, start_map, end_map,
                    text, text_len, include_confidence, include_spans,
                )
                for name, instances in record_results.items():
                    if instances:
                        sample[name] = instances

            if joint:
                sample.update(self._decode_joint(
                    batch, i, core, candidates, threshold, offset, start_map, end_map,
                    text, text_len, include_confidence, include_spans,
                    layout, specs,
                ))
                self._decode_classifications(sample, batch, core, i)
                results.append(sample)
                continue

            relation_results = self._decode_relations(
                i, core, candidates, metadata_list[i], threshold, offset,
                start_map, end_map, text, text_len, include_confidence, include_spans,
                layout,
            )
            sample.update(relation_results)

            entity_results: "OrderedDict[str, Any]" = OrderedDict()
            for qid, spec in enumerate(specs):
                if spec["task_type"] != "entities":
                    continue  # events are assembled below; records by the record head
                spans = self._query_spans(
                    qid, i, grouped_candidates, null_probs, overlap_policy,
                    offset, start_map, end_map, text, text_len,
                )
                entity_results[spec["field_name"]] = self._format_spans(
                    spans or [], include_confidence, include_spans,
                    already_finalized=True,
                )

            if entity_results:
                sample["entities"] = [entity_results]

            event_results = self._decode_events(
                specs, i, grouped_candidates, null_probs, overlap_policy,
                offset, start_map, end_map, text, text_len,
                include_confidence, include_spans,
            )
            for event_type, instances in event_results.items():
                sample[event_type] = instances

            self._decode_classifications(sample, batch, core, i)
            results.append(sample)

        return results

    def _query_spans(
        self,
        qid: int,
        sample_index: int,
        grouped_candidates,
        null_probs,
        overlap_policy,
        offset: int,
        start_map,
        end_map,
        text: str,
        text_len: int,
    ) -> Optional[List[Tuple[str, float, int, int]]]:
        """Resolve one extractive query's candidates to character-anchored spans.

        Returns ``None`` when the query abstains, which the caller distinguishes
        from "decoded, found nothing" (an empty list).
        """
        if (
            null_probs is not None
            and float(null_probs[sample_index, qid])
            > self.boundary_settings.abstention_threshold
        ):
            return None
        spans: List[Tuple[str, float, int, int]] = []
        for p, s, e in _resolve_spans(grouped_candidates[sample_index][qid], overlap_policy):
            ts, te = s - offset, e - offset
            if ts < 0 or te > text_len or te <= ts:
                continue
            char_start, char_end = token_boundaries_to_character_offsets(
                ts, te, start_map, end_map
            )
            surface = text[char_start:char_end].strip()
            if surface:
                spans.append((surface, p, char_start, char_end))
        return spans

    def _decode_events(
        self,
        specs: List[Dict[str, Any]],
        sample_index: int,
        grouped_candidates,
        null_probs,
        overlap_policy,
        offset: int,
        start_map,
        end_map,
        text: str,
        text_len: int,
        include_confidence: bool,
        include_spans: bool,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Assemble events from their ``[V]`` mention queries.

        Events are supervised as MENTIONS -- one extractive query per trigger and
        per role -- NOT as record instances: an events schema never produces
        ``record_metadata``, so ``compile_record_specs`` returns ``{}`` and the
        record head is inert for them (verified in training collation, not just
        inference). Before this, event queries were skipped here on the assumption
        the record head would decode them, so the trigger/role spans the model
        genuinely learned were never emitted and every event metric read 0.0 at
        every threshold -- including 0.01.

        Shape mirrors the span engine's ``_extract_events``:
        ``{event_type: [{"triggers": [...], "arguments": [{"role", "entity"}]}]}``.
        Field 0 of a group is the trigger (as in the span engine, which reads
        ``scores[0]``); the rest are roles. A group with no trigger span is
        dropped, matching the span engine, since gold arguments are keyed by
        their trigger.

        One instance per event type: the mention path carries no instance
        dimension, so multi-instance separation needs the record head.
        """
        groups: "OrderedDict[str, List[Tuple[int, Dict[str, Any]]]]" = OrderedDict()
        for qid, spec in enumerate(specs):
            if spec["task_type"] == "events":
                groups.setdefault(spec["task_name"], []).append((qid, spec))

        out: Dict[str, List[Dict[str, Any]]] = {}
        for event_type, entries in groups.items():
            triggers: List[Any] = []
            arguments: List[Dict[str, Any]] = []
            for qid, spec in sorted(entries, key=lambda e: e[1]["field_index"]):
                spans = self._query_spans(
                    qid, sample_index, grouped_candidates, null_probs,
                    overlap_policy, offset, start_map, end_map, text, text_len,
                )
                if not spans:
                    continue
                formatted = self._format_spans(
                    spans, include_confidence, include_spans, already_finalized=True
                )
                if spec["field_index"] == 0:
                    triggers.extend(formatted)
                else:
                    arguments.extend(
                        {"role": spec["field_name"], "entity": value}
                        for value in formatted
                    )
            if triggers:
                out[event_type] = [{"triggers": triggers, "arguments": arguments}]
        return out

    def _decode_classifications(
        self, sample: Dict[str, Any], batch, core: Dict[str, Any], sample_index: int,
    ) -> None:
        """Add this sample's classification results in place."""
        schema = batch.original_schemas[sample_index]
        for cls in core["cls_specs"][sample_index]:
            self._extract_classification_result(
                sample, cls["task_name"], schema,
                cls["group_embs"], cls["schema_tokens"],
                temperature=self.boundary_settings.classification_temperature,
            )

    def _relation_pairs_and_logits(
        self,
        sample_index: int,
        core: Dict[str, Any],
        candidates,
        layout: QueryLayout,
    ):
        """Propose typed relation pairs for one sample and score them.

        Returns ``(pairs, logits)`` with *raw* logits — each caller applies its own
        temperature. Shared by the greedy and joint decode paths. Passing the real
        ``layout`` types the pair endpoint keys by `role_name`.
        """
        rel_specs = core["rel_specs"][sample_index]
        sample_candidates = self._single_sample_candidates(candidates, sample_index)
        pairs = self.relation_pair_generator.generate(
            sample_candidates,
            [layout],
            [entry["spec"] for entry in rel_specs],
        )
        if not len(pairs):
            return pairs, None
        query_states = torch.stack(
            [entry["query_state"] for entry in rel_specs]
        ).unsqueeze(0)
        logits = self.relation_scorer(
            core["text_states"][sample_index:sample_index + 1],
            query_states,
            sample_candidates,
            pairs,
        )
        return pairs, logits.detach()

    def _decode_relations(
        self,
        sample_index: int,
        core: Dict[str, Any],
        candidates,
        metadata: Dict[str, Any],
        threshold: float,
        offset: int,
        start_map,
        end_map,
        text: str,
        text_len: int,
        include_confidence: bool,
        include_spans: bool,
        layout: QueryLayout,
    ) -> Dict[str, Any]:
        """Decode sparse relation pairs for one sample."""
        if not getattr(self, "enable_relations", False) or candidates is None:
            return {}
        if not core["rel_specs"][sample_index]:
            return {}
        pairs, logits = self._relation_pairs_and_logits(
            sample_index, core, candidates, layout
        )
        if logits is None:
            return {}
        probabilities = torch.sigmoid(
            logits / self.boundary_settings.relation_temperature
        )
        out: Dict[str, Any] = {}
        relation_metadata = metadata.get("relation_metadata", {})
        for pair_index, probability in enumerate(probabilities):
            relation_type = pairs.relation_types[pair_index]
            relation_threshold = relation_metadata.get(relation_type, {}).get(
                "threshold", threshold
            )
            if relation_threshold is None:
                relation_threshold = threshold
            score = float(probability.detach())
            if score < relation_threshold:
                continue
            head = self._token_span_to_char(
                int(pairs.head_start[pair_index]), int(pairs.head_end[pair_index]),
                offset, start_map, end_map, text, text_len,
            )
            tail = self._token_span_to_char(
                int(pairs.tail_start[pair_index]), int(pairs.tail_end[pair_index]),
                offset, start_map, end_map, text, text_len,
            )
            if head is None or tail is None:
                continue
            out.setdefault(relation_type, []).append(
                _relation_value(
                    head, tail, score, include_confidence, include_spans
                )
            )
        return out

    def _token_span_to_char(
        self, start: int, end: int, offset: int, start_map, end_map,
        text: str, text_len: int,
    ):
        """Token span in the pre-offset frame -> ``(surface, char_start, char_end)``.

        Returns ``None`` when the span falls outside the chunk or is empty after
        stripping — the same guard the greedy decode applies.
        """
        ts, te = start - offset, end - offset
        if not (0 <= ts < te <= text_len):
            return None
        c0, c1 = token_boundaries_to_character_offsets(ts, te, start_map, end_map)
        surface = text[c0:c1].strip()
        return (surface, c0, c1) if surface else None

    def _decode_joint(
        self,
        batch,
        sample_index: int,
        core: Dict[str, Any],
        candidates,
        threshold: float,
        offset: int,
        start_map,
        end_map,
        text: str,
        text_len: int,
        include_confidence: bool,
        include_spans: bool,
        layout: QueryLayout,
        specs: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Decode one sample with the joint_ie typed-constraint beam.

        Entities and relations are selected *together* over the boundary candidate
        scores rather than per query, so a relation survives only with both of its
        endpoints. Mentions and relation endpoints are typed from the same
        ``layout``, so edges reference mention nodes by construction.

        Returns the same output shape as the greedy path. Note that adaptive
        thresholding and null-abstention remain greedy-only.
        """
        from gliner2.joint_ie.candidate_scores import (
            boundary_record_groups_to_role_edges,
            boundary_record_instance_nodes,
            joint_decode,
        )
        from gliner2.joint_ie.constraints import TypedEndpoints

        pairs, logits = self._relation_pairs_and_logits(
            sample_index, core, candidates, layout
        )
        if logits is None:
            logits = []
        # Qualified per query, not by bare field name: two relation types sharing the
        # Standard Format's head/tail would otherwise key the same span identically and
        # JointProblem would reject the problem. `relations.py::_query_type` qualifies the
        # pair endpoint keys the same way, so edges keep resolving to these nodes.
        query_types = [
            qualified_query_type(q, spec["field_name"]) for q, spec in enumerate(specs)
        ]
        groups = self._record_groups(batch, sample_index, core, candidates)
        # Mirror decode_group's instance gate exactly: it passes
        # object_threshold=record_anchor_threshold for BOTH modes, so one threshold
        # covers natural and anchorless. Without this the joint arm has no existence
        # gate and emits events greedy would reject (JOINT_IE_SCALING decision D).
        gate = dict(
            instance_threshold=self.boundary_settings.record_anchor_threshold,
            temperature=self.boundary_settings.record_temperature,
        )
        role_edges = boundary_record_groups_to_role_edges(groups, query_types, **gate)
        instance_nodes = boundary_record_instance_nodes(groups, query_types, **gate)
        constraints = [
            # Qualified for the same reason, and this is where it bites hardest: with bare
            # role names every relation's constraint is ("head",)/("tail",), so the
            # endpoint typing is vacuous across relation types and constrains nothing.
            TypedEndpoints(
                entry["spec"].relation_type,
                tuple(qualified_query_type(q, layout.query(q).role_name)
                      for q in entry["spec"].head_query_ids),
                tuple(qualified_query_type(q, layout.query(q).role_name)
                      for q in entry["spec"].tail_query_ids),
            )
            for entry in core["rel_specs"][sample_index]
        ]
        solution = joint_decode(
            candidates,
            query_types,
            pairs,
            logits,
            constraints=constraints,
            sample_index=sample_index,
            text=text,
            mention_threshold=threshold,
            beam_width=self.boundary_settings.joint_beam_width,
            pair_temperature=self.boundary_settings.pair_temperature,
            relation_temperature=self.boundary_settings.relation_temperature,
            extra_edges=role_edges,
            extra_nodes=instance_nodes,
        )

        sample: Dict[str, Any] = {}
        # Relation head/tail-role queries are mentions too; only entity queries
        # belong in the entities section.
        # Keyed by the qualified type the nodes actually carry; unqualified for display.
        entity_results: "OrderedDict[str, List]" = OrderedDict(
            (query_types[q], [])
            for q, spec in enumerate(specs) if spec["task_type"] == "entities"
        )
        entity_types = set(entity_results)
        chars: Dict[Any, Tuple[str, int, int]] = {}
        for node in solution.nodes:
            span = self._token_span_to_char(
                node.start, node.end, offset, start_map, end_map, text, text_len
            )
            if span is None:
                continue
            chars[node.candidate_id] = span
            if node.entity_type in entity_types:
                entity_results[node.entity_type].append(
                    (span[0], node.probability, span[1], span[2])
                )

        formatted = OrderedDict(
            (display_query_type(name), self._format_spans(
                spans, include_confidence, include_spans, already_finalized=True
            ))
            for name, spans in entity_results.items()
        )
        if formatted:
            sample["entities"] = [formatted]

        for edge in solution.edges:
            if "::" in edge.relation_type:
                continue  # a record role edge, emitted as a record below
            head, tail = chars.get(edge.head), chars.get(edge.tail)
            if head is None or tail is None:
                continue
            sample.setdefault(edge.relation_type, []).append(
                _relation_value(
                    head, tail, edge.head_probability,
                    include_confidence, include_spans,
                )
            )

        sample.update(self._format_joint_records(
            groups, solution, query_types, offset, start_map, end_map,
            text, text_len, include_confidence, include_spans,
        ))
        return sample

    def _format_joint_records(
        self, groups, solution, query_types: List[str], offset: int,
        start_map, end_map, text: str, text_len: int,
        include_confidence: bool, include_spans: bool,
    ) -> Dict[str, Any]:
        """Rebuild record instances from the solution's role edges (§3b).

        Role edges carry ``hypothesis`` = the trigger node key, so grouping by it
        reconstitutes instances. Output matches the greedy record shape exactly, so
        the eval harness runs both arms unchanged.
        """
        by_task: Dict[str, Dict[Any, Dict[str, List]]] = {}
        for edge in solution.edges:
            task, sep, role = edge.relation_type.partition("::")
            if not sep:
                continue
            instances = by_task.setdefault(task, {}).setdefault(edge.hypothesis, {})
            instances.setdefault(role, []).append((edge.tail[1], edge.tail[2]))

        out: Dict[str, Any] = {}
        for group in groups:
            spec = group.spec
            found = by_task.get(spec.task_name)
            if not found:
                continue
            instances = []
            for trigger, roles in found.items():
                inst: "OrderedDict[str, Any]" = OrderedDict()
                for fspec in spec.fields:
                    role = _query_type_name(query_types, fspec.query_id)
                    spans = [(trigger[1], trigger[2])] if fspec.is_anchor \
                        else roles.get(role, [])
                    value = self._format_record_field(
                        spans, fspec.cardinality.is_scalar, offset, start_map,
                        end_map, text, text_len, include_confidence, include_spans,
                    )
                    if not fspec.cardinality.is_scalar and value is None:
                        value = []
                    inst[fspec.name] = value
                if any(v is not None and v != [] for v in inst.values()):
                    instances.append(inst)
            if instances:
                out[spec.task_name] = instances
        return out

    def _format_record_field(
        self, spans, is_scalar: bool, offset: int, start_map, end_map,
        text: str, text_len: int, include_confidence: bool, include_spans: bool,
    ):
        """Format one record field's token spans; shared by the greedy and joint paths.

        Scalar -> str/dict/None, list -> the standard span list.
        """
        formatted: List[Tuple[str, float, int, int]] = []
        for (ts_raw, te_raw) in spans:
            span = self._token_span_to_char(
                ts_raw, te_raw, offset, start_map, end_map, text, text_len
            )
            if span is not None:
                formatted.append((span[0], 1.0, span[1], span[2]))
        if is_scalar:
            if not formatted:
                return None
            s, conf, cs, ce = formatted[0]
            if include_spans and include_confidence:
                return {"text": s, "confidence": conf, "start": cs, "end": ce}
            if include_spans:
                return {"text": s, "start": cs, "end": ce}
            if include_confidence:
                return {"text": s, "confidence": conf}
            return s
        return self._format_spans(
            formatted, include_confidence, include_spans, already_finalized=True
        )

    def _record_groups(self, batch, sample_index: int, core: Dict[str, Any], candidates):
        """Raw ``RecordGroupOutput``s for one sample (each carries its own ``spec``).

        The pre-greedy record lattice, shared by the greedy record decode and the
        joint path -- the `_relation_pairs_and_logits` analogue. Feeding the joint
        path from `decode_group` output instead would only re-rank greedy decisions.
        """
        if not getattr(self, "enable_records", False):
            return []
        if candidates is None or candidates.candidate_states is None:
            return []
        record_specs = getattr(batch, "record_specs", ())
        if sample_index >= len(record_specs) or not record_specs[sample_index]:
            return []
        query_states_i = core["query_states"][sample_index]
        return [
            self.record_decoder.forward_group(
                spec, query_states_i, candidates, sample_index)
            for spec in record_specs[sample_index].values()
        ]

    def _decode_records(
        self,
        batch,
        sample_index: int,
        core: Dict[str, Any],
        candidates,
        offset: int,
        start_map,
        end_map,
        text: str,
        text_len: int,
        include_confidence: bool,
        include_spans: bool,
    ) -> Dict[str, Any]:
        """Decode record/event groups into public structure output shapes."""
        groups = self._record_groups(batch, sample_index, core, candidates)
        if not groups:
            return {}

        settings = self.boundary_settings
        out: Dict[str, Any] = {}

        for group in groups:
            spec = group.spec
            decoded = decode_group(
                group,
                anchor_threshold=settings.record_anchor_threshold,
                field_threshold=settings.record_field_threshold,
                object_threshold=settings.record_anchor_threshold,
                temperature=settings.record_temperature,
            )
            instances = []
            for rec in decoded:
                inst: "OrderedDict[str, Any]" = OrderedDict()
                # Emit every declared field in schema order (legacy shape):
                # scalar -> str/None, list -> list[str] (possibly empty).
                for fspec in spec.fields:
                    spans = rec.fields.get(fspec.query_id, [])
                    value = self._format_record_field(
                        spans, fspec.cardinality.is_scalar, offset, start_map,
                        end_map, text, text_len, include_confidence, include_spans,
                    )
                    if not fspec.cardinality.is_scalar and value is None:
                        value = []
                    inst[fspec.name] = value
                if any(v is not None and v != [] for v in inst.values()):
                    instances.append(inst)
            if instances:
                out[spec.task_name] = instances
        return out


__all__ = ["BoundaryExtractor"]

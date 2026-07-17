"""Tests for the greedy global event assembler (gliner2.inference.global_decode).

Inputs mimic the per-chunk results the long-doc path produces: dicts with an
``event_extraction`` block whose spans are already in global offsets and carry
``{text,confidence,start,end}``.
"""

from gliner2.inference.chunking import TextChunk, merge_chunk_results
from gliner2.inference.global_decode import (
    GlobalDecodeConfig,
    assemble_events_global,
    beam_decode,
)


def span(text, start, end, conf=0.9):
    return {"text": text, "confidence": conf, "start": start, "end": end}


def mention(triggers, args):
    return {"triggers": triggers, "arguments": [{"role": r, "entity": e} for r, e in args]}


def chunk(event_extraction):
    return {"event_extraction": event_extraction}


class TestAssembly:
    def test_unions_args_split_across_windows(self):
        trig = span("bombed", 10, 16)
        c1 = chunk({"Attack": [mention([trig], [("Target", span("base", 30, 34))])]})
        c2 = chunk({"Attack": [mention([trig], [("Attacker", span("rebels", 0, 6))])]})

        out = assemble_events_global([c1, c2])

        assert list(out.keys()) == ["Attack"]
        assert len(out["Attack"]) == 1
        ev = out["Attack"][0]
        assert len(ev["triggers"]) == 1
        assert {a["role"] for a in ev["arguments"]} == {"Target", "Attacker"}

    def test_same_arg_deduped_across_windows(self):
        trig = span("bombed", 10, 16)
        base = ("Target", span("base", 30, 34))
        out = assemble_events_global([
            chunk({"Attack": [mention([trig], [base])]}),
            chunk({"Attack": [mention([trig], [base])]}),
        ])
        ev = out["Attack"][0]
        assert len(ev["arguments"]) == 1

    def test_distinct_triggers_not_merged(self):
        c = chunk({"Attack": [
            mention([span("bombed", 10, 16)], []),
            mention([span("shot", 200, 204)], []),
        ]})
        out = assemble_events_global([c])
        assert len(out["Attack"]) == 2

    def test_higher_confidence_span_kept_on_dedup(self):
        trig = span("bombed", 10, 16)
        out = assemble_events_global([
            chunk({"Attack": [mention([trig], [("Target", span("base", 30, 34, conf=0.70))])]}),
            chunk({"Attack": [mention([trig], [("Target", span("base", 30, 34, conf=0.92))])]}),
        ])
        arg = out["Attack"][0]["arguments"][0]
        assert arg["entity"]["confidence"] == 0.92

    def test_invalid_role_dropped_when_roles_provided(self):
        trig = span("bombed", 10, 16)
        c = chunk({"Attack": [mention([trig], [
            ("Target", span("base", 30, 34)),
            ("Bogus", span("noise", 40, 45)),
        ])]})
        out = assemble_events_global([c], event_roles={"Attack": ["Target"]})
        roles = {a["role"] for a in out["Attack"][0]["arguments"]}
        assert roles == {"Target"}

    def test_no_events_returns_empty_dict(self):
        assert assemble_events_global([{"entities": {"PER": ["John"]}}]) == {}

    def test_requested_empty_type_preserved(self):
        assert assemble_events_global([chunk({"Attack": []})]) == {"Attack": []}

    def test_trigger_iou_threshold_respected(self):
        # near-miss triggers below the IoU threshold stay distinct events
        c = chunk({"Attack": [
            mention([span("a", 0, 10)], []),
            mention([span("b", 8, 18)], []),  # IoU = 2/18 ~ 0.11 < 0.5
        ]})
        out = assemble_events_global([c], cfg=GlobalDecodeConfig(trigger_iou=0.5))
        assert len(out["Attack"]) == 2


class TestBeam:
    def _greedy(self, assembled):
        # sanity: with defaults the beam keeps everything (equals greedy input)
        return beam_decode(assembled, GlobalDecodeConfig())

    def test_beam_is_noop_without_conflicts(self):
        assembled = {"Attack": [{
            "triggers": [span("bombed", 10, 16)],
            "arguments": [
                {"role": "Target", "entity": span("base", 30, 34)},
                {"role": "Attacker", "entity": span("rebels", 0, 6)},
            ],
        }]}
        out = self._greedy(assembled)
        assert len(out["Attack"][0]["arguments"]) == 2

    def test_trigger_floor_drops_low_conf_event_but_keeps_type(self):
        assembled = {"Attack": [{
            "triggers": [span("bombed", 10, 16, conf=0.2)],
            "arguments": [{"role": "Target", "entity": span("base", 30, 34)}],
        }]}
        out = beam_decode(assembled, GlobalDecodeConfig(min_trigger_conf=0.5))
        assert out["Attack"] == []

    def test_single_filler_cardinality_keeps_top_one(self):
        assembled = {"Attack": [{
            "triggers": [span("bombed", 10, 16)],
            "arguments": [
                {"role": "Target", "entity": span("base", 30, 34, conf=0.9)},
                {"role": "Target", "entity": span("camp", 40, 44, conf=0.6)},
            ],
        }]}
        out = beam_decode(assembled, GlobalDecodeConfig(single_filler_roles=frozenset({"Target"})))
        args = out["Attack"][0]["arguments"]
        assert len(args) == 1 and args[0]["entity"]["text"] == "base"

    def test_span_conflict_drops_lower_confidence_reuse(self):
        # same span "base" claimed by two events; the low-confidence reuse loses
        shared_hi = span("base", 30, 34, conf=0.9)
        shared_lo = span("base", 30, 34, conf=0.4)
        assembled = {
            "Attack": [{"triggers": [span("bombed", 10, 16)],
                        "arguments": [{"role": "Target", "entity": shared_hi}]}],
            "Movement": [{"triggers": [span("fled", 100, 104)],
                          "arguments": [{"role": "Origin", "entity": shared_lo}]}],
        }
        out = beam_decode(assembled, GlobalDecodeConfig(conflict_penalty=0.5))
        assert len(out["Attack"][0]["arguments"]) == 1
        assert out["Movement"][0]["arguments"] == []


class TestMergeWiring:
    """Drive the real merge path (remap + assemble) via merge_chunk_results."""

    def _fixture(self):
        # Same "bombed" trigger at global [150,156]; its Target argument is only
        # in window A, its Attacker only in window B.
        text = "z" * 300
        chunks = [
            TextChunk(text="", start_char=0, end_char=200, start_word=0, end_word=0),
            TextChunk(text="", start_char=100, end_char=300, start_word=0, end_word=0),
        ]
        results = [
            {"event_extraction": {"Attack": [{
                "triggers": [span("bombed", 150, 156)],
                "arguments": [{"role": "Target", "entity": span("base", 170, 174)}],
            }]}},
            {"event_extraction": {"Attack": [{  # chunk-local offsets (start_char 100)
                "triggers": [span("bombed", 50, 56)],
                "arguments": [{"role": "Attacker", "entity": span("rebels", 60, 66)}],
            }]}},
        ]
        return text, chunks, results

    def test_global_decode_unions_cross_window_args(self):
        text, chunks, results = self._fixture()
        merged = merge_chunk_results(
            text, chunks, results, include_spans=True,
            global_decode=True, event_roles={"Attack": ["Target", "Attacker"]},
        )
        events = merged["event_extraction"]["Attack"]
        assert len(events) == 1
        assert {a["role"] for a in events[0]["arguments"]} == {"Target", "Attacker"}

    def test_without_global_decode_events_are_not_stitched(self):
        text, chunks, results = self._fixture()
        merged = merge_chunk_results(text, chunks, results, include_spans=True)
        # today's behavior: the two windows' events are concatenated, not merged
        assert len(merged["event_extraction"]["Attack"]) == 2

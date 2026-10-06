"""event_cluster / event_cluster_argument: event identity by CLUSTER (COREFERENT_LINK_SPEC.md section 5).

A prediction that finds a 3-mention gold event through ONE non-first mention, with every argument right,
scores 0.0 on today's event_argument (its key is the whole trigger set) -- the new keys must score it a
hit, and must still score 0.0 when no mention is shared. Existing keys are unchanged.
"""
import pytest

from gliner2.training.eval_metrics import compute_metrics


class _Stub:
    def __init__(self, pred):
        self.pred = pred

    def batch_extract(self, texts, schemas, **k):
        return [self.pred]


TEXT = "Rebels attacked the base. The assault was quick; it ended fast."
GOLD = {"events": [{"event_type": "Conflict.Attack", "triggers": ["attacked", "the assault", "it"],
                    "arguments": [{"role": "Attacker", "entity": "Rebels"}, {"role": "Target", "entity": "base"}]}]}
ARGS = [{"role": "Attacker", "entity": "Rebels"}, {"role": "Target", "entity": "base"}]


def _m(triggers, args=ARGS):
    return compute_metrics(_Stub({"event_extraction": {"Conflict.Attack": [{"triggers": triggers, "arguments": args}]}}),
                           [(TEXT, GOLD)])


def test_found_through_a_non_first_mention_is_a_hit_on_cluster_keys_only():
    m = _m(["the assault"])
    assert m["eval_event_cluster_strict_micro_f1"] == 1.0
    assert m["eval_event_cluster_argument_strict_micro_f1"] == 1.0
    assert m["eval_event_argument_strict_micro_f1"] == 0.0        # today's key misses every argument


def test_no_shared_mention_is_a_miss():
    m = _m(["raided"], ARGS[:1])
    assert m["eval_event_cluster_strict_micro_f1"] == 0.0
    assert m["eval_event_cluster_argument_strict_micro_f1"] == 0.0


def test_wrong_argument_inside_a_matched_event_costs_argument_f1_only():
    m = _m(["attacked"], [{"role": "Attacker", "entity": "Police"}, {"role": "Target", "entity": "base"}])
    assert m["eval_event_cluster_strict_micro_f1"] == 1.0
    assert m["eval_event_cluster_argument_strict_micro_f1"] == pytest.approx(0.5)

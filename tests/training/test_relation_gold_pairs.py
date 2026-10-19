"""relation_gold_pairs: "all" labels every head x tail occurrence pair; "nearest" only the closest (TODO #40)."""
import random

from gliner2.processing.boundary_preprocessing import _pair_gap
from gliner2.training.trainer import ExtractorCollator
from tests.models.boundary.test_record_head_pipeline import _build_tiny_records_model

TEXT = "Alice met Bob in Paris . Later Alice and Bob left Paris for Rome while Alice stayed"
REC = {"relations": [{"met": {"head": "Alice", "tail": "Bob"}}]}


def test_pair_gap_orders_by_token_distance_then_position():
    pairs = [(0, 1, 20, 21), (6, 7, 8, 9), (0, 1, 2, 3), (14, 15, 8, 9)]
    assert min(pairs, key=_pair_gap) == (0, 1, 2, 3)
    assert _pair_gap((5, 8, 6, 7)) == (0, 5, 8)


def _gold_pairs(mode):
    model = _build_tiny_records_model()
    coll = ExtractorCollator(model.processor, is_training=True, architecture="boundary", max_gold_per_query=32,
                             on_capacity_exceeded="raise", error_policy="raise", relation_gold_pairs=mode)
    random.seed(0)
    out = coll([(TEXT, REC)])
    pairs, mask = out.targets.edge_targets[:2]
    tokens = [t.lower() for t in out.text_tokens[0]]
    return [(" ".join(tokens[a:b]), " ".join(tokens[c:d]), c - b) for a, b, c, d in pairs[0, 0][mask[0, 0]].tolist()]


def test_all_labels_every_occurrence_pair():
    assert len(_gold_pairs("all")) == 3 * 2  # Alice x3, Bob x2


def test_nearest_keeps_only_the_closest_pair():
    pairs = _gold_pairs("nearest")
    assert pairs == [("alice", "bob", 1)]  # "Alice met Bob": one token apart
    assert {p for p in _gold_pairs("all")} >= set(pairs)

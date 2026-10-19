"""The entity-mention coreference link: clusters ride record -> batch -> loss, and nothing else moves."""
import torch

from gliner2.models.boundary import mention_link as L
from gliner2.processing import coref
from gliner2.training.chunking import _chunk_clusters
from gliner2.training.data import DataLoader_Factory

CLUSTERS = [[{"text": "Ann", "start": 0, "end": 3}, {"text": "her", "start": 13, "end": 16}],
            [{"text": "Bea", "start": 8, "end": 11}]]


def test_attach_and_take_carry_clusters_without_touching_the_record():
    output = {"entities": {"person": ["Ann"]}}
    schema = coref.attach(output, CLUSTERS)
    assert coref.SCHEMA_KEY not in output and schema[coref.SCHEMA_KEY] == CLUSTERS
    clean, clusters = coref.take([("Ann met Bea. her", schema)])
    assert clean == [("Ann met Bea. her", output)] and clusters == [CLUSTERS]


def test_word_mentions_survive_a_schema_prefix_longer_than_the_text():
    """text_word_counts counts schema-prefix words; an unaligned mention must not index past the text (eb22)."""
    starts, ends = [0, 4, 8, 13], [3, 7, 11, 16]
    bad = [[{"text": "nn", "start": 1, "end": 3}]]
    assert coref.word_mentions(CLUSTERS + bad, starts, ends, 9) == [(0, 0, 0), (3, 3, 0), (2, 2, 1)]


def test_batch_mentions_follow_kept_indices_not_text():
    from types import SimpleNamespace
    out = SimpleNamespace(original_texts=["Ann met Bea. her"], start_mappings=[[0, 4, 8, 13]], end_mappings=[[3, 7, 11, 16]],
                          text_word_counts=[4], kept_indices=[1])
    assert coref.batch_mentions(out, [None, CLUSTERS]) == ([(0, 0, 0), (3, 3, 0), (2, 2, 1)],)
    out.kept_indices = [0]
    assert coref.batch_mentions(out, [None, CLUSTERS]) == ([],)


def test_same_text_twice_in_a_batch_gets_only_its_own_clusters():
    """A replay record and the coref copy of the same article: the replay copy must NOT inherit the clusters."""
    import random
    from gliner2.training.trainer import ExtractorCollator
    from tests.models.boundary.test_record_head_pipeline import _build_tiny_records_model
    model = _build_tiny_records_model()
    coll = ExtractorCollator(model.processor, is_training=True, architecture="boundary", max_gold_per_query=32,
                             on_capacity_exceeded="raise", error_policy="skip")
    text = "Ann met Bea. her"
    random.seed(0)
    out = coll([(text, {"entities": {"person": ["Ann"]}}), (text, coref.attach({}, CLUSTERS))])
    assert out.kept_indices == [0, 1]
    assert out.coref_mentions[0] == [] and len(out.coref_mentions[1]) == 3


def test_word_mentions_keep_word_aligned_mentions_and_count_the_rest():
    starts, ends = [0, 4, 8, 13], [3, 7, 11, 16]          # Ann met Bea her
    for k in coref.COUNTS:
        coref.COUNTS[k] = 0
    bad = CLUSTERS + [[{"text": "nn", "start": 1, "end": 3}]]
    assert coref.word_mentions(bad, starts, ends, 4) == [(0, 0, 0), (3, 3, 0), (2, 2, 1)]
    assert coref.COUNTS["unaligned"] == 1
    assert coref.word_mentions(CLUSTERS, starts, ends, 2) == [(0, 0, 0)]   # her, Bea outside the window


def test_chunk_clusters_rebase_and_drop_mentions_crossing_the_window():
    assert _chunk_clusters(CLUSTERS, 8, 16) == [[{"text": "her", "start": 5, "end": 8}], [{"text": "Bea", "start": 0, "end": 3}]]
    assert _chunk_clusters(CLUSTERS, 2, 12) == [[{"text": "Bea", "start": 6, "end": 9}]]


def test_validation_keeps_clusters_and_coref_only_records():
    recs = [{"input": "Ann met Bea. her", "output": {"entities": {"person": ["Ann"]}}, "coref_clusters": CLUSTERS},
            {"input": "Ann met Bea. her", "output": {}, "coref_clusters": CLUSTERS},
            {"input": "nothing", "output": {}}]
    kept = DataLoader_Factory.load(data=recs, max_samples=-1, shuffle=False, seed=0, validate=True)
    assert [r.get("coref_clusters") for r in kept] == [CLUSTERS, CLUSTERS]


def test_mention_states_are_span_means():
    states = torch.arange(12.0).view(4, 3)
    mean, mid = L.mention_states(states, [(0, 1, 0), (3, 3, 1)])
    assert torch.equal(mean, torch.tensor([[1.5, 2.5, 3.5], [9.0, 10.0, 11.0]])) and mid.tolist() == [0.5, 3.0]


def test_pair_loss_takes_every_positive_and_the_hardest_negatives():
    logits = torch.tensor([[0.0, 6.0, 1.0, 0.1],
                           [6.0, 0.0, 5.0, 4.0],
                           [1.0, 5.0, 0.0, 0.0],
                           [0.1, 4.0, 0.0, 0.0]])
    clusters = torch.tensor([0, 0, 1, 2])
    loss, n, wins, ranked = L.pair_loss(logits, clusters, k=1)
    # positive (0,1); hardest negative per mention: 0->2, 1->2, 2->1, 3->1 -- (1,3) is picked ONLY by
    # mention 3, from below the diagonal, so it needs the symmetrisation
    assert n == 4 and (wins, ranked) == (3, 3)
    want = torch.nn.functional.binary_cross_entropy_with_logits(
        torch.tensor([6.0, 1.0, 5.0, 4.0]), torch.tensor([1.0, 0, 0, 0]), reduction="sum")
    assert torch.isclose(loss, want)


def test_batch_loss_skips_samples_without_two_mentions():
    link = L.MentionLink(3, 4)
    loss, n = L.batch_loss(link, torch.randn(2, 5, 3), [[(0, 0, 0)], []], k=2)
    assert n == 0 and float(loss) == 0.0

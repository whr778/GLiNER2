"""relation_keep_gold_pairs: gold pairs the proposal ranking dropped are appended to the relation loss's pair set."""
import torch

from gliner2.configuration import validate_boundary_head
from gliner2.models.boundary.relations import RelationPairBatch, append_gold_pairs


def _pairs(rows, mask=None):
    t = torch.tensor(rows, dtype=torch.long)
    n = t.shape[0]
    return RelationPairBatch(batch_index=t[:, 0], relation_index=t[:, 1], head_start=t[:, 2], head_end=t[:, 3],
                             tail_start=t[:, 4], tail_end=t[:, 5], head_prob=torch.full((n,), 0.5), tail_prob=torch.full((n,), 0.5),
                             pair_mask=None if mask is None else torch.tensor(mask))


def _gold(entries, shape):
    """entries: {(b, r, k): (hs, he, ts, te)} -> (gold_pairs [B,R,K,4], gold_mask [B,R,K])."""
    pairs = torch.zeros(*shape, 4, dtype=torch.long)
    mask = torch.zeros(*shape, dtype=torch.bool)
    for (b, r, k), span in entries.items():
        pairs[b, r, k] = torch.tensor(span)
        mask[b, r, k] = True
    return pairs, mask


def test_missing_gold_pairs_are_appended_with_their_spans_and_relation():
    kept = _pairs([[0, 0, 1, 2, 5, 6], [0, 0, 1, 2, 9, 10]])          # one gold pair already kept, one false pair
    gold, mask = _gold({(0, 0, 0): (1, 2, 5, 6), (0, 0, 1): (20, 22, 30, 31), (1, 1, 0): (3, 4, 7, 8)}, (2, 2, 2))
    out, by_ranking, appended = append_gold_pairs(kept, gold, mask)
    assert (by_ranking, appended) == (1, 2)
    assert len(out) == 4 and out.pair_mask.tolist() == [True] * 4
    rows = torch.stack((out.batch_index, out.relation_index, out.head_start, out.head_end, out.tail_start, out.tail_end), -1).tolist()
    assert rows[2:] == [[0, 0, 20, 22, 30, 31], [1, 1, 3, 4, 7, 8]]
    assert out.head_prob[2:].tolist() == [1.0, 1.0]


def test_already_kept_gold_is_not_duplicated_and_masked_slots_do_not_count_as_kept():
    kept = _pairs([[0, 0, 1, 2, 5, 6], [0, 0, 20, 22, 30, 31]], mask=[True, False])   # second slot is padding
    gold, mask = _gold({(0, 0, 0): (1, 2, 5, 6), (0, 0, 1): (20, 22, 30, 31)}, (1, 1, 2))
    out, by_ranking, appended = append_gold_pairs(kept, gold, mask)
    assert (by_ranking, appended) == (1, 1)                 # the padded slot does not count as kept
    assert len(out) == 3 and out.pair_mask.tolist() == [True, False, True]


def test_no_gold_returns_the_pairs_unchanged():
    kept = _pairs([[0, 0, 1, 2, 5, 6]])
    gold, mask = _gold({}, (1, 1, 1))
    out, by_ranking, appended = append_gold_pairs(kept, gold, mask)
    assert out is kept and (by_ranking, appended) == (0, 0)


def test_flag_reaches_boundary_settings_and_defaults_off():
    assert validate_boundary_head({})["relation_keep_gold_pairs"] is False
    assert validate_boundary_head({"relation_keep_gold_pairs": True})["relation_keep_gold_pairs"] is True

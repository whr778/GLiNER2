"""The typed-margin mask: who gets penalised, and -- mostly -- who does not.

Every assertion here is an ABSTENTION check. The mask exists to penalise a candidate whose
entity type the role-type map forbids for this query's role, and its correctness is almost
entirely about the cases it must leave alone: an unconstrained role, an untyped span, a span
carrying several types of which one is allowed, and gold itself.
"""

import torch

from gliner2.models.boundary.losses import (
    apply_typed_margin, proposal_listwise_loss,
)
from gliner2.models.boundary.model import _typed_margin_mask


PERSON, ORG, DATE = 1, 2, 4      # bits, as _typed_margin_tables assigns them


def _indices(spans):
    """[1, Q, C, 2] from a list-of-lists of (start, end)."""
    return torch.tensor([[list(s) for s in row] for row in spans]).unsqueeze(0)


def _table(rows):
    """[1, N, 3] span table of (start, end, type_bits)."""
    return torch.tensor([rows], dtype=torch.long)


def test_returns_none_without_inputs():
    idx = _indices([[(0, 1), (2, 3)]])
    assert _typed_margin_mask(idx, None, None) is None
    assert _typed_margin_mask(idx, torch.zeros(1, 1, dtype=torch.long), None) is None


def test_disallowed_type_is_masked():
    # query allows PERSON; candidate (0,1) is an ORG -> disallowed
    idx = _indices([[(0, 1), (2, 3)]])
    mask = _typed_margin_mask(idx, torch.tensor([[PERSON]]), _table([(0, 1, ORG)]))
    assert mask[0, 0, 0].item() is True
    assert mask[0, 0, 1].item() is False   # untyped span: unknown, not wrong


def test_allowed_type_is_not_masked():
    idx = _indices([[(0, 1)]])
    mask = _typed_margin_mask(idx, torch.tensor([[PERSON]]), _table([(0, 1, PERSON)]))
    assert not bool(mask.any())


def test_unconstrained_query_is_left_alone():
    """allowed_bits == 0 means the map has NO OPINION -- it must not penalise anything."""
    idx = _indices([[(0, 1)]])
    mask = _typed_margin_mask(idx, torch.tensor([[0]]), _table([(0, 1, ORG)]))
    assert not bool(mask.any())


def test_multi_type_span_survives_if_any_type_is_allowed():
    """A surface tagged both ORG and PERSON is fine for a PERSON role."""
    idx = _indices([[(0, 1)]])
    mask = _typed_margin_mask(idx, torch.tensor([[PERSON]]), _table([(0, 1, ORG | PERSON)]))
    assert not bool(mask.any())


def test_span_must_match_exactly():
    """A candidate overlapping a typed span but not equal to it carries no type."""
    idx = _indices([[(0, 2)]])
    mask = _typed_margin_mask(idx, torch.tensor([[PERSON]]), _table([(0, 1, ORG)]))
    assert not bool(mask.any())


def test_padding_rows_are_ignored():
    idx = _indices([[(0, 1)]])
    mask = _typed_margin_mask(
        idx, torch.tensor([[PERSON]]), _table([(0, 1, ORG), (-1, -1, -1)])
    )
    assert mask[0, 0, 0].item() is True


def test_gold_is_never_penalised_even_when_disallowed():
    """4% of gold arguments carry a type the map's tail filters dropped."""
    # NOT constant: the margin scales by the batch's own sd, so all-equal logits give
    # sd == 0 and the function correctly declines to do anything at all.
    logits = torch.tensor([[[0.0, 1.0, 2.0]]])
    valid = torch.ones(1, 1, 3, dtype=torch.bool)
    gold = torch.tensor([[[True, False, False]]])
    typed = torch.tensor([[[True, True, False]]])     # gold itself is "disallowed"
    out, used = apply_typed_margin(logits, valid, gold, typed, 1.0)
    assert out[0, 0, 0].item() == 0.0                 # gold untouched
    assert out[0, 0, 1].item() > 1.0                  # the real negative moved up
    assert out[0, 0, 2].item() == 2.0                 # allowed negative untouched
    assert used.item() == 1


def test_mask_is_transposed_with_the_logits():
    """REGRESSION: the shared-pool branch calls the loss with (query, candidate) = (2, 1).

    Every other mask was normalised through `_to_query_candidate` and this one was not, so a
    [B, Q, C] mask was applied against a [B, C, Q] tensor. With Q != C that raises; the test
    uses Q != C so the bug cannot hide.
    """
    batch, n_query, n_cand = 1, 2, 3
    logits = torch.zeros(batch, n_cand, n_query)          # candidate-major
    valid = torch.ones(batch, n_cand, n_query, dtype=torch.bool)
    gold = torch.zeros(batch, n_cand, n_query, dtype=torch.bool)
    gold[0, 0, 0] = True
    typed = torch.zeros(batch, n_query, n_cand, dtype=torch.bool)   # query-major
    typed[0, 0, 1] = True
    loss = proposal_listwise_loss(
        logits, gold, valid, torch.ones(batch, n_query, dtype=torch.bool),
        query_axis=2, candidate_axis=1,
        typed_margin_mask=typed, typed_margin_k=1.0,
    )
    assert torch.isfinite(loss)


def test_k_zero_is_a_no_op():
    logits = torch.randn(1, 2, 4)
    valid = torch.ones(1, 2, 4, dtype=torch.bool)
    gold = torch.zeros(1, 2, 4, dtype=torch.bool)
    gold[0, :, 0] = True
    typed = torch.ones(1, 2, 4, dtype=torch.bool)
    out, used = apply_typed_margin(logits, valid, gold, typed, 0.0)
    assert torch.equal(out, logits) and used.item() == 0

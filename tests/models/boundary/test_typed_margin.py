"""The typed margin: self-normalising, gold-safe, and reaching BOTH listwise paths.

`reranker_listwise_loss` delegates to `proposal_listwise_loss`, and it accepted the two
margin arguments while forwarding NEITHER. A rerank arm would have logged the flag as set
and trained exactly as its control -- a treatment that reads as configured and is inert.
Traced: with the old delegation, k=0.25 and k=5.0 both gave 0.794673.
"""
import torch

from gliner2.models.boundary.losses import (
    apply_typed_margin,
    proposal_listwise_loss,
    reranker_listwise_loss,
)
from gliner2.models.boundary.constants import MASK_LOGIT


def _case(seed=0, B=1, Q=2, C=6):
    torch.manual_seed(seed)
    logits = torch.randn(B, Q, C)
    labels = torch.zeros(B, Q, C)
    labels[0, 0, 0] = 1.0
    labels[0, 1, 1] = 1.0
    valid = torch.ones(B, Q, C, dtype=torch.bool)
    qmask = torch.ones(B, Q, dtype=torch.bool)
    tm = torch.zeros(B, Q, C, dtype=torch.bool)
    tm[0, 0, 3] = True
    tm[0, 1, 4] = True
    return logits, labels, valid, qmask, tm


def test_margin_reaches_the_rerank_path():
    """THE REGRESSION. Fails outright if the wrapper stops forwarding."""
    lg, lb, v, q, tm = _case()
    off = reranker_listwise_loss(lg, lb, v, q, typed_margin_mask=tm, typed_margin_k=0.0)
    on = reranker_listwise_loss(lg, lb, v, q, typed_margin_mask=tm, typed_margin_k=0.25)
    assert not torch.allclose(off, on), "the rerank path ignored the margin"


def test_margin_reaches_the_proposal_path():
    lg, lb, v, q, tm = _case()
    gold = lb > 0.5
    off = proposal_listwise_loss(lg, gold, v, q, typed_margin_mask=tm, typed_margin_k=0.0)
    on = proposal_listwise_loss(lg, gold, v, q, typed_margin_mask=tm, typed_margin_k=0.25)
    assert not torch.allclose(off, on)


def test_loss_is_monotone_in_k():
    lg, lb, v, q, tm = _case()
    vals = [float(reranker_listwise_loss(lg, lb, v, q, typed_margin_mask=tm, typed_margin_k=k))
            for k in (0.0, 0.25, 1.0, 5.0)]
    assert vals == sorted(vals), f"a bigger margin must not lower the loss: {vals}"


def test_gold_is_never_penalised():
    """4% of gold arguments carry a type the map disallows -- the map's own tail filters drop
    legitimate minorities. Marginalising gold would train against the data."""
    lg, lb, v, q, _ = _case()
    gold_only = torch.zeros_like(v)
    gold_only[0, 0, 0] = True          # a GOLD candidate, marked incompatible
    gold_only[0, 1, 1] = True
    off = reranker_listwise_loss(lg, lb, v, q, typed_margin_k=0.0)
    on = reranker_listwise_loss(lg, lb, v, q, typed_margin_mask=gold_only, typed_margin_k=5.0)
    assert torch.allclose(off, on), "gold must be excluded from the margin"


def test_no_mask_or_zero_k_is_bit_identical():
    lg, lb, v, q, tm = _case()
    base = reranker_listwise_loss(lg, lb, v, q)
    assert torch.equal(base, reranker_listwise_loss(lg, lb, v, q, typed_margin_k=0.25))
    assert torch.equal(base, reranker_listwise_loss(lg, lb, v, q, typed_margin_mask=tm,
                                                    typed_margin_k=0.0))


def test_margin_is_k_times_the_live_sd():
    """SELF-NORMALISING: the logits are not normalised and their scale moves with the data --
    measured p5-p95 spread 12.2 (English) to 20.0 (Chinese), sd 2.30 (proposal) vs 7.18
    (rerank). The added amount must track the batch's own sd, not a fixed constant."""
    lg, _, v, _, tm = _case()
    gold = torch.zeros_like(v)
    out, used = apply_typed_margin(lg, v, gold, tm, 0.25)
    sd = lg[v].std()
    delta = (out - lg)[tm]
    assert torch.allclose(delta, 0.25 * sd.expand_as(delta), atol=1e-5)
    assert float(used) == float(tm.sum())


def test_scaling_every_logit_scales_the_margin_with_it():
    lg, _, v, _, tm = _case()
    gold = torch.zeros_like(v)
    d1 = (apply_typed_margin(lg, v, gold, tm, 0.25)[0] - lg)[tm]
    d2 = (apply_typed_margin(lg * 10, v, gold, tm, 0.25)[0] - lg * 10)[tm]
    assert torch.allclose(d2, d1 * 10, atol=1e-4), "a fixed margin would not scale"


def test_all_masked_logits_is_a_safe_no_op():
    lg, _, v, _, tm = _case()
    dead = torch.full_like(lg, MASK_LOGIT)
    gold = torch.zeros_like(v)
    out, used = apply_typed_margin(dead, v, gold, tm, 0.25)
    assert torch.equal(out, dead) and float(used) == 0.0

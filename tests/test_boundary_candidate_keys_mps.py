"""Candidate spans must survive long documents on every backend, MPS included.

`torch.gather` on an int64 tensor rounds through float32 on MPS: above 2^24 only even
integers are representable, so odd values round down. GLiNER2 used to pack a candidate
span into one integer, `start * n + end`, and gather that. For a document with more than
~4,096 boundaries -- every Chinese article, since the word splitter yields about one word
per character -- n^2 crosses 2^24 and a gathered key unpacks to a DIFFERENT span.

Observed 2026-09-03: gold (4340, 4341) with n=5593 packs to 24,277,961, MPS returned
24,277,960, which unpacks to (4340, 4340). `validate_candidate_indices` rejected the
zero-width span and killed training at step 2. A key rounding to a different but still
valid pair would have been silently wrong at inference instead.

The fix carries (start, end) through the permutations rather than the packed key; both
stay below n, where gather is exact. These tests pin that: the components must round-trip,
and both call sites must agree with CPU.
"""

import pytest
import torch

from gliner2.models.boundary.pool import DocumentCandidatePool
from gliner2.models.boundary.proposal import assemble_candidates

mps_only = pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="MPS-specific gather precision"
)

N = 5593  # a 6,000-character Chinese article, ~one word per character


def _pairs(b=2, q=2, p=64, g=8, n=N, seed=0):
    torch.manual_seed(seed)
    starts = torch.randint(0, n - 1, (b, q, p))
    ends = torch.clamp(starts + torch.randint(1, 40, (b, q, p)), max=n - 1)
    scores = torch.randn(b, q, p)
    valid = torch.rand(b, q, p) > 0.2
    query_mask = torch.ones(b, q, dtype=torch.bool)
    gs = torch.randint(0, n - 1, (b, q, g))
    ge = torch.clamp(gs + torch.randint(1, 40, (b, q, g)), max=n - 1)
    return (starts, ends, scores, valid, query_mask,
            torch.stack((gs, ge), -1), torch.ones(b, q, g, dtype=torch.bool))


@mps_only
def test_int64_gather_is_lossy_on_mps_above_2_24():
    """The platform fact the fix exists for. If this ever fails, the fix can be simpler."""
    key = torch.tensor([[24277961]], dtype=torch.long, device="mps")
    gathered = int(torch.gather(key, -1, torch.zeros_like(key)))
    assert gathered != 24277961, (
        "MPS int64 gather is exact now; the component-carrying workaround in "
        "assemble_candidates/_deduplicate_pool can be reconsidered"
    )


@mps_only
def test_assembled_candidates_are_not_degenerate_on_mps():
    s, e, sc, v, qm, gp, gm = _pairs()
    idx, valid, _, _, _ = assemble_candidates(
        *(t.to("mps") for t in (s, e, sc, v, qm)),
        capacity=384, n_boundaries=N,
        gold_pairs=gp.to("mps"), gold_mask=gm.to("mps"), gold_injection_prob=1.0,
    )
    active = idx[valid]
    assert (active[..., 1] > active[..., 0]).all(), "zero-width or inverted candidate span"
    assert (active[..., 1] <= N).all()


@mps_only
def test_assembled_candidates_match_cpu_on_mps():
    s, e, sc, v, qm, gp, gm = _pairs(seed=1)
    kwargs = dict(capacity=384, n_boundaries=N, gold_injection_prob=1.0)
    cpu = assemble_candidates(s, e, sc, v, qm, gold_pairs=gp, gold_mask=gm, **kwargs)
    mps = assemble_candidates(
        *(t.to("mps") for t in (s, e, sc, v, qm)),
        gold_pairs=gp.to("mps"), gold_mask=gm.to("mps"), **kwargs,
    )
    mask = cpu[1]
    assert torch.equal(cpu[1], mps[1].cpu())
    assert torch.equal(cpu[0][mask], mps[0].cpu()[mask])


@mps_only
def test_document_pool_matches_cpu_on_mps():
    torch.manual_seed(2)
    b, q, d = 2, 2, 32
    pool = DocumentCandidatePool(d, pool_boundary_top_k=16, pool_size=64,
                                 min_pool_per_query=8).eval()
    states = torch.randn(b, N, d)
    bmask = torch.ones(b, N, dtype=torch.bool)
    qmask = torch.ones(b, q, dtype=torch.bool)
    sl, el = torch.randn(b, q, N), torch.randn(b, q, N)
    with torch.no_grad():
        cpu = pool(states, bmask, qmask, sl, el)
        mps = pool.to("mps")(states.to("mps"), bmask.to("mps"), qmask.to("mps"),
                             sl.to("mps"), el.to("mps"))
    assert torch.equal(cpu.mask, mps.mask.cpu())
    assert torch.equal(cpu.indices[cpu.mask], mps.indices.cpu()[cpu.mask])
    active = mps.indices.cpu()[mps.mask.cpu()]
    assert (active[..., 1] > active[..., 0]).all(), "zero-width or inverted pooled span"

"""GIST veto: a mined negative the guide prefers to the positive is dropped, not penalized."""
import torch
from gliner2.models.boundary.losses import (
    apply_guide_veto, select_hard_negative_candidates,
)


def _fixture():
    # [B=1, Q=3, C=2].  q0 is the gold type for both candidates.
    # candidate 0: rival q1 outscores q0 in the LIVE model -- a genuine confusion to train on.
    # candidate 1: rival q2 outscores q0 in the live model AND the guide agrees it belongs
    #              there, so it is a false negative and must be vetoed.
    live = torch.tensor([[[5.0, 1.0],
                          [4.0, -8.0],
                          [-8.0, 4.0]]])
    guide = torch.tensor([[[5.0, 1.0],
                           [2.0, -8.0],      # guide does NOT prefer q1 on candidate 0
                           [-8.0, 6.0]]])    # guide DOES prefer q2 on candidate 1
    labels = torch.tensor([[[1.0, 1.0], [0.0, 0.0], [0.0, 0.0]]])
    valid = torch.ones_like(labels, dtype=torch.bool)
    return live, guide, labels, valid


def test_veto_drops_only_the_guide_endorsed_negative():
    live, guide, labels, valid = _fixture()
    mined = select_hard_negative_candidates(
        live, labels, valid, negatives_per_positive=2, minimum_negatives=1,
        query_axis=2, candidate_axis=1,
    )
    assert bool(mined[0, 1, 0]) and bool(mined[0, 2, 1]), "both rivals should be mined"

    # NOTE the axes differ from the mining call above, deliberately: mining is done with the
    # axes swapped (that is what converts span-mining into query-mining), while the veto
    # needs the true axes because it reduces over queries to find each candidate's positive.
    kept = apply_guide_veto(mined, guide, labels, valid, query_axis=1, candidate_axis=2)
    assert bool(kept[0, 1, 0]), "a real confusion must survive -- it is what we train on"
    assert not bool(kept[0, 2, 1]), "a guide-endorsed negative must be vetoed"
    assert bool(kept[0, 0, 0]) and bool(kept[0, 0, 1]), "positives are never vetoed"


def test_margin_raises_the_bar_for_a_veto():
    live, guide, labels, valid = _fixture()
    mined = select_hard_negative_candidates(
        live, labels, valid, negatives_per_positive=2, minimum_negatives=1,
        query_axis=2, candidate_axis=1,
    )
    # guide prefers q2 on candidate 1 by 6.0 - 1.0 = 5.0; a larger margin must save it.
    kept = apply_guide_veto(mined, guide, labels, valid, margin=6.0,
                            query_axis=1, candidate_axis=2)
    assert bool(kept[0, 2, 1]), "margin above the guide's preference should block the veto"


def test_swapped_axes_on_the_veto_silently_do_nothing():
    """Guard the API trap: the veto must get the TRUE axes, not the mining call's swapped ones."""
    live, guide, labels, valid = _fixture()
    mined = select_hard_negative_candidates(
        live, labels, valid, negatives_per_positive=2, minimum_negatives=1,
        query_axis=2, candidate_axis=1,
    )
    right = apply_guide_veto(mined, guide, labels, valid, query_axis=1, candidate_axis=2)
    wrong = apply_guide_veto(mined, guide, labels, valid, query_axis=2, candidate_axis=1)
    assert not torch.equal(right, wrong), "the two differ -- so the axes are load-bearing"
    assert torch.equal(wrong, mined), "swapped axes veto nothing, which is the silent failure"


def test_candidates_without_a_positive_are_untouched():
    live, guide, labels, valid = _fixture()
    labels = torch.zeros_like(labels)          # no positive anywhere
    mined = select_hard_negative_candidates(
        live, labels, valid, negatives_per_positive=1, minimum_negatives=1,
        query_axis=2, candidate_axis=1,
    )
    kept = apply_guide_veto(mined, guide, labels, valid, query_axis=1, candidate_axis=2)
    assert torch.equal(kept, mined), "no positive means no reference, so no veto"


def test_no_veto_when_the_guide_has_no_opinion():
    """A tie at zero is abstention, not endorsement, and must not delete a negative.

    Measured on real cache entries: a gold span scored 0.00 under its own type and 0.00
    under a randomly drawn cross-domain rival. Ranking the rival "above" the positive there
    is float noise.
    """
    live = torch.tensor([[[5.0], [4.0]]])          # [B=1, Q=2, C=1]
    guide = torch.tensor([[[0.0], [0.0001]]])      # both effectively zero
    labels = torch.tensor([[[1.0], [0.0]]])
    valid = torch.ones_like(labels, dtype=torch.bool)
    mined = select_hard_negative_candidates(
        live, labels, valid, negatives_per_positive=1, minimum_negatives=1,
        query_axis=2, candidate_axis=1,
    )
    assert bool(mined[0, 1, 0])
    no_floor = apply_guide_veto(mined, guide, labels, valid, query_axis=1, candidate_axis=2)
    with_floor = apply_guide_veto(mined, guide, labels, valid, floor=0.05,
                                  query_axis=1, candidate_axis=2)
    assert not bool(no_floor[0, 1, 0]), "without a floor the noise wins and vetoes"
    assert bool(with_floor[0, 1, 0]), "with a floor the negative survives"

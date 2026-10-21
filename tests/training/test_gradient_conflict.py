"""gradient_conflict: parameter grouping, the per-task share gate and the weighted rebuild of the span total."""
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "train"))
sys.path.insert(0, str(ROOT))
from gradient_conflict import check_shares, check_weights, group_of, task_loss  # noqa: E402


def _model(proposal_weight=0.3, consistency_weight=0.1):
    settings = NS(soft_iou_aux_weight=0.5, rerank_listwise_weight=0.3, proposal_loss_weight=proposal_weight,
                  abstention_loss_weight=0.2, count_loss_weight=0.1, consistency_loss_weight=consistency_weight)
    head = NS(loss_weights={"start": 1.0, "end": 1.0, "pair": 1.0, "inside": 0.5}, settings=settings,
              _soft_iou_scale=0.5, _consistency_scale=1.0)
    return NS(boundary_head=head)


def _losses():
    """Two tasks with shares of pair and proposal; total = 1.0*pair + 0.3*proposal + 0.1*consistency."""
    ent = {"entities_pair_loss": 0.2, "entities_proposal_loss": 1.0}
    evt = {"events_pair_loss": 0.1, "events_proposal_loss": 2.0}
    t = {k: torch.tensor(v) for k, v in {**ent, **evt}.items()}
    t.update(pair_loss=torch.tensor(0.3), proposal_loss=torch.tensor(3.0), consistency_loss=torch.tensor(0.5))
    t["total_loss"] = torch.tensor(1.0 * 0.3 + 0.3 * 3.0 + 0.1 * 0.5)
    return t


def test_group_of():
    assert group_of("encoder.layers.7.attn.Wqkv.weight") == "encoder.layer07"
    assert group_of("encoder.final_norm.weight") == "encoder.other"
    assert group_of("boundary_head.pair_scorer.query_gate.weight") == "boundary_head.pair_scorer"


def test_task_loss_applies_term_weights():
    losses = _losses()
    assert float(task_loss(_model(), losses, "entities")) == pytest.approx(0.2 + 0.3 * 1.0)
    assert float(task_loss(_model(), losses, "events")) == pytest.approx(0.1 + 0.3 * 2.0)


def test_check_weights_rebuilds_span_total_and_refuses_a_wrong_weight():
    check_weights(_model(), _losses())
    with pytest.raises(AssertionError):
        check_weights(_model(proposal_weight=0.5), _losses())
    with pytest.raises(AssertionError):
        check_weights(_model(consistency_weight=0.0), _losses())


def test_check_shares_refuses_shares_that_do_not_sum():
    losses = _losses()
    check_shares(losses)
    losses["events_pair_loss"] = torch.tensor(0.5)
    with pytest.raises(AssertionError):
        check_shares(losses)

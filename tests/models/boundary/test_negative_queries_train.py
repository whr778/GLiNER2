"""A real training step with ABSENT label queries in the batch.

The unit tests prove the injector builds absent queries; these prove the MODEL trains on
them. The distinction matters because every consumer involved -- `negative_query_ratio`
(0.5) selecting absent queries into the pair loss, `abstention_loss` (0.2) whose target is 1
for an absent query, `count_log_rate_loss` (0.2) supervising a count of zero -- has been live
in every run this project has trained and has always selected from an empty set. A loss that
silently contributes nothing, or goes non-finite on an empty gold set, would be invisible.
"""

import torch

from gliner2.training import ExtractorCollator
from tests.fixtures.tiny_boundary_checkpoint import build_tiny_boundary_model

TEXT = "apple released iphone ."
GOLD = {"entities": {"company": ["apple"], "product": ["iphone"]}}
# `product2` and `venue` are absent: the label is offered, the empty list is the gold.
WITH_ABSENT = {"entities": {"company": ["apple"], "product": ["iphone"],
                            "venue": [], "person": []}}


def _step(schema):
    model = build_tiny_boundary_model()
    model.train()
    collator = ExtractorCollator(model.processor, is_training=True, architecture="boundary")
    batch = collator([(TEXT, schema)])
    out = model(batch)
    return model, out


def test_a_batch_carrying_absent_queries_trains_finitely():
    model, out = _step(WITH_ABSENT)
    assert out.total_loss is not None
    assert torch.isfinite(out.total_loss), "absent queries must not produce a non-finite loss"

    out.total_loss.backward()
    grads = [p.grad for p in model.boundary_head.parameters() if p.grad is not None]
    assert grads and any(float(g.abs().sum()) > 0 for g in grads), \
        "no gradient reached the head, so the absent queries contributed nothing"


def test_absent_queries_CHANGE_the_loss_rather_than_being_ignored():
    """The failure this guards is silence: a query that costs tokens and teaches nothing."""
    torch.manual_seed(0)
    _, gold_out = _step(GOLD)
    torch.manual_seed(0)
    _, absent_out = _step(WITH_ABSENT)

    assert float(gold_out.total_loss) != float(absent_out.total_loss), \
        "adding two absent labels left the loss bit-identical -- they are being dropped"


def test_a_relation_with_ZERO_GOLD_still_trains_finitely():
    """`_relation_loss` builds 'gold-inclusive proposals'. With a zero-gold relation spec the
    proposal set can be empty, which would make the negative contribute nothing at all."""
    schema = {"entities": {"company": ["apple"], "product": ["iphone"]},
              "relations": [{"released": {"head": "apple", "tail": "iphone"}}],
              "absent_relations": ["acquired", "sued"]}
    model, out = _step(schema)
    assert out.total_loss is not None and torch.isfinite(out.total_loss)
    out.total_loss.backward()
    grads = [p.grad for p in model.boundary_head.parameters() if p.grad is not None]
    assert grads and any(float(g.abs().sum()) > 0 for g in grads)


def test_an_absent_EVENT_type_trains_finitely():
    schema = {"events": [{"event_type": "release",
                          "triggers": ["released"],
                          "arguments": [{"role": "agent", "entity": "apple"}]}],
              "absent_events": {"lawsuit": ["plaintiff"], "merger": ["acquirer"]}}
    model, out = _step(schema)
    assert out.total_loss is not None and torch.isfinite(out.total_loss)

"""Can the event loss overfit ONE example -- through both decode paths?

The classic test for a broken loss. It was written after the label-negatives A/B produced
`event_trigger` F1 of exactly 0.0000 in both arms and the suspicion fell on the event loss:
with `event_records: true` events are supervised through the RECORD head rather than the
mention path, which is a genuinely different loss, and "exactly zero" is the signature of a
term that never fires.

It is not broken. Both paths drive the loss down and recover the trigger from a single
example. What produced the scary reading was the HARNESS: `SamplingConfig.remove_events_prob`
is 0.2 and live during training, so on a one-example batch collated once outside the loop, a
single unlucky draw removes the only event and every step sees a zero loss. Any test that
asserts on loss magnitude with a small batch must disable schema dropout first -- this is the
third time that sampling has produced a false reading in this project.
"""

import dataclasses

import torch

from gliner2.training import ExtractorCollator
from tests.fixtures.tiny_boundary_checkpoint import build_tiny_boundary_model

TEXT = "the navy tested a missile on tuesday ."
GOLD = {"events": [{"event_type": "Experiment", "triggers": ["tested"],
                    "arguments": [{"role": "subject", "entity": "the navy"},
                                  {"role": "date", "entity": "tuesday"}]}]}


def _model(event_records: bool):
    torch.manual_seed(0)
    m = build_tiny_boundary_model()
    s = dataclasses.replace(m.boundary_settings, event_records=event_records,
                            enable_records=True)
    m.boundary_settings = s
    m.boundary_head.settings = s
    m.enable_records = True
    sc = m.processor.sampling_config
    for f in dataclasses.fields(sc):
        if f.name.endswith("_prob"):
            setattr(sc, f.name, 0.0)     # see the module docstring
    m.train()
    return m


def _overfit(event_records: bool, steps: int = 150):
    m = _model(event_records)
    collator = ExtractorCollator(m.processor, is_training=True, architecture="boundary",
                                 event_records=event_records)
    batch = collator([(TEXT, GOLD)])
    n_records = sum(len(r) for r in (getattr(batch.targets, "records", None) or []))
    opt = torch.optim.AdamW(m.parameters(), lr=3e-4)
    losses = []
    for _ in range(steps):
        opt.zero_grad()
        out = m(batch)
        losses.append(float(out.total_loss.detach()))
        out.total_loss.backward()
        opt.step()
    m.eval()
    with torch.no_grad():
        pred = m.extract_events(TEXT, {"Experiment": ["subject", "date"]}, threshold=0.3)
    inst = (pred.get("event_extraction") or {}).get("Experiment") or []
    triggers = [t for i in inst if isinstance(i, dict) for t in (i.get("triggers") or [])]
    return n_records, losses, inst, triggers


def test_the_MENTION_path_can_overfit_one_event():
    n_records, losses, inst, triggers = _overfit(event_records=False)

    assert n_records == 0, "event_records is off, so events must NOT become record targets"
    assert losses[0] > 1.0, "a zero starting loss means nothing was supervised at all"
    assert losses[-1] < losses[0] / 4, f"loss did not fall: {losses[0]:.3f} -> {losses[-1]:.3f}"
    assert "tested" in triggers


def test_the_RECORD_path_can_overfit_one_event():
    """`event_records: true` supervises events through the record head -- a different loss,
    and the one under suspicion when the A/B read exactly 0.0000."""
    n_records, losses, inst, triggers = _overfit(event_records=True)

    assert n_records == 1, "the event must become a record target"
    assert losses[0] > 1.0, "a zero starting loss means the record loss never fired"
    assert losses[-1] < losses[0] / 4, f"loss did not fall: {losses[0]:.3f} -> {losses[-1]:.3f}"
    assert "tested" in triggers, f"the record path lost the trigger: {triggers}"
    assert len(inst) < 5, f"degenerate decode: {len(inst)} instances for a one-event document"

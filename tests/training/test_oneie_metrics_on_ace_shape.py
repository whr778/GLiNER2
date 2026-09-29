"""The OneIE-comparable metrics, exercised on ACE-SHAPED gold rather than toy dicts.

WHY A FIXTURE AND NOT MORE INLINE DICTS. `test_argc_metric.py` pins the key shapes on
hand-built records, which proves the functions work and nothing about whether they survive
a real corpus layout. This fixture carries ACE's TAXONOMY -- `Conflict.Attack`, `Life.Die`,
`Personnel.Elect`, roles Attacker/Target/Victim/Place/Person/Entity, dotted entity types
like `PER.Individual` -- in our own JSONL shape, with the three structures that actually
break things:

  * TWO `Conflict.Attack` instances in one document (the case `event_records` exists for,
    and the one a type-keyed decoder collapses);
  * a trigger surface that REPEATS across instances ("elected" twice), which is exactly
    where surface-keying is more lenient than OneIE's offsets;
  * an event with NO trigger, which our strict key drops and Arg-C must keep and count.

NO LDC CONTENT. ace2005 is licensed and must not be redistributed. Every sentence here is
invented; only the taxonomy is ACE's, and a taxonomy is not copyrightable content.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from gliner2.training.eval_metrics import _gold_oneie_sets, _pred_oneie_sets

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "ace_analog.jsonl"


@pytest.fixture(scope="module")
def records():
    assert FIXTURE.is_file(), f"missing fixture {FIXTURE}"
    return [json.loads(line) for line in FIXTURE.open(encoding="utf-8") if line.strip()]


def _as_pred(rec):
    """Turn gold into a PERFECT prediction in `event_extraction` shape."""
    block = {}
    for ev in rec["output"]["events"]:
        block.setdefault(ev["event_type"], []).append(
            {"triggers": list(ev["triggers"]), "arguments": list(ev["arguments"])}
        )
    return {"event_extraction": block}


def test_the_fixture_carries_the_three_hard_shapes(records):
    """The premise. If the fixture loses these, the tests below prove much less."""
    evs = [e for r in records for e in r["output"]["events"]]
    assert len(evs) >= 6
    types_per_doc = [[e["event_type"] for e in r["output"]["events"]] for r in records]
    assert any(len(t) != len(set(t)) for t in types_per_doc), "need a same-type pair"
    assert any(not e["triggers"] for e in evs), "need a trigger-less event"
    trigs = [t for e in evs for t in e["triggers"]]
    assert len(trigs) != len(set(trigs)), "need a repeated trigger surface"


def test_a_perfect_prediction_scores_perfectly_on_all_four(records):
    """The control that makes every other number readable. If gold-as-prediction does not
    score 1.0, the keys disagree with themselves and nothing downstream means anything."""
    for rec in records:
        g, _ = _gold_oneie_sets(rec["output"])
        p, _ = _pred_oneie_sets(_as_pred(rec))
        for key in ("trigi", "trigc", "argi", "argc"):
            assert g[key] == p[key], f"{key} disagrees on a perfect prediction"


def test_the_repeated_trigger_collapses_under_surface_keying(records):
    """DOCUMENTS the deviation rather than hiding it. "elected" fires twice in one
    document; offsets would give two Trig-I nodes, a surface gives ONE. That is the
    leniency recorded in METRICS.md, and it is asserted here so it cannot drift silently."""
    rec = next(r for r in records
               if sum(len(e["triggers"]) for e in r["output"]["events"]) == 2
               and len({t for e in r["output"]["events"] for t in e["triggers"]}) == 1)
    g, _ = _gold_oneie_sets(rec["output"])
    assert len(g["trigi"]) == 1, "two occurrences of one surface are ONE Trig-I key"
    assert len(g["trigc"]) == 1, "and one Trig-C key, since both share the event type"
    # AND THE ARGUMENTS COLLAPSE TOO, which is the criterion working, not a bug.
    # Marin is `Person` in BOTH Elect instances, and Arg-C drops the instance -- so the
    # two contribute ONE key, while the two different `Entity` fillers stay distinct:
    #   (personnel.elect, person, marin)        <- both instances, collapsed
    #   (personnel.elect, entity, the council)
    #   (personnel.elect, entity, the union)
    # This is precisely the leniency that makes Arg-C unusable as an internal metric: the
    # same person elected twice is indistinguishable from elected once.
    assert len(g["argc"]) == 3, sorted(g["argc"])
    assert ("personnel.elect", "person", "marin") in g["argc"]
    assert len({k for k in g["argc"] if k[1] == "entity"}) == 2


def test_the_trigger_less_event_is_counted_not_dropped(records):
    rec = next(r for r in records
               if any(not e["triggers"] for e in r["output"]["events"]))
    g, triggerless = _gold_oneie_sets(rec["output"])
    assert triggerless == 1
    assert any(k[0] == "transaction.transfer-ownership" for k in g["argc"])
    assert not g["trigi"], "it contributes no trigger key, correctly"


def test_multi_instance_same_type_keeps_its_arguments_distinct(records):
    """The event_records case. Two Conflict.Attack instances must not pool their
    arguments into one key set."""
    rec = records[0]
    g, _ = _gold_oneie_sets(rec["output"])
    attack_args = {k for k in g["argc"] if k[0] == "conflict.attack"}
    assert ("conflict.attack", "target", "the market") in attack_args
    assert ("conflict.attack", "target", "the bridge") in attack_args, (
        "the second instance's Target must survive as its own key"
    )

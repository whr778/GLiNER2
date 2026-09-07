"""The shared annotator rules live in markdown and are loaded, not restated.

The point of `tools/data/annotation/GUIDELINES.md` is that reviewing it IS reviewing the
prompt. That only holds if every annotator actually loads it, so these tests fail when a
new annotator reinvents a rule locally instead.
"""
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools" / "data"))
sys.path.insert(0, str(ROOT / "tools"))

from annotation import GUIDELINES, available, rules  # noqa: E402

ANNOTATORS = sorted((ROOT / "tools" / "data").glob("annotate_*.py"))


def test_guidelines_file_defines_the_expected_rules():
    assert set(available()) == {"json_only", "verbatim", "no_inference", "minority",
                               "ambiguity", "uncertain_field"}


def test_rules_returns_the_markdown_bytes_verbatim():
    """No paraphrase between the document and the prompt -- that gap is the whole risk."""
    text = GUIDELINES.read_text(encoding="utf-8")
    body = re.search(r"<!--\s*rule:\s*json_only\s*-->\n(.*?)\n<!--\s*end\s*-->",
                     text, re.DOTALL).group(1)
    assert rules("json_only") == " ".join(body.split())


def test_unknown_rule_raises_rather_than_contributing_nothing():
    """A prompt silently missing a rule it believes it has is the defect being prevented."""
    with pytest.raises(KeyError):
        rules("no_such_rule")


def test_rules_compose_in_the_order_requested():
    assert rules("json_only", "minority") != rules("minority", "json_only")
    assert rules("json_only", "minority").startswith(rules("json_only"))


@pytest.mark.parametrize("path", ANNOTATORS, ids=lambda p: p.name)
def test_every_annotator_loads_the_shared_rules(path):
    src = path.read_text(encoding="utf-8")
    assert "from annotation import rules" in src, (
        f"{path.name} does not load the shared guidelines; a locally restated rule can "
        f"drift from the reviewed document without any error")
    assert "SYSTEM = (" in src and "+ rules(" in src, (
        f"{path.name} does not build SYSTEM from the shared rules")


@pytest.mark.parametrize("path", ANNOTATORS, ids=lambda p: p.name)
def test_annotators_do_not_restate_shared_rules_locally(path):
    """One statement per rule. Two makes the shared file non-authoritative."""
    m = re.search(r"SYSTEM = \(\n(.*?)\n\) \+ ", path.read_text(encoding="utf-8"), re.DOTALL)
    assert m, f"{path.name}: unexpected SYSTEM shape"
    task = m.group(1).lower()
    for banned in ("single json object", "character for character", "never infer beyond"):
        assert banned not in task, (
            f"{path.name} restates '{banned}' in its task text; it comes from GUIDELINES.md")


def test_gate_does_not_get_the_minority_rule():
    """annotate_gate's fallback `no_toll` is a REAL balanced class, not an uncertainty
    sink, and the gate's measured failure mode is false positives. Discouraging the
    negative would make that worse -- so the anti-fallback rule must not apply here."""
    src = (ROOT / "tools" / "data" / "annotate_gate.py").read_text(encoding="utf-8")
    spec = re.search(r"SYSTEM = \(\n.*?\n\) \+ \" \" \+ rules\(([^)]*)\)", src, re.DOTALL)
    assert spec and "minority" not in spec.group(1)


def test_ambiguity_rule_ranks_guessing_above_omitting():
    """Omission is NOT neutral in this pipeline: mint_entity_negatives seeds 12 absent
    types per record with [], so ~1 in 9 omissions becomes an explicit "not present".
    The rule must therefore prefer a considered guess, and must still forbid the
    catch-all -- if this text is ever softened, that reasoning went with it."""
    text = rules("ambiguity").lower()
    assert "emit all of them" in text, "the multi-label escape must come first"
    assert "considered guess is more useful than an omission" in text
    assert "never resolve an ambiguity by reaching for a catch-all" in text


def test_gate_gets_neither_minority_nor_ambiguity():
    """annotate_gate is a binary relevance filter whose fallback is a real class and
    whose measured failure mode is false positives. Neither rule belongs there."""
    import re
    src = (ROOT / "tools" / "data" / "annotate_gate.py").read_text(encoding="utf-8")
    spec = re.search(r"SYSTEM = \(\n.*?\n\) \+ \" \" \+ rules\(([^)]*)\)", src, re.DOTALL)
    assert spec
    assert "minority" not in spec.group(1) and "ambiguity" not in spec.group(1)


def test_uncertain_is_a_field_and_says_so():
    """A trained `Uncertain` CLASS rebuilds the sink it exists to avoid, so the rule must
    keep saying that nothing is filed under it."""
    text = rules("uncertain_field").lower()
    assert "not a category" in text and "nothing is filed under it" in text
    assert "never use it to avoid deciding" in text


def test_uncertain_types_are_never_seeded_as_negatives():
    """The whole point: 'declined to use it, therefore absent' is backwards for a type the
    annotator considered and could not decide."""
    import random
    sys.path.insert(0, str(ROOT / "tools" / "data" / "synthetic"))
    from validate import mint_entity_negatives

    ents = {"person": ["a"]}
    mint_entity_negatives(ents, random.Random(0), 12, uncertain=["weapon", "aircraft"])
    seeded = {k for k, v in ents.items() if not v}
    assert len(seeded) == 12, "still seeds the requested number of negatives"
    assert not ({"weapon", "aircraft"} & seeded), "uncertain types must never be asserted absent"


def test_negative_seeding_is_unchanged_without_uncertainty():
    """The default path must behave exactly as before -- this is an addition, not a change."""
    import random
    sys.path.insert(0, str(ROOT / "tools" / "data" / "synthetic"))
    from validate import mint_entity_negatives

    a, b = {"person": ["x"]}, {"person": ["x"]}
    mint_entity_negatives(a, random.Random(7), 12)
    mint_entity_negatives(b, random.Random(7), 12, uncertain=[])
    assert a == b


# --- structure scoring shape tolerance (2026-09-07) -------------------------------------

def test_scorer_reads_greedy_and_joint_field_shapes_identically():
    """The two decode paths emit DIFFERENT shapes for the same field, and the scorer used
    to silently drop one of them.

    Greedy emits {"text": ...} per field; joint emits [{"text": ...}], because the joint
    path compiles non-anchor fields as list-valued while greedy compiles them scalar --
    the same schema, two cardinalities. `_pred_structure_set` had a dict branch and a str
    branch and no else, so every correctly-extracted joint field was discarded BEFORE
    comparison. Measured: that alone produced "joint decode destroys structures"
    (structure F1 0.1208 -> 0.0343) on a decode that had found the same fields.
    """
    from gliner2.training.eval_metrics import _pred_structure_set
    greedy = {"record": [{"ticket_id": {"text": "HD-1"}, "reporter": None}]}
    joint = {"record": [{"ticket_id": [{"text": "HD-1"}], "reporter": []}]}
    assert _pred_structure_set(greedy) == _pred_structure_set(joint)
    assert _pred_structure_set(joint) == {("record", "ticket_id", "HD-1")}


def test_scorer_keeps_every_filler_of_a_genuine_multi_value_field():
    """Tolerating lists must not collapse a real multi-instance field to its first value."""
    from gliner2.training.eval_metrics import _pred_structure_set
    multi = {"record": [{"author": [{"text": "Chen"}, {"text": "Okafor"}]}]}
    assert _pred_structure_set(multi) == {("record", "author", "Chen"),
                                          ("record", "author", "Okafor")}


def test_scorer_still_accepts_bare_strings():
    """include_confidence=False emits a bare string; that path predates both of the above."""
    from gliner2.training.eval_metrics import _pred_structure_set
    assert _pred_structure_set({"record": [{"f": "v"}]}) == {("record", "f", "v")}
    assert _pred_structure_set({"record": [{"f": ["a", "b"]}]}) == {("record", "f", "a"),
                                                                   ("record", "f", "b")}

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
    assert set(available()) == {"json_only", "verbatim", "no_inference", "minority"}


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

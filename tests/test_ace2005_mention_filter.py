"""ACE 2005 mention-type filtering + cascade, on a synthetic .apf.xml/.sgm pair.

Validates that filtering an entity mention by type (NAM/NOM/PRO) cascades to the
relations and event arguments that referenced it, and that non-entity event
arguments (e.g. time expressions, whose REFID is not an entity mention) survive.

NOTE: the fixture encodes the assumed raw-LDC APF structure -- `TYPE` attribute
on <entity_mention>, and mention-level REFIDs on both relation_mention_argument
and event_mention_argument. If real ACE data is shaped differently, these tests
and the converter's reads move together.
"""

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools" / "data"))
from convert_ace2005 import iter_apf_files, parse_apf  # noqa: E402
from _mention_filter import MentionFilter  # noqa: E402


SGM = """<DOC>
<TEXT>
John Smith works for the company . He attacked the base in Baghdad today .
</TEXT>
</DOC>
"""

APF = """<?xml version="1.0"?>
<source_file>
 <document DOCID="TEST">
  <entity ID="E1" TYPE="PER">
    <entity_mention ID="E1-1" TYPE="NAM"><extent><charseq>John Smith</charseq></extent></entity_mention>
    <entity_mention ID="E1-2" TYPE="PRO"><extent><charseq>He</charseq></extent></entity_mention>
  </entity>
  <entity ID="E2" TYPE="ORG">
    <entity_mention ID="E2-1" TYPE="NOM"><extent><charseq>the company</charseq></extent></entity_mention>
  </entity>
  <entity ID="E3" TYPE="GPE">
    <entity_mention ID="E3-1" TYPE="NAM"><extent><charseq>Baghdad</charseq></extent></entity_mention>
  </entity>
  <entity ID="E4" TYPE="FAC">
    <entity_mention ID="E4-1" TYPE="NOM"><extent><charseq>the base</charseq></extent></entity_mention>
  </entity>
  <relation ID="R1" TYPE="ORG-AFF">
    <relation_mention ID="R1-1">
      <relation_mention_argument REFID="E1-1" ROLE="Arg-1"/>
      <relation_mention_argument REFID="E2-1" ROLE="Arg-2"/>
    </relation_mention>
  </relation>
  <relation ID="R2" TYPE="PHYS">
    <relation_mention ID="R2-1">
      <relation_mention_argument REFID="E1-2" ROLE="Arg-1"/>
      <relation_mention_argument REFID="E4-1" ROLE="Arg-2"/>
    </relation_mention>
  </relation>
  <event ID="EV1" TYPE="Conflict" SUBTYPE="Attack">
    <event_mention ID="EV1-1">
      <anchor><charseq>attacked</charseq></anchor>
      <event_mention_argument REFID="E1-2" ROLE="Attacker"><extent><charseq>He</charseq></extent></event_mention_argument>
      <event_mention_argument REFID="E4-1" ROLE="Target"><extent><charseq>the base</charseq></extent></event_mention_argument>
      <event_mention_argument REFID="E3-1" ROLE="Place"><extent><charseq>Baghdad</charseq></extent></event_mention_argument>
      <event_mention_argument REFID="T1-1" ROLE="Time-Within"><extent><charseq>today</charseq></extent></event_mention_argument>
    </event_mention>
  </event>
 </document>
</source_file>
"""


def _write_pair(tmp_path):
    (tmp_path / "doc.sgm").write_text(SGM, encoding="utf-8")
    apf = tmp_path / "doc.apf.xml"
    apf.write_text(APF, encoding="utf-8")
    return apf


def _roles(event):
    return {a["role"] for a in event["arguments"]}


def _rel_names(out):
    return {next(iter(r)) for r in out.get("relations", [])}


def test_no_filter_keeps_everything(tmp_path):
    out = parse_apf(_write_pair(tmp_path), keep_subtypes=True)["output"]
    assert out["entities"]["PER"] == ["John Smith", "He"]
    assert out["entities"]["ORG"] == ["the company"]
    assert out["entities"]["GPE"] == ["Baghdad"]
    assert out["entities"]["FAC"] == ["the base"]
    assert _rel_names(out) == {"ORG-AFF", "PHYS"}
    ev = out["events"][0]
    assert ev["event_type"] == "Conflict.Attack"
    assert _roles(ev) == {"Attacker", "Target", "Place", "Time-Within"}


def test_drop_pronoun_cascades(tmp_path):
    stats = Counter()
    out = parse_apf(_write_pair(tmp_path), keep_subtypes=True,
                    mention_filter=MentionFilter({"NAM", "NOM"}), stats=stats)["output"]
    # PRO mention "He" gone from PER; everything else stays
    assert out["entities"]["PER"] == ["John Smith"]
    # PHYS (PRO head) dropped; ORG-AFF (NAM+NOM) survives
    assert _rel_names(out) == {"ORG-AFF"}
    # Attacker (PRO) dropped; Target/Place (entities) and Time-Within (non-entity) kept
    assert _roles(out["events"][0]) == {"Target", "Place", "Time-Within"}
    assert stats["filtered_mentions"] == 1
    assert stats["filtered_relations"] == 1
    assert stats["filtered_event_args"] == 1


def test_nam_only_drops_nom_and_pro(tmp_path):
    out = parse_apf(_write_pair(tmp_path), keep_subtypes=True,
                    mention_filter=MentionFilter({"NAM"}))["output"]
    # only NAM mentions survive; NOM-only entity types vanish entirely
    assert out["entities"]["PER"] == ["John Smith"]
    assert out["entities"]["GPE"] == ["Baghdad"]
    assert "ORG" not in out["entities"]
    assert "FAC" not in out["entities"]
    # both relations need a filtered mention -> all dropped
    assert "relations" not in out
    # only Place (NAM) and the non-entity Time-Within survive
    assert _roles(out["events"][0]) == {"Place", "Time-Within"}


# Fixture WITH <head> spans that differ from <extent>, to exercise head vs extent
# offsets across entities, relations, and event arguments.
SGM_HEAD = """<DOC>
<TEXT>
The vice president met Acme Corp in Paris .
</TEXT>
</DOC>
"""

APF_HEAD = """<?xml version="1.0"?>
<source_file>
 <document DOCID="TESTHEAD">
  <entity ID="E1" TYPE="PER">
    <entity_mention ID="E1-1" TYPE="NOM">
      <extent><charseq>The vice president</charseq></extent>
      <head><charseq>president</charseq></head>
    </entity_mention>
  </entity>
  <entity ID="E2" TYPE="ORG">
    <entity_mention ID="E2-1" TYPE="NAM">
      <extent><charseq>Acme Corp</charseq></extent>
      <head><charseq>Acme Corp</charseq></head>
    </entity_mention>
  </entity>
  <entity ID="E3" TYPE="GPE">
    <entity_mention ID="E3-1" TYPE="NAM">
      <extent><charseq>Paris</charseq></extent>
      <head><charseq>Paris</charseq></head>
    </entity_mention>
  </entity>
  <relation ID="R1" TYPE="ORG-AFF">
    <relation_mention ID="R1-1">
      <relation_mention_argument REFID="E1-1" ROLE="Arg-1"/>
      <relation_mention_argument REFID="E2-1" ROLE="Arg-2"/>
    </relation_mention>
  </relation>
  <event ID="EV1" TYPE="Contact" SUBTYPE="Meet">
    <event_mention ID="EV1-1">
      <anchor><charseq>met</charseq></anchor>
      <event_mention_argument REFID="E1-1" ROLE="Entity"><extent><charseq>The vice president</charseq></extent></event_mention_argument>
      <event_mention_argument REFID="E3-1" ROLE="Place"><extent><charseq>Paris</charseq></extent></event_mention_argument>
    </event_mention>
  </event>
 </document>
</source_file>
"""


def _write_head_pair(tmp_path):
    (tmp_path / "doc.sgm").write_text(SGM_HEAD, encoding="utf-8")
    apf = tmp_path / "doc.apf.xml"
    apf.write_text(APF_HEAD, encoding="utf-8")
    return apf


def test_head_offsets_are_the_default(tmp_path):
    # Default: head spans, so "The vice president" -> "president" everywhere.
    out = parse_apf(_write_head_pair(tmp_path), keep_subtypes=True)["output"]
    assert out["entities"]["PER"] == ["president"]
    assert out["entities"]["ORG"] == ["Acme Corp"]  # head == extent
    # Relation Arg-1 uses the same head surface.
    assert out["relations"] == [{"ORG-AFF": {"head": "president", "tail": "Acme Corp"}}]
    # Event entity-argument inherits the head surface too.
    ev = out["events"][0]
    entity_arg = next(a for a in ev["arguments"] if a["role"] == "Entity")
    assert entity_arg["entity"] == "president"


def test_extent_offsets_flag_keeps_full_span(tmp_path):
    # use_head=False -> full extent, determiners/modifiers retained.
    out = parse_apf(_write_head_pair(tmp_path), keep_subtypes=True, use_head=False)["output"]
    assert out["entities"]["PER"] == ["The vice president"]
    assert out["relations"] == [{"ORG-AFF": {"head": "The vice president", "tail": "Acme Corp"}}]
    ev = out["events"][0]
    entity_arg = next(a for a in ev["arguments"] if a["role"] == "Entity")
    assert entity_arg["entity"] == "The vice president"


def test_iter_apf_files_only_reads_adj_folder(tmp_path):
    """The raw ACE 2005 LDC delivery re-annotates the same documents in
    several sibling folders per genre (adj, fp1, fp2, timex2norm, ...).
    Only 'adj' (adjudicated gold) should be converted -- walking the whole
    tree duplicates every document once per folder."""
    for folder in ("adj", "fp1", "fp2", "timex2norm"):
        d = tmp_path / "bc" / folder
        d.mkdir(parents=True)
        (d / "DOC1.apf.xml").write_text("<x/>", encoding="utf-8")

    found = sorted(p.relative_to(tmp_path) for p in iter_apf_files(tmp_path))
    assert [str(p) for p in found] == ["bc/adj/DOC1.apf.xml"]


def test_iter_apf_files_no_adj_folder_yields_nothing(tmp_path):
    """A genre with no 'adj' folder contributes no documents -- there is no
    fallback to a non-adjudicated annotation pass."""
    d = tmp_path / "bn" / "timex2norm"
    d.mkdir(parents=True)
    (d / "DOC1.apf.xml").write_text("<x/>", encoding="utf-8")

    assert list(iter_apf_files(tmp_path)) == []

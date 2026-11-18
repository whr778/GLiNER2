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


def _write_ldc_full_pair(tmp_path):
    """Realistic LDC v5.1.1 shape: <BODY><HEADLINE> + <TEXT>, &amp; entities,
    START/END charseq offsets, entity subtypes, relation subtypes, multi-mention
    events, and timex2 mentions."""
    import html
    import re

    sgm = (
        "<DOC>\n"
        "<DOCID> NW_ENG_20261118.0001 </DOCID>\n"
        "<BODY>\n"
        "<HEADLINE>\n"
        "AT&amp;T acquires Acme Wireless in Dallas\n"
        "</HEADLINE>\n"
        "<TEXT>\n"
        "John Smith works for AT&amp;T Corp in Dallas. He met Mary Lee there on Tuesday. "
        "Later, Smith attacked the facility, and the strike damaged the plant.\n"
        "</TEXT>\n"
        "</BODY>\n"
        "</DOC>\n"
    )
    tagless = html.unescape(re.sub(r"<[^>]+>", "", sgm))

    def span(sub, start_from=0):
        i = tagless.index(sub, start_from)
        return i, i + len(sub) - 1

    s_att0, e_att0 = span("AT&T")
    s_dal0, e_dal0 = span("Dallas", e_att0 + 1)
    s_js1, e_js1 = span("John Smith")
    s_att1, e_att1 = span("AT&T Corp")
    s_he, e_he = span("He", e_att1)
    s_met, e_met = span("met", s_he)
    s_ml, e_ml = span("Mary Lee")
    s_there, e_there = span("there", s_ml)
    s_tue, e_tue = span("Tuesday")
    s_js2, e_js2 = span("Smith", s_tue)
    s_atk1, e_atk1 = span("attacked")
    s_fac_ext, e_fac_ext = span("the facility")
    s_fac_hd, e_fac_hd = span("facility", s_fac_ext)
    s_atk2, e_atk2 = span("strike")
    s_plt_ext, e_plt_ext = span("the plant")
    s_plt_hd, e_plt_hd = span("plant", s_plt_ext)

    apf = f"""<?xml version="1.0"?>
<source_file>
 <document DOCID="NW_ENG_20261118.0001">
  <entity ID="E1" TYPE="PER" SUBTYPE="Individual">
    <entity_mention ID="E1-1" TYPE="NAM">
      <extent><charseq START="{s_js1}" END="{e_js1}">John Smith</charseq></extent>
      <head><charseq START="{s_js1}" END="{e_js1}">John Smith</charseq></head>
    </entity_mention>
    <entity_mention ID="E1-2" TYPE="PRO">
      <extent><charseq START="{s_he}" END="{e_he}">He</charseq></extent>
      <head><charseq START="{s_he}" END="{e_he}">He</charseq></head>
    </entity_mention>
    <entity_mention ID="E1-3" TYPE="NAM">
      <extent><charseq START="{s_js2}" END="{e_js2}">Smith</charseq></extent>
      <head><charseq START="{s_js2}" END="{e_js2}">Smith</charseq></head>
    </entity_mention>
  </entity>
  <entity ID="E2" TYPE="ORG" SUBTYPE="Commercial">
    <entity_mention ID="E2-0" TYPE="NAM">
      <extent><charseq START="{s_att0}" END="{e_att0}">AT&amp;T</charseq></extent>
      <head><charseq START="{s_att0}" END="{e_att0}">AT&amp;T</charseq></head>
    </entity_mention>
    <entity_mention ID="E2-1" TYPE="NAM">
      <extent><charseq START="{s_att1}" END="{e_att1}">AT&amp;T Corp</charseq></extent>
      <head><charseq START="{s_att1}" END="{e_att1}">AT&amp;T Corp</charseq></head>
    </entity_mention>
  </entity>
  <entity ID="E3" TYPE="GPE" SUBTYPE="Population-Center">
    <entity_mention ID="E3-1" TYPE="NAM">
      <extent><charseq START="{s_dal0}" END="{e_dal0}">Dallas</charseq></extent>
      <head><charseq START="{s_dal0}" END="{e_dal0}">Dallas</charseq></head>
    </entity_mention>
    <entity_mention ID="E3-2" TYPE="PRO">
      <extent><charseq START="{s_there}" END="{e_there}">there</charseq></extent>
      <head><charseq START="{s_there}" END="{e_there}">there</charseq></head>
    </entity_mention>
  </entity>
  <entity ID="E4" TYPE="PER" SUBTYPE="Individual">
    <entity_mention ID="E4-1" TYPE="NAM">
      <extent><charseq START="{s_ml}" END="{e_ml}">Mary Lee</charseq></extent>
      <head><charseq START="{s_ml}" END="{e_ml}">Mary Lee</charseq></head>
    </entity_mention>
  </entity>
  <entity ID="E5" TYPE="FAC" SUBTYPE="Plant">
    <entity_mention ID="E5-1" TYPE="NOM">
      <extent><charseq START="{s_fac_ext}" END="{e_fac_ext}">the facility</charseq></extent>
      <head><charseq START="{s_fac_hd}" END="{e_fac_hd}">facility</charseq></head>
    </entity_mention>
    <entity_mention ID="E5-2" TYPE="NOM">
      <extent><charseq START="{s_plt_ext}" END="{e_plt_ext}">the plant</charseq></extent>
      <head><charseq START="{s_plt_hd}" END="{e_plt_hd}">plant</charseq></head>
    </entity_mention>
  </entity>
  <timex2 ID="T1">
    <timex2_mention ID="T1-1">
      <extent><charseq START="{s_tue}" END="{e_tue}">Tuesday</charseq></extent>
    </timex2_mention>
  </timex2>
  <relation ID="R1" TYPE="ORG-AFF" SUBTYPE="Employment">
    <relation_mention ID="R1-1">
      <relation_mention_argument REFID="E1-1" ROLE="Arg-1"/>
      <relation_mention_argument REFID="E2-1" ROLE="Arg-2"/>
    </relation_mention>
  </relation>
  <event ID="EV1" TYPE="Contact" SUBTYPE="Meet">
    <event_mention ID="EV1-1">
      <anchor><charseq START="{s_met}" END="{e_met}">met</charseq></anchor>
      <event_mention_argument REFID="E1-2" ROLE="Entity"><extent><charseq START="{s_he}" END="{e_he}">He</charseq></extent></event_mention_argument>
      <event_mention_argument REFID="E4-1" ROLE="Entity"><extent><charseq START="{s_ml}" END="{e_ml}">Mary Lee</charseq></extent></event_mention_argument>
      <event_mention_argument REFID="E3-2" ROLE="Place"><extent><charseq START="{s_there}" END="{e_there}">there</charseq></extent></event_mention_argument>
      <event_mention_argument REFID="T1-1" ROLE="Time-Within"><extent><charseq START="{s_tue}" END="{e_tue}">Tuesday</charseq></extent></event_mention_argument>
    </event_mention>
  </event>
  <event ID="EV2" TYPE="Conflict" SUBTYPE="Attack">
    <event_mention ID="EV2-1">
      <anchor><charseq START="{s_atk1}" END="{e_atk1}">attacked</charseq></anchor>
      <event_mention_argument REFID="E1-3" ROLE="Attacker"><extent><charseq START="{s_js2}" END="{e_js2}">Smith</charseq></extent></event_mention_argument>
      <event_mention_argument REFID="E5-1" ROLE="Target"><extent><charseq START="{s_fac_ext}" END="{e_fac_ext}">the facility</charseq></extent></event_mention_argument>
    </event_mention>
    <event_mention ID="EV2-2">
      <anchor><charseq START="{s_atk2}" END="{e_atk2}">strike</charseq></anchor>
      <event_mention_argument REFID="E5-2" ROLE="Target"><extent><charseq START="{s_plt_ext}" END="{e_plt_ext}">the plant</charseq></extent></event_mention_argument>
    </event_mention>
  </event>
 </document>
</source_file>
"""
    (tmp_path / "doc.sgm").write_text(sgm, encoding="utf-8")
    apf_path = tmp_path / "doc.apf.xml"
    apf_path.write_text(apf, encoding="utf-8")
    return apf_path


def test_canonical_taxonomy_and_html_unescape_and_headline(tmp_path):
    rec = parse_apf(_write_ldc_full_pair(tmp_path))
    out = rec["output"]
    # Canonical default: top-level entities (7) + top-level relations (6) + subtyped events (33)
    assert set(out["entities"]) == {"PER", "ORG", "GPE", "FAC"}
    assert out["entities"]["ORG"] == ["AT&T", "AT&T Corp"]
    assert out["relations"] == [{"ORG-AFF": {"head": "John Smith", "tail": "AT&T Corp"}}]
    assert [e["event_type"] for e in out["events"]] == [
        "Contact.Meet", "Conflict.Attack", "Conflict.Attack"
    ]


def test_coref_clusters_and_offset_spans_slice_verbatim_after_dumps_record(tmp_path):
    import json
    from _split import dumps_record

    apf_path = _write_ldc_full_pair(tmp_path)
    rec = json.loads(dumps_record(parse_apf(apf_path, offsets=True)))
    text = rec["input"]

    clusters = rec["coref_clusters"]
    assert len(clusters) == 5
    for cl in clusters:
        for m in cl:
            assert text[m["start"]:m["end"]] == m["text"]

    # Headline "Dallas" (start 31) is placed rather than body "Dallas" (start 72).
    dallas_cluster = next(c for c in clusters if any(m["text"] == "Dallas" for m in c))
    assert [m["text"] for m in dallas_cluster] == ["Dallas", "there"]
    assert dallas_cluster[0]["start"] < text.index("John Smith")

    for ev in rec["output"]["events"]:
        for t in ev["triggers"]:
            assert text[t["start"]:t["end"]] == t["text"]
        for arg in ev["arguments"]:
            ent = arg["entity"]
            assert text[ent["start"]:ent["end"]] == ent["text"]

    # Mention filter cascades into coref_clusters too (PRO removed).
    rec_no_pro = parse_apf(apf_path, mention_filter=MentionFilter({"NAM", "NOM"}))
    flat_texts = {m["text"] for c in rec_no_pro["coref_clusters"] for m in c}
    assert "He" not in flat_texts
    assert "there" not in flat_texts
    assert {"AT&T", "AT&T Corp", "John Smith", "Smith"} <= flat_texts


def test_merge_event_mentions_clusters_coreferent_triggers_and_args(tmp_path):
    apf_path = _write_ldc_full_pair(tmp_path)
    rec = parse_apf(apf_path, offsets=True, merge_event_mentions=True)
    events = rec["output"]["events"]
    assert len(events) == 2
    attack = next(e for e in events if e["event_type"] == "Conflict.Attack")
    assert [t["text"] for t in attack["triggers"]] == ["attacked", "strike"]
    assert [(a["role"], a["entity"]["text"]) for a in attack["arguments"]] == [
        ("Attacker", "Smith"),
        ("Target", "facility"),
        ("Target", "plant"),
    ]


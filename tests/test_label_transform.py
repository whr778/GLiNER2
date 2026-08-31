"""Unit tests for the label roll-up / remap transforms in tools/train/train.py.

The transform logic lives in the config-driven trainer; we load it by path
(it is a script, not an installed module). Covers roll-up, map-after-rollup
ordering, and -- critically -- merge-on-collision so no entities are lost when
two child labels roll up to the same parent.
"""

import importlib.util
from pathlib import Path

import pytest

from gliner2.training.data import DataLoader_Factory

_TRAIN_PY = Path(__file__).resolve().parents[1] / "tools" / "train" / "train.py"


def _load_train_module():
    spec = importlib.util.spec_from_file_location("train_cli", _TRAIN_PY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


train = _load_train_module()


def test_rollup_keeps_parent_segment():
    fn = train._label_fn(rollup=True, separator=".", mapping={})
    assert fn("ORG.Media") == "ORG"
    assert fn("Cognitive.IdentifyCategorize.Unspecified") == "Cognitive"
    assert fn("PER") == "PER"  # no separator -> unchanged


def test_map_applies_after_rollup():
    fn = train._label_fn(rollup=True, separator=".", mapping={"ORG": "ORGANIZATION"})
    assert fn("ORG.Media") == "ORGANIZATION"  # rolled to ORG, then mapped
    assert fn("LOC.Address") == "LOC"          # unmapped parent passes through


def test_map_without_rollup_uses_full_label():
    fn = train._label_fn(rollup=False, separator=".", mapping={"ORG.Media": "MEDIA"})
    assert fn("ORG.Media") == "MEDIA"
    assert fn("ORG.Government") == "ORG.Government"


def test_entities_merge_no_surfaces_lost():
    """Two children of one parent must merge, not overwrite."""
    fn = train._label_fn(rollup=True, separator=".", mapping={})
    ents = {"ORG.Media": ["BBC", "CNN"], "ORG.Government": ["CNN", "Senate"], "LOC.Address": ["5th Ave"]}
    out = train._transform_entities(ents, fn)
    assert out == {"ORG": ["BBC", "CNN", "Senate"], "LOC": ["5th Ave"]}  # CNN deduped, nothing lost


def test_entity_descriptions_keep_first_nonempty_on_collision():
    fn = train._label_fn(rollup=True, separator=".", mapping={})
    desc = {"ORG.Media": "a media org", "ORG.Government": "a gov org"}
    assert train._transform_descriptions(desc, fn) == {"ORG": "a media org"}


def test_relations_name_rolled_up():
    fn = train._label_fn(rollup=True, separator=".", mapping={})
    rels = [{"PHYS.Located": {"head": "a", "tail": "b"}}]
    assert train._transform_relations(rels, fn) == [{"PHYS": {"head": "a", "tail": "b"}}]


def test_events_type_and_role_rolled_up():
    fn = train._label_fn(rollup=True, separator=".", mapping={})
    events = [{
        "event_type": "Conflict.Attack",
        "triggers": ["bombed"],
        "arguments": [{"role": "Place.City", "entity": "Paris"}],
    }]
    out = train._transform_events(events, fn)
    assert out[0]["event_type"] == "Conflict"
    assert out[0]["triggers"] == ["bombed"]
    assert out[0]["arguments"][0] == {"role": "Place", "entity": "Paris"}


def test_classifications_labels_and_true_label_rolled_up():
    fn = train._label_fn(rollup=True, separator=".", mapping={})
    cls = [{"task": "topic", "labels": ["Sci.Physics", "Sci.Math"], "true_label": "Sci.Physics"}]
    out = train._transform_classifications(cls, fn)
    assert out[0]["labels"] == ["Sci"]          # deduped after rollup
    assert out[0]["true_label"] == "Sci"
    assert out[0]["task"] == "topic"            # task name untouched


def test_no_transform_is_identity():
    fns = train._category_fns({})
    rec = {"input": "x", "output": {"entities": {"ORG.Media": ["BBC"]}}}
    assert train.transform_record(rec, fns) == rec


def test_schema_format_entities_transformed():
    """The {text, schema} record form is handled too (entities as label->desc)."""
    fns = train._category_fns({"entities": {"rollup": True}})
    rec = {"text": "x", "schema": {"entities": {"ORG.Media": "media", "ORG.Government": "gov"}}}
    out = train.transform_record(rec, fns)
    assert out["schema"]["entities"] == {"ORG": "media"}


def test_transformed_record_roundtrips_through_factory():
    """A transformed dict loads through DataLoader_Factory unchanged (in-memory path)."""
    fns = train._category_fns({"entities": {"rollup": True, "map": {"ORG": "ORGANIZATION"}}})
    rec = {"input": "BBC reported", "output": {"entities": {"ORG.Media": ["BBC"]}}}
    out = train.transform_record(rec, fns)
    loaded = DataLoader_Factory.load([out], shuffle=False)
    assert loaded == [{"input": "BBC reported", "output": {"entities": {"ORGANIZATION": ["BBC"]}}}]


# --- per-category dispatch -------------------------------------------------

def test_category_fns_selects_only_active_categories():
    fns = train._category_fns({
        "entities": {"rollup": True},
        "relations": {"rollup": False, "map": {}},                 # inactive -> skipped
        "events": {"rollup": False, "map": {"Conflict.Attack": "attack"}},  # active via map
    })
    assert set(fns) == {"entities", "events"}


def test_category_fns_rejects_flat_form():
    with pytest.raises(ValueError):
        train._category_fns({"rollup": True, "separator": "."})


def test_entities_active_leaves_events_untouched():
    fns = train._category_fns({"entities": {"rollup": True}})
    rec = {"input": "x", "output": {
        "entities": {"ORG.Media": ["BBC"]},
        "events": [{"event_type": "Conflict.Attack", "triggers": ["t"], "arguments": []}],
    }}
    out = train.transform_record(rec, fns)["output"]
    assert out["entities"] == {"ORG": ["BBC"]}
    assert out["events"][0]["event_type"] == "Conflict.Attack"  # events not configured -> untouched


def test_events_active_leaves_entities_untouched():
    fns = train._category_fns({"events": {"rollup": True}})
    rec = {"input": "x", "output": {
        "entities": {"ORG.Media": ["BBC"]},
        "events": [{"event_type": "Conflict.Attack", "triggers": ["t"],
                    "arguments": [{"role": "Place.City", "entity": "Paris"}]}],
    }}
    out = train.transform_record(rec, fns)["output"]
    assert out["entities"] == {"ORG.Media": ["BBC"]}             # entities not configured -> untouched
    assert out["events"][0]["event_type"] == "Conflict"
    assert out["events"][0]["arguments"][0]["role"] == "Place"  # events covers roles too


def test_per_category_separator_and_map_are_independent():
    fns = train._category_fns({
        "entities": {"rollup": True, "separator": ".", "map": {"ORG": "ORGANIZATION"}},
        "relations": {"rollup": True, "separator": "."},
    })
    rec = {"input": "x", "output": {
        "entities": {"ORG.Media": ["BBC"]},
        "relations": [{"PHYS.Located": {"head": "a", "tail": "b"}}],
    }}
    out = train.transform_record(rec, fns)["output"]
    assert out["entities"] == {"ORGANIZATION": ["BBC"]}
    assert out["relations"] == [{"PHYS": {"head": "a", "tail": "b"}}]


# --- structures: the EKF surface -------------------------------------------

def _casualty_record():
    return {"input": "66 missing in Orleans", "output": {
        "json_structures": [{"casualty_report": {"missing": "66", "location": "Orleans"}}],
        "record_metadata": {"casualty_report": {"mode": "natural", "anchor": "missing"}}}}


def test_structures_renames_name_fields_metadata_and_anchor():
    """record_metadata is keyed by structure name and its anchor names a field.

    A rename that misses either leaves a structure the boundary head decodes to {}
    with no error, so all four sites have to move together.
    """
    fns = train._category_fns({"structures": {"map": {
        "casualty_report": "casualty", "missing": "unaccounted", "location": "place"}}})
    out = train.transform_record(_casualty_record(), fns)["output"]
    assert out["json_structures"] == [{"casualty": {"unaccounted": "66", "place": "Orleans"}}]
    assert out["record_metadata"] == {"casualty": {"mode": "natural", "anchor": "unaccounted"}}


def test_structures_merge_on_collision_keeps_first_nonempty():
    fns = train._category_fns({"structures": {"map": {"dead": "casualties", "killed": "casualties"}}})
    record = {"output": {"json_structures": [{"r": {"dead": "5", "killed": "7"}}]}}
    assert train.transform_record(record, fns)["output"]["json_structures"] == [
        {"r": {"casualties": "5"}}]


def test_structures_rollup_applies_to_names_and_fields():
    fns = train._category_fns({"structures": {"rollup": True, "separator": "."}})
    record = {"output": {"json_structures": [{"report.casualty": {"count.dead": "3"}}]}}
    assert train.transform_record(record, fns)["output"]["json_structures"] == [
        {"report": {"count": "3"}}]


def test_structures_inactive_leaves_record_untouched():
    record = _casualty_record()
    fns = train._category_fns({"entities": {"map": {"a": "b"}}})
    assert train.transform_record(record, fns) == record


def test_structures_is_a_recognised_category():
    assert "structures" in train.LABEL_CATEGORIES


# --- shared labels_file ----------------------------------------------------

def test_labels_file_is_loaded_and_inline_config_wins(tmp_path):
    """Warm-started models share one map file; an inline category overrides it."""
    shared = tmp_path / "unified.yaml"
    shared.write_text(
        "labels:\n"
        "  entities:\n    rollup: false\n    map:\n      'Company Name': CompanyName\n"
        "  events:\n    rollup: false\n    map:\n      death: Death\n",
        encoding="utf-8")
    cfg = {"labels_file": str(shared), "labels": {"events": {"map": {"death": "Fatality"}}}}
    merged = train.load_labels_cfg(cfg, "")
    fns = train._category_fns(merged)
    assert fns["entities"]("Company Name") == "CompanyName"   # from the shared file
    assert fns["events"]("death") == "Fatality"               # inline wins


def test_labels_file_resolves_relative_to_the_config(tmp_path):
    (tmp_path / "labels").mkdir()
    (tmp_path / "labels" / "u.yaml").write_text(
        "labels:\n  entities:\n    map:\n      a: b\n", encoding="utf-8")
    cfg_path = tmp_path / "run.yaml"
    cfg_path.write_text("labels_file: labels/u.yaml\n", encoding="utf-8")
    fns = train._category_fns(train.load_labels_cfg({"labels_file": "labels/u.yaml"}, str(cfg_path)))
    assert fns["entities"]("a") == "b"


def test_no_labels_file_is_unchanged():
    assert train.load_labels_cfg({"labels": {"entities": {"map": {"a": "b"}}}}, "") == {
        "entities": {"map": {"a": "b"}}}
    assert train.load_labels_cfg({}, "") == {}

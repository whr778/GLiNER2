"""Unit tests for the label roll-up / remap transforms in tools/train/train.py.

The transform logic lives in the config-driven trainer; we load it by path
(it is a script, not an installed module). Covers roll-up, map-after-rollup
ordering, and -- critically -- merge-on-collision so no entities are lost when
two child labels roll up to the same parent.
"""

import importlib.util
from pathlib import Path

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
        "trigger": "bombed",
        "arguments": [{"role": "Place.City", "entity": "Paris"}],
    }]
    out = train._transform_events(events, fn)
    assert out[0]["event_type"] == "Conflict"
    assert out[0]["trigger"] == "bombed"
    assert out[0]["arguments"][0] == {"role": "Place", "entity": "Paris"}


def test_classifications_labels_and_true_label_rolled_up():
    fn = train._label_fn(rollup=True, separator=".", mapping={})
    cls = [{"task": "topic", "labels": ["Sci.Physics", "Sci.Math"], "true_label": "Sci.Physics"}]
    out = train._transform_classifications(cls, fn)
    assert out[0]["labels"] == ["Sci"]          # deduped after rollup
    assert out[0]["true_label"] == "Sci"
    assert out[0]["task"] == "topic"            # task name untouched


def test_no_transform_is_identity():
    fn = train._label_fn(rollup=False, separator=".", mapping={})
    rec = {"input": "x", "output": {"entities": {"ORG.Media": ["BBC"]}}}
    assert train.transform_record(rec, fn) == rec


def test_schema_format_entities_transformed():
    """The {text, schema} record form is handled too (entities as label->desc)."""
    fn = train._label_fn(rollup=True, separator=".", mapping={})
    rec = {"text": "x", "schema": {"entities": {"ORG.Media": "media", "ORG.Government": "gov"}}}
    out = train.transform_record(rec, fn)
    assert out["schema"]["entities"] == {"ORG": "media"}


def test_transformed_record_roundtrips_through_factory():
    """A transformed dict loads through DataLoader_Factory unchanged (in-memory path)."""
    fn = train._label_fn(rollup=True, separator=".", mapping={"ORG": "ORGANIZATION"})
    rec = {"input": "BBC reported", "output": {"entities": {"ORG.Media": ["BBC"]}}}
    out = train.transform_record(rec, fn)
    loaded = DataLoader_Factory.load([out], shuffle=False)
    assert loaded == [{"input": "BBC reported", "output": {"entities": {"ORGANIZATION": ["BBC"]}}}]

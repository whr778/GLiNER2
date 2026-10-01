"""apply_label_map replays the training label transform on an inference schema."""

from gliner2.configuration import ExtractorConfig
from gliner2.inference.label_map import apply_label_map, label_fn

LABEL_MAP = {
    "entities": {"rollup": False, "separator": ".", "map": {"person": "Person", "PER": "Person"}},
    "events": {"rollup": False, "separator": ".", "map": {"attack": "Attack", "Origin": "origin"}},
    "relations": {"rollup": True, "separator": ".", "map": {"org": "Organization"}},
    "classifications": {"map": {"pos": "positive"}},
}


def test_entities_list_and_dict_forms_and_report():
    schema, applied = apply_label_map({"entities": ["person", "PER", "date"]}, LABEL_MAP)
    assert schema["entities"] == ["Person", "date"]          # collision deduped, order kept
    assert applied == {"entities": {"person": "Person", "PER": "Person"}}
    schema, _ = apply_label_map({"entities": {"person": "a human"}}, LABEL_MAP)
    assert schema["entities"] == {"Person": "a human"}


def test_events_types_roles_and_rich_config():
    schema, applied = apply_label_map(
        {"events": {"attack": ["Origin", "Place"],
                    "Transport": {"roles": ["Origin"], "role_descriptions": {"Origin": "from"},
                                  "exclusive_roles": ["Origin"]}}}, LABEL_MAP)
    assert schema["events"]["Attack"] == ["origin", "Place"]
    assert schema["events"]["Transport"] == {"roles": ["origin"], "role_descriptions": {"origin": "from"},
                                             "exclusive_roles": ["origin"]}
    assert applied == {"events": {"attack": "Attack", "Origin": "origin"}}


def test_rollup_runs_before_map_and_classification_labels_map():
    assert label_fn(LABEL_MAP["relations"])("org.member_of") == "Organization"
    schema, _ = apply_label_map({"classifications": [{"task": "s", "labels": ["pos", "neg"]}]}, LABEL_MAP)
    assert schema["classifications"][0]["labels"] == ["positive", "neg"]


def test_no_map_and_structures_untouched():
    src = {"entities": ["person"], "structures": {"contact": {"fields": [{"name": "person"}]}}}
    assert apply_label_map(src, None) == (src, {})
    schema, _ = apply_label_map(src, LABEL_MAP)
    assert schema["structures"] == src["structures"]
    assert src["entities"] == ["person"]                     # input not mutated


def test_config_round_trips_label_map_and_inference_defaults(tmp_path):
    cfg = ExtractorConfig(model_name="bert-base-uncased", architecture="boundary",
                          label_map=LABEL_MAP, inference_defaults={"threshold": 0.3, "global_decode": True})
    cfg.save_pretrained(tmp_path)
    back = ExtractorConfig.from_pretrained(tmp_path)
    assert back.label_map == LABEL_MAP
    assert back.inference_defaults == {"threshold": 0.3, "global_decode": True}

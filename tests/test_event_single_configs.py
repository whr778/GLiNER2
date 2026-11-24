"""The events/ single-dataset pairs stay aligned: same data and dose, pretrained is the axis."""

from pathlib import Path

import yaml

from gliner2.configuration import unreachable_config_keys

ROOT = Path(__file__).resolve().parents[1] / "tools/train/config/events"

# Train-split ontology counted 2026-11-24. No relations.
DOSE = {
    "wikievents": {"entities": 17, "events": 48},
    "rams": {"events": 139},
    "casie": {"entities": 21, "events": 5},
    "maven": {"events": 168},
    "cmnee": {"events": 8},
    "duee": {"events": 65},
    "docee": {"entities": 356},
    "docfee": {"entities": 30},
    "chfinann": {"entities": 23},
    "mendeley": {"events": 1},
    "wojood": {"events": 1},
}


def _load(slug, arm):
    return yaml.safe_load((ROOT / f"{slug}-{arm}.yaml").read_text(encoding="utf-8"))


def test_each_event_corpus_has_an_eb23a_arm_and_a_multi_control():
    for slug, dose in DOSE.items():
        eb, mu = _load(slug, "eb23a"), _load(slug, "multi")
        assert unreachable_config_keys(eb["model"]) == []
        assert unreachable_config_keys(mu["model"]) == []
        assert eb["model"]["pretrained"] == "whr778/gliner2-eb23a"
        assert mu["model"]["pretrained"] == "fastino/gliner2.5-multi-v1"
        assert eb["training"]["negative_labels_per_dim"] == dose
        assert mu["training"]["negative_labels_per_dim"] == dose
        assert "relations" not in dose
        assert eb["data"] == mu["data"]
        assert eb["training"]["metric_for_best"] == mu["training"]["metric_for_best"]
        assert eb["training"]["num_epochs"] == mu["training"]["num_epochs"]
        assert eb["labels_file"] == mu["labels_file"] == "labels/unified-full-v4.yaml"
        eb_train = dict(eb["training"])
        mu_train = dict(mu["training"])
        for block in (eb_train, mu_train):
            block.pop("output_dir")
            block.pop("experiment_name")
        assert eb_train == mu_train
        eb_head = dict(eb["model"]["boundary_head"])
        mu_head = dict(mu["model"]["boundary_head"])
        eb_head.pop("record_coref_link_weight")
        mu_head.pop("record_coref_link_weight")
        assert eb_head == mu_head
        assert eb_head["loss_reduction"] == "global"
        assert eb["eval"] == mu["eval"]
        # One trigger per event, so TriggerLink has no positives. Mendeley is the exception.
        eb_w = eb["model"]["boundary_head"]["record_coref_link_weight"]
        mu_w = mu["model"]["boundary_head"]["record_coref_link_weight"]
        if slug == "mendeley":
            assert (eb_w, mu_w) == (0.3, 0.0)
        else:
            assert (eb_w, mu_w) == (0.0, 0.0)


def test_cmnee_withholds_entity_negatives_on_the_offset_train_file():
    data = _load("cmnee", "eb23a")["data"]
    assert data["event_files"]["cmnee"]["train"].endswith("cmnee_ner_off.train.jsonl")
    assert data["partial_annotation"] == {"cmnee_ner_off": ["entities"]}
